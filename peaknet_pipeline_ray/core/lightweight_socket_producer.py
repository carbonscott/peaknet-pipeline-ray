#!/usr/bin/env python3
"""
Socket Producer with Integrated HDF5 Parsing - Optimized for GPU Resource Utilization

This producer eliminates GPU actor overhead by performing HDF5 parsing in the producer stage:
1. Receive raw bytes from socket
2. Parse HDF5 data and create PipelineInput objects
3. Push ready-to-use data to queue for GPU actors

Key improvements over previous architecture:
- HDF5 parsing moved from GPU actors to dedicated producer CPU cores
- Eliminates 1-2ms parsing overhead per batch from GPU actors
- Creates PipelineInput objects (same as random data sources)
- Configurable HDF5 field mapping support
- Better resource separation: I/O processing vs pure GPU compute
"""

import ray
import time
import logging
import uuid
import numpy as np
from io import BytesIO
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import h5py
import hdf5plugin  # For compression support
from pynng import Pull0

from ..config.schemas import DataSourceConfig
from ..config.data_structures import PipelineInput
from ..utils.queue import ShardedQueueManager

logging.basicConfig(level=logging.INFO)


@dataclass
class RawSocketData:
    """Lightweight wrapper for raw socket bytes with minimal metadata."""
    raw_bytes: bytes
    timestamp: float
    producer_id: int
    batch_id: str

    def __len__(self):
        """Return size of raw data for monitoring."""
        return len(self.raw_bytes)


@dataclass
class LightweightProducerStats:
    """Statistics for socket producer with HDF5 parsing."""
    packets_received: int = 0
    bytes_received: int = 0
    batches_generated: int = 0
    parsing_errors: int = 0
    connection_retries: int = 0
    queue_errors: int = 0


@ray.remote
class LightweightSocketProducer:
    """
    Socket producer with integrated HDF5 parsing for optimal pipeline performance.

    This producer now performs HDF5 parsing in the producer stage (moved from GPU actors)
    to eliminate the 1-2ms parsing overhead from GPU actors. This improves resource
    utilization by separating I/O processing from GPU compute operations.

    Key features:
    - HDF5 parsing in dedicated producer CPU cores (not GPU-bound cores)
    - Creates ready-to-use PipelineInput objects (same as random data sources)
    - Configurable field mapping for different HDF5 formats
    - Error handling and statistics tracking for parsing operations
    """

    def __init__(
        self,
        producer_id: int,
        config: DataSourceConfig,
        deterministic: bool = False
    ):
        """Initialize lightweight socket producer.

        Args:
            producer_id: Unique identifier for this producer
            config: Data source configuration with socket details
            deterministic: Use deterministic batch IDs for testing
        """
        self.producer_id = producer_id
        self.config = config
        self.deterministic = deterministic

        # Socket connection
        self.socket = None
        self.socket_address = f"tcp://{config.socket_hostname}:{config.socket_port}"

        # Statistics tracking
        self.stats = LightweightProducerStats()

        # Batch tracking
        self.batch_counter = 0

        logging.info(
            f"LightweightSocketProducer {producer_id} initialized: "
            f"socket={self.socket_address}"
        )

    def _connect_socket(self) -> bool:
        """Connect to socket for receiving data."""
        try:
            if self.socket:
                self.socket.close()

            # Use Pull0(listen=address) pattern like the working pull script
            self.socket = Pull0(listen=self.socket_address)

            logging.info(f"Producer {self.producer_id}: Listening on {self.socket_address}")
            return True

        except Exception as e:
            self.stats.connection_retries += 1
            logging.error(f"Producer {self.producer_id}: Failed to connect to {self.socket_address}: {e}")
            return False

    def _receive_raw_bytes(self) -> Optional[bytes]:
        """Receive raw bytes from socket (minimal processing).

        Returns:
            Raw bytes data or None if error
        """
        try:
            # Block until data arrives - this is the ONLY blocking operation
            raw_bytes = self.socket.recv()

            self.stats.packets_received += 1
            self.stats.bytes_received += len(raw_bytes)

            return raw_bytes

        except Exception as e:
            logging.error(f"Producer {self.producer_id}: Socket receive error: {e}")
            return None

    def _create_pipeline_input(self, raw_bytes: bytes) -> Optional[PipelineInput]:
        """Parse HDF5 data and create PipelineInput object.

        This method moves HDF5 parsing from GPU actors to the producer stage
        for better performance and resource utilization.

        Args:
            raw_bytes: Raw HDF5 data from socket

        Returns:
            PipelineInput object with parsed data or None if parsing failed
        """
        try:
            # Parse HDF5 data from raw bytes
            with h5py.File(BytesIO(raw_bytes), 'r') as h5_file:
                extracted_data = {}

                # Extract each configured field
                for field_name, hdf5_path in self.config.fields.items():
                    if hdf5_path in h5_file:
                        try:
                            data = h5_file[hdf5_path][:]
                            extracted_data[field_name] = data
                        except Exception as e:
                            logging.warning(
                                f"Producer {self.producer_id}: Failed to read {hdf5_path}: {e}"
                            )
                    else:
                        logging.debug(f"Producer {self.producer_id}: Field {hdf5_path} not found in HDF5")

                # Validate required fields
                missing_fields = [field for field in self.config.required_fields
                                if field not in extracted_data]
                if missing_fields:
                    logging.error(
                        f"Producer {self.producer_id}: Missing required fields: {missing_fields}"
                    )
                    return None

                # Extract detector data (main image data)
                detector_data = extracted_data.get("detector_data")
                if detector_data is None:
                    logging.error(f"Producer {self.producer_id}: No detector_data found in extracted data")
                    return None

                # Convert to numpy array if needed
                if not isinstance(detector_data, np.ndarray):
                    detector_data = np.array(detector_data)

                # Ensure proper data format
                # Handle different input shapes and convert to expected format
                if len(detector_data.shape) == 2:
                    # Single sample: (H, W) -> add batch and channel dimensions -> (1, 1, H, W)
                    detector_data = detector_data[np.newaxis, np.newaxis, ...]
                elif len(detector_data.shape) == 3:
                    # Could be (B, H, W) or (C, H, W)
                    # Assume it's (B, H, W) and add channel dimension -> (B, 1, H, W)
                    detector_data = detector_data[:, np.newaxis, ...]
                elif len(detector_data.shape) == 4:
                    # Already in (B, C, H, W) format
                    pass
                else:
                    raise ValueError(f"Unexpected detector data shape: {detector_data.shape}")

                # Create metadata from other fields
                metadata = {}
                for field_name, field_value in extracted_data.items():
                    if field_name != "detector_data":
                        # Convert numpy arrays to native Python types for metadata
                        if isinstance(field_value, np.ndarray):
                            if field_value.size == 1:
                                metadata[field_name] = field_value.item()
                            else:
                                metadata[field_name] = field_value.tolist()
                        else:
                            metadata[field_name] = field_value

                # Add producer metadata
                metadata['producer_id'] = self.producer_id
                metadata['reception_timestamp'] = time.time()

                # Create batch ID
                if self.deterministic:
                    batch_id = f"producer_{self.producer_id}_batch_{self.batch_counter}"
                else:
                    batch_id = f"producer_{self.producer_id}_{uuid.uuid4().hex[:8]}"

                # Create PipelineInput using the efficient ObjectRef mode
                return PipelineInput.from_numpy_array(
                    numpy_array=detector_data.astype(np.float32),
                    metadata=metadata,
                    batch_id=batch_id
                )

        except Exception as e:
            self.stats.parsing_errors += 1
            logging.error(f"Producer {self.producer_id}: HDF5 parsing error: {e}")
            return None

    def _push_to_queue(self, queue_manager: ShardedQueueManager, pipeline_input: PipelineInput) -> bool:
        """Push parsed pipeline input to queue.

        Args:
            queue_manager: Queue manager for pipeline
            pipeline_input: Parsed PipelineInput object

        Returns:
            True if successful, False if backpressure
        """
        try:
            # PipelineInput already uses efficient Ray ObjectRef storage internally
            # Just push the PipelineInput object directly to queue
            obj_ref = ray.put(pipeline_input)

            # Push to queue
            success = queue_manager.put(obj_ref)
            if success:
                self.stats.batches_generated += 1
                logging.debug(f"Producer {self.producer_id}: Pushed parsed batch {self.batch_counter}")
                return True
            else:
                logging.warning(f"Producer {self.producer_id}: Queue full for batch {self.batch_counter}")
                return False

        except Exception as e:
            self.stats.queue_errors += 1
            logging.error(f"Producer {self.producer_id}: Failed to push to queue: {e}")
            return False

    def stream_raw_bytes_to_queue(
        self,
        queue_manager: ShardedQueueManager,
        total_batches: Optional[int],
        coordinator: ray.ObjectRef,
        progress_interval: int = 10
    ) -> Dict[str, Any]:
        """
        Main streaming loop: socket→parse→queue flow with HDF5 processing.

        This method now performs HDF5 parsing in the producer stage (moved from GPU actors)
        to eliminate parsing overhead from GPU actors and improve resource utilization.

        Args:
            queue_manager: Pipeline input queue manager
            total_batches: Maximum number of batches to produce (None = stream indefinitely)
            coordinator: Pipeline coordinator for shutdown signaling
            progress_interval: Log progress every N batches

        Returns:
            Dictionary with production statistics
        """
        logging.info(
            f"Producer {self.producer_id}: Starting lightweight streaming from {self.socket_address}"
        )

        # Connect to socket
        if not self._connect_socket():
            return self._get_final_stats(success=False, error="Failed to connect to socket")

        start_time = time.time()

        try:
            # Stream until total_batches reached (finite) or indefinitely (None)
            while total_batches is None or self.batch_counter < total_batches:
                # STEP 1: Receive raw bytes (only blocking operation)
                raw_bytes = self._receive_raw_bytes()
                if raw_bytes is None:
                    # Socket timeout or connection lost - normal termination for streaming
                    logging.info(f"Producer {self.producer_id}: Socket closed/timeout, streaming completed normally")
                    break

                # STEP 2: Parse HDF5 and create PipelineInput (moved from GPU actors)
                pipeline_input = self._create_pipeline_input(raw_bytes)
                if pipeline_input is None:
                    # HDF5 parsing failed - skip this batch and continue
                    logging.warning(f"Producer {self.producer_id}: Skipping batch due to parsing error")
                    continue

                # STEP 3: Push to queue (fast Ray operation)
                success = self._push_to_queue(queue_manager, pipeline_input)
                if not success:
                    # Backpressure - wait briefly and retry
                    time.sleep(0.01)
                    continue

                self.batch_counter += 1

                # Progress logging
                if self.batch_counter % progress_interval == 0:
                    elapsed = time.time() - start_time
                    bytes_per_sec = self.stats.bytes_received / elapsed if elapsed > 0 else 0
                    if total_batches is not None:
                        logging.info(
                            f"Producer {self.producer_id}: {self.batch_counter}/{total_batches} batches, "
                            f"{bytes_per_sec/1024/1024:.1f} MB/s, {self.stats.packets_received} packets"
                        )
                    else:
                        logging.info(
                            f"Producer {self.producer_id}: {self.batch_counter} batches (streaming), "
                            f"{bytes_per_sec/1024/1024:.1f} MB/s, {self.stats.packets_received} packets"
                        )

        except KeyboardInterrupt:
            logging.info(f"Producer {self.producer_id}: Interrupted by user")
        except Exception as e:
            logging.error(f"Producer {self.producer_id}: Unexpected error: {e}")
            return self._get_final_stats(success=False, error=str(e))
        finally:
            # Clean up socket connection
            if self.socket:
                self.socket.close()

        # Register completion with coordinator
        producer_id_str = f"producer_{self.producer_id}"
        ray.get(coordinator.register_producer_finished.remote(producer_id_str))

        logging.info(f"Producer {self.producer_id}: Lightweight streaming completed")
        return self._get_final_stats(success=True)

    def _get_final_stats(self, success: bool, error: Optional[str] = None) -> Dict[str, Any]:
        """Get final production statistics.

        Args:
            success: Whether production completed successfully
            error: Error message if production failed

        Returns:
            Statistics dictionary
        """
        return {
            'producer_id': self.producer_id,
            'success': success,
            'error': error,
            'batches_generated': self.stats.batches_generated,
            'total_samples': self.stats.batches_generated,  # Expected by pipeline
            'packets_received': self.stats.packets_received,
            'bytes_received': self.stats.bytes_received,
            'parsing_errors': self.stats.parsing_errors,
            'connection_retries': self.stats.connection_retries,
            'queue_errors': self.stats.queue_errors,
            'backpressure_events': 0,  # Expected by pipeline (lightweight producer doesn't track this)
            'producer_type': 'lightweight_socket_with_parsing'
        }


def create_lightweight_socket_producers(
    num_producers: int,
    config: DataSourceConfig,
    deterministic: bool = False
) -> List[ray.ObjectRef]:
    """Create multiple lightweight socket producer actors.

    Args:
        num_producers: Number of producer actors to create
        config: Data source configuration
        deterministic: Use deterministic IDs for testing

    Returns:
        List of Ray actor references
    """
    producers = []
    for i in range(num_producers):
        producer = LightweightSocketProducer.remote(
            producer_id=i,
            config=config,
            deterministic=deterministic
        )
        producers.append(producer)

    logging.info(f"Created {num_producers} lightweight socket producers")
    return producers


if __name__ == "__main__":
    print("🚀 Lightweight Socket Producer")
    print("Ultra-fast socket→queue flow for zero-gap pipeline performance")
    print("HDF5 parsing moved to pipeline actors for CPU/GPU overlap")