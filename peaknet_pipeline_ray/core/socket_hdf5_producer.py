#!/usr/bin/env python3
"""
Socket HDF5 Data Producer for PeakNet Pipeline.

This module provides a Ray actor that connects to a socket data stream (from LCLStreamer),
receives HDF5 binary data, extracts detector data and metadata according to configuration,
and feeds PipelineInput objects to the pipeline queue system.

Key features:
1. Configurable HDF5 field mapping to match LCLStreamer output
2. Batch assembly from individual socket messages
3. Robust error handling and connection retry
4. Integration with pipeline coordinator for clean shutdown
5. Statistics tracking for monitoring
"""

import ray
import torch
import numpy as np
import time
import logging
import uuid
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import h5py
import hdf5plugin  # For compression support
from pynng import Pull0

from ..config.data_structures import PipelineInput
from ..config.schemas import DataSourceConfig
from ..utils.queue import ShardedQueueManager

logging.basicConfig(level=logging.INFO)


@dataclass
class SocketProducerStats:
    """Statistics tracked by the socket producer."""
    packets_received: int = 0
    batches_generated: int = 0
    total_samples: int = 0
    connection_retries: int = 0
    parsing_errors: int = 0
    backpressure_events: int = 0


@ray.remote
class SocketHDF5Producer:
    """Ray actor that receives HDF5 data from socket and feeds pipeline.

    This producer connects to a socket (typically from LCLStreamer), receives
    HDF5 binary data, extracts image data and metadata based on configuration,
    and creates PipelineInput objects for the pipeline.
    """

    def __init__(
        self,
        producer_id: int,
        config: DataSourceConfig,
        batch_size: int = 16,
        deterministic: bool = False
    ):
        """Initialize the socket HDF5 producer.

        Args:
            producer_id: Unique identifier for this producer
            config: Data source configuration with socket and field mapping
            batch_size: Number of samples per batch to create
            deterministic: Use deterministic batch IDs for testing
        """
        self.producer_id = producer_id
        self.config = config
        self.batch_size = batch_size
        self.deterministic = deterministic

        # Statistics tracking
        self.stats = SocketProducerStats()

        # Socket connection
        self.socket = None
        self.socket_address = f"tcp://{config.socket_hostname}:{config.socket_port}"

        # Batch assembly
        self.current_batch = []
        self.batch_counter = 0

        logging.info(
            f"SocketHDF5Producer {producer_id} initialized: "
            f"socket={self.socket_address}, batch_size={batch_size}, "
            f"fields={list(config.fields.keys())}"
        )

    def _connect_socket(self) -> bool:
        """Listen on the socket for Push connections from LCLStreamer.

        Returns:
            True if socket bind successful, False otherwise
        """
        try:
            if self.socket:
                self.socket.close()

            # Use Pull0(listen=address) pattern like the working pull script
            self.socket = Pull0(listen=self.socket_address)

            logging.info(f"Producer {self.producer_id}: Listening on {self.socket_address} for LCLStreamer connections")
            return True

        except Exception as e:
            self.stats.connection_retries += 1
            logging.error(f"Producer {self.producer_id}: Failed to listen on {self.socket_address}: {e}")
            return False

    def _receive_socket_data(self, blocking: bool = True) -> Optional[bytes]:
        """Receive data from socket.

        Args:
            blocking: If True, block indefinitely until data arrives.
                     If False, use configured timeout.

        Returns:
            Raw bytes data or None if timeout/error (only when not blocking)
        """
        try:
            if blocking:
                # Block indefinitely until data arrives (no timeout)
                self.socket.recv_timeout = -1  # Infinite timeout
            else:
                # Use configured timeout
                self.socket.recv_timeout = int(self.config.socket_timeout * 1000)  # Convert to ms

            data = self.socket.recv()
            self.stats.packets_received += 1
            return data
        except Exception as e:
            if blocking:
                # In blocking mode, this is a real error (connection lost, etc.)
                logging.error(f"Producer {self.producer_id}: Socket connection error: {e}")
            else:
                # In non-blocking mode, timeout is expected
                logging.debug(f"Producer {self.producer_id}: Socket receive timeout: {e}")
            return None

    def _parse_hdf5_data(self, raw_data: bytes) -> Optional[Dict[str, Any]]:
        """Parse HDF5 data and extract configured fields.

        Args:
            raw_data: Raw HDF5 binary data from socket

        Returns:
            Dictionary with extracted fields or None if parsing failed
        """
        try:
            # Create HDF5 file from bytes
            with h5py.File(BytesIO(raw_data), 'r') as h5_file:
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

                return extracted_data

        except Exception as e:
            self.stats.parsing_errors += 1
            logging.error(f"Producer {self.producer_id}: HDF5 parsing error: {e}")
            return None

    def _parse_hdf5_batch(self, raw_data: bytes) -> Optional[Dict[str, Any]]:
        """Parse HDF5 data as a complete batch (supports variable batch sizes).

        This method handles both single-sample and multi-sample HDF5 messages.
        It trusts the pusher to provide appropriately batched data.

        Args:
            raw_data: Raw HDF5 binary data from socket

        Returns:
            Dictionary with extracted batch data or None if parsing failed
        """
        try:
            # Create HDF5 file from bytes
            with h5py.File(BytesIO(raw_data), 'r') as h5_file:
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

                return extracted_data

        except Exception as e:
            self.stats.parsing_errors += 1
            logging.error(f"Producer {self.producer_id}: HDF5 batch parsing error: {e}")
            return None

    def _create_pipeline_input(self, data: Dict[str, Any], sample_idx: int) -> PipelineInput:
        """Create PipelineInput object from extracted data.

        Args:
            data: Extracted data from HDF5
            sample_idx: Index within current batch

        Returns:
            PipelineInput object ready for pipeline
        """
        # Extract image data (detector_data is the main field)
        image_data = data.get("detector_data")
        if image_data is None:
            raise ValueError("No detector_data found in extracted data")

        # Convert to torch tensor if needed
        if isinstance(image_data, np.ndarray):
            # Ensure proper shape (add channel dimension if needed)
            if len(image_data.shape) == 2:
                image_data = image_data[np.newaxis, ...]  # Add channel dimension
            image_tensor = torch.from_numpy(image_data.astype(np.float32))
        else:
            image_tensor = torch.tensor(image_data, dtype=torch.float32)

        # Create metadata from other fields
        metadata = {}
        for field_name, field_value in data.items():
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
        metadata['sample_idx'] = sample_idx
        metadata['reception_timestamp'] = time.time()

        # Create batch ID
        if self.deterministic:
            batch_id = f"producer_{self.producer_id}_batch_{self.batch_counter}_sample_{sample_idx}"
        else:
            batch_id = f"producer_{self.producer_id}_{uuid.uuid4().hex[:8]}_sample_{sample_idx}"

        return PipelineInput(
            image_data=image_tensor,
            metadata=metadata,
            batch_id=batch_id
        )

    def _create_pipeline_batch(self, batch_data: Dict[str, Any]) -> List[PipelineInput]:
        """Create batch of PipelineInput objects from HDF5 batch data.

        This method handles variable batch sizes - the batch size is determined
        by the shape of the detector_data field from the pusher.

        Args:
            batch_data: Extracted batch data from HDF5

        Returns:
            List of PipelineInput objects (one per sample in the batch)
        """
        # Extract image data (detector_data is the main field)
        image_data = batch_data.get("detector_data")
        if image_data is None:
            raise ValueError("No detector_data found in extracted batch data")

        # Convert to numpy array if needed
        if not isinstance(image_data, np.ndarray):
            image_data = np.array(image_data)

        # Determine batch size from the data shape
        # Support both single sample: (H, W) and batched samples: (B, H, W)
        if len(image_data.shape) == 2:
            # Single sample: (H, W) - treat as batch of 1
            batch_size = 1
            samples = [image_data]
        elif len(image_data.shape) == 3:
            # Multiple samples: (B, H, W) - trust the pusher's batch size
            batch_size = image_data.shape[0]
            samples = [image_data[i] for i in range(batch_size)]
        else:
            raise ValueError(f"Unexpected detector_data shape: {image_data.shape}")

        # Create PipelineInput for each sample in the batch
        pipeline_inputs = []
        for sample_idx, sample_image in enumerate(samples):
            # Convert to torch tensor
            if len(sample_image.shape) == 2:
                sample_image = sample_image[np.newaxis, ...]  # Add channel dimension
            image_tensor = torch.from_numpy(sample_image.astype(np.float32))

            # Create metadata for this sample
            metadata = {}
            for field_name, field_value in batch_data.items():
                if field_name != "detector_data":
                    # Handle both scalar and array metadata
                    if isinstance(field_value, np.ndarray):
                        if field_value.size == 1:
                            metadata[field_name] = field_value.item()
                        elif len(field_value.shape) == 1 and field_value.shape[0] == batch_size:
                            # Per-sample metadata array
                            metadata[field_name] = field_value[sample_idx].item() if field_value[sample_idx].size == 1 else field_value[sample_idx].tolist()
                        else:
                            # Shared metadata for all samples
                            metadata[field_name] = field_value.tolist()
                    else:
                        metadata[field_name] = field_value

            # Add producer metadata
            metadata['producer_id'] = self.producer_id
            metadata['sample_idx'] = sample_idx
            metadata['batch_size'] = batch_size
            metadata['reception_timestamp'] = time.time()

            # Create batch ID
            if self.deterministic:
                batch_id = f"producer_{self.producer_id}_batch_{self.batch_counter}_sample_{sample_idx}"
            else:
                batch_id = f"producer_{self.producer_id}_{uuid.uuid4().hex[:8]}_sample_{sample_idx}"

            pipeline_inputs.append(PipelineInput(
                image_data=image_tensor,
                metadata=metadata,
                batch_id=batch_id
            ))

        return pipeline_inputs

    def _push_batch_to_queue(self, queue_manager: ShardedQueueManager, batch: List[PipelineInput]) -> bool:
        """Push completed batch to pipeline queue.

        Args:
            queue_manager: Queue manager for pipeline
            batch: List of PipelineInput objects

        Returns:
            True if successful, False if backpressure
        """
        try:
            # Convert to Ray object references for efficient storage and push each item individually
            items_pushed = 0
            for pipeline_input in batch:
                obj_ref = ray.put(pipeline_input)

                # Push individual item to queue using the correct put() method
                success = queue_manager.put(obj_ref)
                if success:
                    items_pushed += 1
                else:
                    # If any item fails to push (queue full), this is backpressure
                    self.stats.backpressure_events += 1
                    logging.warning(
                        f"Producer {self.producer_id}: Queue full after pushing {items_pushed}/{len(batch)} items"
                    )
                    return False

            # All items successfully pushed
            self.stats.batches_generated += 1
            self.stats.total_samples += len(batch)
            logging.info(
                f"Producer {self.producer_id}: Pushed batch {self.batch_counter} "
                f"({len(batch)} samples) to queue"
            )
            return True

        except Exception as e:
            logging.error(f"Producer {self.producer_id}: Failed to push batch to queue: {e}")
            return False

    def stream_batches_to_queue(
        self,
        queue_manager: ShardedQueueManager,
        total_batches: int,
        coordinator: ray.ObjectRef,
        progress_interval: int = 10
    ) -> Dict[str, Any]:
        """Main streaming loop: receive data and create batches.

        Args:
            queue_manager: Pipeline input queue manager
            total_batches: Maximum number of batches to produce
            coordinator: Pipeline coordinator for shutdown signaling
            progress_interval: Log progress every N batches

        Returns:
            Dictionary with production statistics
        """
        logging.info(
            f"Producer {self.producer_id}: Starting to stream {total_batches} batches "
            f"from {self.socket_address}"
        )

        # Connect to socket
        if not self._connect_socket():
            return self._get_final_stats(success=False, error="Failed to connect to socket")

        start_time = time.time()

        try:
            while self.batch_counter < total_batches:
                # Direct batch pass-through: receive one HDF5 message = one batch
                # Trust the pusher to send appropriately sized batches

                # Receive data from socket (blocking until data arrives)
                raw_data = self._receive_socket_data(blocking=True)
                if raw_data is None:
                    # This is a real error (connection lost, etc.) - abort
                    logging.error(f"Producer {self.producer_id}: Socket connection lost, aborting")
                    return self._get_final_stats(success=False, error="Socket connection lost")

                # Parse HDF5 data to extract the complete batch
                batch_data = self._parse_hdf5_batch(raw_data)
                if batch_data is None:
                    continue  # Skip malformed data

                # Create batch of PipelineInputs from the HDF5 batch
                try:
                    pipeline_batch = self._create_pipeline_batch(batch_data)
                    if not pipeline_batch:
                        continue  # Skip empty batches
                except Exception as e:
                    logging.error(f"Producer {self.producer_id}: Failed to create pipeline batch: {e}")
                    continue

                # Push the complete batch to queue
                success = self._push_batch_to_queue(queue_manager, pipeline_batch)
                if not success:
                    # Backpressure - wait briefly and retry
                    time.sleep(0.1)
                    continue

                self.batch_counter += 1
                logging.info(f"Producer {self.producer_id}: Pushed batch {self.batch_counter} ({len(pipeline_batch)} samples) to queue")

                # Progress logging
                if self.batch_counter % progress_interval == 0:
                    elapsed = time.time() - start_time
                    rate = self.stats.total_samples / elapsed if elapsed > 0 else 0
                    logging.info(
                        f"Producer {self.producer_id}: {self.batch_counter}/{total_batches} batches, "
                        f"{self.stats.total_samples} samples, {rate:.1f} samples/s"
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

        logging.info(f"Producer {self.producer_id}: Completed streaming {self.batch_counter} batches")
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
            'total_samples': self.stats.total_samples,
            'packets_received': self.stats.packets_received,
            'connection_retries': self.stats.connection_retries,
            'parsing_errors': self.stats.parsing_errors,
            'backpressure_events': self.stats.backpressure_events
        }


def create_socket_hdf5_producers(
    num_producers: int,
    config: DataSourceConfig,
    batch_size: int = 16,
    deterministic: bool = False
) -> List[ray.ObjectRef]:
    """Create multiple socket HDF5 producer actors.

    Args:
        num_producers: Number of producer actors to create
        config: Data source configuration
        batch_size: Batch size for each producer
        deterministic: Use deterministic IDs for testing

    Returns:
        List of Ray actor references
    """
    producers = []
    for i in range(num_producers):
        producer = SocketHDF5Producer.remote(
            producer_id=i,
            config=config,
            batch_size=batch_size,
            deterministic=deterministic
        )
        producers.append(producer)

    logging.info(f"Created {num_producers} socket HDF5 producers")
    return producers