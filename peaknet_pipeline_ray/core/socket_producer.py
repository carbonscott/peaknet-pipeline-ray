#!/usr/bin/env python3
"""
Socket Producer - Configurable Parsing for Performance Testing

This producer supports two parsing strategies to compare throughput:

Mode 1 (parse_location="consumer", default): Q1→P parsing
1. Receive raw bytes from socket
2. Push raw bytes directly to queue
3. Let pipeline actors parse DURING GPU compute for perfect overlap

Mode 2 (parse_location="producer"): S→Q1 parsing
1. Receive raw bytes from socket
2. Parse immediately in producer
3. Push parsed tensors to queue

This configurability allows A/B testing to determine optimal parsing location.
"""

import ray
import time
import logging
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from pynng import Pull0

from ..config.schemas import DataSourceConfig
from ..utils.queue import ShardedQueueManager

logging.basicConfig(level=logging.INFO)


@dataclass
class RawSocketData:
    """Lightweight wrapper for raw socket bytes with minimal metadata.

    Used when parse_location="consumer" - stores raw bytes as ObjectRef.
    """
    raw_bytes_ref: ray.ObjectRef  # ObjectRef to raw bytes
    timestamp: float
    producer_id: int
    batch_id: str

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata dict for Q2 output queue compatibility."""
        return {
            'timestamp': self.timestamp,
            'producer_id': self.producer_id,
            'batch_id': self.batch_id
        }


@dataclass
class ParsedSocketData:
    """Wrapper for pre-parsed tensors with ObjectRefs to prevent GC.

    Used when parse_location="producer" - stores parsed tensors as ObjectRefs.
    """
    tensor_refs: List[ray.ObjectRef]  # List of ObjectRefs to individual tensors
    timestamp: float
    producer_id: int
    batch_id: str

    def __len__(self):
        """Return number of tensors."""
        return len(self.tensor_refs)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata dict for Q2 output queue compatibility."""
        return {
            'timestamp': self.timestamp,
            'producer_id': self.producer_id,
            'batch_id': self.batch_id
        }


@dataclass
class SocketProducerStats:
    """Minimal statistics for socket producer."""
    packets_received: int = 0
    bytes_received: int = 0
    batches_generated: int = 0
    connection_retries: int = 0
    queue_errors: int = 0
    backpressure_events: int = 0


@ray.remote
class SocketProducer:
    """
    Ultra-fast socket producer that only handles raw bytes.

    This producer maximizes socket throughput by eliminating all processing
    overhead. HDF5 parsing and tensor creation are deferred to pipeline actors
    where they can overlap with GPU compute.
    """

    def __init__(
        self,
        producer_id: int,
        config: DataSourceConfig,
        socket_address: str,
        deterministic: bool = False
    ):
        """Initialize socket producer.

        Args:
            producer_id: Unique identifier for this producer
            config: Data source configuration with socket details
            socket_address: Socket address to connect to (e.g., "tcp://sdfada012:12321")
            deterministic: Use deterministic batch IDs for testing
        """
        self.producer_id = producer_id
        self.config = config
        self.deterministic = deterministic

        # Socket connection
        self.socket = None
        self.socket_address = socket_address

        # Parsing configuration (extracted from config for convenience)
        self.parse_location = config.parse_location
        self.serialization_format = config.serialization_format
        self.fields = config.fields

        # Statistics tracking
        self.stats = SocketProducerStats()

        # Batch tracking
        self.batch_counter = 0

        logging.info(
            f"SocketProducer {producer_id} initialized: "
            f"socket={self.socket_address}, parse_location={self.parse_location}"
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

    def _parse_raw_bytes(self, raw_bytes: bytes) -> ParsedSocketData:
        """
        Parse raw socket bytes and wrap in ParsedSocketData with ObjectRefs.

        This method executes parsing in the producer (S→Q1 stage) and wraps
        tensors in ObjectRefs to prevent GC during queue backpressure.

        Supports two serialization formats:
        - NumPy .npz format (fast, recommended): Uses np.load() for simple dictionary access
        - HDF5 format (legacy): Uses h5py for hierarchical structure parsing

        Args:
            raw_bytes: Raw bytes from socket containing detector data in (B, C, H, W) format

        Returns:
            ParsedSocketData with tensor ObjectRefs
        """
        from io import BytesIO
        import numpy as np
        import torch

        try:
            if self.serialization_format == "numpy":
                # Fast NumPy .npz parsing (3-7x faster than HDF5)
                arrays = np.load(BytesIO(raw_bytes))

                # Extract detector data using field mapping
                detector_data_key = self.fields.get("detector_data", "data")

                if detector_data_key not in arrays.files:
                    raise ValueError(f"Detector data key '{detector_data_key}' not found in .npz. Available keys: {arrays.files}")

                detector_data = arrays[detector_data_key]
                arrays.close()

            elif self.serialization_format == "hdf5":
                # Legacy HDF5 parsing (slower, for backward compatibility)
                import h5py
                import hdf5plugin

                with h5py.File(BytesIO(raw_bytes), 'r') as h5_file:
                    detector_data_path = self.fields.get("detector_data", "/data/data")

                    if detector_data_path in h5_file:
                        detector_data = h5_file[detector_data_path][:]
                    else:
                        # Try common fallback paths
                        possible_paths = ['/data/data', '/entry/data/detector_data', '/detector_data']
                        detector_data = None
                        for path in possible_paths:
                            if path in h5_file:
                                detector_data = h5_file[path][:]
                                break

                        # Last resort: use first available dataset
                        if detector_data is None:
                            for key in h5_file.keys():
                                if hasattr(h5_file[key], 'shape'):
                                    detector_data = h5_file[key][:]
                                    logging.debug(f"Producer {self.producer_id}: Using fallback field '{key}' for detector data")
                                    break

                        if detector_data is None:
                            raise ValueError(f"Detector data path '{detector_data_path}' not found in HDF5. Available paths: {list(h5_file.keys())}")
            else:
                raise ValueError(f"Unknown serialization format: {self.serialization_format}. Must be 'numpy' or 'hdf5'")

            # Convert to numpy array if needed
            if not isinstance(detector_data, np.ndarray):
                detector_data = np.array(detector_data)

            # Expect preprocessed data in (B, C, H, W) format
            if len(detector_data.shape) != 4:
                raise ValueError(f"Unexpected detector data shape: {detector_data.shape}. Expected (B, C, H, W) format from preprocessed producer")

            # Extract individual (C, H, W) samples from (B, C, H, W) batch
            cpu_tensors = [torch.from_numpy(detector_data[i].astype(np.float32))
                         for i in range(detector_data.shape[0])]

            logging.debug(
                f"Producer {self.producer_id}: Parsed raw {self.serialization_format} data -> {len(cpu_tensors)} tensors, "
                f"each with shape {cpu_tensors[0].shape if cpu_tensors else 'empty'}"
            )

            # Create ObjectRefs for each tensor ONCE
            tensor_refs = [ray.put(tensor) for tensor in cpu_tensors]

            # Create batch ID
            if self.deterministic:
                batch_id = f"producer_{self.producer_id}_batch_{self.batch_counter}"
            else:
                batch_id = f"producer_{self.producer_id}_{uuid.uuid4().hex[:8]}"

            return ParsedSocketData(
                tensor_refs=tensor_refs,
                timestamp=time.time(),
                producer_id=self.producer_id,
                batch_id=batch_id
            )

        except Exception as e:
            logging.error(f"Producer {self.producer_id}: Failed to parse raw bytes: {e}")
            import traceback
            logging.error(traceback.format_exc())
            # Return None to signal parsing failure
            return None

    def _create_raw_socket_data(self, raw_bytes: bytes) -> RawSocketData:
        """Create wrapper with ObjectRef to raw bytes.

        Args:
            raw_bytes: Raw HDF5 data from socket

        Returns:
            RawSocketData wrapper with minimal metadata
        """
        # Create batch ID (minimal overhead)
        if self.deterministic:
            batch_id = f"producer_{self.producer_id}_batch_{self.batch_counter}"
        else:
            batch_id = f"producer_{self.producer_id}_{uuid.uuid4().hex[:8]}"

        # Create ObjectRef ONCE for these bytes (prevents GC during retry)
        raw_bytes_ref = ray.put(raw_bytes)

        return RawSocketData(
            raw_bytes_ref=raw_bytes_ref,
            timestamp=time.time(),
            producer_id=self.producer_id,
            batch_id=batch_id
        )

    def stream_raw_bytes_to_queue(
        self,
        queue_manager: ShardedQueueManager,
        total_batches: Optional[int],
        coordinator: ray.ObjectRef,
        progress_interval: int = 10
    ) -> Dict[str, Any]:
        """
        Main streaming loop: configurable socket→queue flow with optional parsing.

        Supports two parsing strategies (controlled by parse_location config):
        1. "consumer" (Q1→P): Push raw bytes, let actor parse during GPU overlap (default)
        2. "producer" (S→Q1): Parse in producer before queueing for comparison

        Args:
            queue_manager: Pipeline input queue manager
            total_batches: Maximum number of batches to produce (None = stream indefinitely)
            coordinator: Pipeline coordinator for shutdown signaling
            progress_interval: Log progress every N batches

        Returns:
            Dictionary with production statistics
        """
        logging.info(
            f"Producer {self.producer_id}: Starting streaming from {self.socket_address}"
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

                # STEP 2: Process data based on parse_location config
                if self.parse_location == "producer":
                    # S→Q1 parsing: Parse immediately before queueing
                    data = self._parse_raw_bytes(raw_bytes)
                    if data is None:  # None means parsing failed
                        logging.warning(f"Producer {self.producer_id}: Parsing failed for batch {self.batch_counter}, skipping")
                        continue
                else:
                    # Q1→P parsing: Create minimal wrapper, defer parsing to consumer
                    data = self._create_raw_socket_data(raw_bytes)

                # STEP 3: Push to queue directly with backpressure retry (NO extra ray.put()!)
                backoff_delay = 0.001  # Start with 1ms delay
                while True:
                    success = queue_manager.put(data)  # Direct put - data already contains ObjectRefs!
                    if success:
                        self.stats.batches_generated += 1
                        logging.debug(
                            f"Producer {self.producer_id}: Pushed batch {self.batch_counter} "
                            f"({len(data)} {'tensors' if isinstance(data, ParsedSocketData) else 'bytes'})"
                        )
                        break

                    # Queue full - retry with same data (small wrapper stays alive → ObjectRefs preserved)
                    self.stats.backpressure_events += 1
                    time.sleep(min(backoff_delay, 0.01))
                    backoff_delay = min(backoff_delay * 1.5, 0.01)  # Exponential backoff up to 10ms

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

        logging.info(f"Producer {self.producer_id}: Streaming completed")
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
            'connection_retries': self.stats.connection_retries,
            'queue_errors': self.stats.queue_errors,
            'backpressure_events': self.stats.backpressure_events,
            'producer_type': 'socket'
        }


def create_socket_producers(
    num_producers: int,
    config: DataSourceConfig,
    deterministic: bool = False
) -> List[ray.ObjectRef]:
    """Create multiple socket producer actors.

    In socket mode, num_producers parameter is IGNORED - the function always creates
    one producer per socket address (1:1 mapping). This eliminates configuration
    redundancy and prevents mismatches.

    Args:
        num_producers: Ignored in socket mode (kept for API compatibility)
        config: Data source configuration with socket_addresses list
        deterministic: Use deterministic IDs for testing

    Returns:
        List of Ray actor references

    Raises:
        ValueError: If socket_addresses is not provided or empty for socket mode
    """
    if config.socket_addresses is None or len(config.socket_addresses) == 0:
        raise ValueError(
            "socket_addresses must be provided for socket data source. "
            "Example: socket_addresses: [['sdfada012', 12321]]"
        )

    # Socket mode: Always create 1 producer per socket (ignore num_producers parameter)
    num_sockets = len(config.socket_addresses)
    actual_num_producers = num_sockets

    logging.info(
        f"Socket mode: Creating {actual_num_producers} producer(s) "
        f"(1 per socket, num_producers parameter ignored)"
    )

    producers = []
    for i in range(actual_num_producers):
        hostname, port = config.socket_addresses[i]
        socket_address = f"tcp://{hostname}:{port}"

        producer = SocketProducer.remote(
            producer_id=i,
            config=config,
            socket_address=socket_address,
            deterministic=deterministic
        )
        producers.append(producer)
        logging.info(f"Producer {i} → {socket_address}")

    logging.info(f"Created {actual_num_producers} socket producer(s) for {num_sockets} socket(s)")
    return producers


if __name__ == "__main__":
    print("🚀 Socket Producer")
    print("Ultra-fast socket→queue flow for zero-gap pipeline performance")
    print("HDF5 parsing moved to pipeline actors for CPU/GPU overlap")