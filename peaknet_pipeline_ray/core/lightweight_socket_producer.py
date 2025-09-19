#!/usr/bin/env python3
"""
Lightweight Socket Producer - Optimized for Zero-Gap Pipeline Performance

This producer eliminates pipeline gaps by doing the absolute minimum work:
1. Receive raw bytes from socket
2. Push raw bytes directly to queue
3. Let pipeline actors do HDF5 parsing DURING GPU compute for perfect overlap

Key improvements over socket_hdf5_producer.py:
- NO HDF5 parsing (moved to pipeline actor)
- NO tensor creation (moved to pipeline actor)
- NO Ray object serialization overhead
- Ultra-fast socket→queue flow keeps pipeline fed
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
    """Minimal statistics for lightweight producer."""
    packets_received: int = 0
    bytes_received: int = 0
    batches_generated: int = 0
    connection_retries: int = 0
    queue_errors: int = 0


@ray.remote
class LightweightSocketProducer:
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

    def _create_raw_socket_data(self, raw_bytes: bytes) -> RawSocketData:
        """Create minimal wrapper for raw bytes (ultra-fast).

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

        return RawSocketData(
            raw_bytes=raw_bytes,
            timestamp=time.time(),
            producer_id=self.producer_id,
            batch_id=batch_id
        )

    def _push_to_queue(self, queue_manager: ShardedQueueManager, raw_data: RawSocketData) -> bool:
        """Push raw data to pipeline queue (fast operation).

        Args:
            queue_manager: Queue manager for pipeline
            raw_data: Raw socket data wrapper

        Returns:
            True if successful, False if backpressure
        """
        try:
            # Single Ray object store operation for the entire raw data
            obj_ref = ray.put(raw_data)

            # Push to queue
            success = queue_manager.put(obj_ref)
            if success:
                self.stats.batches_generated += 1
                logging.debug(f"Producer {self.producer_id}: Pushed raw batch {self.batch_counter} ({len(raw_data)} bytes)")
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
        total_batches: int,
        coordinator: ray.ObjectRef,
        progress_interval: int = 10
    ) -> Dict[str, Any]:
        """
        Main streaming loop: ultra-fast socket→queue flow.

        This is the KEY OPTIMIZATION: Minimal work per iteration to maximize
        socket throughput and eliminate pipeline gaps.

        Args:
            queue_manager: Pipeline input queue manager
            total_batches: Maximum number of batches to produce
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
            while self.batch_counter < total_batches:
                # STEP 1: Receive raw bytes (only blocking operation)
                raw_bytes = self._receive_raw_bytes()
                if raw_bytes is None:
                    logging.error(f"Producer {self.producer_id}: Socket connection lost, aborting")
                    return self._get_final_stats(success=False, error="Socket connection lost")

                # STEP 2: Create minimal wrapper (ultra-fast)
                raw_data = self._create_raw_socket_data(raw_bytes)

                # STEP 3: Push to queue (fast Ray operation)
                success = self._push_to_queue(queue_manager, raw_data)
                if not success:
                    # Backpressure - wait briefly and retry
                    time.sleep(0.01)
                    continue

                self.batch_counter += 1

                # Progress logging
                if self.batch_counter % progress_interval == 0:
                    elapsed = time.time() - start_time
                    bytes_per_sec = self.stats.bytes_received / elapsed if elapsed > 0 else 0
                    logging.info(
                        f"Producer {self.producer_id}: {self.batch_counter}/{total_batches} batches, "
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
            'connection_retries': self.stats.connection_retries,
            'queue_errors': self.stats.queue_errors,
            'backpressure_events': 0,  # Expected by pipeline (lightweight producer doesn't track this)
            'producer_type': 'lightweight_socket'
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