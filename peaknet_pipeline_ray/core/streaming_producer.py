"""Streaming data producer for continuous pipeline data generation.

This module provides a Ray actor that generates synthetic data and continuously
streams it to the pipeline queues. It supports realistic streaming scenarios
with configurable data rates, batch sizes, and inter-batch delays.

Key features:
1. Continuous data generation with configurable patterns
2. Backpressure handling when queues are full
3. Integration with pipeline coordinator for clean termination
4. Support for both synthetic and real data sources
5. Metadata generation for downstream processing
"""

import ray
import torch
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..config.data_structures import PipelineInput
from ..utils.queue import ShardedQueueManager

logging.basicConfig(level=logging.INFO)


@dataclass
class StreamingStats:
    """Statistics tracked by the streaming producer."""
    batches_generated: int = 0
    total_samples: int = 0
    generation_time: float = 0.0
    queue_put_time: float = 0.0
    backpressure_events: int = 0
    queue_full_time: float = 0.0
    
    @property
    def generation_rate(self) -> float:
        """Samples per second generation rate."""
        if self.generation_time > 0:
            return self.total_samples / self.generation_time
        return 0.0
    
    @property
    def effective_rate(self) -> float:
        """Effective samples per second including queue delays."""
        total_time = self.generation_time + self.queue_put_time + self.queue_full_time
        if total_time > 0:
            return self.total_samples / total_time
        return 0.0


@ray.remote
class StreamingDataProducer:
    """Ray actor that generates streaming data for the pipeline.
    
    This producer generates synthetic image data at a configurable rate and
    pushes it to the pipeline input queue. It handles backpressure gracefully
    and integrates with the coordinator for clean shutdown.
    """
    
    def __init__(
        self,
        producer_id: int,
        batch_size: int = 16,
        tensor_shape: Tuple[int, int, int] = (1, 1920, 1920),
        inter_batch_delay: float = 0.0,
        max_queue_retries: int = 50,
        deterministic: bool = False
    ):
        """Initialize the streaming data producer.
        
        Args:
            producer_id: Unique identifier for this producer
            batch_size: Number of samples per batch
            tensor_shape: Shape of each tensor (C, H, W)
            inter_batch_delay: Delay between batches in seconds
            max_queue_retries: Maximum retries when queue is full
            deterministic: Use deterministic data generation
        """
        self.producer_id = producer_id
        self.batch_size = batch_size
        self.tensor_shape = tensor_shape
        self.inter_batch_delay = inter_batch_delay
        self.max_queue_retries = max_queue_retries
        self.deterministic = deterministic
        
        # Statistics tracking
        self.stats = StreamingStats()
        
        # Set random seed for deterministic generation
        if deterministic:
            np.random.seed(42 + producer_id)
            torch.manual_seed(42 + producer_id)
        
        logging.info(
            f"StreamingDataProducer {producer_id} initialized: "
            f"batch_size={batch_size}, shape={tensor_shape}, "
            f"delay={inter_batch_delay}s, deterministic={deterministic}"
        )
    
    def generate_synthetic_batch(self, batch_idx: int) -> PipelineInput:
        """Generate a synthetic batch with realistic metadata.
        
        Args:
            batch_idx: Index of the batch for tracking
            
        Returns:
            PipelineInput with synthetic data and metadata
        """
        gen_start = time.time()
        
        # Generate synthetic image data
        if self.deterministic:
            # Deterministic data for testing
            batch_data = torch.ones(
                self.batch_size, *self.tensor_shape,
                dtype=torch.float32
            ) * (0.5 + 0.1 * batch_idx)
        else:
            # Random data for realistic simulation
            batch_data = torch.randn(
                self.batch_size, *self.tensor_shape,
                dtype=torch.float32
            )
        
        # Convert to numpy for Ray ObjectRef storage
        batch_numpy = batch_data.numpy()
        
        # Generate realistic metadata
        metadata = {
            'producer_id': self.producer_id,
            'batch_index': batch_idx,
            'timestamp': time.time(),
            'photon_energies': [8000.0 + np.random.randn() * 100 for _ in range(self.batch_size)],
            'run_id': f"run_{self.producer_id}_{batch_idx // 100}",
            'detector_settings': {
                'gain': 'high',
                'exposure_time': 0.001,
                'binning': '1x1'
            },
            'experimental_conditions': {
                'temperature': 295.0 + np.random.randn() * 2.0,
                'pressure': 1013.25 + np.random.randn() * 10.0
            }
        }
        
        batch_id = f"producer_{self.producer_id}_batch_{batch_idx}"
        
        self.stats.generation_time += time.time() - gen_start
        
        # Use ObjectRef mode for optimal performance
        return PipelineInput.from_numpy_array(
            numpy_array=batch_numpy,
            metadata=metadata,
            batch_id=batch_id
        )
    
    def stream_batches_to_queue(
        self,
        q1_manager: ShardedQueueManager,
        total_batches: int,
        coordinator: Optional[ray.actor.ActorHandle] = None,
        progress_interval: int = 50
    ) -> Dict[str, Any]:
        """Main streaming method - generates batches and pushes to queue.
        
        Args:
            q1_manager: Input queue manager to push data to
            total_batches: Total number of batches to generate
            coordinator: Optional coordinator for registration
            progress_interval: Report progress every N batches
            
        Returns:
            Dictionary with production statistics
        """
        start_time = time.time()
        
        logging.info(
            f"Producer {self.producer_id}: Starting to stream {total_batches} batches "
            f"(batch_size={self.batch_size}, total_samples={total_batches * self.batch_size})"
        )
        
        for batch_idx in range(total_batches):
            # Generate batch
            batch_data = self.generate_synthetic_batch(batch_idx)
            
            # Push to queue with backpressure handling
            success = self._push_with_backpressure(q1_manager, batch_data, batch_idx)
            
            if not success:
                logging.error(
                    f"Producer {self.producer_id}: Failed to push batch {batch_idx} "
                    f"after {self.max_queue_retries} retries"
                )
                break
            
            self.stats.batches_generated += 1
            self.stats.total_samples += self.batch_size
            
            # Progress reporting
            if progress_interval > 0 and (batch_idx + 1) % progress_interval == 0:
                elapsed = time.time() - start_time
                rate = self.stats.total_samples / elapsed if elapsed > 0 else 0
                logging.info(
                    f"Producer {self.producer_id}: Generated {batch_idx + 1}/{total_batches} "
                    f"batches ({self.stats.total_samples} samples) at {rate:.1f} samples/s"
                )
            
            # Inter-batch delay
            if self.inter_batch_delay > 0:
                time.sleep(self.inter_batch_delay)
        
        total_time = time.time() - start_time
        
        # Register completion with coordinator
        if coordinator is not None:
            producer_id = f"producer_{self.producer_id}"
            try:
                ray.get(coordinator.register_producer_finished.remote(producer_id))
                logging.info(f"Producer {self.producer_id}: Registered completion with coordinator")
            except Exception as e:
                logging.warning(f"Producer {self.producer_id}: Failed to register completion: {e}")
        
        # Final statistics
        final_stats = {
            'producer_id': self.producer_id,
            'batches_generated': self.stats.batches_generated,
            'total_samples': self.stats.total_samples,
            'total_time': total_time,
            'generation_rate': self.stats.generation_rate,
            'effective_rate': self.stats.effective_rate,
            'backpressure_events': self.stats.backpressure_events,
            'queue_full_time': self.stats.queue_full_time,
            'avg_batch_time': total_time / max(self.stats.batches_generated, 1)
        }
        
        logging.info(
            f"Producer {self.producer_id} completed: "
            f"{final_stats['batches_generated']} batches, "
            f"{final_stats['total_samples']} samples, "
            f"{final_stats['effective_rate']:.1f} samples/s effective rate"
        )
        
        return final_stats
    
    def _push_with_backpressure(
        self,
        q1_manager: ShardedQueueManager,
        batch_data: PipelineInput,
        batch_idx: int
    ) -> bool:
        """Push batch to queue with exponential backoff on backpressure.
        
        Args:
            q1_manager: Queue manager to push to
            batch_data: The batch data to push
            batch_idx: Batch index for logging
            
        Returns:
            True if successful, False if all retries exhausted
        """
        put_start = time.time()
        retry_count = 0
        backoff_delay = 0.001  # Start with 1ms
        
        while retry_count < self.max_queue_retries:
            success = q1_manager.put(batch_data)
            
            if success:
                self.stats.queue_put_time += time.time() - put_start
                return True
            
            # Backpressure event
            self.stats.backpressure_events += 1
            retry_count += 1
            
            # Exponential backoff (cap at 100ms)
            time.sleep(min(backoff_delay, 0.1))
            backoff_delay *= 1.5
            
            # Track time spent waiting for queue space
            self.stats.queue_full_time += backoff_delay
            
            if retry_count % 10 == 0:
                logging.warning(
                    f"Producer {self.producer_id}: Batch {batch_idx} "
                    f"queue full, retry {retry_count}/{self.max_queue_retries}"
                )
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current producer statistics.
        
        Returns:
            Dictionary with current statistics
        """
        return {
            'producer_id': self.producer_id,
            'batches_generated': self.stats.batches_generated,
            'total_samples': self.stats.total_samples,
            'generation_rate': self.stats.generation_rate,
            'effective_rate': self.stats.effective_rate,
            'backpressure_events': self.stats.backpressure_events,
            'queue_full_time': self.stats.queue_full_time
        }


def create_streaming_producers(
    num_producers: int,
    batch_size: int = 16,
    tensor_shape: Tuple[int, int, int] = (1, 1920, 1920),
    inter_batch_delay: float = 0.0,
    deterministic: bool = False
) -> list:
    """Convenience function to create multiple streaming producers.
    
    Args:
        num_producers: Number of producer actors to create
        batch_size: Batch size for each producer
        tensor_shape: Tensor shape for generated data
        inter_batch_delay: Delay between batches
        deterministic: Use deterministic generation
        
    Returns:
        List of Ray actor handles
    """
    producers = []
    for i in range(num_producers):
        producer = StreamingDataProducer.remote(
            producer_id=i,
            batch_size=batch_size,
            tensor_shape=tensor_shape,
            inter_batch_delay=inter_batch_delay,
            deterministic=deterministic
        )
        producers.append(producer)
    
    logging.info(f"Created {num_producers} streaming data producers")
    return producers


if __name__ == "__main__":
    # Simple test of the streaming producer
    if not ray.is_initialized():
        ray.init()
    
    from ..utils.queue import ShardedQueueManager
    
    print("Testing StreamingDataProducer...")
    
    # Create queue and producer
    queue_manager = ShardedQueueManager("test_queue", num_shards=2, maxsize_per_shard=10)
    producer = StreamingDataProducer.remote(
        producer_id=0,
        batch_size=4,
        tensor_shape=(1, 64, 64),
        deterministic=True
    )
    
    # Stream some data
    stats = ray.get(producer.stream_batches_to_queue.remote(
        queue_manager, total_batches=5, progress_interval=2
    ))
    
    print(f"Producer stats: {stats}")
    print(f"Queue size after production: {queue_manager.size()}")
    
    print("✅ StreamingDataProducer test passed!")