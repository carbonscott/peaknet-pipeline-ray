# Streaming Queue Pipeline Implementation Plan

## Current State Analysis

We are starting from the `v0.1-no-queue-version` tag, which contains:

### ✅ Working Components
1. **Double Buffered Pipeline**: The `DoubleBufferedPipeline` class with proper overlapping H2D/Compute/D2H operations
2. **Pipeline Actor**: `PeakNetPipelineActor` with `process_batch_list()` method that:
   - Processes multiple batches WITHOUT per-batch synchronization
   - Only syncs once at the very end (`pipeline.wait_for_completion()`)
   - Preserves double buffering overlapping operations
3. **Event-based synchronization**: Uses CUDA events for fine-grained pipeline synchronization
4. **Ray integration**: Proper Ray actor with GPU assignment

### 🚫 Missing Components  
1. **Queue infrastructure**: No streaming data input/output queues
2. **Producer/Consumer pattern**: No continuous data flow
3. **Coordination mechanism**: No intelligent termination logic
4. **Streaming processing**: Only batch-list processing (finite batches)

## Root Cause of Previous Issues

The broken implementation had per-batch synchronization (`pipeline.wait_for_completion()` after EACH batch), which completely defeated double buffering and caused `cudaStreamSynchronize` blocking visible in profiling.

## Implementation Strategy

### Phase 1: Add Queue Infrastructure
1. **Create Queue Classes**: Implement Ray-based queues for streaming data
2. **Batch Data Structures**: Define structures for queue items (BatchData, BatchOutput) 
3. **Test Queue Operations**: Verify queue put/get performance

### Phase 2: Add Streaming Producers
1. **Streaming Data Producer**: Actor that generates data and pushes to Q1
2. **Batch Assembly**: Convert streaming items into batches compatible with existing pipeline
3. **Backpressure Handling**: Producer should handle queue full scenarios

### Phase 3: Convert Pipeline Actor to Streaming
1. **New Method**: `process_from_queue()` that pulls from Q1, processes, pushes to Q2
2. **Preserve Double Buffering**: Use the EXACT same approach as `process_batch_list()`:
   - Accumulate multiple batches before processing
   - Process accumulated batches with NO per-batch sync
   - Only sync at the end of each batch group
3. **Streaming Loop**: Convert from finite batch processing to continuous streaming

### Phase 4: Add Query-Based Coordination
1. **Coordinator Actor**: Lightweight Ray actor for termination coordination
2. **Producer Registration**: Producers register completion 
3. **Intelligent Actor Queries**: Actors query coordinator when queues empty
4. **Natural Termination**: No timeout-based or poison pill approaches

## Key Design Principles

### 1. Preserve Working Double Buffering
```python
# GOOD - Original working approach
for batch in batch_accumulator:
    if batch_idx > 0:
        pipeline.swap()
    pipeline.process_batch(...)  # NO SYNC HERE!

# Only sync once for entire group
pipeline.wait_for_completion()
```

### 2. Batch Accumulation for Streaming
```python
# Convert streaming to batch groups for double buffering
accumulator = []
while streaming:
    # Collect N batches
    for _ in range(batch_group_size):
        batch = queue.get_nowait()
        if batch: accumulator.append(batch)
    
    # Process entire group with original double buffering
    if accumulator:
        process_batch_group(accumulator)  # No per-batch sync!
        accumulator.clear()
```

### 3. Smart Termination  
```python
# Only query coordinator when queue is empty
batch = q1.get(timeout=short)
if batch is None:
    # Queue empty - intelligent decision
    should_stop = coordinator.should_actor_shutdown(queue_empty=True)
    if should_stop:
        break  # All producers done
    else:
        continue  # Keep waiting, producers still working
```

## Detailed Technical Design

### 1. Queue Infrastructure (`peaknet_pipeline_ray/utils/queue.py`)

#### Queue Architecture
```python
@ray.remote
class QueueActor:
    """Ray actor implementing a thread-safe queue with backpressure."""
    def __init__(self, maxsize: int = 100):
        self.queue = queue.Queue(maxsize=maxsize)
        self.closed = False
    
    def put(self, item, timeout: float = None) -> bool:
        """Non-blocking put with timeout. Returns success/failure."""
        try:
            self.queue.put(item, timeout=timeout)
            return True
        except queue.Full:
            return False
    
    def get(self, timeout: float = None):
        """Get item with timeout. Returns None if timeout/empty."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def size(self) -> int:
        return self.queue.qsize()

class ShardedQueueManager:
    """Client-side manager for multiple queue shards with load balancing."""
    def __init__(self, name: str, num_shards: int = 4, maxsize_per_shard: int = 25):
        self.shards = [QueueActor.remote(maxsize_per_shard) for _ in range(num_shards)]
        self.name = name
        self.put_counter = 0
    
    def put(self, item, timeout: float = 1.0) -> bool:
        """Round-robin put across shards."""
        shard_idx = self.put_counter % len(self.shards)
        self.put_counter += 1
        return ray.get(self.shards[shard_idx].put.remote(item, timeout))
    
    def get(self, timeout: float = 1.0):
        """Try all shards for get (random order to avoid hot spots)."""
        import random
        shard_indices = list(range(len(self.shards)))
        random.shuffle(shard_indices)
        
        for shard_idx in shard_indices:
            item = ray.get(self.shards[shard_idx].get.remote(timeout=0.1))
            if item is not None:
                return item
        return None
```

#### Data Structures (`peaknet_pipeline_ray/config/batch_structures.py`)
```python
from dataclasses import dataclass
from typing import List
import ray

@dataclass
class BatchData:
    """Input batch data structure for Q1."""
    tensor_ref: ray.ObjectRef  # Ray reference to batch tensor (B, C, H, W)
    photon_energies: List[float]  # Per-sample photon energies
    batch_id: str  # Unique identifier
    batch_size: int  # Actual number of samples in batch
    metadata: dict = None  # Optional additional metadata

@dataclass  
class BatchOutput:
    """Output batch data structure for Q2."""
    predictions_ref: ray.ObjectRef  # Ray reference to predictions tensor
    photon_energies: List[float]  # Same as input (pass-through)
    batch_id: str  # Same as input batch ID
    processing_stats: dict = None  # Optional processing statistics
```

### 2. Coordination Mechanism (`peaknet_pipeline_ray/core/coordinator.py`)

#### Coordinator State Machine
```python
@ray.remote(num_cpus=0)  # Lightweight coordination actor
class PipelineCoordinator:
    """Centralized coordinator for intelligent pipeline termination."""
    
    def __init__(self):
        self.producers_finished = set()  # Set of finished producer IDs
        self.actors_ready = set()  # Set of actors ready for shutdown
        self.expected_producers = 0
        self.expected_actors = 0
        self.state = "INITIALIZED"  # INITIALIZED -> WAITING -> PRODUCERS_FINISHED -> COMPLETED
    
    def set_expected_counts(self, producers: int, actors: int):
        """Set expected number of producers and actors."""
        self.expected_producers = producers
        self.expected_actors = actors  
        self.state = "WAITING"
        logging.info(f"Coordinator expects {producers} producers, {actors} actors")
    
    def register_producer_finished(self, producer_id: str):
        """Producer calls this when it finishes generating data."""
        self.producers_finished.add(producer_id)
        logging.info(f"📊 Producer {producer_id} finished ({len(self.producers_finished)}/{self.expected_producers})")
        
        if len(self.producers_finished) == self.expected_producers:
            self.state = "PRODUCERS_FINISHED"
            logging.info("🎉 All producers finished!")
    
    def should_actor_shutdown(self, queue_empty: bool = True) -> bool:
        """CORE METHOD: Actor queries this when queue is empty."""
        if not queue_empty:
            return False  # Queue not empty, keep processing
            
        if self.state != "PRODUCERS_FINISHED":
            return False  # Producers still working, keep waiting
            
        # All producers done AND queue empty = safe to shutdown
        return True
    
    def register_ready_for_shutdown(self, actor_id: str):
        """Actor calls this when ready to shutdown."""
        self.actors_ready.add(actor_id)
        logging.info(f"🎭 Actor {actor_id} ready for shutdown ({len(self.actors_ready)}/{self.expected_actors})")
        
        if len(self.actors_ready) == self.expected_actors:
            self.state = "COMPLETED" 
            logging.info("🏁 All actors ready - pipeline complete!")
    
    def get_state(self) -> dict:
        """Get current coordinator state for debugging."""
        return {
            "state": self.state,
            "producers_finished": len(self.producers_finished),
            "expected_producers": self.expected_producers,
            "actors_ready": len(self.actors_ready),
            "expected_actors": self.expected_actors,
            "all_producers_finished": len(self.producers_finished) == self.expected_producers
        }
```

### 3. Streaming Producer (`peaknet_pipeline_ray/core/streaming_producer.py`)

#### Producer Implementation
```python
@ray.remote
class StreamingDataProducer:
    """Generates streaming data and pushes to Q1."""
    
    def __init__(self, producer_id: int, batch_size: int = 16, shape: tuple = (1, 1920, 1920)):
        self.producer_id = producer_id
        self.batch_size = batch_size
        self.shape = shape
        self.batch_counter = 0
    
    def stream_batches_to_queue(self, q1_manager, total_batches: int, coordinator=None):
        """Main streaming method - generates batches and pushes to Q1."""
        logging.info(f"Producer {self.producer_id}: Starting to stream {total_batches} batches")
        
        for batch_idx in range(total_batches):
            # Generate batch tensor (B, C, H, W)
            batch_tensor = torch.randn(self.batch_size, *self.shape, dtype=torch.float32)
            batch_ref = ray.put(batch_tensor.numpy())  # Store in Ray object store
            
            # Create batch data
            batch_data = BatchData(
                tensor_ref=batch_ref,
                photon_energies=[8000.0 + np.random.randn() * 100 for _ in range(self.batch_size)],
                batch_id=f"producer_{self.producer_id}_batch_{batch_idx}",
                batch_size=self.batch_size
            )
            
            # Push to Q1 with backpressure handling
            success = False
            retries = 0
            while not success and retries < 50:
                success = q1_manager.put(batch_data, timeout=0.1)
                if not success:
                    retries += 1
                    time.sleep(0.1)  # Backpressure delay
            
            if not success:
                logging.error(f"Producer {self.producer_id}: Failed to push batch {batch_idx}")
                break
                
            self.batch_counter += 1
        
        # Register completion with coordinator
        if coordinator is not None:
            producer_id = f"producer_{self.producer_id}"
            ray.get(coordinator.register_producer_finished.remote(producer_id))
            logging.info(f"Producer {self.producer_id}: Registered completion, exiting")
        
        return self.batch_counter
```

### 4. Streaming Pipeline Actor Integration

#### Core Streaming Method
```python
def process_from_queue(
    self, 
    q1_manager: "ShardedQueueManager", 
    q2_manager: "ShardedQueueManager",
    coordinator=None,
    batch_group_size: int = 3
) -> Dict[str, Any]:
    """Stream processing using batch accumulation with preserved double buffering."""
    
    logging.info(f"Actor {self.gpu_id}: Starting streaming with batch groups of {batch_group_size}")
    
    all_results = []
    total_batches_processed = 0
    first_batch_seen = False
    
    while True:
        # PHASE 1: Accumulate batch group
        batch_group = []
        for _ in range(batch_group_size):
            batch_data = q1_manager.get(timeout=0.5)
            if batch_data is None:
                break  # No more data available right now
            batch_group.append(batch_data)
            first_batch_seen = True
        
        # PHASE 2: Handle empty queue with coordination
        if len(batch_group) == 0:
            if coordinator is not None:
                should_shutdown = ray.get(coordinator.should_actor_shutdown.remote(queue_empty=True))
                if should_shutdown:
                    logging.info(f"Actor {self.gpu_id}: Coordinator confirms shutdown")
                    break
                else:
                    continue  # Producers still working, keep waiting
            else:
                # No coordinator - legacy behavior
                if first_batch_seen:
                    break  # We've seen data before, now it's empty = done
                else:
                    continue  # Haven't seen any data yet, keep waiting
        
        # PHASE 3: Process batch group using ORIGINAL double buffering approach
        group_results = self._process_batch_group_with_double_buffering(batch_group, q2_manager, total_batches_processed)
        
        all_results.extend(group_results)
        total_batches_processed += len(batch_group)
        
        if total_batches_processed % 10 == 0:
            logging.info(f"Actor {self.gpu_id}: Processed {total_batches_processed} batches")
    
    # Register completion with coordinator
    if coordinator is not None:
        actor_id = f"actor_{self.gpu_id}"
        ray.get(coordinator.register_ready_for_shutdown.remote(actor_id))
    
    return {
        'batches_processed': total_batches_processed,
        'total_samples': sum(r['batch_size'] for r in all_results),
        'results': all_results
    }

def _process_batch_group_with_double_buffering(self, batch_group, q2_manager, batch_offset):
    """Process a group of batches using EXACT same logic as process_batch_list."""
    
    results = []
    
    # Convert BatchData to Ray ObjectRefs (same as original process_batch_list input)
    batch_refs_list = []
    for batch_data in batch_group:
        # Extract tensor from BatchData and convert to list of ObjectRefs per sample
        batch_tensor = ray.get(batch_data.tensor_ref)  # Get (B, C, H, W) tensor
        sample_refs = [ray.put(batch_tensor[i]) for i in range(batch_data.batch_size)]
        batch_refs_list.append(sample_refs)
    
    # CRITICAL: Use EXACT same processing logic as working process_batch_list
    for batch_idx, batch_refs in enumerate(batch_refs_list):
        # Call the ORIGINAL working method
        result = self.process_batch_from_ray_object_store(batch_refs, batch_offset + batch_idx)
        results.append({
            'batch_id': batch_group[batch_idx].batch_id,
            'batch_size': result['batch_size'],
            'success': True
        })
    
    # CRITICAL: Only sync once at the end (same as original)
    with nvtx.range("final_group_sync"):
        self.pipeline.wait_for_completion()
    
    # Push results to Q2
    for i, result_info in enumerate(results):
        batch_data = batch_group[i]
        # Get predictions from pipeline output buffer
        predictions = self.pipeline.cpu_output_buffers[self.pipeline.current][:batch_data.batch_size].clone()
        
        output_batch = BatchOutput(
            predictions_ref=ray.put(predictions.numpy()),
            photon_energies=batch_data.photon_energies,
            batch_id=batch_data.batch_id
        )
        
        success = q2_manager.put(output_batch)
        result_info['success'] = success
    
    return results
```

### 5. Pipeline Orchestration (`peaknet_pipeline_ray/pipeline.py`)

#### Main Pipeline Runner
```python
def run_streaming_pipeline(
    num_producers: int = 2,
    num_actors: int = 2, 
    total_batches_per_producer: int = 100,
    batch_size: int = 16,
    shape: tuple = (1, 1920, 1920)
):
    """Main orchestration for streaming pipeline."""
    
    # Create coordinator
    coordinator = PipelineCoordinator.remote()
    ray.get(coordinator.set_expected_counts.remote(num_producers, num_actors))
    
    # Create queues
    q1_manager = ShardedQueueManager("pipeline_q1", num_shards=4, maxsize_per_shard=25) 
    q2_manager = ShardedQueueManager("pipeline_q2", num_shards=4, maxsize_per_shard=25)
    
    # Launch producers
    producer_tasks = []
    for i in range(num_producers):
        producer = StreamingDataProducer.remote(i, batch_size, shape)
        task = producer.stream_batches_to_queue.remote(q1_manager, total_batches_per_producer, coordinator)
        producer_tasks.append(task)
    
    # Launch actors
    actor_tasks = []
    for i in range(num_actors):
        actor = PeakNetPipelineActor.remote(
            input_shape=shape,
            batch_size=batch_size,
            peaknet_config=None,  # No-op mode for testing
            gpu_id=i
        )
        task = actor.process_from_queue.remote(q1_manager, q2_manager, coordinator)
        actor_tasks.append(task)
    
    # Wait for natural completion
    logging.info("Waiting for producers to complete...")
    producer_results = ray.get(producer_tasks)
    
    logging.info("Waiting for actors to complete...")  
    actor_results = ray.get(actor_tasks)
    
    return {
        'producer_results': producer_results,
        'actor_results': actor_results,
        'total_produced': sum(producer_results),
        'total_processed': sum(r['batches_processed'] for r in actor_results)
    }
```

## Implementation Files

### New Files to Create
1. `peaknet_pipeline_ray/utils/queue.py` - Queue infrastructure (detailed above)
2. `peaknet_pipeline_ray/core/coordinator.py` - Coordination actor (detailed above) 
3. `peaknet_pipeline_ray/core/streaming_producer.py` - Data producer (detailed above)
4. `peaknet_pipeline_ray/config/batch_structures.py` - Data structures (detailed above)

### Files to Modify
1. `peaknet_pipeline_ray/core/peaknet_ray_pipeline_actor.py` - Add streaming methods (detailed above)
2. `peaknet_pipeline_ray/pipeline.py` - Add pipeline orchestration (detailed above)

### Critical Implementation Details

1. **Batch Group Size**: Use 3-5 batches per group for optimal double buffering
2. **Queue Timeouts**: Short timeouts (0.5-1.0s) for responsive coordination queries  
3. **No Debug Prints**: Remove all debug prints from pipeline that cause blocking
4. **Preserve Buffer Swapping**: Exact same swap logic as original `process_batch_list`

## Success Criteria

1. **Performance**: Should match original double buffering performance (no blocking)
2. **Completeness**: All streaming data processed, no premature exits
3. **Profiling**: nsys profile shows overlapping operations, no `cudaStreamSynchronize` blocks
4. **Termination**: Clean shutdown when all producers finish and queues empty

## Testing Strategy

1. **Unit Tests**: Test each component (queues, coordinator, producer) separately  
2. **Integration Test**: Full streaming pipeline with small data set
3. **Performance Test**: Compare with original batch processing performance
4. **Profiling Validation**: Verify double buffering preserved with nsys

This plan builds on the working foundation while adding streaming capabilities, preserving the critical double buffering performance that was working in v0.1-no-queue-version.