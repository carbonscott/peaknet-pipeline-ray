#!/usr/bin/env python3
"""
Ray Pipeline Actor - Wrap DoubleBufferedPipeline in Ray actor for multi-GPU scaling

Each actor maintains a DoubleBufferedPipeline instance and processes data from
Ray's object store. Preserves all nvtx annotations for nsys profiling.

Adapted for PeakNet segmentation models.
"""

import ray
import torch
import torch.cuda.nvtx as nvtx
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import psutil
import os

# Import existing pipeline components
from .peaknet_pipeline import NWayBufferedPipeline, create_peaknet_model, get_numa_info, get_gpu_info
from .peaknet_utils import create_autocast_context
# GPU health validation now handled at system level before Ray initialization


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PeakNetPipelineActorBase:
    """
    Ray actor that wraps NWayBufferedPipeline for distributed processing.

    Each actor maintains:
    - A loaded PeakNet model
    - NWayBufferedPipeline instance with configurable concurrency
    - GPU assignment from Ray
    - Statistics tracking
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 512, 512),
        batch_size: int = 10,
        peaknet_config: dict = None,
        weights_path: str = None,
        pin_memory: bool = True,
        compile_mode: Optional[str] = None,
        warmup_iterations: int = 50,
        deterministic: bool = False,
        gpu_id: int = None,
        precision_dtype: str = "float32",
        fields: Dict[str, str] = None,
        num_buffers: int = 2,
    ):
        """
        Initialize the pipeline actor.

        Args:
            input_shape: Input tensor shape (C, H, W)
            batch_size: Batch size for processing
            peaknet_config: PeakNet configuration dict with model parameters
            weights_path: Path to PeakNet model weights
            pin_memory: Use pinned memory
            compile_mode: Torch compile mode (None = no compilation)
            warmup_iterations: Number of warmup iterations (0 = skip warmup)
            deterministic: Use deterministic operations
            gpu_id: Explicit GPU ID to use (None for Ray auto-assignment)
            precision_dtype: Precision type for mixed precision ('float32', 'bfloat16', 'float16')
            fields: Field mapping dictionary for data extraction
            num_buffers: Number of concurrent in-flight batches (2=double, 3=triple, 4+=quad+ buffering)
        """
        logging.info("=== Initializing PeakNetPipelineActor ===")

        # Set deterministic behavior if requested
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(42)

        # GPU assignment: Trust Ray's resource allocation
        # Ray automatically manages GPU assignment via CUDA_VISIBLE_DEVICES
        if gpu_id is not None:
            # Explicit GPU ID provided (for testing/debugging only)
            self.gpu_id = gpu_id
            logging.info(f"Using explicitly assigned GPU {self.gpu_id}")
        else:
            # Ray-native approach: assigned GPU always appears as cuda:0
            self.gpu_id = 0
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
            if cuda_visible != 'not set':
                physical_gpu = cuda_visible.split(',')[0]
                logging.info(f"Using Ray-assigned GPU device 0 (physical GPU {physical_gpu})")
            else:
                logging.warning("CUDA_VISIBLE_DEVICES not set by Ray - using device 0")

        logging.info(f"Actor GPU assignment complete - using CUDA device {self.gpu_id}")

        # Verify CUDA_VISIBLE_DEVICES is set correctly
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        logging.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

        # CUDA context establishment: Minimal warmup to ensure GPU is accessible
        try:
            # Simple GPU context establishment
            with torch.cuda.device(self.gpu_id):
                # Create small tensor to initialize CUDA context
                warmup_tensor = torch.ones(2, 2, device=f'cuda:{self.gpu_id}')
                _ = warmup_tensor + warmup_tensor  # Simple operation
                torch.cuda.synchronize(self.gpu_id)
                del warmup_tensor

            logging.info(f"CUDA context established on device {self.gpu_id}")

        except Exception as e:
            logging.error(f"CUDA context initialization failed: {e}")
            raise RuntimeError(f"CUDA context failed: {e}")

        # Store configuration
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.peaknet_config = peaknet_config
        self.weights_path = weights_path
        self.warmup_completed = False
        self.configured_warmup_iterations = warmup_iterations
        self.pin_memory = pin_memory
        self.deterministic = deterministic
        self.fields = fields or {"detector_data": "data"}
        self.num_buffers = num_buffers

        # Circular metadata buffer to track N batches' metadata for N-way buffering
        self.metadata_buffer = [None] * num_buffers

        # Get system info
        self.numa_info = get_numa_info()
        self.gpu_info = get_gpu_info(self.gpu_id)

        logging.info(f"Actor GPU: {self.gpu_info.get('name', 'Unknown')} ({self.gpu_info.get('memory_mb', 0):.0f} MB)")
        logging.info(f"Actor CPU affinity: {self.numa_info.get('cpu_ranges', 'unknown')}")

        # Create PeakNet model or use no-op mode
        if peaknet_config and peaknet_config.get('model'):
            # PeakNet mode: create actual model
            from .peaknet_utils import create_peaknet_model, get_peaknet_shapes

            self.peaknet_model = create_peaknet_model(
                peaknet_config=peaknet_config,
                weights_path=weights_path,
                device=f'cuda:{self.gpu_id}'
            )

            # Get shapes from configuration
            model_input_shape, model_output_shape = get_peaknet_shapes(peaknet_config, batch_size=1)
            self.input_shape = model_input_shape[1:]  # Remove batch dimension
            self.output_shape = model_output_shape[1:]  # Remove batch dimension
            logging.info(f"PeakNet mode: input_shape={self.input_shape}, output_shape={self.output_shape}")

            # Apply model compilation if requested
            if compile_mode is not None:
                try:
                    self.peaknet_model = torch.compile(self.peaknet_model, mode=compile_mode)
                    logging.info(f"torch.compile enabled (mode={compile_mode})")
                except Exception as e:
                    logging.warning(f"Model compilation failed: {e}")
        else:
            # No-op mode: no model, use provided input shape
            self.peaknet_model = None
            self.input_shape = input_shape
            self.output_shape = input_shape
            logging.info(f"No-op mode: input_shape={self.input_shape}, output_shape={self.output_shape}")

        # Create autocast context for mixed precision inference
        device_str = f'cuda:{self.gpu_id}'
        autocast_context = create_autocast_context(device_str, precision_dtype)
        logging.info(f"Created autocast context: dtype={precision_dtype}")

        # Create N-way buffered pipeline - use the same GPU device as assigned to this actor
        pipeline_gpu_id = self.gpu_id

        self.pipeline = NWayBufferedPipeline(
            model=self.peaknet_model,
            batch_size=batch_size,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            gpu_id=pipeline_gpu_id,
            pin_memory=pin_memory,
            autocast_context=autocast_context,
            num_buffers=num_buffers
        )

        logging.info(f"Created N-way buffered pipeline with {num_buffers} buffers (concurrency level={num_buffers})")

        # Initialize statistics
        self.stats = {
            'batches_processed': 0,
            'samples_processed': 0,
            'total_time': 0.0,
            'gpu_id': self.gpu_id,
            'actor_id': f"actor_{self.gpu_id}_{os.getpid()}",
            'model_config': {
                'peaknet_config': peaknet_config,
                'weights_path': weights_path,
                'input_shape': self.input_shape,
                'output_shape': self.output_shape
            }
        }

        # Model warmup if requested (only when using torch.compile)
        if warmup_iterations > 0 and self.peaknet_model is not None and compile_mode is not None:
            logging.info(f"Running model warmup with {warmup_iterations} iterations...")
            self._run_warmup(warmup_iterations, self.input_shape)
            self.warmup_completed = True

        logging.info(f"PeakNetPipelineActor initialized successfully on GPU {self.gpu_id}")


    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        try:
            # Check GPU availability
            torch.cuda.is_available()

            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(self.gpu_id).total_memory
            gpu_memory_used = torch.cuda.memory_allocated(self.gpu_id)
            gpu_memory_cached = torch.cuda.memory_reserved(self.gpu_id)

            # Simple GPU computation test
            with torch.cuda.device(self.gpu_id):
                test_tensor = torch.randn(100, 100, device=f'cuda:{self.gpu_id}')
                _ = test_tensor.sum()

            return {
                'status': 'healthy',
                'gpu_id': self.gpu_id,
                'gpu_memory_total_mb': gpu_memory / (1024 * 1024),
                'gpu_memory_used_mb': gpu_memory_used / (1024 * 1024),
                'gpu_memory_cached_mb': gpu_memory_cached / (1024 * 1024),
                'model_loaded': self.peaknet_model is not None,
                'pipeline_initialized': self.pipeline is not None
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'gpu_id': self.gpu_id
            }

    def process_from_queue(
        self,
        q1_manager,
        memory_sync_interval: int,
        q2_manager=None,
        coordinator=None,
        max_empty_polls: int = 10,
        poll_timeout: float = 0.01
    ) -> Dict[str, Any]:
        """
        TRUE STREAMING: Process data continuously from queue without batch accumulation.

        CRITICAL DESIGN: This method preserves the double buffering by NEVER calling
        pipeline.wait_for_completion() per batch. Only syncs at the very end or
        periodically for memory management.

        Args:
            q1_manager: Input queue manager to pull data from
            memory_sync_interval: Sync every N batches for memory management
            q2_manager: Optional output queue manager to push results to
            coordinator: Optional coordinator for termination logic
            max_empty_polls: Check coordinator after N consecutive empty polls
            poll_timeout: Timeout for queue polling in seconds

        Returns:
            Dictionary with processing statistics
        """
        with nvtx.range(f"ray_actor_stream_processing_gpu_{self.gpu_id}"):
            logging.info(f"Actor {self.gpu_id}: Starting streaming from queue")
            logging.info(f"Actor {self.gpu_id}: q1_manager type={type(q1_manager).__name__}, "
                        f"shards={getattr(q1_manager, 'num_shards', 'N/A')}, "
                        f"shard_actors={len(getattr(q1_manager, 'shard_actors', []))}, "
                        f"poll_timeout={poll_timeout}, max_empty_polls={max_empty_polls}")

            processed_count = 0
            consecutive_empty_polls = 0
            start_time = time.time()
            last_output_data = None
            _debug_logged = False

            while True:
                # Pull single batch from queue (non-blocking with short timeout)
                batch_data = q1_manager.get(timeout=poll_timeout)

                if not _debug_logged and batch_data is not None:
                    logging.info(f"Actor {self.gpu_id}: First batch received! type={type(batch_data).__name__}")
                    _debug_logged = True
                elif not _debug_logged and consecutive_empty_polls > 0 and consecutive_empty_polls % 1000 == 0:
                    # Log every 1000 empty polls to show the actor is alive
                    logging.info(f"Actor {self.gpu_id}: Still polling Q1 (empty_polls={consecutive_empty_polls})")

                if batch_data is None:
                    # Queue is empty
                    consecutive_empty_polls += 1

                    # Only check coordinator after multiple empty polls to avoid overhead
                    if consecutive_empty_polls >= max_empty_polls:
                        if coordinator is not None:
                            should_stop = ray.get(coordinator.should_actor_shutdown.remote(queue_empty=True))
                            if should_stop:
                                logging.info(f"Actor {self.gpu_id}: Coordinator confirms shutdown")
                                break
                        else:
                            # No coordinator - if we've seen data before and now empty, we're done
                            if processed_count > 0:
                                logging.info(f"Actor {self.gpu_id}: No coordinator, processed {processed_count} batches, queue empty - shutting down")
                                break

                        consecutive_empty_polls = 0  # Reset counter after coordinator check
                    continue

                # Reset empty poll counter on successful get
                consecutive_empty_polls = 0

                # CRITICAL: Process batch WITHOUT synchronization to maintain double buffering
                if processed_count > 0:
                    # Swap buffers but DO NOT sync - this is key to maintaining overlap
                    self.pipeline.swap()

                # Initialize metadata variables (will be populated based on data source)
                preprocessing_metadata = None
                original_image_ref = None

                # Extract data from different sources
                if hasattr(batch_data, 'tensor_refs'):
                    # ParsedSocketData - tensors already parsed in producer, dereference ObjectRefs
                    cpu_tensors = ray.get(batch_data.tensor_refs)

                    # NEW: Extract preprocessing metadata for Q2->W reconstruction
                    preprocessing_metadata = getattr(batch_data, 'preprocessing_metadata', None)

                    # NEW: Store preprocessed detector image for CXI writer
                    # Reconstruct batch from individual tensor_refs
                    import numpy as np
                    batch_array = np.stack([t.numpy() for t in cpu_tensors])
                    original_image_ref = ray.put(batch_array)
                elif hasattr(batch_data, 'get_torch_tensor'):
                    # PipelineInput object - get as individual tensors
                    batch_tensor = batch_data.get_torch_tensor(device='cpu')
                    # Convert to list of individual tensors for pipeline compatibility
                    cpu_tensors = [batch_tensor[i] for i in range(batch_tensor.shape[0])]
                elif isinstance(batch_data, list):
                    # Check if list contains tensors or ObjectRefs
                    if len(batch_data) > 0 and isinstance(batch_data[0], torch.Tensor):
                        # Legacy producer-side parsing - tensors directly in list
                        cpu_tensors = batch_data
                    else:
                        # Legacy: list of ObjectRefs - dereference them
                        cpu_tensors = ray.get(batch_data)
                else:
                    # Unknown type - try ray.get as last resort
                    cpu_tensors = ray.get(batch_data)

                actual_batch_size = len(cpu_tensors)

                # Validate batch size matches configuration
                if actual_batch_size != self.batch_size:
                    error_msg = f"Batch size mismatch: expected {self.batch_size}, got {actual_batch_size} (batch #{processed_count}, actor {self.gpu_id})"
                    logging.error(error_msg)
                    raise ValueError(error_msg)

                # Process through pipeline - this starts async H2D/Compute/D2H
                with nvtx.range(f"stream_process_batch_{processed_count}"):
                    self.pipeline.process_batch(
                        cpu_batch=cpu_tensors,
                        batch_idx=processed_count,
                        current_batch_size=actual_batch_size,
                        nvtx_prefix=f"stream_actor_{self.gpu_id}"
                    )

                # Store current batch metadata in circular buffer BEFORE handling output
                # (will be retrieved (N-1) iterations later)
                metadata_idx = processed_count % self.num_buffers
                self.metadata_buffer[metadata_idx] = {
                    'input': batch_data if hasattr(batch_data, 'metadata') else None,
                    'batch_size': actual_batch_size,
                    'start_time': time.time(),
                    'original_image_ref': original_image_ref,
                    'preprocessing_metadata': preprocessing_metadata
                }

                # Handle output from (N-1) iterations ago
                # Only start after pipeline has filled (processed >= N-1 batches)
                if processed_count >= self.num_buffers - 1 and q2_manager is not None:
                    # Calculate which buffer to read from (N-1 iterations old)
                    output_idx = (self.pipeline.current_idx - (self.num_buffers - 1)) % self.num_buffers

                    # Retrieve metadata for the batch we're outputting
                    output_metadata_idx = (processed_count - (self.num_buffers - 1)) % self.num_buffers
                    output_data = self.metadata_buffer[output_metadata_idx]

                    # CRITICAL: Synchronize on D2H event for this specific buffer
                    # With N>=3, this completes instantly (D2H finished long ago)
                    with nvtx.range(f"sync_d2h_buffer_{output_idx}"):
                        self.pipeline.d2h_done_event[output_idx].synchronize()

                    # Now safe to read output buffer (D2H complete)
                    with nvtx.range(f"clone_buffer_{output_idx}"):
                        output_tensor = self.pipeline.cpu_output_buffers[output_idx][:output_data['batch_size']].clone()

                    # Package and send to Q2 (CPU-bound work, overlaps with GPU work on current buffer)
                    with nvtx.range(f"package_output_{output_idx}"):
                        if output_data['input'] is not None and hasattr(output_data['input'], 'metadata'):
                            from ..config.data_structures import PipelineOutput
                            output_batch = PipelineOutput.from_input_and_predictions(
                                pipeline_input=output_data['input'],
                                predictions=output_tensor,
                                start_time=output_data['start_time'],
                                original_image_ref=output_data.get('original_image_ref'),
                                preprocessing_metadata=output_data.get('preprocessing_metadata')
                            )
                            q2_manager.put(output_batch)

                    logging.debug(f"Actor {self.gpu_id}: Delivered output from buffer {output_idx} (batch {processed_count - (self.num_buffers - 1)}) to Q2")

                processed_count += 1

                # Update statistics
                self.stats['batches_processed'] += 1
                self.stats['samples_processed'] += actual_batch_size

                # Periodic logging
                if processed_count % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = self.stats['samples_processed'] / elapsed if elapsed > 0 else 0
                    logging.info(f"Actor {self.gpu_id}: Processed {processed_count} batches, {rate:.1f} samples/s")

                # OPTIONAL: Periodic sync for memory management (NOT per batch!)
                # This is optional and should be infrequent to maintain performance
                if memory_sync_interval > 0 and processed_count % memory_sync_interval == 0:
                    logging.debug(f"Actor {self.gpu_id}: Periodic memory sync at batch {processed_count}")
                    self.pipeline.wait_for_completion()

            # FINAL synchronization - wait for all pending operations to complete
            if processed_count > 0:
                with nvtx.range("final_streaming_sync"):
                    self.pipeline.wait_for_completion()

                # Handle remaining (N-1) outputs still in flight
                # These are the last (num_buffers - 1) batches that were processed but not yet output
                if q2_manager is not None:
                    logging.info(f"Actor {self.gpu_id}: Draining pipeline - handling final {self.num_buffers - 1} outputs")

                    for i in range(1, self.num_buffers):
                        # Calculate which batch we're outputting
                        remaining_batch_idx = processed_count - (self.num_buffers - 1) + i

                        # Check if this batch exists (might have fewer than N batches processed)
                        if remaining_batch_idx >= 0 and remaining_batch_idx < processed_count:
                            # Calculate buffer index for this remaining output
                            output_idx = (self.pipeline.current_idx - (self.num_buffers - 1 - i)) % self.num_buffers
                            metadata_idx = remaining_batch_idx % self.num_buffers
                            output_data = self.metadata_buffer[metadata_idx]

                            if output_data is not None:
                                # Sync and handle output
                                with nvtx.range(f"final_sync_buffer_{output_idx}"):
                                    self.pipeline.d2h_done_event[output_idx].synchronize()

                                with nvtx.range(f"final_clone_buffer_{output_idx}"):
                                    output_tensor = self.pipeline.cpu_output_buffers[output_idx][:output_data['batch_size']].clone()

                                with nvtx.range(f"final_package_output_{output_idx}"):
                                    if output_data['input'] is not None and hasattr(output_data['input'], 'metadata'):
                                        from ..config.data_structures import PipelineOutput
                                        final_output = PipelineOutput.from_input_and_predictions(
                                            pipeline_input=output_data['input'],
                                            predictions=output_tensor,
                                            start_time=output_data['start_time'],
                                            original_image_ref=output_data.get('original_image_ref'),
                                            preprocessing_metadata=output_data.get('preprocessing_metadata')
                                        )
                                        q2_manager.put(final_output)
                                        logging.debug(f"Actor {self.gpu_id}: Delivered final output from buffer {output_idx} (batch {remaining_batch_idx}) to Q2")

            # Register with coordinator that this actor is done
            if coordinator is not None:
                actor_id = f"actor_{self.gpu_id}"
                ray.get(coordinator.register_actor_finished.remote(actor_id))

            total_time = time.time() - start_time

            result = {
                'actor_id': self.gpu_id,
                'batches_processed': processed_count,
                'total_samples': self.stats['samples_processed'],
                'processing_time': total_time,
                'average_throughput': self.stats['samples_processed'] / total_time if total_time > 0 else 0,
                'success': True
            }

            logging.info(
                f"Actor {self.gpu_id}: Streaming completed - "
                f"{processed_count} batches, {self.stats['samples_processed']} samples, "
                f"{result['average_throughput']:.1f} samples/s"
            )

            return result

    def _run_warmup(self, warmup_iterations: int, input_shape: Tuple[int, int, int]) -> None:
        """Run model warmup with synthetic data.

        Args:
            warmup_iterations: Number of warmup iterations to process
            input_shape: Shape of input tensors (C, H, W)
        """
        try:
            import torch

            # Generate synthetic warmup data using provided shape
            warmup_tensor = torch.randn(self.batch_size, *input_shape, dtype=torch.float32)
            warmup_tensors = [warmup_tensor[i] for i in range(self.batch_size)]

            logging.info(f"Warmup: Processing {warmup_iterations} iterations of batch size {self.batch_size}")

            for batch_idx in range(warmup_iterations):
                # Process batch through pipeline (similar to regular processing)
                if batch_idx > 0:
                    self.pipeline.swap()  # No sync during warmup for performance

                self.pipeline.process_batch(
                    cpu_batch=warmup_tensors,
                    batch_idx=batch_idx,
                    current_batch_size=self.batch_size,
                    nvtx_prefix=f"warmup_actor_{self.gpu_id}"
                )

            # Final synchronization to ensure all warmup work completes
            # Step 1: Sync pipeline streams (H2D, compute, D2H)
            self.pipeline.wait_for_completion()

            # Step 2: Explicit device synchronization (ensures ALL CUDA operations complete)
            # This is critical for:
            # - Preventing hangs when transitioning to production streaming
            # - Creating clear profiling boundaries in nsys
            # - Ensuring compiled kernels are fully initialized
            torch.cuda.synchronize(self.gpu_id)

            # Step 3: Clear CUDA cache to prevent memory fragmentation
            torch.cuda.empty_cache()

            logging.info(f"Warmup completed: {warmup_iterations} iterations processed, device synchronized")

        except Exception as e:
            logging.warning(f"Warmup failed: {e}, continuing without warmup")

    def graceful_shutdown(self) -> Dict[str, Any]:
        """
        Gracefully shutdown the actor with profiling data preservation.

        This method ensures that:
        1. Current work is completed
        2. Pipeline is properly synchronized
        3. Profiling data is flushed (for nsys profiling)
        4. Resources are cleaned up

        Returns:
            Dictionary with shutdown status and statistics
        """
        try:
            logging.info(f"Actor {self.gpu_id}: Starting graceful shutdown...")
            start_time = time.time()

            # Complete any pending pipeline work
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                with nvtx.range(f"actor_shutdown_sync_gpu_{self.gpu_id}"):
                    logging.info(f"Actor {self.gpu_id}: Synchronizing pipeline...")
                    self.pipeline.wait_for_completion()

                    # Additional GPU synchronization to ensure all CUDA work is done
                    with torch.cuda.device(self.gpu_id):
                        torch.cuda.synchronize(self.gpu_id)

                    logging.info(f"Actor {self.gpu_id}: Pipeline synchronization completed")

            # Give nsys profiling time to flush data (important for profile data integrity)
            if hasattr(self, '_is_profiling_enabled') or 'nsight' in os.environ.get('RAY_RUNTIME_ENV', ''):
                logging.info(f"Actor {self.gpu_id}: Flushing profiling data...")
                time.sleep(0.5)  # Allow profiling data to flush

            # Get final statistics
            final_stats = self.get_statistics()
            shutdown_time = time.time() - start_time

            logging.info(
                f"Actor {self.gpu_id}: Graceful shutdown completed in {shutdown_time:.3f}s - "
                f"processed {final_stats.get('batches_processed', 0)} batches"
            )

            # Use Ray's graceful exit mechanism
            ray.actor.exit_actor()

            return {
                'success': True,
                'gpu_id': self.gpu_id,
                'shutdown_time': shutdown_time,
                'final_stats': final_stats
            }

        except Exception as e:
            error_msg = f"Actor {self.gpu_id}: Graceful shutdown failed: {e}"
            logging.error(error_msg)

            # Force exit if graceful shutdown fails
            try:
                ray.actor.exit_actor()
            except:
                pass

            return {
                'success': False,
                'gpu_id': self.gpu_id,
                'error': str(e)
            }



# Create Ray actor classes from the base
@ray.remote(num_gpus=1)
class PeakNetPipelineActor(PeakNetPipelineActorBase):
    """Ray actor for PeakNet pipeline processing without profiling."""
    pass


@ray.remote(num_gpus=1, runtime_env={"nsight": {
    "t": "cuda,cudnn,cublas,nvtx,osrt",
    "cuda-graph-trace": "node",
    "cuda-memory-usage": "true",
    "stop-on-exit": "true",
}})
class PeakNetPipelineActorWithProfiling(PeakNetPipelineActorBase):
    """Ray actor with enhanced nsys profiling for CUDA graphs."""
    pass


def create_pipeline_actors(
    num_actors: int,
    enable_profiling: bool = False,
    validate_gpus: bool = True,
    **pipeline_kwargs
) -> List[ray.actor.ActorHandle]:
    """
    Create multiple PeakNet pipeline actors with optional GPU health validation.

    Args:
        num_actors: Number of actors to create
        enable_profiling: Whether to enable nsys profiling
        validate_gpus: Whether to pre-validate GPU health before actor creation
        **pipeline_kwargs: Arguments passed to actor constructor

    Returns:
        List of Ray actor handles
    """
    logging.info(f"Creating {num_actors} PeakNet pipeline actors (profiling={'enabled' if enable_profiling else 'disabled'})")

    # GPU validation is now handled at system level before Ray initialization
    # All GPUs Ray sees are guaranteed healthy

    # Pass CUDA_VISIBLE_DEVICES to actors so they respect user's GPU selection
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    actor_runtime_env = {}
    if cuda_visible:
        actor_runtime_env['env_vars'] = {'CUDA_VISIBLE_DEVICES': cuda_visible}
        logging.info(f"Passing CUDA_VISIBLE_DEVICES={cuda_visible} to actors")

    # Choose actor class based on profiling preference
    actor_class = PeakNetPipelineActorWithProfiling if enable_profiling else PeakNetPipelineActor

    actors = []
    for i in range(num_actors):
        try:
            if actor_runtime_env:
                actor = actor_class.options(runtime_env=actor_runtime_env).remote(**pipeline_kwargs)
            else:
                actor = actor_class.remote(**pipeline_kwargs)
            actors.append(actor)
            logging.info(f"Created actor {i+1}/{num_actors}")
        except Exception as e:
            logging.error(f"Failed to create actor {i+1}/{num_actors}: {e}")
            # Continue trying to create remaining actors

    if len(actors) == 0:
        raise RuntimeError("Failed to create any pipeline actors")
    elif len(actors) < num_actors:
        logging.warning(f"Only created {len(actors)}/{num_actors} requested actors")

    logging.info(f"Successfully created {len(actors)} pipeline actors")
    return actors


def test_pipeline_actor():
    """Simple test of pipeline actor functionality."""
    if not ray.is_initialized():
        ray.init()

    logging.info("Testing Ray pipeline actor...")

    # Create single actor
    actor = PeakNetPipelineActor.remote(
        input_shape=(1, 512, 512),
        batch_size=4,
        peaknet_config=None,  # No-op mode for testing
        deterministic=True
    )

    # Test health check
    health = ray.get(actor.health_check.remote())
    logging.info(f"Actor health: {health['status']}")

    # Generate test data
    test_tensors = []
    for i in range(4):
        tensor = torch.randn(1, 512, 512)
        test_tensors.append(ray.put(tensor))

    # Process batch
    result = ray.get(actor.process_batch_from_ray_object_store.remote(test_tensors, 0))
    logging.info(f"Batch result: {result}")

    # Get statistics
    stats = ray.get(actor.get_statistics.remote())
    logging.info(f"Actor stats: {stats}")

    logging.info("Pipeline actor test passed!")


if __name__ == "__main__":
    test_pipeline_actor()
