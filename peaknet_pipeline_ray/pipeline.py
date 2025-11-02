#!/usr/bin/env python3
"""
Main PeakNet Pipeline Ray implementation.

Provides the high-level PeakNetPipeline class that orchestrates the entire
streaming ML inference workflow with metadata pass-through support.
"""

import os
import logging
import time
import sys
import signal
import threading
import glob
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

import ray

from .config import PipelineConfig
from .config.data_structures import PipelineInput, PipelineOutput
from .core.gpu_health_validator import get_healthy_gpus_for_ray
from .core.peaknet_ray_pipeline_actor import create_pipeline_actors

# Module-level logger for categorized output
logger = logging.getLogger(__name__)


class PipelineResults:
    """Container for pipeline execution results."""

    def __init__(self, success: bool, performance: Dict[str, Any], error: Optional[str] = None):
        self.success = success
        self.performance = performance
        self.error = error

    def __repr__(self) -> str:
        if self.success:
            samples = self.performance.get('total_samples', 0)
            throughput = self.performance.get('overall_throughput', 0)
            return f"PipelineResults(success=True, samples={samples:,}, throughput={throughput:.1f} samples/s)"
        else:
            return f"PipelineResults(success=False, error='{self.error}')"


class PeakNetPipeline:
    """
    High-level interface for running PeakNet inference pipeline with Ray.

    This class orchestrates the entire streaming ML inference workflow:
    - GPU health validation
    - Ray cluster setup
    - Data generation with streaming
    - Multi-GPU inference with double buffering
    - Results collection with metadata pass-through
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Pipeline configuration containing all runtime parameters
        """
        self.config = config
        self.logger = self._setup_logging()

        # Signal handling for graceful shutdown
        self.shutdown_event = threading.Event()
        self.actors = []  # Store actor references for cleanup
        self.coordinator = None  # Store coordinator reference for cleanup
        self._original_sigint_handler = None
        self._original_sigterm_handler = None

    def _setup_logging(self) -> logging.Logger:
        """Configure logging based on configuration."""
        logger = logging.getLogger('peaknet_pipeline')

        # Avoid duplicate handlers
        if logger.handlers:
            return logger

        level = logging.DEBUG if self.config.output.verbose else logging.INFO
        format_str = '%(asctime)s - %(levelname)s - %(message)s' if self.config.output.verbose else '%(message)s'

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(handler)
        logger.setLevel(level)

        return logger

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown with profiling."""
        def signal_handler(signum, frame):
            signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            if not self.config.output.quiet:
                print(f"\nReceived {signal_name} - shutting down...")
                if self.config.profiling.enable_profiling:
                    print("  Saving profiling data...")

            self.shutdown_event.set()

            # Notify coordinator of signal shutdown if available
            if self.coordinator is not None:
                try:
                    ray.get(self.coordinator.request_signal_shutdown.remote(f"{signal_name} received"))
                except Exception as e:
                    if not self.config.output.quiet:
                        print(f"     Failed to notify coordinator of shutdown: {e}")

        # Store original handlers to restore them later
        self._original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, signal_handler)

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
        if self._original_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)

    def _graceful_shutdown_actors(self) -> None:
        """Gracefully shutdown all pipeline actors."""
        if not self.actors:
            return

        if not self.config.output.quiet:
            print(f"    Gracefully shutting down {len(self.actors)} actors...")

        shutdown_futures = []

        # Try graceful shutdown first
        for i, actor in enumerate(self.actors):
            try:
                # Send shutdown signal to actor
                future = actor.graceful_shutdown.remote()
                shutdown_futures.append((i, future))
            except Exception as e:
                if self.config.output.verbose:
                    print(f"     Actor {i}: Failed to send shutdown signal: {e}")

        # Wait for graceful shutdowns with timeout
        timeout = 10.0  # seconds
        start_time = time.time()

        for i, future in shutdown_futures:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                if not self.config.output.quiet:
                    print(f"    Timeout waiting for actor {i} shutdown")
                break

            try:
                ray.get(future, timeout=min(remaining_time, 2.0))
                if self.config.output.verbose:
                    print(f"    Actor {i}: Graceful shutdown completed")
            except ray.exceptions.GetTimeoutError:
                if not self.config.output.quiet:
                    print(f"    Actor {i}: Shutdown timeout, forcing termination")
            except Exception as e:
                if self.config.output.verbose:
                    print(f"     Actor {i}: Shutdown error: {e}")

        # Clear actor references
        self.actors.clear()

        if not self.config.output.quiet:
            print("    Actor shutdown completed")

    def _collect_profiling_data(self) -> Dict[str, Any]:
        """Collect and validate nsys profiling data after shutdown.

        Returns:
            Dictionary with profiling data collection results
        """
        if not self.config.profiling.enable_profiling:
            return {'enabled': False, 'message': 'Profiling not enabled'}

        try:
            # Find Ray session directory
            ray_temp_dir = "/tmp/ray"
            if not os.path.exists(ray_temp_dir):
                return {'success': False, 'error': 'Ray temp directory not found'}

            # Look for the most recent session
            session_pattern = os.path.join(ray_temp_dir, "session_*")
            session_dirs = glob.glob(session_pattern)
            if not session_dirs:
                return {'success': False, 'error': 'No Ray session directories found'}

            # Get the most recent session directory
            latest_session = max(session_dirs, key=os.path.getmtime)
            nsight_dir = os.path.join(latest_session, "logs", "nsight")

            if not os.path.exists(nsight_dir):
                return {'success': False, 'error': f'Nsight logs directory not found: {nsight_dir}'}

            # Find all .nsys-rep files
            nsys_pattern = os.path.join(nsight_dir, "**", "*.nsys-rep")
            nsys_files = glob.glob(nsys_pattern, recursive=True)

            if not nsys_files:
                return {'success': False, 'error': f'No .nsys-rep files found in {nsight_dir}'}

            # Validate and collect file information
            valid_files = []
            total_size = 0

            for nsys_file in nsys_files:
                try:
                    file_stat = os.stat(nsys_file)
                    file_size = file_stat.st_size

                    # Basic validation - file should be > 1KB (very basic check)
                    if file_size > 1024:
                        valid_files.append({
                            'path': nsys_file,
                            'size_mb': file_size / (1024 * 1024),
                            'mtime': file_stat.st_mtime
                        })
                        total_size += file_size
                    else:
                        if self.config.output.verbose:
                            print(f"     Skipping small profile file: {nsys_file} ({file_size} bytes)")

                except OSError as e:
                    if self.config.output.verbose:
                        print(f"     Cannot access profile file: {nsys_file} - {e}")

            # Copy files to output directory if specified
            copied_files = []
            if self.config.profiling.output_dir and valid_files:
                output_dir = Path(self.config.profiling.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                for file_info in valid_files:
                    src_path = file_info['path']
                    filename = os.path.basename(src_path)
                    # Add timestamp to avoid conflicts
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    dest_filename = f"{timestamp}_{filename}"
                    dest_path = output_dir / dest_filename

                    try:
                        shutil.copy2(src_path, dest_path)
                        copied_files.append(str(dest_path))
                        if self.config.output.verbose:
                            print(f"    Copied profile: {dest_path}")
                    except Exception as e:
                        if self.config.output.verbose:
                            print(f"     Failed to copy {src_path}: {e}")

            return {
                'success': True,
                'nsight_dir': nsight_dir,
                'files_found': len(valid_files),
                'total_size_mb': total_size / (1024 * 1024),
                'valid_files': valid_files,
                'copied_files': copied_files
            }

        except Exception as e:
            return {'success': False, 'error': f'Profile collection failed: {e}'}

    def _print_profiling_summary(self, profile_results: Dict[str, Any]) -> None:
        """Print summary of profiling data collection."""
        if not self.config.profiling.enable_profiling or self.config.output.quiet:
            return

        if not profile_results.get('success', False):
            print(f"\nProfiling data collection failed: {profile_results.get('error', 'Unknown')}")
            return

        files_found = profile_results.get('files_found', 0)
        total_size = profile_results.get('total_size_mb', 0)
        print(f"\nProfiling: {files_found} files ({total_size:.1f} MB) in {profile_results.get('nsight_dir', 'N/A')}")

        copied_files = profile_results.get('copied_files', [])
        if copied_files:
            print(f"Copied {len(copied_files)} files to {self.config.profiling.output_dir}")

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        runtime = self.config.runtime
        system = self.config.system
        data = self.config.data

        if runtime.max_actors is not None and runtime.max_actors <= 0:
            raise ValueError("max_actors must be positive")

        if system.min_gpus <= 0:
            raise ValueError("min_gpus must be positive")

        # Only validate num_producers for non-socket modes (socket mode auto-infers from socket_addresses)
        if self.config.data_source.source_type != "socket":
            if runtime.num_producers is None or runtime.num_producers <= 0:
                raise ValueError("num_producers must be positive for non-socket data sources")

        if runtime.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if len(data.shape) != 3:
            raise ValueError("data.shape must be (C, H, W)")

        if any(dim <= 0 for dim in data.shape):
            raise ValueError("all data.shape dimensions must be positive")

        if runtime.inter_batch_delay < 0:
            raise ValueError("inter_batch_delay cannot be negative")

    def _print_banner(self) -> None:
        """Print pipeline banner and configuration summary."""
        print("PeakNet Pipeline - Ray Multi-GPU Streaming")
        print("=" * 45)

        if self.config.output.verbose:
            print(f"Config: batch_size={self.config.runtime.batch_size}, shape={self.config.data.shape}, min_gpus={self.config.system.min_gpus}")
            print()

    def _setup_gpu_environment(self) -> List[int]:
        """Set up GPU environment with health validation."""
        if self.config.system.skip_gpu_validation:
            if not self.config.output.quiet:
                logger.info(f"[GPU] Validation skipped (assuming {self.config.system.min_gpus}+ GPUs available)")
            return list(range(self.config.system.min_gpus))

        try:
            healthy_gpus = get_healthy_gpus_for_ray(min_gpus=self.config.system.min_gpus)
            return healthy_gpus

        except RuntimeError as e:
            logger.error(f"[GPU] Validation failed: {e}")
            raise

    def _setup_ray_cluster(self) -> None:
        """Initialize Ray cluster connection."""
        max_actors = self.config.runtime.max_actors

        # Check if the current Python process has connected to Ray
        # NOTE: This checks if THIS process called ray.init(), NOT whether a Ray cluster is running.
        # If you manually started a cluster with `ray start --head`, this will still be False
        # until we call ray.init() below, which will connect to your existing cluster.
        if not ray.is_initialized():
            try:
                ray.init(namespace=self.config.ray.namespace)
                cluster_resources = ray.cluster_resources()
                gpu_count = int(cluster_resources.get('GPU', 0))
                cpu_count = int(cluster_resources.get('CPU', 0))

                if not self.config.output.quiet:
                    logger.info(f"[Ray] Cluster connected: {gpu_count} GPUs, {cpu_count} cores available")
                    if max_actors:
                        logger.debug(f"[Ray] Actor limit: {max_actors} (user-specified)")

            except Exception as e:
                error_msg = f"Ray initialization failed: {e}"
                logger.error(f"[Ray] {error_msg}")
                raise RuntimeError(error_msg)
        else:
            cluster_resources = ray.cluster_resources()
            gpu_count = int(cluster_resources.get('GPU', 0))
            cpu_count = int(cluster_resources.get('CPU', 0))

            if not self.config.output.quiet:
                logger.info(f"[Ray] Cluster already running: {gpu_count} GPUs, {cpu_count} cores available")
                if max_actors:
                    logger.debug(f"[Ray] Actor limit: {max_actors} (user-specified)")

    def _create_gpu_actors(self, healthy_gpus: List[int]) -> List[Any]:
        """Create GPU pipeline actors with automatic scaling."""
        # Determine actual number of actors to create
        max_possible_actors = len(healthy_gpus)
        actual_num_actors = max_possible_actors

        max_actors = self.config.runtime.max_actors
        if max_actors is not None:
            actual_num_actors = min(max_actors, max_possible_actors)
            if max_actors > max_possible_actors:
                if not self.config.output.quiet:
                    logger.warning(f"[Actors] Requested {max_actors} but only {max_possible_actors} GPUs available")

        enable_profiling = self.config.profiling.enable_profiling

        if not self.config.output.quiet:
            logger.info(f"[Actors] Creating {actual_num_actors} on {max_possible_actors} GPU(s)")
            if enable_profiling:
                logger.debug(f"[Actors] Profiling: enabled (NSys profile files will be generated)")

        try:
            # Determine input shape based on data source configuration
            if self.config.data_source.source_type == "socket":
                # Socket source: detector shape from config (may be overridden by model's image_size)
                detector_shape = self.config.data_source.shape
                input_shape = detector_shape
                if not self.config.output.quiet:
                    # Show both detector size and actual model input (if PeakNet mode)
                    if self.config.model.peaknet_config and 'model' in self.config.model.peaknet_config:
                        image_size = self.config.model.peaknet_config['model'].get('image_size', 512)
                        model_input = (1, image_size, image_size)
                        logger.debug(f"[Actors] Detector: {detector_shape}, Model: {model_input}")
                    else:
                        logger.debug(f"[Actors] Socket shape: {detector_shape}")
            else:
                # Random source: use data.shape
                input_shape = self.config.data.shape
                if not self.config.output.quiet:
                    logger.debug(f"[Actors] Random data shape: {input_shape}")

            actors = create_pipeline_actors(
                num_actors=actual_num_actors,
                enable_profiling=enable_profiling,
                validate_gpus=False,  # Already validated at system level
                # Pipeline configuration
                input_shape=input_shape,
                batch_size=self.config.runtime.batch_size,
                num_buffers=self.config.runtime.pipeline_concurrency,  # N-way buffering concurrency
                # PeakNet configuration
                weights_path=self.config.model.weights_path,
                peaknet_config=self.config.model.peaknet_config,
                compile_mode=self.config.model.compile_mode,
                warmup_iterations=self.config.model.warmup_iterations,
                deterministic=True,
                pin_memory=self.config.system.pin_memory,
                # Mixed precision configuration
                precision_dtype=self.config.precision.dtype,
                # Data source configuration for socket mode
                fields=self.config.data_source.fields,
            )

            if not self.config.output.quiet:
                logger.info(f"[Actors] Created {len(actors)} successfully")

            # Verify actor health if enabled
            if self.config.system.verify_actors:
                health_futures = [actor.health_check.remote() for actor in actors]

                try:
                    health_results = ray.get(health_futures, timeout=30)
                    healthy_count = sum(1 for h in health_results if h.get('status') == 'healthy')

                    if not self.config.output.quiet:
                        logger.debug(f"[Actors] Health check: {healthy_count}/{len(actors)} healthy")

                        if self.config.output.verbose:
                            for i, health in enumerate(health_results):
                                gpu_id = health.get('gpu_id', 'unknown')
                                status = health.get('status', 'unknown')
                                logger.debug(f"[Actors] Actor {i}: GPU {gpu_id} - {status}")

                except Exception as e:
                    if not self.config.output.quiet:
                        logger.warning(f"[Actors] Health check failed: {e} (continuing anyway)")

            return actors

        except Exception as e:
            error_msg = f"Failed to create pipeline actors: {e}"
            logger.error(f"[Actors] {error_msg}")
            raise RuntimeError(error_msg)

    def _print_results(self, performance: Dict[str, Any]) -> None:
        """Print performance results."""
        if not performance['success']:
            print(f"\nProcessing failed: {performance.get('error', 'Unknown error')}")
            return

        print(f"\nResults: {performance['total_samples']:,} samples, {performance['total_batches']:,} batches, {performance['total_processing_time']:.2f}s, {performance['overall_throughput']:.1f} samples/s")

        if self.config.output.verbose:
            actor_stats = performance['actor_stats']
            for actor_idx, stats in actor_stats.items():
                throughput = stats['samples'] / stats['total_time'] if stats['total_time'] > 0 else 0
                print(f"  Actor {actor_idx}: {stats['batches']} batches, {stats['samples']:,} samples, {throughput:.1f} samples/s")

    def _save_results(self, performance: Dict[str, Any]) -> None:
        """Save results to output directory if specified."""
        output_dir_path = self.config.output.output_dir
        if not output_dir_path or not performance['success']:
            return

        output_dir = Path(output_dir_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save performance metrics
        import json
        from datetime import datetime

        results_file = output_dir / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert config to dict for JSON serialization
        config_dict = {
            'model': {
                'yaml_path': self.config.model.yaml_path,
                'weights_path': self.config.model.weights_path
            },
            'runtime': {
                'max_actors': self.config.runtime.max_actors,
                'batch_size': self.config.runtime.batch_size,
                'total_samples': self.config.runtime.total_samples,
                'num_producers': self.config.runtime.num_producers,
                'batches_per_producer': self.config.runtime.batches_per_producer,
                'inter_batch_delay': self.config.runtime.inter_batch_delay
            },
            'data': {
                'shape': list(self.config.data.shape),
                'input_channels': self.config.data.shape[0]  # Extract from shape
            },
            'system': {
                'min_gpus': self.config.system.min_gpus,
                'skip_gpu_validation': self.config.system.skip_gpu_validation,
                'pin_memory': self.config.system.pin_memory,
                'verify_actors': self.config.system.verify_actors
            },
            'profiling': {
                'enable_profiling': self.config.profiling.enable_profiling,
                'output_dir': self.config.profiling.output_dir
            },
            'output': {
                'output_dir': self.config.output.output_dir,
                'verbose': self.config.output.verbose,
                'quiet': self.config.output.quiet
            }
        }

        save_data = {
            'timestamp': datetime.now().isoformat(),
            'configuration': config_dict,
            'performance': performance
        }

        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        if not self.config.output.quiet:
            print(f"\n Results saved to: {results_file}")

    def run_streaming_pipeline(
        self,
        total_samples: Optional[int] = None,
        enable_output_queue: bool = False
    ) -> PipelineResults:
        """
        Run the streaming pipeline with continuous data processing.

        This method launches producers that generate data continuously,
        pipeline actors that process data as it arrives (without accumulation),
        and manages coordination for clean termination.

        Args:
            total_samples: Override total samples to process (optional)
            enable_output_queue: Enable Q2 output queue for downstream processing

        Returns:
            PipelineResults containing success status and performance metrics
        """
        # Override total_samples if provided
        if total_samples is not None:
            self.config.runtime.total_samples = total_samples

        try:
            # Validate configuration
            self._validate_config()

            # Print banner
            if not self.config.output.quiet:
                self._print_streaming_banner()

            # Step 1: GPU Environment Setup
            healthy_gpus = self._setup_gpu_environment()

            # Step 2: Ray Cluster Setup
            self._setup_ray_cluster()

            # Step 3: Create GPU Actors
            actors = self._create_gpu_actors(healthy_gpus)

            # Step 4: Run Streaming Pipeline
            performance = self._run_streaming_workflow(actors, enable_output_queue)

            # Step 5: Results
            if not self.config.output.quiet:
                self._print_streaming_results(performance)

            if self.config.output.output_dir:
                self._save_results(performance)

            return PipelineResults(success=performance['success'], performance=performance)

        except KeyboardInterrupt:
            error = "Streaming pipeline interrupted by user"
            if not self.config.output.quiet:
                print(f"\n  {error}")
            return PipelineResults(success=False, performance={}, error=error)

        except Exception as e:
            error = f"Streaming pipeline failed: {e}"
            if not self.config.output.quiet:
                print(f"\n {error}")
                if self.config.output.verbose:
                    import traceback
                    traceback.print_exc()
            return PipelineResults(success=False, performance={}, error=str(e))

    def _print_streaming_banner(self) -> None:
        """Print streaming pipeline banner."""
        print(" Ray Multi-GPU PeakNet STREAMING Pipeline")
        print("=" * 55)
        print()

        if self.config.output.verbose:
            print(f"Configuration:")
            print(f"  Model: PeakNet (weights: {self.config.model.weights_path is not None})")
            # Show actual model input shape (from image_size) for PeakNet, or data.shape for no-op mode
            if self.config.model.peaknet_config and 'model' in self.config.model.peaknet_config:
                image_size = self.config.model.peaknet_config['model'].get('image_size', 512)
                print(f"  Model input: [1, {image_size}, {image_size}] tensor shape")
            else:
                print(f"  Data: {self.config.data.shape} tensor shape")
            num_producers_str = "auto" if self.config.runtime.num_producers is None else str(self.config.runtime.num_producers)
            print(f"  Runtime: {self.config.runtime.batch_size} batch size, {num_producers_str} producers")
            print(f"  System: min {self.config.system.min_gpus} GPUs required")
            print()

    def _run_streaming_workflow(
        self, 
        actors: List[Any], 
        enable_output_queue: bool
    ) -> Dict[str, Any]:
        """Run the complete streaming workflow with producers and actors."""
        from .utils.queue import ShardedQueueManager
        from .core.coordinator import create_streaming_coordinator
        from .core.streaming_producer import create_streaming_producers

        if not self.config.output.quiet:
            print("\n Starting Streaming Workflow")

        # Calculate production parameters
        runtime = self.config.runtime
        data = self.config.data

        # Determine actual producer count (socket mode auto-infers from socket_addresses)
        if self.config.data_source.source_type == "socket":
            actual_num_producers = len(self.config.data_source.socket_addresses) if self.config.data_source.socket_addresses else 0
            batches_per_producer = None  # Stream until socket closes
            total_expected_batches = None  # Unknown for streaming
            total_expected_samples = None  # Unknown for streaming
        else:
            actual_num_producers = runtime.num_producers
            # Calculate finite batches for non-socket sources
            if runtime.total_samples is not None:
                total_batches_needed = (runtime.total_samples + runtime.batch_size - 1) // runtime.batch_size
                batches_per_producer = max(1, total_batches_needed // runtime.num_producers)
                if total_batches_needed > runtime.num_producers * batches_per_producer:
                    batches_per_producer += 1
            else:
                batches_per_producer = runtime.batches_per_producer

            total_expected_batches = runtime.num_producers * batches_per_producer
            total_expected_samples = total_expected_batches * runtime.batch_size

        if not self.config.output.quiet:
            print(f"   Producers: {actual_num_producers}")
            print(f"   Actors: {len(actors)}")
            if self.config.data_source.source_type == "socket":
                print(f"   Batches per producer: unlimited (stream until socket closes)")
                print(f"   Expected total: unknown (continuous streaming)")
            else:
                print(f"   Batches per producer: {batches_per_producer}")
                print(f"   Expected total: {total_expected_batches} batches, {total_expected_samples} samples")
            # Show actual model input shape (from image_size) for PeakNet, or data.shape for no-op mode
            if self.config.model.peaknet_config and 'model' in self.config.model.peaknet_config:
                image_size = self.config.model.peaknet_config['model'].get('image_size', 512)
                print(f"   Model input shape: [1, {image_size}, {image_size}]")
            else:
                print(f"   Model input shape: {data.shape}")
            print(f"   Inter-batch delay: {runtime.inter_batch_delay}s")

        # Step 1: Create Coordinator
        if not self.config.output.quiet:
            print("\n📡 Step 1: Creating Streaming Coordinator")

        coordinator = create_streaming_coordinator(
            expected_producers=actual_num_producers,
            expected_actors=len(actors)
        )

        # Store coordinator reference for signal handling
        self.coordinator = coordinator

        # Step 2: Create Queues
        if not self.config.output.quiet:
            print(" Step 2: Creating Queue Infrastructure")

        # Input queue (Q1) - producers -> actors
        num_shards = runtime.queue_num_shards
        maxsize_per_shard = runtime.queue_maxsize_per_shard

        q1_manager = ShardedQueueManager(
            runtime.queue_names.input_queue,
            num_shards=num_shards,
            maxsize_per_shard=maxsize_per_shard
        )

        q2_manager = None
        if enable_output_queue:
            q2_manager = ShardedQueueManager(
                runtime.queue_names.output_queue,
                num_shards=num_shards,
                maxsize_per_shard=maxsize_per_shard
            )

        if not self.config.output.quiet:
            print(f"   Q1 (input): {num_shards} shards, {maxsize_per_shard} items/shard")
            if enable_output_queue:
                print(f"   Q2 (output): {num_shards} shards, {maxsize_per_shard} items/shard")

        # Step 3: Launch Streaming Producers
        if not self.config.output.quiet:
            print(f" Step 3: Launching Streaming Producers ({self.config.data_source.source_type})")

        producers = self._create_data_producers(runtime, data)

        producer_tasks = []
        for i, producer in enumerate(producers):
            # Use appropriate method based on producer type
            if self.config.data_source.source_type == "socket":
                # Socket producers stream indefinitely (batches_per_producer is None)
                task = producer.stream_raw_bytes_to_queue.remote(
                    q1_manager,
                    batches_per_producer,  # None for socket sources
                    coordinator,
                    progress_interval=100  # Fixed interval for socket streaming
                )
            else:
                # Standard producers (random data, etc.)
                task = producer.stream_batches_to_queue.remote(
                    q1_manager,
                    batches_per_producer,
                    coordinator,
                    progress_interval=max(10, batches_per_producer // 10)
                )
            producer_tasks.append(task)

        # Step 4: Launch Streaming Pipeline Actors
        if not self.config.output.quiet:
            print(" Step 4: Launching Streaming Pipeline Actors")

        actor_tasks = []
        for i, actor in enumerate(actors):
            task = actor.process_from_queue.remote(
                q1_manager,
                runtime.memory_sync_interval,  # From config
                q2_manager,
                coordinator,
                max_empty_polls=runtime.max_empty_polls,  # From config
                poll_timeout=runtime.poll_timeout   # From config
            )
            actor_tasks.append(task)

        # Step 5: Monitor and Wait for Completion
        if not self.config.output.quiet:
            print("⏳ Step 5: Streaming Processing (Real-time)")

        start_time = time.time()

        # Wait for producers to complete
        if not self.config.output.quiet:
            print("\n   Waiting for producers to finish...")

        # Get results from all producers (some may already be completed)
        remaining_tasks = []
        completed_results = []

        for i, task in enumerate(producer_tasks):
            try:
                # Check if already completed
                result = ray.get(task, timeout=0.001)
                completed_results.append(result)
            except:
                # Still running, add to remaining tasks
                remaining_tasks.append(task)

        # Wait for any remaining producer tasks
        if remaining_tasks:
            remaining_results = ray.get(remaining_tasks)
            producer_results = completed_results + remaining_results
        else:
            producer_results = completed_results
        producer_time = time.time() - start_time

        if not self.config.output.quiet:
            total_produced_samples = sum(r['total_samples'] for r in producer_results)
            total_backpressure = sum(r['backpressure_events'] for r in producer_results)
            print(f"    All producers finished in {producer_time:.2f}s")
            print(f"    Produced: {total_produced_samples} samples, {total_backpressure} backpressure events")

        # Wait for actors to complete
        if not self.config.output.quiet:
            print("   Waiting for actors to finish processing...")

        actor_results = ray.get(actor_tasks)
        total_time = time.time() - start_time

        # Step 6: Calculate Performance Metrics
        total_processed_samples = sum(r['total_samples'] for r in actor_results)
        total_processed_batches = sum(r['batches_processed'] for r in actor_results)

        # Per-actor stats
        actor_stats = {}
        for i, result in enumerate(actor_results):
            actor_stats[i] = {
                'batches': result['batches_processed'],
                'samples': result['total_samples'],
                'throughput': result['average_throughput'],
                'processing_time': result['processing_time']
            }

        return {
            'success': True,
            'streaming_mode': True,
            'total_samples': total_processed_samples,
            'total_batches': total_processed_batches,
            'total_processing_time': total_time,
            'producer_time': producer_time,
            'overall_throughput': total_processed_samples / total_time if total_time > 0 else 0,
            'producer_results': producer_results,
            'actor_results': actor_results,
            'actor_stats': actor_stats,
            'expected_samples': total_expected_samples,
            'sample_completion_rate': total_processed_samples / total_expected_samples if total_expected_samples and total_expected_samples > 0 else 0,
            'queue_config': {
                'q1_shards': num_shards,
                'q1_maxsize_per_shard': maxsize_per_shard,
                'output_queue_enabled': enable_output_queue
            }
        }

    def _create_data_producers(self, runtime, data):
        """Create data producers based on configuration.

        Args:
            runtime: Runtime configuration
            data: Data configuration

        Returns:
            List of producer actors
        """
        if self.config.data_source.source_type == "socket":
            # Create socket producers for optimal CPU/GPU overlap
            from .core.socket_producer import create_socket_producers

            if not self.config.output.quiet:
                num_sockets = len(self.config.data_source.socket_addresses) if self.config.data_source.socket_addresses else 0
                if num_sockets == 1:
                    host, port = self.config.data_source.socket_addresses[0]
                    print(f"   Socket: {host}:{port}")
                else:
                    print(f"   Sockets: {num_sockets} addresses (1 producer per socket)")
                    for i, (host, port) in enumerate(self.config.data_source.socket_addresses):
                        print(f"      [{i}] {host}:{port}")
                print(f"   Optimization: Raw bytes → Pipeline parsing for zero gaps")

            return create_socket_producers(
                num_producers=runtime.num_producers,
                config=self.config.data_source,
                deterministic=False
            )
        else:
            # Create random data producers (default/backward compatibility)
            from .core.streaming_producer import create_streaming_producers

            if not self.config.output.quiet:
                print(f"   Random data: {data.shape}")

            return create_streaming_producers(
                num_producers=runtime.num_producers,
                batch_size=runtime.batch_size,
                tensor_shape=data.shape,
                inter_batch_delay=runtime.inter_batch_delay,
                deterministic=False
            )

    def _print_streaming_results(self, performance: Dict[str, Any]) -> None:
        """Print streaming pipeline performance results."""
        if not performance['success']:
            print(f"\nProcessing failed: {performance.get('error', 'Unknown')}")
            return

        print(f"\nResults: {performance['total_samples']:,} samples in {performance['total_processing_time']:.2f}s ({performance['overall_throughput']:.1f} samples/s)")

        if self.config.output.verbose:
            for actor_id, stats in performance['actor_stats'].items():
                print(f"  Actor {actor_id}: {stats['samples']:,} samples, {stats['throughput']:.1f} samples/s")