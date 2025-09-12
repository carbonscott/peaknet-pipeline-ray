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
from pathlib import Path
from typing import Dict, List, Any, Optional

import ray

from .config import PipelineConfig
from .config.data_structures import PipelineInput, PipelineOutput
from .core.gpu_health_validator import get_healthy_gpus_for_ray
from .core.peaknet_ray_data_producer import RayDataProducerManager
from .core.peaknet_ray_pipeline_actor import create_pipeline_actors


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

    def run(self, total_samples: Optional[int] = None) -> PipelineResults:
        """
        Run the complete pipeline workflow.

        Args:
            total_samples: Override total samples to process (optional)

        Returns:
            PipelineResults containing success status and performance metrics
        """
        # Override total_samples if provided
        if total_samples is not None:
            self.config.runtime.total_samples = total_samples

        try:
            # Validate configuration
            self._validate_config()

            # Print banner and configuration summary
            if not self.config.output.quiet:
                self._print_banner()

            # Step 1: GPU Environment Setup
            healthy_gpus = self._setup_gpu_environment()

            # Step 2: Ray Cluster Setup
            self._setup_ray_cluster()

            # Step 3: Create GPU Actors
            actors = self._create_gpu_actors(healthy_gpus)

            # Step 4: Generate Streaming Data
            all_batches = self._generate_streaming_data()

            # Step 5: Process Data
            performance = self._process_streaming_data(actors, all_batches)

            # Step 6: Results and cleanup
            if not self.config.output.quiet:
                self._print_results(performance)

            if self.config.output.output_dir:
                self._save_results(performance)

            return PipelineResults(success=performance['success'], performance=performance)

        except KeyboardInterrupt:
            error = "Pipeline interrupted by user"
            if not self.config.output.quiet:
                print(f"\n⚠️  {error}")
            return PipelineResults(success=False, performance={}, error=error)

        except Exception as e:
            error = f"Pipeline failed with error: {e}"
            if not self.config.output.quiet:
                print(f"\n💥 {error}")
                if self.config.output.verbose:
                    import traceback
                    traceback.print_exc()
            return PipelineResults(success=False, performance={}, error=str(e))

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        runtime = self.config.runtime
        system = self.config.system
        data = self.config.data

        if runtime.max_actors is not None and runtime.max_actors <= 0:
            raise ValueError("max_actors must be positive")

        if system.min_gpus <= 0:
            raise ValueError("min_gpus must be positive")

        if runtime.num_producers <= 0:
            raise ValueError("num_producers must be positive")

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
        print("🚀 Ray Multi-GPU PeakNet Streaming Pipeline")
        print("=" * 55)

        if self.config.output.verbose:
            print(f"Configuration summary:")
            print(f"  Model: PeakNet (weights: {self.config.model.weights_path is not None})")
            print(f"  Data: {self.config.data.shape} tensor shape")
            print(f"  Runtime: {self.config.runtime.batch_size} batch size, {self.config.runtime.num_producers} producers")
            print(f"  System: min {self.config.system.min_gpus} GPUs required")
            print()

    def _setup_gpu_environment(self) -> List[int]:
        """Set up GPU environment with health validation."""
        if self.config.system.skip_gpu_validation:
            if not self.config.output.quiet:
                print("\n⚡ Step 1: GPU Environment Setup (Production Mode)")
                print(f"✅ Skipping validation - trusting Ray cluster provides {self.config.system.min_gpus}+ healthy GPUs")
            return list(range(self.config.system.min_gpus))

        if not self.config.output.quiet:
            print("\n🔍 Step 1: GPU Health Validation")
            if self.config.output.verbose:
                print("   (Use --skip-gpu-validation for faster startup in production)")

        try:
            healthy_gpus = get_healthy_gpus_for_ray(min_gpus=self.config.system.min_gpus)

            if not self.config.output.quiet:
                print(f"✅ Found {len(healthy_gpus)} healthy GPUs")
                if self.config.output.verbose:
                    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
                    print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")

            return healthy_gpus

        except RuntimeError as e:
            error_msg = f"GPU validation failed: {e}"
            if not self.config.output.quiet:
                print(f"❌ {error_msg}")
                print("\n💡 Troubleshooting:")
                print("   - Ensure CUDA is available: nvidia-smi")
                print("   - Check for GPU hardware issues")
                print("   - Try reducing --min-gpus or --max-actors")
                print("   - Use --skip-gpu-validation for production clusters")
            raise RuntimeError(error_msg)

    def _setup_ray_cluster(self) -> None:
        """Initialize Ray cluster connection."""
        if not self.config.output.quiet:
            print("\n⚡ Step 2: Ray Cluster Setup")

        max_actors = self.config.runtime.max_actors

        if not ray.is_initialized():
            try:
                ray.init()
                cluster_resources = ray.cluster_resources()
                gpu_count = int(cluster_resources.get('GPU', 0))

                if not self.config.output.quiet:
                    print(f"✅ Ray cluster initialized")
                    print(f"   Available GPUs: {gpu_count}")
                    print(f"   CPU cores: {int(cluster_resources.get('CPU', 0))}")

                    if max_actors:
                        print(f"   Will create up to {max_actors} actors (user limit)")
                    else:
                        print(f"   Will auto-scale to use all healthy GPUs")

            except Exception as e:
                error_msg = f"Ray initialization failed: {e}"
                if not self.config.output.quiet:
                    print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
        else:
            cluster_resources = ray.cluster_resources()
            gpu_count = int(cluster_resources.get('GPU', 0))

            if not self.config.output.quiet:
                print(f"✅ Ray cluster already running")
                print(f"   Available GPUs: {gpu_count}")
                print(f"   CPU cores: {int(cluster_resources.get('CPU', 0))}")

                if max_actors:
                    print(f"   Will create up to {max_actors} actors (user limit)")
                else:
                    print(f"   Will auto-scale to use all healthy GPUs")

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
                    print(f"⚠️  Requested {max_actors} actors but only {max_possible_actors} healthy GPUs available")

        enable_profiling = self.config.profiling.enable_profiling
        profiling_text = " (with profiling)" if enable_profiling else ""

        if not self.config.output.quiet:
            print(f"\n🎭 Step 3: Creating {actual_num_actors} GPU Pipeline Actors{profiling_text}")
            print(f"   Available healthy GPUs: {max_possible_actors}")
            if max_actors:
                print(f"   User-specified actor limit: {max_actors}")

            if enable_profiling and self.config.output.verbose:
                print("   📊 NSys profiling enabled - profile files will be generated per actor")

        try:
            actors = create_pipeline_actors(
                num_actors=actual_num_actors,
                enable_profiling=enable_profiling,
                validate_gpus=False,  # Already validated at system level
                # Pipeline configuration
                input_shape=self.config.data.shape,
                batch_size=self.config.runtime.batch_size,
                # PeakNet configuration
                weights_path=self.config.model.weights_path,
                peaknet_config=self.config.model.peaknet_config,
                compile_mode=self.config.model.compile_mode,
                warmup_samples=self.config.model.warmup_samples,
                deterministic=True,
                pin_memory=self.config.system.pin_memory
            )

            if not self.config.output.quiet:
                print(f"✅ Successfully created {len(actors)} GPU actors")

            # Verify actor health if enabled
            if self.config.system.verify_actors:
                if not self.config.output.quiet:
                    print("   Verifying actor health...")
                health_futures = [actor.health_check.remote() for actor in actors]

                try:
                    health_results = ray.get(health_futures, timeout=30)
                    healthy_count = sum(1 for h in health_results if h.get('status') == 'healthy')

                    if not self.config.output.quiet:
                        print(f"   ✅ {healthy_count}/{len(actors)} actors are healthy")

                        if self.config.output.verbose:
                            for i, health in enumerate(health_results):
                                gpu_id = health.get('gpu_id', 'unknown')
                                status = health.get('status', 'unknown')
                                print(f"      Actor {i}: GPU {gpu_id} - {status}")

                except Exception as e:
                    if not self.config.output.quiet:
                        print(f"   ⚠️  Actor health check failed: {e}")
                        print("   Continuing anyway...")

            return actors

        except Exception as e:
            error_msg = f"Failed to create pipeline actors: {e}"
            if not self.config.output.quiet:
                print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    def _generate_streaming_data(self) -> List[Any]:
        """Generate streaming data using Ray tasks."""
        if not self.config.output.quiet:
            print(f"\n📊 Step 4: Generating Streaming Data")

        # Extract config values
        runtime = self.config.runtime
        data = self.config.data

        # Calculate actual production parameters based on total_samples if provided
        if runtime.total_samples is not None:
            # Calculate required batches to reach total_samples
            total_batches_needed = (runtime.total_samples + runtime.batch_size - 1) // runtime.batch_size
            batches_per_producer = max(1, total_batches_needed // runtime.num_producers)
            # Adjust if we need more producers or batches
            if total_batches_needed > runtime.num_producers * batches_per_producer:
                batches_per_producer += 1
            if not self.config.output.quiet:
                print(f"   Using total_samples={runtime.total_samples}")
                print(f"   Adjusted batches per producer: {batches_per_producer}")
        else:
            batches_per_producer = runtime.batches_per_producer

        if not self.config.output.quiet:
            print(f"   Producers: {runtime.num_producers}")
            print(f"   Batches per producer: {batches_per_producer}")
            print(f"   Total batches: {runtime.num_producers * batches_per_producer}")
            print(f"   Batch size: {runtime.batch_size} samples")
            print(f"   Total samples: {runtime.num_producers * batches_per_producer * runtime.batch_size}")
            print(f"   Input shape: {data.shape}")

        manager = RayDataProducerManager()
        start_time = time.time()

        try:
            producer_futures = manager.launch_producers(
                num_producers=runtime.num_producers,
                batches_per_producer=batches_per_producer,
                batch_size=runtime.batch_size,
                tensor_shape=data.shape,
                inter_batch_delay=runtime.inter_batch_delay,
                deterministic=False  # Random data for realistic streaming
            )

            all_batches = manager.get_all_batches()
            generation_time = time.time() - start_time

            total_samples = len(all_batches) * runtime.batch_size
            generation_rate = total_samples / generation_time

            if not self.config.output.quiet:
                print(f"✅ Data generation complete:")
                print(f"   Generated: {len(all_batches)} batches ({total_samples} samples)")
                print(f"   Time: {generation_time:.2f}s")
                print(f"   Rate: {generation_rate:.1f} samples/s")

            return all_batches

        except Exception as e:
            error_msg = f"Data generation failed: {e}"
            if not self.config.output.quiet:
                print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    def _process_streaming_data(self, actors: List[Any], all_batches: List[Any]) -> Dict[str, Any]:
        """Process streaming data across multiple GPU actors."""
        if not self.config.output.quiet:
            print(f"\n⚡ Step 5: Multi-GPU Streaming Processing")
            print(f"   Processing {len(all_batches)} batches across {len(actors)} GPUs")

        # Distribute batches across actors (round-robin)
        processing_futures = []
        actor_assignments = []

        for batch_idx, batch in enumerate(all_batches):
            actor_idx = batch_idx % len(actors)
            actor = actors[actor_idx]

            future = actor.process_batch_from_ray_object_store.remote(batch, batch_idx)
            processing_futures.append(future)
            actor_assignments.append(actor_idx)

        # Process batches and collect results
        if not self.config.output.quiet:
            print("   Processing batches...")
        processing_start = time.time()

        results = []
        completed = 0
        total_batches = len(processing_futures)

        try:
            for i, future in enumerate(processing_futures):
                result = ray.get(future, timeout=30)
                results.append(result)
                completed += 1

                if self.config.output.verbose or (completed % max(1, total_batches // 10) == 0):
                    progress = (completed / total_batches) * 100
                    actor_idx = actor_assignments[i]
                    samples = result['batch_size']
                    proc_time = result['processing_time']

                    if not self.config.output.quiet:
                        print(f"   [{progress:5.1f}%] Batch {completed}/{total_batches} "
                              f"→ GPU Actor {actor_idx}: {samples} samples ({proc_time:.3f}s)")

        except Exception as e:
            return {'success': False, 'error': str(e)}

        total_processing_time = time.time() - processing_start

        # Calculate performance metrics
        total_samples = sum(r['batch_size'] for r in results)
        overall_throughput = total_samples / total_processing_time

        # Per-actor statistics
        actor_stats = {}
        for i, result in enumerate(results):
            actor_idx = actor_assignments[i]
            if actor_idx not in actor_stats:
                actor_stats[actor_idx] = {'batches': 0, 'samples': 0, 'total_time': 0.0}

            actor_stats[actor_idx]['batches'] += 1
            actor_stats[actor_idx]['samples'] += result['batch_size']
            actor_stats[actor_idx]['total_time'] += result['processing_time']

        return {
            'success': True,
            'total_samples': total_samples,
            'total_batches': len(results),
            'total_processing_time': total_processing_time,
            'overall_throughput': overall_throughput,
            'actor_stats': actor_stats,
            'results': results
        }

    def _print_results(self, performance: Dict[str, Any]) -> None:
        """Print comprehensive performance results."""
        print("\n📈 Performance Results")
        print("=" * 30)

        if not performance['success']:
            print(f"❌ Processing failed: {performance.get('error', 'Unknown error')}")
            return

        # Overall metrics
        print(f"✅ Overall Performance:")
        print(f"   Total samples processed: {performance['total_samples']:,}")
        print(f"   Total batches: {performance['total_batches']:,}")
        print(f"   Processing time: {performance['total_processing_time']:.2f}s")
        print(f"   Overall throughput: {performance['overall_throughput']:.1f} samples/s")

        # Per-actor breakdown
        print(f"\n🎭 Per-Actor Performance:")
        actor_stats = performance['actor_stats']

        for actor_idx, stats in actor_stats.items():
            actor_throughput = stats['samples'] / stats['total_time'] if stats['total_time'] > 0 else 0
            avg_batch_time = stats['total_time'] / stats['batches'] if stats['batches'] > 0 else 0

            print(f"   GPU Actor {actor_idx}:")
            print(f"      Batches: {stats['batches']}")
            print(f"      Samples: {stats['samples']:,}")
            print(f"      Throughput: {actor_throughput:.1f} samples/s")
            print(f"      Avg batch time: {avg_batch_time:.3f}s")

        # Profiling information
        if self.config.profiling.enable_profiling:
            import os
            tmpdir = os.environ.get('TMPDIR', '/tmp')
            print(f"\n📊 Profiling Information:")
            print(f"   NSys profiling: enabled")
            print(f"   Profile files: generated per actor (nsys-rep format)")
            print(f"   📁 Files saved to: {tmpdir}/ray/session_latest/logs/nsight/")
            print(f"   💡 Find your .nsys-rep files in Ray's logs directory")
            print(f"   💡 Copy files locally: cp {tmpdir}/ray/session_latest/logs/nsight/*.nsys-rep ./")
            print(f"   💡 Analyze with: nsys-ui <file.nsys-rep> or nsys stats <file.nsys-rep>")

        # Configuration summary
        if self.config.output.verbose:
            runtime = self.config.runtime
            data = self.config.data
            print(f"\n⚙️  Configuration Used:")
            print(f"   Actor limit: {'auto-scale' if runtime.max_actors is None else runtime.max_actors}")
            print(f"   Min GPUs required: {self.config.system.min_gpus}")
            print(f"   Producers: {runtime.num_producers}")
            print(f"   Batch size: {runtime.batch_size}")
            print(f"   Input shape: {data.shape}")
            print(f"   Inter-batch delay: {runtime.inter_batch_delay}s")
            print(f"   Profiling: {'enabled' if self.config.profiling.enable_profiling else 'disabled'}")

        if performance['success']:
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"   Processed {performance['total_samples']:,} samples at {performance['overall_throughput']:.1f} samples/s")

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
            print(f"\n💾 Results saved to: {results_file}")

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
                print(f"\n⚠️  {error}")
            return PipelineResults(success=False, performance={}, error=error)

        except Exception as e:
            error = f"Streaming pipeline failed: {e}"
            if not self.config.output.quiet:
                print(f"\n💥 {error}")
                if self.config.output.verbose:
                    import traceback
                    traceback.print_exc()
            return PipelineResults(success=False, performance={}, error=str(e))

    def _print_streaming_banner(self) -> None:
        """Print streaming pipeline banner."""
        print("🌊 Ray Multi-GPU PeakNet STREAMING Pipeline")
        print("=" * 55)
        print("🚀 TRUE CONTINUOUS STREAMING - No batch accumulation!")
        print("🔄 Preserves double buffering - No per-batch sync!")
        print()

        if self.config.output.verbose:
            print(f"Configuration:")
            print(f"  Model: PeakNet (weights: {self.config.model.weights_path is not None})")
            print(f"  Data: {self.config.data.shape} tensor shape")
            print(f"  Runtime: {self.config.runtime.batch_size} batch size, {self.config.runtime.num_producers} producers")
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
            print("\n🌊 Starting Streaming Workflow")

        # Calculate production parameters
        runtime = self.config.runtime
        data = self.config.data

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
            print(f"   Producers: {runtime.num_producers}")
            print(f"   Actors: {len(actors)}")  
            print(f"   Batches per producer: {batches_per_producer}")
            print(f"   Expected total: {total_expected_batches} batches, {total_expected_samples} samples")
            print(f"   Input shape: {data.shape}")
            print(f"   Inter-batch delay: {runtime.inter_batch_delay}s")

        # Step 1: Create Coordinator
        if not self.config.output.quiet:
            print("\n📡 Step 1: Creating Streaming Coordinator")

        coordinator = create_streaming_coordinator(
            expected_producers=runtime.num_producers,
            expected_actors=len(actors)
        )

        # Step 2: Create Queues
        if not self.config.output.quiet:
            print("📦 Step 2: Creating Queue Infrastructure")

        # Input queue (Q1) - producers -> actors
        num_shards = runtime.queue_num_shards
        maxsize_per_shard = runtime.queue_maxsize_per_shard

        q1_manager = ShardedQueueManager(
            "streaming_input_queue", 
            num_shards=num_shards, 
            maxsize_per_shard=maxsize_per_shard
        )

        q2_manager = None
        if enable_output_queue:
            q2_manager = ShardedQueueManager(
                "streaming_output_queue",
                num_shards=num_shards,
                maxsize_per_shard=maxsize_per_shard
            )

        if not self.config.output.quiet:
            print(f"   Q1 (input): {num_shards} shards, {maxsize_per_shard} items/shard")
            if enable_output_queue:
                print(f"   Q2 (output): {num_shards} shards, {maxsize_per_shard} items/shard")

        # Step 3: Launch Streaming Producers
        if not self.config.output.quiet:
            print("🏭 Step 3: Launching Streaming Producers")

        producers = create_streaming_producers(
            num_producers=runtime.num_producers,
            batch_size=runtime.batch_size,
            tensor_shape=data.shape,
            inter_batch_delay=runtime.inter_batch_delay,
            deterministic=False
        )

        producer_tasks = []
        for i, producer in enumerate(producers):
            task = producer.stream_batches_to_queue.remote(
                q1_manager, 
                batches_per_producer, 
                coordinator,
                progress_interval=max(10, batches_per_producer // 10)
            )
            producer_tasks.append(task)

        # Step 4: Launch Streaming Pipeline Actors  
        if not self.config.output.quiet:
            print("🎭 Step 4: Launching Streaming Pipeline Actors")

        actor_tasks = []
        for i, actor in enumerate(actors):
            task = actor.process_from_queue.remote(
                q1_manager,
                q2_manager,
                coordinator,
                max_empty_polls=20,  # Check coordinator after 20 empty polls
                poll_timeout=0.01,   # 10ms poll timeout
                memory_sync_interval=200  # Sync every 200 batches for memory management
            )
            actor_tasks.append(task)

        # Step 5: Monitor and Wait for Completion
        if not self.config.output.quiet:
            print("⏳ Step 5: Streaming Processing (Real-time)")
            print("   📊 Producers generating data...")
            print("   ⚡ Actors processing continuously...")
            print("   🔄 Double buffering preserved - no per-batch sync!")

        start_time = time.time()

        # Wait for producers to complete
        if not self.config.output.quiet:
            print("\n   Waiting for producers to finish...")

        producer_results = ray.get(producer_tasks)
        producer_time = time.time() - start_time

        if not self.config.output.quiet:
            total_produced_samples = sum(r['total_samples'] for r in producer_results)
            total_backpressure = sum(r['backpressure_events'] for r in producer_results)
            print(f"   ✅ All producers finished in {producer_time:.2f}s")
            print(f"   📊 Produced: {total_produced_samples} samples, {total_backpressure} backpressure events")

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
            'sample_completion_rate': total_processed_samples / total_expected_samples if total_expected_samples > 0 else 0,
            'queue_config': {
                'q1_shards': num_shards,
                'q1_maxsize_per_shard': maxsize_per_shard,
                'output_queue_enabled': enable_output_queue
            }
        }

    def _print_streaming_results(self, performance: Dict[str, Any]) -> None:
        """Print streaming pipeline performance results."""
        print("\n📈 Streaming Pipeline Results")
        print("=" * 40)

        if not performance['success']:
            print(f"❌ Processing failed: {performance.get('error', 'Unknown error')}")
            return

        # Overall metrics
        print(f"🌊 Streaming Performance:")
        print(f"   Total samples processed: {performance['total_samples']:,}")
        print(f"   Total batches processed: {performance['total_batches']:,}")
        print(f"   Expected samples: {performance['expected_samples']:,}")
        print(f"   Completion rate: {performance['sample_completion_rate']:.1%}")
        print(f"   Total time: {performance['total_processing_time']:.2f}s")
        print(f"   Overall throughput: {performance['overall_throughput']:.1f} samples/s")

        # Producer performance
        print(f"\n🏭 Producer Performance:")
        for i, result in enumerate(performance['producer_results']):
            print(f"   Producer {i}: {result['total_samples']:,} samples, "
                  f"{result['backpressure_events']} backpressure events")

        # Actor performance
        print(f"\n🎭 Actor Performance (TRUE STREAMING - No per-batch sync!):")
        for actor_id, stats in performance['actor_stats'].items():
            print(f"   GPU Actor {actor_id}: {stats['batches']} batches, "
                  f"{stats['samples']:,} samples, {stats['throughput']:.1f} samples/s")

        # Queue configuration
        queue_config = performance['queue_config']
        print(f"\n📦 Queue Configuration:")
        print(f"   Input queue: {queue_config['q1_shards']} shards × {queue_config['q1_maxsize_per_shard']} items")
        print(f"   Output queue: {'enabled' if queue_config['output_queue_enabled'] else 'disabled'}")

        # Profiling information
        if self.config.profiling.enable_profiling:
            import os
            tmpdir = os.environ.get('TMPDIR', '/tmp')
            print(f"\n📊 Profiling Information:")
            print(f"   🔍 Check nsys profile for continuous overlapping operations")
            print(f"   ✅ Should show NO cudaStreamSynchronize blocking in hot path")
            print(f"   📁 Profile files: {tmpdir}/ray/session_latest/logs/nsight/")

        if performance['success']:
            print(f"\n🎉 Streaming Pipeline Completed Successfully!")
            print(f"   🌊 TRUE CONTINUOUS PROCESSING: {performance['total_samples']:,} samples")
            print(f"   ⚡ DOUBLE BUFFERING PRESERVED: No per-batch synchronization")
            print(f"   📈 Throughput: {performance['overall_throughput']:.1f} samples/s")