#!/usr/bin/env python3
"""
Command-line interface for PeakNet Pipeline Ray.

A production-ready interface for running PeakNet segmentation model inference at scale 
with streaming data sources across multiple GPUs using Ray.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..config import PipelineConfig


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description='PeakNet Pipeline Ray - Scalable ML inference with streaming data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file (e.g., examples/configs/production.yaml)'
    )

    # Model configuration
    model_group = parser.add_argument_group('model', 'Model configuration options')
    model_group.add_argument(
        '--model-yaml',
        dest='model_yaml_path',
        type=str,
        help='Path to PeakNet model YAML configuration file'
    )
    model_group.add_argument(
        '--model-weights',
        dest='model_weights_path', 
        type=str,
        help='Path to pretrained model weights file'
    )
    model_group.add_argument(
        '--compile-mode',
        type=str,
        choices=[None, 'default', 'reduce-overhead', 'max-autotune'],
        help='PyTorch compilation mode (None = no compilation, default: None)'
    )
    model_group.add_argument(
        '--warmup-samples',
        type=int,
        help='Number of warmup samples (0 = skip warmup, default: 500)'
    )

    # Runtime configuration
    runtime_group = parser.add_argument_group('runtime', 'Pipeline runtime options')
    runtime_group.add_argument(
        '--max-actors',
        type=int,
        help='Maximum number of pipeline actors (default: auto-scale to available GPUs)'
    )
    runtime_group.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for processing'
    )
    runtime_group.add_argument(
        '--total-samples',
        type=int,
        help='Total number of samples to process'
    )
    runtime_group.add_argument(
        '--num-producers',
        type=int,
        help='Number of data producers'
    )
    runtime_group.add_argument(
        '--batches-per-producer',
        type=int,
        help='Number of batches per producer'
    )
    runtime_group.add_argument(
        '--inter-batch-delay',
        type=float,
        help='Delay between batches in seconds'
    )
    runtime_group.add_argument(
        '--queue-shards',
        type=int,
        help='Number of queue shards for parallel access'
    )
    runtime_group.add_argument(
        '--queue-size-per-shard',
        type=int,
        help='Maximum items per queue shard'
    )

    # Data configuration
    data_group = parser.add_argument_group('data', 'Data configuration options')
    data_group.add_argument(
        '--shape',
        nargs=3,
        type=int,
        metavar=('C', 'H', 'W'),
        help='Input tensor shape as channels, height, width (e.g., --shape 1 512 512)'
    )

    # Data source configuration
    data_source_group = parser.add_argument_group('data_source', 'Data source configuration options')
    data_source_group.add_argument(
        '--data-source',
        type=str,
        choices=['random', 'socket'],
        help='Data source type: random (synthetic data) or socket (from LCLStreamer)'
    )
    data_source_group.add_argument(
        '--socket-hostname',
        type=str,
        help='Socket hostname for data source (e.g., sdfada015)'
    )
    data_source_group.add_argument(
        '--socket-port',
        type=int,
        help='Socket port for data source (default: 12321)'
    )
    data_source_group.add_argument(
        '--socket-timeout',
        type=float,
        help='Socket receive timeout in seconds (default: 10.0)'
    )

    # System configuration
    system_group = parser.add_argument_group('system', 'System configuration options')
    system_group.add_argument(
        '--min-gpus',
        type=int,
        help='Minimum number of GPUs required'
    )
    system_group.add_argument(
        '--skip-gpu-validation',
        action='store_true',
        help='Skip GPU health validation'
    )
    system_group.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='Disable memory pinning'
    )
    system_group.add_argument(
        '--no-verify-actors',
        action='store_true',
        help='Disable actor verification'
    )

    # Profiling and output
    profiling_group = parser.add_argument_group('profiling', 'Profiling and output options')
    profiling_group.add_argument(
        '--enable-profiling',
        action='store_true',
        help='Enable performance profiling'
    )
    profiling_group.add_argument(
        '--profiling-output-dir',
        type=str,
        help='Directory for profiling output'
    )
    profiling_group.add_argument(
        '--output-dir',
        type=str,
        help='Directory for pipeline output'
    )
    profiling_group.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    profiling_group.add_argument(
        '--quiet',
        action='store_true',
        help='Enable quiet mode (minimal output)'
    )

    return parser


def load_config(args: argparse.Namespace) -> PipelineConfig:
    """Load configuration from file and command-line overrides."""
    # Start with default config
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        config = PipelineConfig.from_yaml(str(config_path))
    else:
        config = PipelineConfig()

    # Apply command-line overrides
    if args.model_yaml_path is not None:
        config.model.yaml_path = args.model_yaml_path
    if args.model_weights_path is not None:
        config.model.weights_path = args.model_weights_path
    if args.compile_mode is not None:
        config.model.compile_mode = args.compile_mode
    if args.warmup_samples is not None:
        config.model.warmup_samples = args.warmup_samples

    if args.max_actors is not None:
        config.runtime.max_actors = args.max_actors
    if args.batch_size is not None:
        config.runtime.batch_size = args.batch_size
    if args.total_samples is not None:
        config.runtime.total_samples = args.total_samples
    if args.num_producers is not None:
        config.runtime.num_producers = args.num_producers
    if args.batches_per_producer is not None:
        config.runtime.batches_per_producer = args.batches_per_producer
    if args.inter_batch_delay is not None:
        config.runtime.inter_batch_delay = args.inter_batch_delay
    if args.queue_shards is not None:
        config.runtime.queue_num_shards = args.queue_shards
    if args.queue_size_per_shard is not None:
        config.runtime.queue_maxsize_per_shard = args.queue_size_per_shard

    if args.shape is not None:
        config.data.shape = tuple(args.shape)

    # Data source overrides
    if hasattr(args, 'data_source') and args.data_source is not None:
        config.data_source.source_type = args.data_source
    if hasattr(args, 'socket_hostname') and args.socket_hostname is not None:
        config.data_source.socket_hostname = args.socket_hostname
    if hasattr(args, 'socket_port') and args.socket_port is not None:
        config.data_source.socket_port = args.socket_port
    if hasattr(args, 'socket_timeout') and args.socket_timeout is not None:
        config.data_source.socket_timeout = args.socket_timeout

    if args.min_gpus is not None:
        config.system.min_gpus = args.min_gpus
    if args.skip_gpu_validation:
        config.system.skip_gpu_validation = True
    if args.no_pin_memory:
        config.system.pin_memory = False
    if args.no_verify_actors:
        config.system.verify_actors = False

    if args.enable_profiling:
        config.profiling.enable_profiling = True
    if args.profiling_output_dir is not None:
        config.profiling.output_dir = args.profiling_output_dir
    if args.output_dir is not None:
        config.output.output_dir = args.output_dir
    if args.verbose:
        config.output.verbose = True
    if args.quiet:
        config.output.quiet = True

    return config


def main(argv: Optional[list] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Validate conflicting arguments
    if args.verbose and args.quiet:
        print("Error: --verbose and --quiet are mutually exclusive", file=sys.stderr)
        return 1

    try:
        # Load configuration
        config = load_config(args)

        # Print configuration if verbose
        if config.output.verbose:
            print("Pipeline Configuration:")
            print(f"  Model YAML: {config.model.yaml_path}")
            print(f"  Model weights: {config.model.weights_path}")
            print(f"  Max actors: {config.runtime.max_actors}")
            print(f"  Batch size: {config.runtime.batch_size}")
            print(f"  Total samples: {config.runtime.total_samples}")
            print(f"  Data shape: {config.data.shape}")
            print(f"  Data source: {config.data_source.source_type}")
            if config.data_source.source_type == "socket":
                print(f"  Socket: {config.data_source.socket_hostname}:{config.data_source.socket_port}")
            print(f"  Min GPUs: {config.system.min_gpus}")
            print(f"  Profiling: {config.profiling.enable_profiling}")

        # Import and run the actual pipeline
        from ..pipeline import PeakNetPipeline

        # Create and run streaming pipeline
        pipeline = PeakNetPipeline(config)
        results = pipeline.run_streaming_pipeline()

        # Return appropriate exit code
        return 0 if results.success else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())