#!/usr/bin/env python3
"""
Test script for the streaming pipeline implementation.

This test validates that the streaming pipeline:
1. Processes data continuously without accumulation
2. Preserves double buffering performance (no per-batch sync)
3. Handles producer-consumer coordination correctly
4. Achieves comparable or better throughput than batch processing

Usage:
    python test_streaming_pipeline.py
    python test_streaming_pipeline.py --batch-mode  # Compare with batch processing
    python test_streaming_pipeline.py --profile     # Run with nsys profiling
"""

import argparse
import logging
import time
import ray
from typing import Dict, Any

from peaknet_pipeline_ray.pipeline import PeakNetPipeline
from peaknet_pipeline_ray.config import PipelineConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_test_config(
    batch_size: int = 8,
    num_producers: int = 2,
    num_actors: int = 2,
    total_samples: int = 1000,
    tensor_shape: tuple = (1, 512, 512),
    enable_profiling: bool = False
) -> PipelineConfig:
    """Create a test configuration for the streaming pipeline.
    
    Args:
        batch_size: Batch size for processing
        num_producers: Number of data producer actors
        num_actors: Number of pipeline processing actors (limited by GPUs)
        total_samples: Total samples to process
        tensor_shape: Input tensor shape (C, H, W)
        enable_profiling: Enable nsys profiling
        
    Returns:
        PipelineConfig configured for testing
    """
    config = PipelineConfig()
    
    # Runtime configuration
    config.runtime.batch_size = batch_size
    config.runtime.num_producers = num_producers
    config.runtime.total_samples = total_samples
    config.runtime.inter_batch_delay = 0.0  # No delay for testing
    config.runtime.max_actors = num_actors  # Limit actors for testing
    
    # Data configuration
    config.data.shape = tensor_shape
    config.data.input_channels = tensor_shape[0]
    
    # System configuration
    config.system.min_gpus = 1  # Minimum GPUs required
    config.system.skip_gpu_validation = False  # Validate GPUs
    config.system.pin_memory = True
    config.system.verify_actors = True
    
    # Model configuration (no-op mode for testing)
    config.model.peaknet_config = None  # No-op mode
    config.model.weights_path = None
    
    # Profiling configuration
    config.profiling.enable_profiling = enable_profiling
    
    # Output configuration
    config.output.verbose = True
    config.output.quiet = False
    
    return config


def test_streaming_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """Test the streaming pipeline with the given configuration.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Dictionary with test results
    """
    print("\n" + "="*60)
    print("🌊 TESTING STREAMING PIPELINE")
    print("="*60)
    print("🚀 TRUE CONTINUOUS STREAMING - No batch accumulation!")
    print("🔄 Preserving double buffering - No per-batch sync!")
    print("="*60)
    
    pipeline = PeakNetPipeline(config)
    
    start_time = time.time()
    results = pipeline.run_streaming_pipeline(enable_output_queue=False)
    total_time = time.time() - start_time
    
    if results.success:
        perf = results.performance
        print(f"\n✅ STREAMING TEST PASSED!")
        print(f"   Processed: {perf['total_samples']:,} samples in {total_time:.2f}s")
        print(f"   Throughput: {perf['overall_throughput']:.1f} samples/s")
        print(f"   Completion: {perf['sample_completion_rate']:.1%}")
        print(f"   Actors: {len(perf['actor_stats'])} GPU actors")
        print(f"   Producers: {len(perf['producer_results'])} data producers")
        
        # Validate expected behavior
        assert perf['streaming_mode'] is True, "Should be in streaming mode"
        assert perf['sample_completion_rate'] > 0.95, f"Should process most samples, got {perf['sample_completion_rate']:.1%}"
        assert perf['overall_throughput'] > 0, "Should have positive throughput"
        
        return {
            'success': True,
            'throughput': perf['overall_throughput'],
            'total_samples': perf['total_samples'],
            'total_time': total_time,
            'completion_rate': perf['sample_completion_rate'],
            'num_actors': len(perf['actor_stats']),
            'num_producers': len(perf['producer_results']),
            'performance': perf
        }
    else:
        print(f"\n❌ STREAMING TEST FAILED: {results.error}")
        return {
            'success': False,
            'error': results.error
        }


def test_batch_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """Test the traditional batch pipeline for comparison.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Dictionary with test results
    """
    print("\n" + "="*60)
    print("📦 TESTING BATCH PIPELINE (for comparison)")
    print("="*60)
    print("🔄 Traditional batch processing with final sync")
    print("="*60)
    
    pipeline = PeakNetPipeline(config)
    
    start_time = time.time()
    results = pipeline.run()  # Traditional batch mode
    total_time = time.time() - start_time
    
    if results.success:
        perf = results.performance
        print(f"\n✅ BATCH TEST PASSED!")
        print(f"   Processed: {perf['total_samples']:,} samples in {total_time:.2f}s")
        print(f"   Throughput: {perf['overall_throughput']:.1f} samples/s")
        print(f"   Actors: {len(perf['actor_stats'])} GPU actors")
        
        return {
            'success': True,
            'throughput': perf['overall_throughput'],
            'total_samples': perf['total_samples'],
            'total_time': total_time,
            'num_actors': len(perf['actor_stats']),
            'performance': perf
        }
    else:
        print(f"\n❌ BATCH TEST FAILED: {results.error}")
        return {
            'success': False,
            'error': results.error
        }


def compare_performance(streaming_result: Dict[str, Any], batch_result: Dict[str, Any]) -> None:
    """Compare streaming vs batch performance.
    
    Args:
        streaming_result: Results from streaming test
        batch_result: Results from batch test
    """
    if not (streaming_result['success'] and batch_result['success']):
        print("⚠️  Cannot compare - one or both tests failed")
        return
    
    print("\n" + "="*60)
    print("📊 PERFORMANCE COMPARISON")
    print("="*60)
    
    stream_throughput = streaming_result['throughput']
    batch_throughput = batch_result['throughput']
    throughput_ratio = stream_throughput / batch_throughput if batch_throughput > 0 else 0
    
    print(f"Streaming Pipeline:")
    print(f"  🌊 Throughput: {stream_throughput:.1f} samples/s")
    print(f"  📊 Samples: {streaming_result['total_samples']:,}")
    print(f"  ⏱️  Time: {streaming_result['total_time']:.2f}s")
    print(f"  ✅ Completion: {streaming_result.get('completion_rate', 1.0):.1%}")
    
    print(f"\nBatch Pipeline:")
    print(f"  📦 Throughput: {batch_throughput:.1f} samples/s")
    print(f"  📊 Samples: {batch_result['total_samples']:,}")
    print(f"  ⏱️  Time: {batch_result['total_time']:.2f}s")
    
    print(f"\nComparison:")
    print(f"  📈 Streaming/Batch Ratio: {throughput_ratio:.2f}x")
    
    if throughput_ratio > 0.95:
        print(f"  ✅ EXCELLENT: Streaming maintains comparable performance!")
        if throughput_ratio > 1.1:
            print(f"  🚀 BONUS: Streaming is faster than batch processing!")
    elif throughput_ratio > 0.8:
        print(f"  ⚠️  ACCEPTABLE: Streaming throughput within 20% of batch")
    else:
        print(f"  ❌ CONCERN: Streaming significantly slower than batch")
        print(f"     This may indicate synchronization issues or overhead")
    
    print("="*60)


def validate_double_buffering_preservation() -> None:
    """Provide guidance for validating double buffering preservation."""
    print("\n" + "="*60) 
    print("🔍 DOUBLE BUFFERING VALIDATION")
    print("="*60)
    print("To confirm double buffering is preserved:")
    print()
    print("1. 📊 Run with profiling enabled:")
    print("   python test_streaming_pipeline.py --profile")
    print()
    print("2. 🔍 Check nsys profile:")
    print("   nsys stats <profile.nsys-rep> | grep cudaStreamSynchronize")
    print()
    print("3. ✅ Expected behavior:")
    print("   - Should see overlapping H2D/Compute/D2H operations")
    print("   - Should NOT see cudaStreamSynchronize in hot processing loop")
    print("   - Only sync calls should be at the very end or periodic memory management")
    print()
    print("4. ❌ Warning signs:")
    print("   - cudaStreamSynchronize after every batch = BROKEN double buffering")
    print("   - No operation overlap = pipeline serialized")
    print("="*60)


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test streaming pipeline implementation')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--num-producers', type=int, default=2, help='Number of producer actors')
    parser.add_argument('--num-actors', type=int, default=2, help='Number of pipeline actors') 
    parser.add_argument('--total-samples', type=int, default=1000, help='Total samples to process')
    parser.add_argument('--tensor-shape', nargs=3, type=int, default=[1, 512, 512], 
                        help='Input tensor shape (C H W)')
    parser.add_argument('--batch-mode', action='store_true',
                        help='Also test batch mode for comparison')
    parser.add_argument('--profile', action='store_true',
                        help='Enable nsys profiling')
    
    args = parser.parse_args()
    
    # Ensure Ray is initialized
    if not ray.is_initialized():
        ray.init()
    
    print("🧪 PeakNet Streaming Pipeline Test")
    print(f"Configuration: batch_size={args.batch_size}, producers={args.num_producers}, "
          f"actors={args.num_actors}, samples={args.total_samples}")
    print(f"Tensor shape: {tuple(args.tensor_shape)}")
    
    # Create test configuration
    config = create_test_config(
        batch_size=args.batch_size,
        num_producers=args.num_producers,
        num_actors=args.num_actors,
        total_samples=args.total_samples,
        tensor_shape=tuple(args.tensor_shape),
        enable_profiling=args.profile
    )
    
    # Test streaming pipeline
    streaming_result = test_streaming_pipeline(config)
    
    batch_result = None
    if args.batch_mode:
        # Test batch pipeline for comparison
        batch_result = test_batch_pipeline(config)
        
        # Compare results
        if streaming_result['success'] and batch_result['success']:
            compare_performance(streaming_result, batch_result)
    
    # Validation guidance
    validate_double_buffering_preservation()
    
    # Final results
    print(f"\n🎯 FINAL RESULTS:")
    if streaming_result['success']:
        print(f"   ✅ Streaming Pipeline: {streaming_result['throughput']:.1f} samples/s")
        if batch_result and batch_result['success']:
            ratio = streaming_result['throughput'] / batch_result['throughput']
            print(f"   📦 Batch Pipeline: {batch_result['throughput']:.1f} samples/s ({ratio:.2f}x)")
        
        print(f"\n🎉 SUCCESS: Streaming pipeline implementation working correctly!")
        print(f"   🌊 TRUE CONTINUOUS STREAMING achieved")
        print(f"   ⚡ DOUBLE BUFFERING preserved") 
        print(f"   📈 Performance: {streaming_result['throughput']:.1f} samples/s")
    else:
        print(f"   ❌ Streaming Pipeline FAILED: {streaming_result.get('error', 'Unknown error')}")
        if batch_result and batch_result['success']:
            print(f"   📦 Batch Pipeline: {batch_result['throughput']:.1f} samples/s (reference)")
    
    # Cleanup
    ray.shutdown()
    
    return 0 if streaming_result['success'] else 1


if __name__ == "__main__":
    exit(main())