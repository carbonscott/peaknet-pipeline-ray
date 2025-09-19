#!/usr/bin/env python3
"""Test script to verify coordination timing configuration."""

from peaknet_pipeline_ray.config import PipelineConfig

def test_coordination_config():
    """Test coordination timing parameters in TEST_DIR config."""

    test_config_path = "/sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml/peaknet.yaml"

    print("Testing coordination timing configuration...")
    print(f"Loading config from: {test_config_path}")

    # Load configuration
    config = PipelineConfig.from_yaml(test_config_path)

    # Test coordination timing
    print("\n=== Coordination Timing Configuration ===")
    print(f"  max_empty_polls: {config.runtime.max_empty_polls}")
    print(f"  poll_timeout: {config.runtime.poll_timeout}s")
    print(f"  Coordinator check delay: {config.runtime.max_empty_polls * config.runtime.poll_timeout:.1f}s")

    # Test queue configuration
    print("\n=== Queue Configuration ===")
    print(f"  queue_num_shards: {config.runtime.queue_num_shards}")
    print(f"  queue_maxsize_per_shard: {config.runtime.queue_maxsize_per_shard}")
    print(f"  Total capacity: {config.runtime.queue_num_shards * config.runtime.queue_maxsize_per_shard}")

    # Verify values
    expected_polls = 500
    expected_timeout = 0.001

    assert config.runtime.max_empty_polls == expected_polls, f"Expected {expected_polls}, got {config.runtime.max_empty_polls}"
    assert config.runtime.poll_timeout == expected_timeout, f"Expected {expected_timeout}, got {config.runtime.poll_timeout}"

    # Calculate gap elimination benefit
    old_delay = 20 * 0.01  # Old configuration: 0.2s
    new_delay = config.runtime.max_empty_polls * config.runtime.poll_timeout  # New: 0.5s

    print(f"\n=== Gap Analysis ===")
    print(f"  OLD: 20 polls × 10ms = {old_delay:.1f}s coordinator check delay")
    print(f"  NEW: {config.runtime.max_empty_polls} polls × {config.runtime.poll_timeout*1000:.1f}ms = {new_delay:.1f}s coordinator check delay")
    print(f"  Trade-off: Longer delay before coordination, but faster individual polls")
    print(f"  Result: Reduced coordination overhead during active processing")

    print(f"\n✅ Coordination timing configuration test passed!")
    return True

if __name__ == '__main__':
    test_coordination_config()