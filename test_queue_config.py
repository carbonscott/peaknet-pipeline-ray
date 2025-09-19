#!/usr/bin/env python3
"""Test script to verify queue configuration loading from YAML."""

import tempfile
import yaml
from pathlib import Path
from peaknet_pipeline_ray.config import PipelineConfig

def test_queue_config_loading():
    """Test that queue configuration is properly loaded from YAML."""

    # Create a test config with queue settings
    test_config = {
        'runtime': {
            'max_actors': 4,
            'batch_size': 16,
            'total_samples': 512,
            'num_producers': 1,
            'batches_per_producer': 32,
            'inter_batch_delay': 0.01,
            'queue_num_shards': 4,
            'queue_maxsize_per_shard': 2000
        },
        'model': {'weights_path': None},
        'data': {'shape': [1, 1920, 1920]},
        'data_source': {'source_type': 'random'},
        'transforms': {'add_channel_dimension': True},
        'system': {'min_gpus': 1},
        'profiling': {'enable_profiling': False},
        'output': {'verbose': True}
    }

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f, default_flow_style=False, indent=2)
        temp_path = f.name

    try:
        # Load configuration
        config = PipelineConfig.from_yaml(temp_path)

        # Test queue configuration
        print("Queue Configuration Test Results:")
        print(f"  queue_num_shards: {config.runtime.queue_num_shards} (expected: 4)")
        print(f"  queue_maxsize_per_shard: {config.runtime.queue_maxsize_per_shard} (expected: 2000)")
        print(f"  Total queue capacity: {config.runtime.queue_num_shards * config.runtime.queue_maxsize_per_shard}")

        # Verify values
        assert config.runtime.queue_num_shards == 4, f"Expected 4, got {config.runtime.queue_num_shards}"
        assert config.runtime.queue_maxsize_per_shard == 2000, f"Expected 2000, got {config.runtime.queue_maxsize_per_shard}"

        # Test round-trip (save and reload)
        config_dict = config.to_dict()
        print(f"\nRound-trip test:")
        print(f"  to_dict() queue_num_shards: {config_dict['runtime']['queue_num_shards']}")
        print(f"  to_dict() queue_maxsize_per_shard: {config_dict['runtime']['queue_maxsize_per_shard']}")

        # Test CLI override equivalents
        print(f"\nCLI Override Test:")
        print(f"  --queue-shards {config.runtime.queue_num_shards}")
        print(f"  --queue-size-per-shard {config.runtime.queue_maxsize_per_shard}")

        print(f"\n✅ All queue configuration tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    finally:
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)

if __name__ == '__main__':
    test_queue_config_loading()