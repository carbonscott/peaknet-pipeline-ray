#!/usr/bin/env python3
"""
Complete integration test for shape transform functionality.

Tests the full pipeline from configuration parsing to transform application,
simulating the real scenario where pusher provides (B, H, W) data and
model expects (B, C, H, W) data.
"""

import sys
import os
import torch
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add paths
sys.path.insert(0, '/sdf/home/c/cwang31/codes/peaknet')

# Import pipeline components
from peaknet_pipeline_ray.core.peaknet_ray_pipeline_actor import PeakNetPipelineActorBase

def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_config_with_transforms():
    """Test configuration with transforms enabled."""
    print("=== Testing Configuration with Transforms ===")

    config_path = "examples/configs/peaknet.yaml"
    config = load_config(config_path)

    print(f"Loaded config from: {config_path}")
    print(f"Transform config: {config.get('transforms', {})}")

    # Extract transform configuration
    transform_config = config.get('transforms', {})
    data_shape = tuple(config['data']['shape'])  # (C, H, W)

    print(f"Data shape: {data_shape}")
    print(f"Add channel dimension: {transform_config.get('add_channel_dimension', False)}")
    print(f"Pad to target: {transform_config.get('pad_to_target', False)}")

    # Create mock actor to test transform creation
    class TestActor(PeakNetPipelineActorBase):
        def __init__(self):
            pass

    test_actor = TestActor()
    transforms = test_actor._create_transform_chain(transform_config, data_shape)

    print(f"Created {len(transforms)} transforms:")
    for i, transform in enumerate(transforms):
        print(f"  {i+1}. {transform.__class__.__name__}")

    assert len(transforms) == 2, f"Expected 2 transforms, got {len(transforms)}"
    print("✅ Configuration with transforms test passed!")

def test_config_without_transforms():
    """Test configuration without transforms."""
    print("\n=== Testing Configuration without Transforms ===")

    config_path = "examples/configs/peaknet-no-transforms.yaml"
    config = load_config(config_path)

    print(f"Loaded config from: {config_path}")
    print(f"Transform config: {config.get('transforms', {})}")

    # Extract transform configuration
    transform_config = config.get('transforms', {})
    data_shape = tuple(config['data']['shape'])  # (C, H, W)

    print(f"Data shape: {data_shape}")
    print(f"Add channel dimension: {transform_config.get('add_channel_dimension', False)}")
    print(f"Pad to target: {transform_config.get('pad_to_target', False)}")

    # Create mock actor to test transform creation
    class TestActor(PeakNetPipelineActorBase):
        def __init__(self):
            pass

    test_actor = TestActor()
    transforms = test_actor._create_transform_chain(transform_config, data_shape)

    print(f"Created {len(transforms)} transforms:")
    for i, transform in enumerate(transforms):
        print(f"  {i+1}. {transform.__class__.__name__}")

    assert len(transforms) == 0, f"Expected 0 transforms, got {len(transforms)}"
    print("✅ Configuration without transforms test passed!")

def test_realistic_scenario():
    """Test realistic scenario with actual data shapes from LCLStreamer."""
    print("\n=== Testing Realistic LCLStreamer Scenario ===")

    # Realistic scenario:
    # - LCLStreamer pushes detector data with shape (1691, 1691) per sample
    # - Pipeline receives batches of size 16: list of 16 tensors, each (1691, 1691)
    # - Model expects: (16, 1, 1920, 1920)

    batch_size = 16
    pusher_shape_per_sample = (1691, 1691)  # From real detector data
    target_model_shape = (1, 1920, 1920)    # Model expectation (C, H, W)

    print(f"Scenario:")
    print(f"  Pusher shape per sample: {pusher_shape_per_sample}")
    print(f"  Batch size: {batch_size}")
    print(f"  Target model shape: {target_model_shape}")

    # Create realistic CPU batch (what pipeline receives)
    cpu_batch = []
    for i in range(batch_size):
        # Each tensor has shape (1691, 1691) - no channel dimension
        tensor = torch.randn(*pusher_shape_per_sample)
        cpu_batch.append(tensor)

    print(f"  Created CPU batch: {len(cpu_batch)} tensors, each shape {cpu_batch[0].shape}")

    # Test transform chain
    from peaknet.tensor_transforms import AddChannelDimension, Pad

    transforms = [
        AddChannelDimension(channel_dim=1, num_channels=1),  # (B,H,W) -> (B,1,H,W)
        Pad(1920, 1920, pad_style='center')                  # (B,1,1691,1691) -> (B,1,1920,1920)
    ]

    # Stack into batch tensor (what pipeline does internally)
    batch_tensor = torch.stack(cpu_batch)
    print(f"  Stacked batch tensor: {batch_tensor.shape}")

    # Apply transforms
    result = batch_tensor
    for transform in transforms:
        before_shape = result.shape
        result = transform(result)
        print(f"  {transform.__class__.__name__}: {before_shape} -> {result.shape}")

    # Verify final shape
    expected_final = torch.Size([batch_size, 1, 1920, 1920])
    assert result.shape == expected_final, f"Expected {expected_final}, got {result.shape}"

    print(f"  Final shape: {result.shape} ✅")
    print("✅ Realistic scenario test passed!")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")

    # Test 1: Smaller input than target (needs padding)
    print("1. Testing smaller input (800x800 -> 1920x1920)...")
    from peaknet.tensor_transforms import AddChannelDimension, Pad

    small_batch = torch.randn(4, 800, 800)  # (B, H, W)
    transforms = [
        AddChannelDimension(),
        Pad(1920, 1920)
    ]

    result = small_batch
    for transform in transforms:
        result = transform(result)

    expected = torch.Size([4, 1, 1920, 1920])
    assert result.shape == expected, f"Expected {expected}, got {result.shape}"
    print("   ✅ Small input padded correctly")

    # Test 2: Larger input than target (should still work - pad adds 0)
    print("2. Testing larger input (2500x2500 -> 1920x1920)...")
    large_batch = torch.randn(4, 2500, 2500)  # (B, H, W)

    result = large_batch
    for transform in transforms:
        result = transform(result)

    # Pad doesn't crop, so this should become larger, not smaller
    expected_large = torch.Size([4, 1, 2500, 2500])  # Pad doesn't crop
    assert result.shape == expected_large, f"Expected {expected_large}, got {result.shape}"
    print("   ✅ Large input handled correctly (Pad doesn't crop)")

    # Test 3: Exact size input (no padding needed)
    print("3. Testing exact size input (1920x1920 -> 1920x1920)...")
    exact_batch = torch.randn(4, 1920, 1920)  # (B, H, W)

    result = exact_batch
    for transform in transforms:
        result = transform(result)

    expected_exact = torch.Size([4, 1, 1920, 1920])
    assert result.shape == expected_exact, f"Expected {expected_exact}, got {result.shape}"
    print("   ✅ Exact size input handled correctly")

    print("✅ Edge cases test passed!")

if __name__ == "__main__":
    try:
        test_config_with_transforms()
        test_config_without_transforms()
        test_realistic_scenario()
        test_edge_cases()

        print("\n🎉 ALL INTEGRATION TESTS PASSED! 🎉")
        print("\nSummary:")
        print("✅ Transform configuration parsing works")
        print("✅ Transform chain creation works")
        print("✅ Realistic LCLStreamer scenario works")
        print("✅ Edge cases handled correctly")
        print("\nThe shape transform solution is ready for production!")

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)