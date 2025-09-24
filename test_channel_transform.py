#!/usr/bin/env python3
"""
Test script for AddChannelDimension transform.

Tests the new AddChannelDimension transform to ensure it properly converts
(B, H, W) tensors to (B, C, H, W) format as needed by the pipeline.
"""

import sys
import os
import torch

# Add peaknet to path
sys.path.insert(0, '/sdf/home/c/cwang31/codes/peaknet')

from peaknet.tensor_transforms import AddChannelDimension, Pad, NoTransform

def test_add_channel_dimension():
    """Test AddChannelDimension transform functionality."""
    print("=== Testing AddChannelDimension Transform ===")

    # Test 1: Basic functionality (B, H, W) -> (B, C, H, W)
    print("\n1. Testing basic channel dimension addition...")
    transform = AddChannelDimension()

    # Simulate pusher data shape (B=10, H=1691, W=1691)
    input_tensor = torch.randn(10, 1691, 1691)
    print(f"Input shape: {input_tensor.shape}")

    output_tensor = transform(input_tensor)
    print(f"Output shape: {output_tensor.shape}")

    expected_shape = torch.Size([10, 1, 1691, 1691])
    assert output_tensor.shape == expected_shape, f"Expected {expected_shape}, got {output_tensor.shape}"
    print("✅ Basic test passed!")

    # Test 2: Multiple channels
    print("\n2. Testing multiple channels...")
    transform_multi = AddChannelDimension(num_channels=3)
    output_multi = transform_multi(input_tensor)
    print(f"Output shape (3 channels): {output_multi.shape}")

    expected_multi_shape = torch.Size([10, 3, 1691, 1691])
    assert output_multi.shape == expected_multi_shape, f"Expected {expected_multi_shape}, got {output_multi.shape}"
    print("✅ Multi-channel test passed!")

    # Test 3: Different channel dimension position
    print("\n3. Testing different channel dimension position...")
    transform_pos = AddChannelDimension(channel_dim=2)
    output_pos = transform_pos(input_tensor)
    print(f"Output shape (channel_dim=2): {output_pos.shape}")

    expected_pos_shape = torch.Size([10, 1691, 1, 1691])
    assert output_pos.shape == expected_pos_shape, f"Expected {expected_pos_shape}, got {output_pos.shape}"
    print("✅ Channel position test passed!")


def test_transform_chain():
    """Test the complete transform chain: AddChannelDimension + Pad."""
    print("\n=== Testing Transform Chain ===")

    # Simulate real pipeline scenario
    # Pusher provides: (10, 1691, 1691)
    # Pipeline needs: (10, 1, 1920, 1920)

    input_tensor = torch.randn(10, 1691, 1691)
    print(f"Input tensor (from pusher): {input_tensor.shape}")

    # Create transform chain
    add_channel = AddChannelDimension()
    pad_to_target = Pad(1920, 1920)  # Target size from config

    # Apply transforms in sequence
    step1 = add_channel(input_tensor)
    print(f"After AddChannelDimension: {step1.shape}")

    step2 = pad_to_target(step1)
    print(f"After Pad: {step2.shape}")

    # Verify final shape
    expected_final = torch.Size([10, 1, 1920, 1920])
    assert step2.shape == expected_final, f"Expected {expected_final}, got {step2.shape}"
    print("✅ Transform chain test passed!")

    # Test with conditional logic (like in training code)
    print("\n--- Testing conditional transform pattern ---")
    needs_channel_dim = True
    needs_padding = True

    pre_transforms = (
        add_channel if needs_channel_dim else NoTransform(),
        pad_to_target if needs_padding else NoTransform(),
    )

    # Apply all transforms
    result = input_tensor
    for transform in pre_transforms:
        result = transform(result)
        print(f"Shape after {transform.__class__.__name__}: {result.shape}")

    assert result.shape == expected_final, f"Expected {expected_final}, got {result.shape}"
    print("✅ Conditional transform pattern test passed!")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")

    # Test 1: Single sample (no batch dimension)
    print("\n1. Testing single sample (2D input)...")
    transform = AddChannelDimension()
    single_sample = torch.randn(1691, 1691)
    print(f"Single sample input: {single_sample.shape}")

    output_single = transform(single_sample)
    print(f"Single sample output: {output_single.shape}")

    expected_single = torch.Size([1691, 1, 1691])  # Channel added at position 1 (index 1)
    assert output_single.shape == expected_single, f"Expected {expected_single}, got {output_single.shape}"
    print("✅ Single sample test passed!")

    # Test 2: Already has channel dimension (should add another)
    print("\n2. Testing input that already has channel dimension...")
    input_with_channel = torch.randn(10, 1, 1691, 1691)
    print(f"Input with channel: {input_with_channel.shape}")

    output_extra_channel = transform(input_with_channel)
    print(f"Output with extra channel: {output_extra_channel.shape}")

    expected_extra = torch.Size([10, 1, 1, 1691, 1691])  # Another channel added
    assert output_extra_channel.shape == expected_extra, f"Expected {expected_extra}, got {output_extra_channel.shape}"
    print("✅ Extra channel test passed!")


if __name__ == "__main__":
    try:
        test_add_channel_dimension()
        test_transform_chain()
        test_edge_cases()

        print("\n🎉 All tests passed! AddChannelDimension transform is working correctly.")
        print("\nTransform is ready for integration into the pipeline.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)