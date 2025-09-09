#!/usr/bin/env python3
"""
PeakNet Utilities for Profiling and Performance Testing

Provides modified PeakNet models optimized for different profiling scenarios,
particularly for testing memory transfer patterns and pipeline performance.
"""

import torch
from torch import nn
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# PeakNet imports
try:
    from peaknet.modeling.convnextv2_bifpn_net import (
        PeakNet, PeakNetConfig, SegHeadConfig
    )
    from peaknet.modeling.bifpn_config import (
        BiFPNConfig, BiFPNBlockConfig, BNConfig, FusionConfig
    )
    from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config
    PEAKNET_AVAILABLE = True
except ImportError:
    PEAKNET_AVAILABLE = False
    print("Warning: PeakNet not available. Make sure peaknet is installed and accessible")


class PeakNetForProfiling(nn.Module):
    """
    PeakNet wrapper optimized for profiling pipeline performance.

    Wraps a PeakNet model to provide consistent interface with the inference
    pipeline and optimize for profiling different memory transfer patterns.
    """

    def __init__(self, peaknet_model: nn.Module, num_classes: int = 2):
        """
        Initialize wrapper around PeakNet model.

        Args:
            peaknet_model: Initialized PeakNet model
            num_classes: Number of classes (default: 2 for peak/background)
        """
        super().__init__()
        self.peaknet_model = peaknet_model
        # Get num_classes from PeakNet model config (correct access path)
        try:
            if hasattr(peaknet_model, 'config') and hasattr(peaknet_model.config, 'seg_head'):
                self.num_classes = peaknet_model.config.seg_head.num_classes
                print(f"✓ Got num_classes from model config: {self.num_classes}")
            else:
                self.num_classes = num_classes
                print(f"⚠ Could not access model.config.seg_head.num_classes, using default: {num_classes}")
        except Exception as e:
            self.num_classes = num_classes
            print(f"⚠ Error accessing model config ({e}), using default num_classes: {num_classes}")
        
        # Add device verification
        try:
            model_device = next(peaknet_model.parameters()).device
            if model_device.type == 'cuda':
                print(f"✓ PeakNet model on GPU: {model_device}")
            else:
                print(f"⚠ PeakNet model on CPU: {model_device}")
        except Exception as e:
            print(f"⚠ Could not determine model device: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns segmentation output.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Segmentation output of shape [batch_size, num_classes, height, width]
        """
        # Run PeakNet inference
        output = self.peaknet_model(x)
        return output

    def eval(self):
        """Set model to evaluation mode"""
        self.peaknet_model.eval()
        return self

    def to(self, device):
        """Move model to device"""
        self.peaknet_model = self.peaknet_model.to(device)
        return self




def create_peaknet_model(
    peaknet_config: dict,
    weights_path: Optional[str] = None,
    device: str = 'cuda:0'
) -> PeakNetForProfiling:
    """
    Create PeakNet model from configuration dictionary.
    
    Args:
        peaknet_config: PeakNet configuration dict with model parameters
        weights_path: Optional path to pre-trained weights
        device: Device to place model on
        
    Returns:
        PeakNetForProfiling model ready for inference
    """
    if not PEAKNET_AVAILABLE:
        raise ImportError("PeakNet not available. Please install peaknet package")

    # Import our native configuration
    from ..config.peaknet_config import create_peaknet_config_from_dict

    print(f"Creating PeakNet model from native configuration")

    # Extract model configuration - should be under 'model' key to match original structure
    model_config = peaknet_config.get("model", peaknet_config)
    
    # Create simplified configuration from dict
    simple_config = create_peaknet_config_from_dict(model_config)
    
    print(f"Model image_size: {simple_config.image_size}")
    print(f"Model num_channels: {simple_config.num_channels}")
    print(f"Model num_classes: {simple_config.num_classes}")

    # Convert to PeakNet configuration objects
    backbone_config, bifpn_config, seg_head_config = simple_config.to_peaknet_configs()
    
    # Create PeakNet configuration
    peaknet_model_config = PeakNetConfig(
        backbone=backbone_config,
        bifpn=bifpn_config,
        seg_head=seg_head_config,
    )

    # Create model
    model = PeakNet(peaknet_model_config)
    model.init_weights()

    # Load weights if provided
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"PeakNet model created: {num_params/1e6:.1f}M parameters")
    print(f"Backbone: ConvNextV2 {simple_config.backbone_hidden_sizes}")
    print(f"BiFPN: {simple_config.bifpn_num_blocks} blocks, {simple_config.bifpn_num_features} features")
    print(f"Input size: {simple_config.image_size}×{simple_config.image_size}")

    # Create profiling wrapper
    wrapper = PeakNetForProfiling(model, num_classes=simple_config.num_classes)
    wrapper = wrapper.to(device)
    
    # Set to eval mode for consistent timing
    wrapper.eval()

    return wrapper


def get_peaknet_shapes(peaknet_config: dict, batch_size: int = 1) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Calculate input and output shapes from PeakNet configuration.
    
    Args:
        peaknet_config: PeakNet configuration dict with model parameters
        batch_size: Batch size
        
    Returns:
        tuple: (input_shape, output_shape) both as (batch_size, channels, height, width)
    """
    # Import our native configuration
    from ..config.peaknet_config import create_peaknet_config_from_dict
    
    # Extract model configuration - should be under 'model' key to match original structure
    model_config = peaknet_config.get("model", peaknet_config)
    
    # Create simplified configuration from dict
    simple_config = create_peaknet_config_from_dict(model_config)
    
    input_shape = (batch_size, simple_config.num_channels, simple_config.image_size, simple_config.image_size)
    output_shape = (batch_size, simple_config.num_classes, simple_config.image_size, simple_config.image_size)
    
    return input_shape, output_shape


if __name__ == '__main__':
    """Demo and testing"""
    if PEAKNET_AVAILABLE:
        print("=== PeakNet Utils Demo ===")

        # Test with a sample configuration file path
        # You would replace this with an actual path to a PeakNet config
        sample_yaml_path = "/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet/convnext_seg_config.yaml"

        if os.path.exists(sample_yaml_path):
            print(f"Using configuration: {sample_yaml_path}")

            # Test shape calculation
            batch_size = 4
            input_shape = get_peaknet_input_shape(sample_yaml_path, batch_size)
            output_shape = get_peaknet_output_shape(sample_yaml_path, batch_size)

            print(f"\nBatch size: {batch_size}")
            print(f"Input shape: {input_shape}")
            print(f"Output shape: {output_shape}")

            # Test transfer size estimation
            transfer_info = estimate_transfer_size(sample_yaml_path, batch_size)
            print(f"\nTransfer size estimation:")
            print(f"  Input: {transfer_info['input_size_mb']:.2f} MB")
            print(f"  Output: {transfer_info['output_size_mb']:.2f} MB")
            print(f"  Total: {transfer_info['total_transfer_mb']:.2f} MB")

            # Test model creation if CUDA available
            if torch.cuda.is_available():
                print(f"\nTesting model creation...")
                try:
                    model = create_peaknet_for_profiling(sample_yaml_path)
                    print(f"Model created successfully on {next(model.parameters()).device}")

                    # Test forward pass with random data
                    C, H, W = input_shape[1], input_shape[2], input_shape[3]
                    test_input = torch.randn(2, C, H, W).cuda()
                    with torch.no_grad():
                        output = model(test_input)
                        print(f"Forward pass successful: {test_input.shape} -> {output.shape}")
                except Exception as e:
                    print(f"Model test failed: {e}")
            else:
                print("CUDA not available, skipping model test")
        else:
            print(f"Sample configuration file not found: {sample_yaml_path}")
            print("Please provide a valid PeakNet YAML configuration file path")
    else:
        print("PeakNet not available, skipping demo")
