"""Native PeakNet configuration dataclasses for the pipeline package.

This module provides a clean, framework-agnostic way to configure PeakNet models
without depending on Hydra/OmegaConf or other external configuration frameworks.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import torch


@dataclass
class ConvNextBackboneConfig:
    """Configuration for ConvNext backbone."""
    num_channels: int = 1
    patch_size: int = 4
    num_stages: int = 4
    hidden_sizes: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    depths: List[int] = field(default_factory=lambda: [3, 3, 9, 3])
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    drop_path_rate: float = 0.0
    image_size: int = 1920
    out_features: List[str] = field(default_factory=lambda: ['stage1', 'stage2', 'stage3', 'stage4'])
    out_indices: Optional[List[int]] = None


@dataclass
class BiFPNBlockConfig:
    """Configuration for BiFPN block."""
    base_level: int = 4
    num_levels: int = 4
    num_features: int = 256
    up_scale_factor: float = 2.0
    down_scale_factor: float = 0.5
    relu_inplace: bool = False
    # BatchNorm configuration
    bn_eps: float = 1e-5
    bn_momentum: float = 0.1
    # Fusion configuration  
    fusion_eps: float = 1e-5


@dataclass
class BiFPNConfig:
    """Configuration for BiFPN."""
    num_blocks: int = 2
    block: BiFPNBlockConfig = field(default_factory=BiFPNBlockConfig)


@dataclass
class SegmentationHeadConfig:
    """Configuration for segmentation head."""
    num_classes: int = 2
    num_groups: int = 32
    out_channels: int = 256
    base_scale_factor: float = 2.0
    uses_learned_upsample: bool = False
    up_scale_factor: List[int] = field(default_factory=lambda: [4, 8, 16, 32])


@dataclass
class SimplePeakNetConfig:
    """Simplified PeakNet configuration that can be easily created from YAML."""
    # Model architecture
    image_size: int = 1920
    num_channels: int = 1
    num_classes: int = 2

    # Backbone configuration
    backbone_hidden_sizes: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    backbone_depths: List[int] = field(default_factory=lambda: [3, 3, 9, 3])

    # BiFPN configuration
    bifpn_num_blocks: int = 2
    bifpn_num_features: int = 256

    # Segmentation head configuration
    seg_out_channels: int = 256
    uses_learned_upsample: bool = False

    # Other model settings
    from_scratch: bool = False

    def to_peaknet_configs(self) -> tuple:
        """Convert to the PeakNet configuration objects.

        Returns:
            tuple: (backbone_config, bifpn_config, seg_head_config)
        """
        # Import PeakNet config classes
        try:
            from peaknet.modeling.convnextv2_bifpn_net import SegHeadConfig
            from peaknet.modeling.bifpn_config import BiFPNConfig, BiFPNBlockConfig, BNConfig, FusionConfig
            from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config
        except ImportError as e:
            raise ImportError(f"PeakNet not available: {e}")

        # Create ConvNext backbone configuration
        backbone_config = ConvNextV2Config(
            num_channels=self.num_channels,
            patch_size=4,
            num_stages=4,
            hidden_sizes=self.backbone_hidden_sizes,
            depths=self.backbone_depths,
            hidden_act="gelu",
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            drop_path_rate=0.0,
            image_size=self.image_size,
            out_features=['stage1', 'stage2', 'stage3', 'stage4'],
            out_indices=None
        )

        # Create BiFPN configuration
        bn_config = BNConfig(eps=1e-5, momentum=0.1)
        fusion_config = FusionConfig(eps=1e-5)

        bifpn_block_config = BiFPNBlockConfig(
            base_level=4,
            num_levels=4,
            num_features=self.bifpn_num_features,
            up_scale_factor=2.0,
            down_scale_factor=0.5,
            relu_inplace=False,
            bn=bn_config,
            fusion=fusion_config
        )

        bifpn_config = BiFPNConfig(
            num_blocks=self.bifpn_num_blocks,
            block=bifpn_block_config
        )

        # Create segmentation head configuration
        seg_head_config = SegHeadConfig(
            num_classes=self.num_classes,
            num_groups=32,
            out_channels=self.seg_out_channels,
            base_scale_factor=2.0,
            uses_learned_upsample=self.uses_learned_upsample,
            up_scale_factor=[4, 8, 16, 32]
        )

        return backbone_config, bifpn_config, seg_head_config


def create_peaknet_config_from_dict(config_dict: Dict[str, Any]) -> SimplePeakNetConfig:
    """Create SimplePeakNetConfig from a dictionary (e.g., loaded from YAML).

    Args:
        config_dict: Dictionary containing PeakNet configuration

    Returns:
        SimplePeakNetConfig: Parsed configuration object
    """
    # Extract values with defaults
    return SimplePeakNetConfig(
        image_size=config_dict.get('image_size', 1920),
        num_channels=config_dict.get('num_channels', 1),
        num_classes=config_dict.get('num_classes', 2),
        backbone_hidden_sizes=config_dict.get('backbone_hidden_sizes', [96, 192, 384, 768]),
        backbone_depths=config_dict.get('backbone_depths', [3, 3, 9, 3]),
        bifpn_num_blocks=config_dict.get('bifpn_num_blocks', 2),
        bifpn_num_features=config_dict.get('bifpn_num_features', 256),
        seg_out_channels=config_dict.get('seg_out_channels', 256),
        uses_learned_upsample=config_dict.get('uses_learned_upsample', False),
        from_scratch=config_dict.get('from_scratch', False)
    )