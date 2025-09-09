"""Configuration schemas for PeakNet Pipeline using dataclasses."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for PeakNet model."""
    yaml_path: Optional[str] = None
    weights_path: Optional[str] = None
    peaknet_config: Optional[Dict[str, Any]] = None


@dataclass  
class RuntimeConfig:
    """Runtime configuration for pipeline execution."""
    max_actors: Optional[int] = None  # Auto-scale to available GPUs
    batch_size: int = 4
    total_samples: Optional[int] = None
    num_producers: int = 4
    batches_per_producer: int = 5
    inter_batch_delay: float = 0.1


@dataclass
class DataConfig:
    """Data configuration for model input."""
    shape: Tuple[int, int, int] = (1, 512, 512)  # C, H, W for model
    input_channels: int = 1


@dataclass
class SystemConfig:
    """System configuration for hardware and resources."""
    min_gpus: int = 1
    skip_gpu_validation: bool = False
    pin_memory: bool = True
    verify_actors: bool = True


@dataclass
class ProfilingConfig:
    """Configuration for performance profiling."""
    enable_profiling: bool = False
    output_dir: Optional[str] = None


@dataclass
class OutputConfig:
    """Configuration for output and logging."""
    output_dir: Optional[str] = None
    verbose: bool = False
    quiet: bool = False


@dataclass
class PipelineConfig:
    """Main pipeline configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)  
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Load configuration from dictionary."""
        # Extract each section, providing empty dict as default
        model_config = ModelConfig(**config_dict.get('model', {}))
        runtime_config = RuntimeConfig(**config_dict.get('runtime', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        system_config = SystemConfig(**config_dict.get('system', {}))
        profiling_config = ProfilingConfig(**config_dict.get('profiling', {}))
        output_config = OutputConfig(**config_dict.get('output', {}))

        return cls(
            model=model_config,
            runtime=runtime_config,
            data=data_config,
            system=system_config,
            profiling=profiling_config,
            output=output_config
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': {
                'yaml_path': self.model.yaml_path,
                'weights_path': self.model.weights_path,
                'peaknet_config': self.model.peaknet_config
            },
            'runtime': {
                'max_actors': self.runtime.max_actors,
                'batch_size': self.runtime.batch_size,
                'total_samples': self.runtime.total_samples,
                'num_producers': self.runtime.num_producers,
                'batches_per_producer': self.runtime.batches_per_producer,
                'inter_batch_delay': self.runtime.inter_batch_delay
            },
            'data': {
                'shape': list(self.data.shape),
                'input_channels': self.data.input_channels
            },
            'system': {
                'min_gpus': self.system.min_gpus,
                'skip_gpu_validation': self.system.skip_gpu_validation,
                'pin_memory': self.system.pin_memory,
                'verify_actors': self.system.verify_actors
            },
            'profiling': {
                'enable_profiling': self.profiling.enable_profiling,
                'output_dir': self.profiling.output_dir
            },
            'output': {
                'output_dir': self.output.output_dir,
                'verbose': self.output.verbose,
                'quiet': self.output.quiet
            }
        }

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)