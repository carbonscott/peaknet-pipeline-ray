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
    compile_mode: Optional[str] = None  # None/"default"/"reduce-overhead"/"max-autotune"
    warmup_iterations: int = 50  # 0 = skip warmup


@dataclass
class RuntimeConfig:
    """Runtime configuration for pipeline execution."""
    max_actors: Optional[int] = None  # Auto-scale to available GPUs
    batch_size: int = 4
    total_samples: Optional[int] = None
    num_producers: int = 4
    batches_per_producer: int = 5
    inter_batch_delay: float = 0.1
    # Memory management configuration
    memory_sync_interval: int = 100  # Sync every N batches for memory management (0=disable)
    # Queue configuration
    queue_num_shards: int = 4  # Number of queue shards for parallel access
    queue_maxsize_per_shard: int = 100  # Maximum items per shard (total capacity = shards * maxsize)
    # Coordination timing configuration
    max_empty_polls: int = 20  # Check coordinator after N consecutive empty polls
    poll_timeout: float = 0.01  # Timeout for queue polling in seconds


@dataclass
class DataConfig:
    """Data configuration for model input."""
    shape: Tuple[int, int, int] = (1, 512, 512)  # C, H, W for model (channels = shape[0])


@dataclass
class DataSourceConfig:
    """Configuration for data source (random or socket)."""
    source_type: str = "random"  # "random" or "socket"

    # Socket configuration
    socket_hostname: str = "localhost"
    socket_port: int = 12321
    socket_timeout: float = 10.0
    socket_retry_attempts: int = 3

    # Required shape for socket data source (H, W) - detector data shape before transforms
    # Users must specify the expected data shape upfront for proper initialization
    shape: Tuple[int, int] = (1691, 1691)  # H, W - actual detector data size

    # Serialization format for socket data
    serialization_format: str = "numpy"  # "numpy" for .npz format (fast), "hdf5" for legacy HDF5 format

    # Field mapping (format depends on serialization_format)
    # - NumPy format: flat keys like "data", "timestamp", "wavelength"
    # - HDF5 format: hierarchical paths like "/data/data", "/data/timestamp"
    fields: Dict[str, str] = field(default_factory=lambda: {
        "detector_data": "data",
        "timestamp": "timestamp",
        "photon_wavelength": "wavelength",
        "random": "random"
    })

    # Batch assembly configuration
    batch_assembly: bool = True
    batch_timeout: float = 1.0  # Max wait time for batch completion

    # Required fields for validation
    required_fields: list = field(default_factory=lambda: ["detector_data"])




@dataclass
class PrecisionConfig:
    """Configuration for mixed precision inference."""
    dtype: str = "float32"  # Options: "float32", "bfloat16", "float16"


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
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
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

        # Extract data config, filtering out deprecated fields
        data_dict = config_dict.get('data', {})
        data_dict.pop('input_channels', None)  # Remove deprecated field if present
        data_config = DataConfig(**data_dict)

        # Extract data source config and handle shape field conversion
        data_source_dict = config_dict.get('data_source', {})
        if 'shape' in data_source_dict and data_source_dict['shape'] is not None:
            data_source_dict['shape'] = tuple(data_source_dict['shape'])
        data_source_config = DataSourceConfig(**data_source_dict)

        precision_config = PrecisionConfig(**config_dict.get('precision', {}))
        system_config = SystemConfig(**config_dict.get('system', {}))
        profiling_config = ProfilingConfig(**config_dict.get('profiling', {}))
        output_config = OutputConfig(**config_dict.get('output', {}))

        return cls(
            model=model_config,
            runtime=runtime_config,
            data=data_config,
            data_source=data_source_config,
            precision=precision_config,
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
                'peaknet_config': self.model.peaknet_config,
                'compile_mode': self.model.compile_mode,
                'warmup_iterations': self.model.warmup_iterations
            },
            'runtime': {
                'max_actors': self.runtime.max_actors,
                'batch_size': self.runtime.batch_size,
                'total_samples': self.runtime.total_samples,
                'num_producers': self.runtime.num_producers,
                'batches_per_producer': self.runtime.batches_per_producer,
                'inter_batch_delay': self.runtime.inter_batch_delay,
                'memory_sync_interval': self.runtime.memory_sync_interval,
                'queue_num_shards': self.runtime.queue_num_shards,
                'queue_maxsize_per_shard': self.runtime.queue_maxsize_per_shard,
                'max_empty_polls': self.runtime.max_empty_polls,
                'poll_timeout': self.runtime.poll_timeout
            },
            'data': {
                'shape': list(self.data.shape)
            },
            'data_source': {
                'source_type': self.data_source.source_type,
                'socket_hostname': self.data_source.socket_hostname,
                'socket_port': self.data_source.socket_port,
                'socket_timeout': self.data_source.socket_timeout,
                'socket_retry_attempts': self.data_source.socket_retry_attempts,
                'shape': list(self.data_source.shape) if self.data_source.shape else None,
                'serialization_format': self.data_source.serialization_format,
                'fields': self.data_source.fields,
                'batch_assembly': self.data_source.batch_assembly,
                'batch_timeout': self.data_source.batch_timeout,
                'required_fields': self.data_source.required_fields
            },
            'precision': {
                'dtype': self.precision.dtype
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