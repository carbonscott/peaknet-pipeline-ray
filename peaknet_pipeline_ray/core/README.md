# PeakNet Pipeline Ray - Core Components

This directory contains the core components for the PeakNet pipeline with Ray distributed computing support.

## Architecture Overview

The core components follow a clear separation of concerns:

1. **Core Pipeline Logic** - Standalone double-buffered pipeline implementation
2. **Ray Distribution Layer** - Wraps core pipeline in Ray actors for multi-GPU scaling
3. **Testing Utilities** - Data producers for pipeline testing and development
4. **Supporting Components** - Model utilities, GPU validation, and profiling tools

## File Descriptions

### Main Pipeline Components

#### `peaknet_pipeline.py` - Core Double-Buffered Pipeline Implementation
- Contains the standalone `DoubleBufferedPipeline` class that implements H2D → Compute → D2H pipeline
- Includes PeakNet model creation and configuration logic
- Can run as a standalone script for single-GPU profiling and testing
- Handles NUMA binding, GPU memory management, and CUDA stream synchronization
- **Purpose**: Core pipeline logic that can be used independently

#### `peaknet_ray_pipeline_actor.py` - Ray Actor Wrapper
- Wraps the `DoubleBufferedPipeline` in Ray actors for distributed multi-GPU execution
- Handles Ray-specific concerns: object store access, actor lifecycle, GPU assignment
- Creates `PeakNetPipelineActor` and `PeakNetPipelineActorWithProfiling` classes
- **Purpose**: Scale the pipeline to multiple GPUs using Ray's distributed computing

**Key Distinction**: `peaknet_pipeline.py` is the core implementation, while `peaknet_ray_pipeline_actor.py` wraps it for Ray distribution.

### Data Producers (Testing Utilities)

#### `data_producer.py` - Simple Ray Data Producer
- Generates raw tensors and puts them in Ray's object store
- Basic testing utility for pipeline actors
- **Purpose**: Simple tensor generation for testing

#### `peaknet_ray_data_producer.py` - Structured Data Producer
- Generates `PipelineInput` objects (structured data with metadata)
- More sophisticated than `data_producer.py` - includes metadata simulation
- **Purpose**: Generate structured test data that mimics real pipeline inputs

**Key Distinction**: `data_producer.py` generates simple tensors, while `peaknet_ray_data_producer.py` generates structured `PipelineInput` objects with metadata.

### Utility Components

#### `peaknet_utils.py` - PeakNet Model Utilities
- Creates and configures PeakNet models from configuration dictionaries
- Contains `PeakNetForProfiling` wrapper class
- Handles model compilation, device placement, and shape calculations
- **Purpose**: PeakNet-specific model creation and configuration

#### `gpu_health_validator.py` - GPU Health Validation
- Pre-Ray system validation of GPU health
- Configures `CUDA_VISIBLE_DEVICES` to only expose healthy GPUs
- **Purpose**: Ensure only working GPUs are available to Ray

#### `peaknet_profiler.py` - NSys Profiling Orchestrator
- Coordinates NSys profiling across multiple experiments
- Generates individual profile reports for each experiment
- **Purpose**: Systematic performance profiling and analysis

## Usage Patterns

### Standalone Pipeline Testing
```bash
python peaknet_pipeline.py --gpu-id=0
```

### Ray Distributed Pipeline
```python
from peaknet_pipeline_ray.core import create_pipeline_actors

actors = create_pipeline_actors(num_actors=4)
```

### Data Generation for Testing
```python
from peaknet_pipeline_ray.core.data_producer import RayDataProducerManager

producer = RayDataProducerManager()
batches = producer.launch_producers(num_producers=2, ...)
```

### Performance Profiling
```bash
python peaknet_profiler.py experiment=peaknet_small_compiled
```

## Important Notes

- **Testing vs Production**: Data producers are testing utilities only. In production, data comes from external streaming sources.
- **GPU Health**: Always run GPU validation before Ray initialization to ensure system stability.
- **Pipeline Overlap**: The double-buffered design enables overlap of H2D, compute, and D2H operations for maximum throughput.
- **Ray Integration**: Ray actors handle GPU assignment automatically via `CUDA_VISIBLE_DEVICES`.