# PeakNet Pipeline Ray

A package for running PeakNet segmentation model inference at scale using Ray (early development).

## Quick Start with NSys Profiling

1. Start Ray head node:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 ray start --head --block
```

2. In another terminal, run PeakNet pipeline with profiling enabled:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 peaknet-pipeline --config examples/configs/peaknet.yaml --max-actors 4 --total-samples 10240 --verbose
```

This will generate NSys profiling data in `$TMPDIR/ray/session_latest/logs/nsight/` showing actual PeakNet GPU kernel execution.

## Model Optimization with PyTorch Compilation

The pipeline supports PyTorch 2.0+ model compilation for improved performance:

### Basic Usage
```bash
# Enable compilation with default mode
peaknet-pipeline --config examples/configs/peaknet.yaml --compile-mode default

# Use high-performance compilation mode
peaknet-pipeline --config examples/configs/peaknet.yaml --compile-mode reduce-overhead

# Maximum optimization (longer compile time)
peaknet-pipeline --config examples/configs/peaknet.yaml --compile-mode max-autotune
```

### Compilation Modes
- `None` (default): No compilation, fastest startup
- `default`: Standard compilation with balanced compile time and performance
- `reduce-overhead`: Aggressive optimization, reduces Python overhead
- `max-autotune`: Maximum optimization, longest compile time but best performance

### Model Warmup
Compiled models benefit from warmup to avoid recompilation overhead during inference:

```bash
# Default warmup (50 iterations)
peaknet-pipeline --config examples/configs/peaknet.yaml --compile-mode default

# Custom warmup iterations
peaknet-pipeline --config examples/configs/peaknet.yaml --compile-mode default --warmup-iterations 100

# Skip warmup
peaknet-pipeline --config examples/configs/peaknet.yaml --compile-mode default --warmup-iterations 0
```

### Performance Recommendations
- Use `reduce-overhead` mode for production workloads with consistent batch sizes
- Allow sufficient warmup iterations (50-100) for stable compilation
- Monitor first-batch latency vs steady-state performance

### Memory Management

The pipeline performs periodic GPU memory synchronization to prevent memory leaks during long-running inference:

```bash
# Default: disabled (maximum performance)
peaknet-pipeline --config config.yaml

# Custom interval: sync every 100 batches
peaknet-pipeline --config config.yaml --memory-sync-interval 100

# Custom interval: sync every 500 batches
peaknet-pipeline --config config.yaml --memory-sync-interval 500
```

**Performance Impact:** Synchronization occurs very infrequently and has minimal impact on throughput.

## What it does

- Runs PeakNet models (ConvNextV2 + BiFPN) 
- Distributes processing across multiple GPUs using Ray
- Processes 1920×1920 crystallography images
- Native Python configuration

## Installation

```bash
pip install -e .
```

## Configuration

The pipeline uses YAML configuration files. See `examples/configs/peaknet.yaml` for the working example with:

- 1920×1920 image processing
- Multi-GPU distribution (1 actors per GPU)
- NSys profiling enabled
- Native PeakNet model configuration
