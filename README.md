# PeakNet Pipeline Ray

A package for running PeakNet segmentation model inference at scale using Ray (early development).

## Quick Start with NSys Profiling

1. Start Ray head node:
```bash
ray start --head --block
```

2. In another terminal, run PeakNet pipeline with profiling enabled:
```bash
python -m peaknet_pipeline_ray.cli.main --config examples/configs/peaknet.yaml --max-actors 4 --total-samples 10240 --verbose
```

This will generate NSys profiling data in `$TMPDIR/ray/session_latest/logs/nsight/` showing actual PeakNet GPU kernel execution.

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
