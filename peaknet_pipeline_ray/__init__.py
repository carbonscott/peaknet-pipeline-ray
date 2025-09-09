"""
PeakNet Pipeline Ray - Scalable ML inference with streaming data.

A production-ready package for running PeakNet segmentation model inference at scale 
with streaming data sources across multiple GPUs using Ray.
"""

from .config import (
    PipelineConfig,
    ModelConfig,
    RuntimeConfig,
    DataConfig,
    SystemConfig,
    ProfilingConfig,
    OutputConfig,
    PipelineInput,
    PipelineOutput
)

# Import pipeline class
from .pipeline import PeakNetPipeline, PipelineResults

__version__ = "0.1.0"
__author__ = "PeakNet Pipeline Team"

__all__ = [
    'PipelineConfig',
    'ModelConfig',
    'RuntimeConfig', 
    'DataConfig',
    'SystemConfig',
    'ProfilingConfig',
    'OutputConfig',
    'PipelineInput',
    'PipelineOutput',
    'PeakNetPipeline',
    'PipelineResults'
]