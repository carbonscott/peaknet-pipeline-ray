"""Data structures for pipeline input/output with metadata pass-through."""

from dataclasses import dataclass
from typing import Dict, Any
import torch
import time


@dataclass
class PipelineInput:
    """Input data structure for the pipeline with metadata pass-through.
    
    This structure separates the image data that the PeakNet model processes
    from the metadata that needs to be passed through to downstream processing.
    """
    image_data: torch.Tensor        # What PeakNet model processes: shape from DataConfig
    metadata: Dict[str, Any]        # Pass-through data: photon_energy, timestamp, run_id, etc.
    batch_id: str                   # Tracking identifier for this batch
    
    def __post_init__(self):
        """Validate the input data structure."""
        if not isinstance(self.image_data, torch.Tensor):
            raise TypeError("image_data must be a torch.Tensor")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        if not isinstance(self.batch_id, str):
            raise TypeError("batch_id must be a string")


@dataclass 
class PipelineOutput:
    """Output data structure from the pipeline with metadata pass-through.
    
    This structure contains the PeakNet model predictions alongside the
    exact metadata from the input, ensuring downstream processing has
    both the inference results and all necessary metadata.
    """
    predictions: torch.Tensor       # PeakNet model output
    metadata: Dict[str, Any]        # Exact pass-through from input
    batch_id: str                   # Same tracking identifier from input
    processing_time: float          # Time spent in pipeline processing (seconds)
    
    def __post_init__(self):
        """Validate the output data structure."""
        if not isinstance(self.predictions, torch.Tensor):
            raise TypeError("predictions must be a torch.Tensor")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        if not isinstance(self.batch_id, str):
            raise TypeError("batch_id must be a string")
        if not isinstance(self.processing_time, (int, float)):
            raise TypeError("processing_time must be a number")
    
    @classmethod
    def from_input_and_predictions(
        cls, 
        pipeline_input: PipelineInput, 
        predictions: torch.Tensor,
        start_time: float
    ) -> 'PipelineOutput':
        """Create PipelineOutput from PipelineInput and model predictions.
        
        Args:
            pipeline_input: Original input data structure
            predictions: Output from PeakNet model
            start_time: Time when processing started (from time.time())
            
        Returns:
            PipelineOutput with metadata preserved from input
        """
        processing_time = time.time() - start_time
        
        return cls(
            predictions=predictions,
            metadata=pipeline_input.metadata.copy(),  # Deep copy to prevent modification
            batch_id=pipeline_input.batch_id,
            processing_time=processing_time
        )