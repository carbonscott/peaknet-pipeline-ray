"""Data structures for pipeline input/output with metadata pass-through."""

from dataclasses import dataclass
from typing import Dict, Any, Union, Optional
import torch
import numpy as np
import time
import ray


@dataclass
class PipelineInput:
    """Input data structure for the pipeline with metadata pass-through.

    This structure separates the image data that the PeakNet model processes
    from the metadata that needs to be passed through to downstream processing.
    
    Supports two modes:
    1. Direct mode: image_data is a torch.Tensor (for backward compatibility)
    2. ObjectRef mode: image_data_ref is a Ray ObjectRef to numpy array (for performance)
    """
    image_data: Optional[torch.Tensor] = None    # Direct tensor (backward compatibility)
    image_data_ref: Optional[ray.ObjectRef] = None  # ObjectRef to numpy array (performance mode)
    metadata: Dict[str, Any] = None              # Pass-through data: photon_energy, timestamp, run_id, etc.
    batch_id: str = None                         # Tracking identifier for this batch

    def __post_init__(self):
        """Validate the input data structure."""
        if self.image_data is None and self.image_data_ref is None:
            raise ValueError("Either image_data or image_data_ref must be provided")
        if self.image_data is not None and self.image_data_ref is not None:
            raise ValueError("Only one of image_data or image_data_ref should be provided")
        
        if self.image_data is not None and not isinstance(self.image_data, torch.Tensor):
            raise TypeError("image_data must be a torch.Tensor")
        
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        if not isinstance(self.batch_id, str):
            raise TypeError("batch_id must be a string")

    @classmethod
    def from_numpy_array(cls, numpy_array: np.ndarray, metadata: Dict[str, Any], batch_id: str) -> 'PipelineInput':
        """Create PipelineInput with numpy array stored as ObjectRef for optimal performance.
        
        IMPORTANT: Ray ObjectRef Preservation
        ====================================
        This method creates an ObjectRef that will be preserved when passed through
        Ray actors (like our queue system) because ObjectRefs inside custom objects
        are NOT auto-dereferenced by Ray. This gives us optimal memory efficiency.
        
        Args:
            numpy_array: Image data as numpy array
            metadata: Pass-through metadata
            batch_id: Tracking identifier
            
        Returns:
            PipelineInput with image_data_ref pointing to numpy array in Ray object store
        """
        array_ref = ray.put(numpy_array)  # Zero-copy storage in plasma
        return cls(
            image_data_ref=array_ref,
            metadata=metadata,
            batch_id=batch_id
        )
    
    @classmethod
    def from_torch_tensor(cls, tensor: torch.Tensor, metadata: Dict[str, Any], batch_id: str) -> 'PipelineInput':
        """Create PipelineInput with torch tensor (backward compatibility).
        
        Args:
            tensor: Image data as torch tensor
            metadata: Pass-through metadata
            batch_id: Tracking identifier
            
        Returns:
            PipelineInput with direct tensor storage
        """
        return cls(
            image_data=tensor,
            metadata=metadata,
            batch_id=batch_id
        )
    
    def get_torch_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Get image data as torch tensor, regardless of storage mode.
        
        Args:
            device: Target device ("cpu", "cuda:0", etc.)
            
        Returns:
            Torch tensor on specified device
        """
        if self.image_data is not None:
            # Direct tensor mode - move to device if needed
            tensor = self.image_data
            if device != "cpu":
                tensor = tensor.to(device)
            return tensor
        else:
            # ObjectRef mode - get numpy array and convert
            numpy_array = ray.get(self.image_data_ref)  # Zero-copy from plasma
            tensor = torch.from_numpy(numpy_array)      # Zero-copy view
            if device != "cpu":
                tensor = tensor.to(device)  # Copy to GPU only when needed
            return tensor
    
    def get_numpy_array(self) -> np.ndarray:
        """Get image data as numpy array, regardless of storage mode.
        
        Returns:
            Numpy array (zero-copy if stored as ObjectRef)
        """
        if self.image_data is not None:
            # Direct tensor mode - convert to numpy
            return self.image_data.cpu().numpy()
        else:
            # ObjectRef mode - get directly (zero-copy)
            return ray.get(self.image_data_ref)


@dataclass 
class PipelineOutput:
    """Output data structure from the pipeline with metadata pass-through.

    This structure contains the PeakNet model predictions alongside the
    exact metadata from the input, ensuring downstream processing has
    both the inference results and all necessary metadata.
    
    Supports two modes like PipelineInput:
    1. Direct mode: predictions is a torch.Tensor (for backward compatibility)
    2. ObjectRef mode: predictions_ref is a Ray ObjectRef to numpy array (for performance)
    """
    predictions: Optional[torch.Tensor] = None      # Direct tensor (backward compatibility)
    predictions_ref: Optional[ray.ObjectRef] = None # ObjectRef to numpy array (performance mode)  
    metadata: Dict[str, Any] = None                 # Exact pass-through from input
    batch_id: str = None                            # Same tracking identifier from input

    def __post_init__(self):
        """Validate the output data structure."""
        if self.predictions is None and self.predictions_ref is None:
            raise ValueError("Either predictions or predictions_ref must be provided")
        if self.predictions is not None and self.predictions_ref is not None:
            raise ValueError("Only one of predictions or predictions_ref should be provided")
            
        if self.predictions is not None and not isinstance(self.predictions, torch.Tensor):
            raise TypeError("predictions must be a torch.Tensor")
        
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        if not isinstance(self.batch_id, str):
            raise TypeError("batch_id must be a string")

    @classmethod
    def from_numpy_array(cls, numpy_predictions: np.ndarray, metadata: Dict[str, Any], 
                        batch_id: str) -> 'PipelineOutput':
        """Create PipelineOutput with numpy array stored as ObjectRef for optimal performance.
        
        Args:
            numpy_predictions: Predictions as numpy array
            metadata: Pass-through metadata
            batch_id: Tracking identifier
            
        Returns:
            PipelineOutput with predictions_ref pointing to numpy array in Ray object store
        """
        predictions_ref = ray.put(numpy_predictions)  # Zero-copy storage in plasma
        return cls(
            predictions_ref=predictions_ref,
            metadata=metadata,
            batch_id=batch_id
        )

    @classmethod
    def from_input_and_predictions(
        cls, 
        pipeline_input: PipelineInput, 
        predictions: Union[torch.Tensor, np.ndarray],
        start_time: float = None  # Kept for backward compatibility but not used
    ) -> 'PipelineOutput':
        """Create PipelineOutput from PipelineInput and model predictions.

        Args:
            pipeline_input: Original input data structure
            predictions: Output from PeakNet model (torch.Tensor or np.ndarray)
            start_time: Ignored (kept for backward compatibility)

        Returns:
            PipelineOutput with metadata preserved from input
        """
        if isinstance(predictions, torch.Tensor):
            return cls(
                predictions=predictions,
                metadata=pipeline_input.metadata.copy(),  # Deep copy to prevent modification
                batch_id=pipeline_input.batch_id
            )
        else:  # numpy array
            return cls.from_numpy_array(
                numpy_predictions=predictions,
                metadata=pipeline_input.metadata.copy(),
                batch_id=pipeline_input.batch_id
            )
    
    def get_torch_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Get predictions as torch tensor, regardless of storage mode.
        
        Args:
            device: Target device ("cpu", "cuda:0", etc.)
            
        Returns:
            Torch tensor on specified device
        """
        if self.predictions is not None:
            # Direct tensor mode - move to device if needed
            tensor = self.predictions
            if device != "cpu":
                tensor = tensor.to(device)
            return tensor
        else:
            # ObjectRef mode - get numpy array and convert
            numpy_array = ray.get(self.predictions_ref)  # Zero-copy from plasma
            tensor = torch.from_numpy(numpy_array)       # Zero-copy view
            if device != "cpu":
                tensor = tensor.to(device)  # Copy to GPU only when needed
            return tensor
    
    def get_numpy_array(self) -> np.ndarray:
        """Get predictions as numpy array, regardless of storage mode.
        
        Returns:
            Numpy array (zero-copy if stored as ObjectRef)
        """
        if self.predictions is not None:
            # Direct tensor mode - convert to numpy
            return self.predictions.cpu().numpy()
        else:
            # ObjectRef mode - get directly (zero-copy)
            return ray.get(self.predictions_ref)