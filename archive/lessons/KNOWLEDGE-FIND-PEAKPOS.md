# CuPy-Based Peak Position Finding Implementation

This document captures the original cupy-based implementation from `peaknet-pipeline` repository for converting PeakNet segmentation maps to peak positions.

**Source Repository**: `/sdf/home/c/cwang31/codes/peaknet-pipeline/`
**Key Commits**:
- `7e9e09d` - "Sync before cupy" (added critical synchronization)
- `35cc922` - "Modify peak coordinate processing to match [p, y, x] format"

---

## Overview

The cupy routine converts PeakNet's output (segmentation maps showing peak regions) into actual peak coordinates using GPU-accelerated image processing operations.

### Pipeline Flow

```
Input Tensor (BP, 1, H, W)
    ↓
Model Inference → logits.softmax(dim=1).argmax(dim=1, keepdim=True)
    ↓
Segmentation Maps (BP, 1, H, W)
    ↓
[CRITICAL: torch.cuda.synchronize() HERE]
    ↓
CuPy Postprocessing
    ↓
Peak Positions [[p, y, x], ...] per batch item
```

---

## Complete Implementation

### 1. Imports and Setup

```python
import torch
import cupy as cp
from cupyx.scipy import ndimage
from contextlib import nullcontext
from peaknet.tensor_transforms import PadAndCrop, InstanceNorm, MergeBatchChannelDims
```

### 2. InferencePipeline Class - Initialization

```python
class InferencePipeline:
    def __init__(self, model, device, mixed_precision_dtype, H, W):
        self.model = model
        self.device = device
        self.mixed_precision_dtype = mixed_precision_dtype
        self.B = None  # Batch size (determined at runtime)
        self.P = None  # Number of panels per batch item (determined at runtime)
        self.H = H
        self.W = W
        self.stages = [
            PipelineStage("data_transfer", self.data_transfer, device),
            PipelineStage("preprocess", self.preprocess, device),
            PipelineStage("inference", self.inference, device),
            PipelineStage("postprocess", self.postprocess, device)
        ]
        # Structure for connected component labeling (8-connectivity)
        self.structure = cp.ones((3, 3), dtype=cp.float32)
        self.autocast_context = None
```

**Key Point**: `self.structure = cp.ones((3, 3), dtype=cp.float32)` defines 8-connectivity for labeling adjacent pixels as part of the same peak.

### 3. Inference Stage - Convert to Segmentation Map

```python
def inference(self, batch):
    with torch.no_grad():
        with self.autocast_context:
            feature_pred = self.model(batch)
    # Convert logits to binary segmentation map
    return feature_pred.softmax(dim=1).argmax(dim=1, keepdim=True)
```

**Output**: Binary segmentation map where 1 indicates peak region, 0 indicates background.

### 4. Postprocess Stage - CuPy Peak Finding

```python
def postprocess(self, seg_maps):
    """
    Process segmentation maps to find peak positions for each batch item.

    This function takes a 4D tensor of segmentation maps and processes each map
    to find peak positions. The peaks are then grouped by batch item.

    Args:
        seg_maps (tensor): A 4D tensor of shape (BP, 1, H, W) where:
            B is the batch size,
            P is the number of panels per batch item,
            H is the height of each segmentation map,
            W is the width of each segmentation map.

    Returns:
        list of lists: A list containing B sublists, where B is the batch size.
        Each sublist contains peak positions for a single batch item across all
        its panels. Each peak position is represented as [p, y, x], where:
            p is the panel index (0 to P-1),
            y is the y-coordinate of the peak,
            x is the x-coordinate of the peak.
        The [p, y, x] format is required by downstream software for further processing.
    """
    B = seg_maps.size(0) // self.P  # BP//P
    peak_positions = [[] for _ in range(B)]  # Initialize a list for each batch item

    # Loop over all panels: (BP,1,H,W) -> (BP,H,W)
    for idx, seg_map in enumerate(seg_maps.flatten(0,1)):
        # Convert PyTorch tensor to CuPy array (stays on GPU)
        seg_map_cp = cp.asarray(seg_map, dtype=cp.float32)

        # Label connected components (find distinct peak regions)
        labeled_map, num_peaks = ndimage.label(seg_map_cp, self.structure)

        # Calculate center of mass for each labeled region
        peak_coords = ndimage.center_of_mass(
            seg_map_cp,
            cp.asarray(labeled_map, dtype=cp.float32),
            cp.arange(1, num_peaks + 1)
        )

        if len(peak_coords) > 0:
            # Append coordinates for this segmap to the corresponding batch item
            b = idx // self.P  # Batch index
            p = idx % self.P   # Panel index within the batch item
            peak_positions[b].extend(
                [p] + peak.tolist() for peak in peak_coords if len(peak) > 0
            )

    return peak_positions
```

#### Key CuPy Operations Explained

1. **`cp.asarray(seg_map, dtype=cp.float32)`**
   - Converts PyTorch GPU tensor to CuPy array
   - No data copy needed - they share the same GPU memory (via DLPack protocol)

2. **`ndimage.label(seg_map_cp, self.structure)`**
   - Labels connected components in the segmentation map
   - Uses 3x3 structure (8-connectivity) to group adjacent pixels
   - Returns:
     - `labeled_map`: array where each pixel has a label (0=background, 1,2,3...=peak IDs)
     - `num_peaks`: total number of distinct peak regions found

3. **`ndimage.center_of_mass(...)`**
   - Calculates the centroid (center of mass) for each labeled region
   - Parameters:
     - Input image: `seg_map_cp` (binary segmentation)
     - Labels: `cp.asarray(labeled_map, dtype=cp.float32)`
     - Index: `cp.arange(1, num_peaks + 1)` (process labels 1 through num_peaks)
   - Returns: list of (y, x) coordinates for each peak

4. **Format conversion to [p, y, x]**
   ```python
   [p] + peak.tolist()  # Prepend panel index to (y, x) coordinates
   ```

### 5. Process Batch with Critical Synchronization

```python
def process_batch(self, batch):
    data = self.stages[0].process(batch)  # data_transfer (H2D)

    # Run preprocessing and inference stages
    for i in range(1, len(self.stages)-1):
        data = self.stages[i].process(data)  # preprocess, inference

    # CRITICAL: Synchronize before CuPy operations
    torch.cuda.synchronize()  # Ensure all CUDA operations complete before
                              # postprocess with cupy to avoid non-deterministic behaviors

    # Now safe to run CuPy postprocessing
    peak_positions = self.stages[-1].process(data)
    return peak_positions
```

**Why `torch.cuda.synchronize()` is Required:**
- PyTorch and CuPy both use CUDA, but manage their own CUDA streams
- Without synchronization, CuPy might access GPU memory before PyTorch finishes writing to it
- This causes non-deterministic behavior and potential race conditions

---

## Usage Example (from run_mpi.py)

```python
# Initialize pipeline
pipeline = InferencePipeline(model, device, mixed_precision_dtype, args.H, args.W)
pipeline.setup_autocast()

# Process batches from dataloader
accumulated_results = []
for batch in dataloader:
    # batch shape: (B, P, H, W) where B=batch size, P=panels per item
    peak_positions = pipeline.process_batch(batch)
    # peak_positions: list of B sublists, each containing [p, y, x] coords

    # Package results with original images
    batch_results = list(zip(batch.cpu().numpy(), peak_positions))
    accumulated_results.extend(batch_results)

    # Push to output queue
    if (batch_idx + 1) % args.accumulation_steps == 0:
        ray.get(peak_positions_queue.put.remote(accumulated_results))
        accumulated_results = []
```

---

## Data Format Details

### Input to Postprocess
```
Shape: (BP, 1, H, W)
- B: Batch size
- P: Number of panels (e.g., for multi-panel detectors)
- H, W: Height and width of segmentation map
- Values: 0 (background) or 1 (peak region)
```

### Output Format
```python
[
    [  # Batch item 0
        [p0, y0, x0],  # Peak 1 on panel p0
        [p0, y1, x1],  # Peak 2 on panel p0
        [p1, y2, x2],  # Peak 3 on panel p1
        ...
    ],
    [  # Batch item 1
        [p0, y0, x0],
        ...
    ],
    ...
]
```

- `p`: Panel index (0 to P-1)
- `y`: Row coordinate (0 to H-1)
- `x`: Column coordinate (0 to W-1)

---

## Downstream Usage (cxi_consumer.py)

The peak positions are then consumed by a CXI file writer:

```python
# From cxi_consumer.py
def consume_and_write(queue_name, ...):
    peak_positions_queue = ray.get_actor(queue_name, namespace=ray_namespace)

    accumulated_images = []
    accumulated_peak_positions = []

    while not terminate:
        data = ray.get(peak_positions_queue.get.remote())

        for image, peak_positions, photon_energy in data:
            if len(peak_positions) >= min_num_peak:
                accumulated_images.append(image)
                accumulated_peak_positions.append(peak_positions)

        # Write to CXI file periodically
        if iteration % save_every == 0 and accumulated_images:
            write_cxi_file(rank, accumulated_images, accumulated_peak_positions, ...)
```

### CXI File Structure
```python
def write_cxi_file(...):
    with h5py.File(filepath, 'w') as f:
        # Store peak positions in CXI format
        f.create_dataset('/entry_1/result_1/peakSegPosRaw', ...)
        f.create_dataset('/entry_1/result_1/peakXPosRaw', ...)
        f.create_dataset('/entry_1/result_1/peakYPosRaw', ...)
        f.create_dataset('/entry_1/result_1/nPeaks', ...)

        for event_enum_idx, (image, peaks) in enumerate(zip(images, peak_positions)):
            # Convert to Cheetah coordinate system
            cheetah_peaks = cheetah_converter.convert_to_cheetah_coords(peaks)
            num_peaks = len(cheetah_peaks)
            f['/entry_1/result_1/nPeaks'][event_enum_idx] = num_peaks

            for peak_enum_idx, (seg, cheetahRow, cheetahCol) in enumerate(cheetah_peaks):
                f['/entry_1/result_1/peakSegPosRaw'][event_enum_idx, peak_enum_idx] = seg
                f['/entry_1/result_1/peakYPosRaw'][event_enum_idx, peak_enum_idx] = cheetahRow
                f['/entry_1/result_1/peakXPosRaw'][event_enum_idx, peak_enum_idx] = cheetahCol
```

---

## CPU-Based Alternative (NumPy + SciPy)

For Q2→W stage running on CPU, the cupy code can be replaced with:

```python
import numpy as np
from scipy import ndimage

def postprocess_cpu(self, seg_maps):
    """CPU-based version using NumPy and SciPy"""
    B = seg_maps.size(0) // self.P
    peak_positions = [[] for _ in range(B)]

    # Create structure on CPU
    structure = np.ones((3, 3), dtype=np.float32)

    for idx, seg_map in enumerate(seg_maps.flatten(0,1)):
        # Transfer to CPU and convert to NumPy
        seg_map_np = seg_map.cpu().numpy().astype(np.float32)

        # Label connected components (identical API to CuPy)
        labeled_map, num_peaks = ndimage.label(seg_map_np, structure)

        # Calculate center of mass (identical API to CuPy)
        peak_coords = ndimage.center_of_mass(
            seg_map_np,
            labeled_map.astype(np.float32),
            np.arange(1, num_peaks + 1)
        )

        if len(peak_coords) > 0:
            b = idx // self.P
            p = idx % self.P
            peak_positions[b].extend(
                [p] + list(peak) for peak in peak_coords if len(peak) > 0
            )

    return peak_positions
```

**Key Differences**:
- Replace `cupy` → `numpy`
- Replace `cupyx.scipy` → `scipy`
- Add `.cpu().numpy()` to transfer tensor to CPU
- Remove `torch.cuda.synchronize()` (not needed on CPU)
- API is nearly identical between CuPy and NumPy/SciPy

**Trade-offs**:
- CPU: More portable, no CuPy dependency, but slower
- GPU (CuPy): Faster, especially for large batches, but requires CuPy installation

---

## Performance Considerations

1. **Synchronization overhead**: `torch.cuda.synchronize()` blocks until all GPU operations complete
   - Necessary for correctness with CuPy
   - Can be eliminated with CPU-based approach

2. **Memory transfers**:
   - CuPy: No transfer needed (shared GPU memory with PyTorch)
   - CPU: Requires D2H transfer (`.cpu().numpy()`)

3. **Computation speed**:
   - CuPy: GPU-accelerated labeling and center-of-mass calculation
   - CPU: Sequential processing, but simpler for small batches

4. **Batch processing**: Loop over panels is sequential - could be parallelized on CPU with multiprocessing

---

## References

- Original implementation: `/sdf/home/c/cwang31/codes/peaknet-pipeline/peaknet_pipeline/pipeline.py`
- Commit history: `git log --oneline | grep -i "peak\|cupy"`
- CuPy documentation: https://docs.cupy.dev/
- SciPy ndimage docs: https://docs.scipy.org/doc/scipy/reference/ndimage.html
