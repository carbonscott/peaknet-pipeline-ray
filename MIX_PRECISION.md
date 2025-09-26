# Mixed Precision Analysis and Recommendations

## Current Implementation Issues

The current mixed precision implementation provides **minimal memory savings** because:

1. **Model weights remain in float32**: No memory reduction for model parameters
   - `peaknet_utils.py:197` only does `.to(device)` without dtype conversion
   - Model continues to use full float32 precision (~50% more memory than needed)

2. **Input data forced to float32**: All tensors explicitly created as float32
   - `peaknet_ray_pipeline_actor.py:590`: `torch.from_numpy(detector_data[i].astype(np.float32))`
   - `socket_hdf5_producer.py:255`: `torch.from_numpy(image_data.astype(np.float32))`
   - `streaming_producer.py:97`: `dtype=torch.float32`

3. **Autocast only affects computations temporarily**: No persistent memory savings
   - Operations are cast to bfloat16 during computation but results often upcast back
   - Model parameters and activations stored in float32

## Industry Best Practices for Inference (2024)

### For Inference Workloads (like PeakNet pipeline):

**Recommended: True Half-Precision**
- Convert model weights to bfloat16: `model = model.to(torch.bfloat16)`
- Convert input data to bfloat16: `input = input.to(torch.bfloat16)`
- **Memory savings: ~50%** for model weights and activations
- **Stability: Good** - bfloat16 has same dynamic range as float32
- **No gradient updates needed** - stability concerns are primarily for training

### Why Current Approach Is Suboptimal:

**Autocast-only mixed precision** is designed for **training stability**:
- Keeps weights in float32 for precise gradient updates
- Only temporarily casts operations to lower precision
- Minimal memory savings (just intermediate computations)
- **Not ideal for inference** where memory is the primary constraint

## Technical Analysis

### Memory Usage Breakdown:
- **Model parameters**: ~50% reduction with bfloat16 weights
- **Activations/intermediate tensors**: ~50% reduction with bfloat16 inputs
- **Current savings**: ~10-20% (only temporary operation casting)

### Hardware Support:
- **A100/H100 GPUs**: 16x faster bfloat16 operations vs float32
- **RTX 3090+ GPUs**: Significant performance gains
- **Older GPUs**: Limited benefits, may default to float32

## Recommendations

### Immediate Fix (High Impact):
1. **Convert model weights to target precision after loading**:
   ```python
   # In peaknet_utils.py:create_peaknet_model()
   if precision_dtype != 'float32':
       wrapper = wrapper.to(dtype=torch.bfloat16)  # or torch.float16
   ```

2. **Create input tensors in target precision**:
   ```python
   # Replace all float32 tensor creation
   torch.from_numpy(data.astype(np.float32))  # Current
   torch.from_numpy(data).to(target_dtype)    # Proposed
   ```

3. **Keep autocast for additional optimization**:
   - Maintains current autocast implementation
   - Provides operation-level casting for unsupported ops

### Configuration Update:
Current precision config in `$TEST_DIR/peaknet-socket.yaml:77`:
```yaml
precision:
  dtype: "bfloat16"
```

Should drive **both** model weights and input data precision, not just autocast.

### Expected Results:
- **Memory usage reduction**: 40-50% overall
- **Performance improvement**: 1.5-3x speedup on modern GPUs
- **Batch size scaling**: Can process larger batches within same memory limit
- **Numerical stability**: Maintained with bfloat16 (same dynamic range as float32)

## Implementation Priority

**High Priority**: This fix would provide significant memory savings and performance improvements for the streaming inference pipeline, allowing larger batch sizes and better GPU utilization.

**Low Risk**: For inference-only workloads, converting to bfloat16 is standard practice with minimal accuracy impact.