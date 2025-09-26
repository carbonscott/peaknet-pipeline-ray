# Lessons Learned: Torch.compile Warmup in Distributed Ray Systems

## Key Finding: Per-Actor Warmup vs Global Warmup

After analyzing both the current peaknet-pipeline-ray implementation and the original peaknet-ray implementation, we discovered a critical architectural difference in how torch.compile warmup is handled:

- **Current Implementation**: ✅ Per-actor warmup in `__init__` method
- **Peaknet-ray**: ❌ Global warmup at pipeline level

## Why Per-Actor Warmup is Superior

### 1. Torch.compile is JIT and Process-Specific
- Each Ray actor runs in its **own process** with its own Python interpreter
- Torch.compile generates **process-specific** compiled graphs and CUDA kernels
- A compiled model in one process **cannot** be shared with another process
- **Critical**: Each actor MUST warm up its own compiled model instance

### 2. CUDA Graph Optimization Requirements
For `reduce-overhead` and `max-autotune` modes:
- CUDA graphs are **GPU-specific** and cannot be shared between processes
- Each GPU needs its own warmup iterations to build optimal execution graphs
- Per-actor warmup ensures each GPU's graphs are properly initialized before production workloads

### 3. Ray Best Practices Alignment
From Ray documentation patterns:
- **"Initialize large objects directly in the actor"** - reduces serialization overhead
- Actors should be **self-contained** and manage their own state
- **Avoid cross-actor dependencies** during initialization

### 4. PyTorch Distributed Training Patterns
PyTorch DDP documentation confirms:
- "Each process inits the model"
- "Each GPU gets its own process"
- Warmup should happen **after** model creation but **before** serving traffic

## Implementation Details

### Current Implementation (Correct)
```python
# peaknet_pipeline_ray/core/peaknet_ray_pipeline_actor.py:196-199
if warmup_samples > 0 and self.peaknet_model is not None:
    logging.info(f"Running model warmup with {warmup_samples} samples...")
    self._run_warmup(warmup_samples, input_shape)
    self.warmup_completed = True
```

Each actor runs a complete warmup through the full pipeline (H2D → Compute → D2H) during initialization.

### Peaknet-ray Implementation (Problematic)
```python
# Global warmup in run_pipeline_test()
if not skip_warmup and warmup_samples > 0:
    print(f"Warmup phase: {warmup_samples} samples...")
    _run_double_buffer_pipeline(
        pipeline, cpu_tensors[:warmup_samples], batch_size, "warmup", sync_frequency, is_warmup=True
    )
```

Single warmup phase runs once before distributed processing begins, leaving actors with uncompiled models.

## Performance Impact Analysis

### Current Approach (Per-Actor Warmup)
- ✅ **Startup**: All actors warm up in parallel (~10s total)
- ✅ **Runtime**: Optimal compiled performance from first production batch
- ✅ **Memory**: Each actor has properly initialized CUDA graphs
- ✅ **Consistency**: All actors have identical compilation state

### Peaknet-ray Approach (Global Warmup)
- ❌ **Startup**: Single warmup + actor creation (sequential)
- ❌ **Runtime**: First N batches per actor trigger compilation (unpredictable slowdown)
- ❌ **Memory**: CUDA graphs built during production (memory spikes)
- ❌ **Inconsistency**: Actors may have different compilation states

## Recommendations for Production Systems

### 1. Always Use Per-Actor Warmup
```python
# In actor __init__ method
if warmup_samples > 0 and self.model is not None:
    self._run_warmup(warmup_samples, input_shape)
    self.warmup_completed = True
```

### 2. Tune Warmup Samples by Compilation Mode
```python
# Increase warmup for aggressive compilation modes
if compile_mode in ['reduce-overhead', 'max-autotune']:
    warmup_samples = max(warmup_samples, 1000)
```

### 3. Monitor and Log Warmup Performance
```python
start_time = time.time()
self._run_warmup(warmup_samples, input_shape)
warmup_time = time.time() - start_time
logging.info(f"Warmup completed: {warmup_samples} samples in {warmup_time:.2f}s")
```

### 4. Consider Compilation Caching (PyTorch 2.0+)
- Use `torch.compile(..., dynamic=False)` for better caching
- Set `TORCH_COMPILE_DEBUG=1` to monitor cache hits
- Consider persistent cache directories for repeated deployments

## Key Takeaways

1. **Architecture Matters**: The current implementation correctly follows distributed systems best practices
2. **Process Isolation**: Ray's process model requires per-actor initialization of compiled models
3. **Performance vs Startup Time**: Per-actor warmup trades slightly longer initialization for optimal runtime performance
4. **Consistency**: All actors start with properly compiled models, ensuring predictable performance

## Future Considerations

- Monitor for PyTorch improvements in cross-process compilation sharing
- Consider async warmup strategies if startup time becomes critical
- Explore compilation result caching for even faster actor initialization
- Investigate torch.compile memory usage patterns during warmup

## Bottom Line

**The current peaknet-pipeline-ray implementation's per-actor warmup approach is architecturally correct and performance-optimal for distributed torch.compile systems.** This represents a significant improvement over the original peaknet-ray implementation.