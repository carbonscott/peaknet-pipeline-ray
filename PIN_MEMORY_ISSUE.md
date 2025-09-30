# Pinned Memory Exhaustion Issue

## Educational Branch - DO NOT USE IN PRODUCTION

This branch demonstrates an approach to fix CPU blocking during H2D transfers by pinning tensors at the Q1→P boundary. **However, this approach causes pinned memory exhaustion and crashes after ~361 batches.**

## The Problem

### Root Cause
Ray's object store returns tensors in pageable memory. When transferring pageable memory to GPU with `non_blocking=True`, CUDA must stage the transfer through a temporary pinned buffer, which blocks the CPU.

### Attempted Solution (This Branch)
```python
# In process_from_queue() and process_batch_from_ray_object_store()
if self.pin_memory and torch.cuda.is_available():
    cpu_tensors = [t.pin_memory() if not t.is_pinned() else t for t in cpu_tensors]
```

### Why It Fails
1. `tensor.pin_memory()` **creates a NEW tensor** in pinned memory
2. The original pageable tensor remains alive (Ray manages its lifecycle)
3. Each batch creates new pinned tensors → memory accumulates
4. Pinned memory is limited (typically ~half of GPU memory)
5. After ~361 batches: **CUDA error: unspecified launch failure** (pinned memory exhausted)

### Observed Behavior
- ✅ Works for first ~361 batches
- ✅ GPU overlap remains good (double buffering works)
- ❌ Crashes with CUDA error when pinned memory exhausted
- ❌ CPU blocking was minimal anyway (not a critical problem)

## The Proper Solution

Use **staging buffers** (implemented in `producer-bottleneck` branch):
- Allocate pinned staging buffers ONCE during actor initialization
- Reuse buffers across all batches (like pipeline's output buffers)
- Copy incoming tensors into staging buffers before H2D
- Fixed memory footprint, no leaks

See the main `producer-bottleneck` branch for the correct implementation.

## Key Takeaway

**Never repeatedly call `tensor.pin_memory()` in a loop** - it allocates new memory each time. Instead, allocate pinned buffers once and reuse them.

## References
- Error location: `peaknet_ray_pipeline_actor.py:250-253, 461-465`
- Failure point: Batch ~361 (depends on tensor size and system pinned memory limit)
- GPU behavior: Double buffering overlap remained good despite this issue
