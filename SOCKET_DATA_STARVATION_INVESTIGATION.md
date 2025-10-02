# Socket Data Starvation Investigation

## Problem Statement

Socket mode shows signs of potential data starvation compared to random mode, with profiling results suggesting pipeline gaps and possible host-device synchronization issues. Need to investigate whether this is due to network bottlenecks in the data producer (LCLStreamer) or issues in the Q1→P pipeline stage.

## Key Question

**Is the double buffered inference pipeline execution fundamentally the same between random and socket modes?**

**Answer: YES** - The P stage (double buffered inference) is identical:
- Same H2D → Compute → D2H overlap patterns
- Same Buffer A/B alternation with CUDA events
- Same GPU memory management and stream synchronization
- Same pipeline actor implementation and execution flow

## Data Flow Analysis

### Random Mode: Ultra-Fast Local Generation ⚡
```
CPU: torch.randn() → PipelineInput → Q1 → Actor gets ready tensors
Timeline: ~microseconds per batch
Bottlenecks: None (CPU generation is instant)
```

### Socket Mode: Multi-Stage Network Processing 🌐
```
LCLStreamer → Network I/O → Raw HDF5 bytes → Q1 → Actor parses HDF5 → CPU tensors → GPU
Timeline: ~milliseconds per batch (1000x slower than random)
Bottlenecks: Multiple potential chokepoints
```

## Critical Performance Bottlenecks Identified

### 1. LCLStreamer → Network Bottleneck
- **Issue**: Network I/O latency, bandwidth limits, TCP buffering delays
- **Evidence**: Socket timeouts, connection retries in producer logs
- **Impact**: Irregular, bursty data arrival patterns instead of smooth flow

### 2. HDF5 Parsing in GPU Actor
- **Issue**: CPU-bound HDF5 parsing happens during GPU overlap window
- **Evidence**: `_parse_raw_socket_data()` processes raw bytes → tensors every batch
- **Impact**: Additional ~1-2ms processing overhead per batch
- **Location**: `peaknet_ray_pipeline_actor.py:545-604`

### 3. Queue Starvation Pattern
- **Issue**: Q1 queue empties faster than socket producer can refill it
- **Evidence**: GPU idle periods visible in profiling traces
- **Impact**: Pipeline gaps manifest as explicit host-device synchronization

## Investigation Plan

### 1. Profile Queue Occupancy Over Time
```python
# Monitor Q1 queue depth every 100ms during execution
queue_depths = []
while pipeline_running:
    depth = q1_manager.get_total_queue_size()
    max_depth = q1_manager.get_max_queue_size()
    queue_depths.append((time.time(), depth, depth/max_depth))
    time.sleep(0.1)

# Analysis:
# Healthy pipeline: Queue stays 80-100% full consistently
# Data starvation: Queue oscillates 0-50% with periodic empty periods
```

### 2. Compare Producer vs Consumer Rates
```python
# Measure actual throughput rates
socket_producer_rate = total_batches_received / elapsed_time
gpu_actor_rate = total_batches_processed / elapsed_time

# Diagnosis:
# If socket_rate < actor_rate → Producer bottleneck (network/LCLStreamer)
# If socket_rate > actor_rate → Consumer bottleneck (GPU pipeline)
# Rates should match in healthy system
```

### 3. Analyze Profiling Timeline Patterns

**Random Mode Expectations:**
- Continuous GPU utilization (>95%)
- Minimal pipeline gaps
- No unexpected cudaStreamSynchronize calls
- Smooth, predictable execution timeline

**Socket Mode Investigation:**
Look for these starvation signatures:
- GPU idle periods longer than buffer swap time
- Explicit cudaStreamSynchronize calls indicating pipeline stalls
- Irregular batch processing intervals
- Host-device memory transfer gaps

### 4. Network Diagnostics
```bash
# Monitor network bandwidth during streaming
iftop -i eth0              # Real-time network utilization
netstat -i                 # Check packet loss/error rates

# Socket buffer analysis
ss -i | grep 12321         # Monitor socket buffer sizes and congestion
lsof -i :12321             # Check socket connection status

# LCLStreamer performance
# Monitor LCLStreamer logs for:
# - Batch generation rate
# - Network send buffer utilization
# - Any timeout/retry messages
```

### 5. Queue Depth Monitoring Integration
Add real-time queue monitoring to pipeline:
```python
# In pipeline execution loop
if batch_idx % 10 == 0:  # Every 10 batches
    q1_depth = q1_manager.get_queue_depth()
    q1_utilization = q1_depth / q1_manager.max_size
    logging.info(f"Q1 utilization: {q1_utilization:.1%} ({q1_depth} items)")

    if q1_utilization < 0.3:  # Less than 30% full
        logging.warning("Potential data starvation detected")
```

## Root Cause Hypotheses

### Primary Hypothesis: LCLStreamer Network Bottleneck
- **Cause**: Network latency creates bursty, irregular data delivery patterns
- **Mechanism**: Q1 queue drains during network delays → GPU actors wait for data → pipeline stalls
- **Evidence**: Profiling should show synchronization gaps correlated with network activity
- **Verification**: Monitor network utilization patterns and correlate with GPU idle periods

### Secondary Hypothesis: HDF5 Parsing Overhead
- **Cause**: CPU-bound HDF5 parsing adds processing latency per batch in GPU actors
- **Mechanism**: Actor spends extra time parsing instead of pure tensor processing
- **Evidence**: Compare actor processing times: socket vs random mode
- **Verification**: Profile actor execution breakdown: parsing vs inference time

### Tertiary Hypothesis: Ray Queue Overhead
- **Cause**: Socket mode uses different queue item types (raw bytes vs ready tensors)
- **Mechanism**: Ray serialization/deserialization overhead for complex socket data
- **Evidence**: Queue put/get operations take longer in socket mode
- **Verification**: Measure queue operation latencies

## Next Steps

### Immediate Actions (Tomorrow)
1. **Run comparative profiling**: Random vs socket mode with identical batch sizes
2. **Implement queue monitoring**: Add real-time Q1 depth logging
3. **Measure producer rates**: Log actual throughput from both data sources
4. **Network analysis**: Monitor bandwidth utilization during socket streaming

### Data Collection Targets
- Queue occupancy timeline (target: >80% full consistently)
- Producer throughput rates (target: match consumer rates)
- Network utilization patterns (identify bursty vs smooth delivery)
- Actor processing time breakdown (parsing vs inference overhead)

### Success Criteria
- Identify specific bottleneck location (network vs parsing vs queue)
- Quantify performance gap magnitude
- Propose targeted optimization approach
- Validate hypothesis with measurement data

---

## Context Files
- **Socket Producer**: `peaknet_pipeline_ray/core/lightweight_socket_producer.py`
- **Random Producer**: `peaknet_pipeline_ray/core/streaming_producer.py`
- **Pipeline Actor**: `peaknet_pipeline_ray/core/peaknet_ray_pipeline_actor.py:545-604`
- **HDF5 Parsing**: `_parse_raw_socket_data()` method
- **Queue Management**: `peaknet_pipeline_ray/utils/queue.py`

## Investigation Status
- [x] Architecture analysis complete
- [x] Bottleneck identification complete
- [x] Investigation plan designed
- [ ] Data collection implementation
- [ ] Comparative profiling execution
- [ ] Root cause validation
- [ ] Optimization recommendations