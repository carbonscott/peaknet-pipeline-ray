# Q2 to CXI Writer: Implementation Architecture

**Status**: Implementation Planning
**Package Location**: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/cxi-pipeline-ray` (new package)
**Testing Location**: `$TEST_DIR` (test configs and validation)
**Related**: See `PLAN-Q2-CXI-WRITER-v2.md` for technical design details

---

## Design Principles

### 1. Separation of Concerns

**Pipeline (P stage)**: Produces inference results to Q2
- Owns: Model inference, GPU resources, Q1→P→Q2 pipeline
- Package: `peaknet-pipeline-ray` (inference)
- Repo: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray`
- Config: `peaknet-socket-profile-673m-with-output.yaml`

**Writer (W stage)**: Consumes inference results from Q2
- Owns: Peak finding, coordinate conversion, CXI file writing
- Package: `cxi-pipeline-ray` (post-processing & file writing) **NEW PACKAGE**
- Repo: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/cxi-pipeline-ray`
- Config: `cxi_writer.yaml` (new, fully configurable)

**Communication**: Ray object store + ShardedQueue (Q2)

### 2. Configuration Strategy

**Problem**: Pipeline and writer must coordinate on Ray namespace and queue topology, but shouldn't share configs.

**Solution**: Separate but consistent configs
- Each process has its own configuration file
- Critical coordination settings (Ray/queue) are **explicitly duplicated** in both configs
- No config inheritance or extension (keeps it simple)
- Manual validation ensures consistency (can add tooling later)

**Why not extend/reuse pipeline config?**
- Writer has completely different concerns (geometry files, peak filtering, file I/O)
- Extending would pollute pipeline config with writer-specific settings
- Separate processes should have separate configs
- Easier to version, deploy, and maintain independently

---

## Package Structure

### New Package: `cxi-pipeline-ray`

**Location**: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/cxi-pipeline-ray`

**Rationale**: Substantial infrastructure component that deserves proper packaging
- Installable package with entry points
- Version controlled, testable, maintainable
- Reusable across experiments
- Follows same pattern as `peaknet-pipeline-ray`

```
cxi-pipeline-ray/                           # Package root
├── cxi_pipeline_ray/                       # Source code
│   ├── __init__.py
│   ├── core/                               # Core components
│   │   ├── __init__.py
│   │   ├── peak_finding.py                 # Ray task for peak finding
│   │   ├── file_writer.py                  # Ray actor for CXI writing
│   │   └── coordinator.py                  # Main pipeline coordinator
│   ├── utils/                              # Utilities
│   │   ├── __init__.py
│   │   ├── config.py                       # Configuration loading/validation
│   │   └── validation.py                   # Config consistency checker
│   ├── cli.py                              # Command-line interface
│   └── version.py
├── examples/                               # Example configurations
│   └── configs/
│       ├── cxi_writer_default.yaml         # Default config with comments
│       └── cxi_writer_production.yaml      # Production config example
├── tests/                                  # Unit and integration tests
│   ├── test_peak_finding.py
│   ├── test_file_writer.py
│   └── test_integration.py
├── pyproject.toml                          # Package definition (modern)
├── setup.py                                # Package definition (legacy compat)
├── README.md                               # Package documentation
└── LICENSE

# Test configurations (separate from package)
$TEST_DIR/                                  # Testing directory
├── cxi_writer_test.yaml                    # Test config for $TEST_DIR
├── peaknet-socket-profile-673m-with-output.yaml  # Pipeline config (exists)
└── validate_pipeline_configs.py           # Integration validation script
```

### Key Package Components

**1. Entry Point** (like `peaknet-pipeline` command):
```bash
# After pip install -e .
cxi-writer --config examples/configs/cxi_writer_default.yaml
```

**2. Importable Library**:
```python
from cxi_pipeline_ray.core.coordinator import run_cpu_postprocessing_pipeline
from cxi_pipeline_ray.utils.config import load_config
```

**3. No Hard-Coded Values**: Everything configurable via YAML

### Related Packages

```
peaknet-pipeline-ray/                       # Inference package (already exists)
├── peaknet_pipeline_ray/
│   └── utils/
│       └── queue.py                        # ShardedQueueManager (shared)
└── examples/configs/
    └── peaknet-socket-profile-673m-with-output.yaml  # Pipeline config

cxi-pipeline-ray/                           # Post-processing package (NEW)
├── cxi_pipeline_ray/
│   └── ...                                 # Implementation
└── examples/configs/
    └── cxi_writer_default.yaml             # Writer config
```

**Dependency**: `cxi-pipeline-ray` depends on `peaknet-pipeline-ray` for `ShardedQueueManager` (or extract to shared package later)

---

## Configuration Schema

### Pipeline Config (Existing)

**File**: `peaknet-socket-profile-673m-with-output.yaml`

```yaml
runtime:
  queue_num_shards: 8
  queue_maxsize_per_shard: 1600
  queue_names:
    input_queue: "peaknet_q1"   # Q1: Producer → Inference
    output_queue: "peaknet_q2"  # Q2: Inference → Writer ← WRITER CONSUMES THIS
  enable_output_queue: true     # MUST be true for writer to work

ray:
  namespace: "peaknet-pipeline"  # ← CRITICAL: Writer must use same namespace
```

### Writer Config (New)

**Example File**: `cxi-pipeline-ray/examples/configs/cxi_writer_default.yaml`
**Test File**: `$TEST_DIR/cxi_writer_test.yaml`

```yaml
# Q2 Consumer: Inference Results → CXI Files
# Consumes from Q2 queue produced by peaknet-pipeline

# ============================================================
# Ray Configuration (MUST MATCH PIPELINE)
# ============================================================
ray:
  namespace: "peaknet-pipeline"  # CRITICAL: Must match pipeline config

# ============================================================
# Input Queue Configuration (MUST MATCH PIPELINE'S output_queue)
# ============================================================
queue:
  name: "peaknet_q2"             # CRITICAL: Must match pipeline's queue_names.output_queue
  num_shards: 8                  # CRITICAL: Must match pipeline's queue_num_shards
  maxsize_per_shard: 1600        # Should match pipeline for consistency
  poll_timeout: 0.01             # How long to wait for data (seconds)

# ============================================================
# CPU Post-Processing Configuration
# ============================================================
processing:
  num_cpu_workers: 16            # Parallel Ray tasks for peak finding
  max_pending_tasks: 100         # Backpressure limit (prevents OOM)

# ============================================================
# Peak Finding Configuration
# ============================================================
peak_finding:
  min_num_peak: 10               # Minimum peaks to save event (quality filter)
  max_num_peak: 2048             # Maximum peaks per event (CXI array size)
  connectivity: 8                # 8-connectivity for connected components (3x3 structure)

# ============================================================
# Geometry and Coordinate Conversion
# ============================================================
geometry:
  geom_file: "/path/to/detector.geom"  # CrystFEL geometry file for CheetahConverter
  # Example: /sdf/data/lcls/ds/prj/prjcwang31/results/geometry/cxi_detector.geom

# ============================================================
# CXI Output Configuration
# ============================================================
output:
  output_dir: "./cxi_output"     # Directory for CXI files
  file_prefix: "peaknet_cxi"     # Filename prefix (e.g., peaknet_cxi_20250113_120000_chunk0001.cxi)
  buffer_size: 100               # Events to buffer before writing CXI file
  create_output_dir: true        # Auto-create output directory if missing

# ============================================================
# System Configuration
# ============================================================
system:
  log_level: "INFO"              # DEBUG, INFO, WARNING, ERROR
  log_file: null                 # Optional: Path to log file (null = stdout only)
  progress_interval: 50          # Log progress every N batches
```

### Configuration Validation

**Critical settings that must match** between pipeline and writer:

| Setting | Pipeline Location | Writer Location | Why Critical |
|---------|------------------|-----------------|--------------|
| Ray namespace | `ray.namespace` | `ray.namespace` | Actors/queues must be in same namespace |
| Q2 queue name | `runtime.queue_names.output_queue` | `queue.name` | Writer must read from correct queue |
| Q2 num shards | `runtime.queue_num_shards` | `queue.num_shards` | ShardedQueue topology must match |

**Optional validation script**:

```bash
# Check if pipeline and writer configs are consistent
cxi-writer --validate-config \
  --pipeline-config $TEST_DIR/peaknet-socket-profile-673m-with-output.yaml \
  --writer-config $TEST_DIR/cxi_writer_test.yaml

# Or use standalone script
python $TEST_DIR/validate_pipeline_configs.py \
  --pipeline-config $TEST_DIR/peaknet-socket-profile-673m-with-output.yaml \
  --writer-config $TEST_DIR/cxi_writer_test.yaml
```

Output:
```
✓ Ray namespace matches: peaknet-pipeline
✓ Q2 queue name matches: peaknet_q2
✓ Q2 num shards matches: 8
✓ Configuration validation passed
```

---

## Implementation Components

### Component 1: Command-Line Interface (`cxi_pipeline_ray/cli.py`)

**Responsibilities**:
- Parse command-line arguments
- Load configuration file (YAML)
- Initialize Ray (connect to existing cluster)
- Create ShardedQueueManager for Q2
- Launch coordinator function
- Handle graceful shutdown (Ctrl+C)
- Provide config validation mode

**Entry Point** (defined in `pyproject.toml`):
```toml
[project.scripts]
cxi-writer = "cxi_pipeline_ray.cli:main"
```

**Interface**:
```bash
# Normal operation
cxi-writer --config examples/configs/cxi_writer_default.yaml

# With config overrides
cxi-writer --config $TEST_DIR/cxi_writer_test.yaml \
  --num-cpu-workers 32 \
  --output-dir /custom/path

# Validation mode
cxi-writer --validate-config \
  --pipeline-config path/to/pipeline.yaml \
  --writer-config path/to/writer.yaml
```

### Component 2: Core Implementation (`cxi_pipeline_ray/core/`)

**File: `peak_finding.py`**
- Ray Task: `process_samples_task()`
- Stateless CPU peak finding
- Input: mini-batch of logits (ObjectRef)
- Output: list of peak positions
- From: `PLAN-Q2-CXI-WRITER-v2.md` lines 460-506

**File: `file_writer.py`**
- Ray Actor: `CXIFileWriterActor`
- Stateful file writer
- Maintains: CheetahConverter, buffer, chunk counter
- Methods: `submit_processed_batch()`, `flush_final()`, `get_statistics()`
- From: `PLAN-Q2-CXI-WRITER-v2.md` lines 516-726

**File: `coordinator.py`**
- Function: `run_cpu_postprocessing_pipeline()`
- Main consumption loop
- Pulls from Q2, splits into mini-batches, launches tasks
- Implements backpressure, pipelining, batched operations
- From: `PLAN-Q2-CXI-WRITER-v2.md` lines 730-976

### Component 3: Utilities (`cxi_pipeline_ray/utils/`)

**File: `config.py`**
```python
def load_config(config_path: str) -> dict:
    """Load and validate YAML configuration."""
    ...

def merge_config_with_overrides(config: dict, overrides: dict) -> dict:
    """Merge CLI overrides into config."""
    ...
```

**File: `validation.py`**
```python
def validate_consistency(pipeline_config: dict, writer_config: dict):
    """
    Validate that pipeline and writer configs have consistent Ray/queue settings.

    Raises ValueError if critical settings don't match.
    """
    # Check Ray namespace
    if pipeline_config['ray']['namespace'] != writer_config['ray']['namespace']:
        raise ValueError(...)

    # Check Q2 queue name
    if pipeline_config['runtime']['queue_names']['output_queue'] != writer_config['queue']['name']:
        raise ValueError(...)

    # Check Q2 shards
    if pipeline_config['runtime']['queue_num_shards'] != writer_config['queue']['num_shards']:
        raise ValueError(...)
```

---

## Usage Patterns

### Pattern 1: Sequential Deployment (Testing)

**Use case**: Test Q2 writer with recorded inference results

```bash
# Step 1: Run pipeline to produce Q2 data (finite samples)
peaknet-pipeline \
  --config $TEST_DIR/peaknet-socket-profile-673m-with-output.yaml \
  --total-samples 10240

# Step 2: Run writer to consume Q2 data
cxi-writer --config $TEST_DIR/cxi_writer_test.yaml
```

### Pattern 2: Concurrent Deployment (Production)

**Use case**: Streaming production workload

```bash
# Terminal 1: Start pipeline (creates Q2 queue and produces data)
peaknet-pipeline \
  --config $TEST_DIR/peaknet-socket-profile-673m-with-output.yaml

# Terminal 2: Start Q2 writer (connects to Q2 and consumes data)
cxi-writer --config $TEST_DIR/cxi_writer_test.yaml
```

**Note**: Pipeline creates the Q2 queue; writer connects to it. Both launch orders technically work due to ShardedQueueManager's create-or-connect behavior, but starting the producer first follows the natural data flow pattern.

### Pattern 3: Multi-Node Deployment

**Use case**: Writer on different node than pipeline

```bash
# On GPU node (sdfada014): Run inference pipeline
peaknet-pipeline \
  --config $TEST_DIR/peaknet-socket-profile-673m-with-output.yaml

# On CPU-heavy node (sdfrome001): Run writer
cxi-writer --config $TEST_DIR/cxi_writer_test.yaml
```

**Requirement**: Both nodes must be in same Ray cluster with same namespace.

---

## Deployment Checklist

### Before First Run

- [ ] **Package installation**: Install `cxi-pipeline-ray` package
  ```bash
  cd /sdf/data/lcls/ds/prj/prjcwang31/results/codes/cxi-pipeline-ray
  pip install -e .  # Editable mode for development
  ```
- [ ] **Ray cluster**: Ensure Ray cluster is running
- [ ] **Configuration**: Create config in `$TEST_DIR/cxi_writer_test.yaml` with correct paths
- [ ] **Validation**: Check Ray/queue settings match pipeline config
- [ ] **Geometry file**: Verify `geom_file` path exists and is valid
- [ ] **Output directory**: Ensure `output_dir` exists or `create_output_dir: true`
- [ ] **Dependencies**: All dependencies in `pyproject.toml` (installed with package)

### Configuration Checklist

- [ ] `ray.namespace` matches pipeline
- [ ] `queue.name` matches pipeline's `output_queue`
- [ ] `queue.num_shards` matches pipeline's `queue_num_shards`
- [ ] `geometry.geom_file` points to valid geometry file
- [ ] `output.output_dir` is writable
- [ ] `processing.num_cpu_workers` is reasonable for node (start with 16)

### Runtime Monitoring

```bash
# Check Ray dashboard
ray dashboard

# Monitor writer logs
tail -f <log_file>  # or watch stdout

# Check Q2 queue status
python -c "
import ray
ray.init(namespace='peaknet-pipeline', ignore_reinit_error=True)
# Add queue inspection code
"

# Monitor CXI output
watch -n 5 'ls -lh $OUTPUT_DIR/*.cxi'
```

---

## Performance Tuning Guide

### Tuning Parameter: `num_cpu_workers`

**What it controls**: Parallelism of peak finding tasks

**Starting point**:
```python
num_cpu_workers = num_cpu_cores // 4  # Conservative
```

**Symptoms and adjustments**:

| Symptom | Diagnosis | Adjustment |
|---------|-----------|------------|
| Low CPU utilization (<50%) | Underutilized CPUs | Increase `num_cpu_workers` |
| High task scheduling overhead | Too many tiny tasks | Decrease `num_cpu_workers` |
| Q2 queue growing | Writer too slow | Increase `num_cpu_workers` |
| Memory usage growing | Too many pending tasks | Decrease `max_pending_tasks` |

### Tuning Parameter: `max_pending_tasks`

**What it controls**: Backpressure limit (bounds memory usage)

**Trade-off**:
- **Lower** (50): Less memory, may underutilize CPUs
- **Higher** (200): More memory, better throughput

**Recommendation**: Start with 100, adjust based on memory monitoring

### Tuning Parameter: `buffer_size`

**What it controls**: Events per CXI file

**Trade-off**:
- **Smaller** (50): More files, faster flush, less memory
- **Larger** (200): Fewer files, slower flush, more memory

**Recommendation**: 100 events per CXI file (good balance)

---

## Integration with Existing Tools

## Package Dependencies

### pyproject.toml Structure

```toml
[project]
name = "cxi-pipeline-ray"
version = "0.1.0"
description = "Ray-based CPU post-processing pipeline for PeakNet inference results"
authors = [{name = "Your Name", email = "your.email@example.com"}]
requires-python = ">=3.9"

dependencies = [
    "ray>=2.0.0",
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "h5py>=3.0.0",
    "torch>=2.0.0",
    "pyyaml>=6.0",
    "peaknet-pipeline-ray",  # For ShardedQueueManager and PipelineOutput
    "crystfel-stream-parser",  # For CheetahConverter
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[project.scripts]
cxi-writer = "cxi_pipeline_ray.cli:main"

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"
```

### Import ShardedQueueManager from Pipeline Package

```python
# cxi_pipeline_ray/core/coordinator.py
from peaknet_pipeline_ray.utils.queue import ShardedQueueManager

# Connect to Q2 queue (created by pipeline)
q2_manager = ShardedQueueManager(
    base_name=config['queue']['name'],
    num_shards=config['queue']['num_shards'],
    maxsize_per_shard=config['queue']['maxsize_per_shard']
)
```

### Import CheetahConverter from crystfel_stream_parser

```python
# cxi_pipeline_ray/core/file_writer.py
from crystfel_stream_parser.joblib_engine import StreamParser
from crystfel_stream_parser.cheetah_converter import CheetahConverter

# Initialize once in actor
geom_block = StreamParser(geom_file).parse(
    num_cpus=1,
    returns_stream_dict=True
)[0].get('GEOM_BLOCK')
cheetah_converter = CheetahConverter(geom_block)
```

---

## Testing Strategy

### Unit Tests (Future)

- Test logits → segmentation map conversion
- Test peak finding with synthetic data
- Test CheetahConverter integration
- Test CXI file structure

### Integration Tests (Priority)

**Test 1: Dummy Q2 data**
```python
# Generate synthetic PipelineOutput and push to Q2
# Verify writer produces valid CXI files
```

**Test 2: Small-scale end-to-end**
```bash
# Run pipeline with total_samples=100
# Run writer and verify output
# Validate CXI files with h5py
```

**Test 3: Continuous streaming**
```bash
# Run pipeline with socket source
# Run writer concurrently
# Monitor for memory leaks, queue growth
```

### Validation Tests

**CXI format validation**:
```python
import h5py

# Check required datasets exist
with h5py.File(cxi_file, 'r') as f:
    assert '/entry_1/data_1/data' in f
    assert '/entry_1/result_1/nPeaks' in f
    assert '/entry_1/result_1/peakXPosRaw' in f
    # ... etc
```

**Downstream tool validation**:
```bash
# Try to index with CrystFEL
indexamajig -i input.lst -o output.stream --geometry=detector.geom
```

---

## Error Handling

### Common Issues and Solutions

**Issue**: `Ray namespace mismatch`
```
Error: Cannot find queue 'peaknet_q2' in namespace 'peaknet-pipeline'
```
**Solution**: Check `ray.namespace` matches in both configs

**Issue**: `Queue shards mismatch`
```
Error: Queue 'peaknet_q2' has 4 shards, expected 8
```
**Solution**: Check `queue.num_shards` matches pipeline's `queue_num_shards`

**Issue**: `Geometry file not found`
```
FileNotFoundError: /path/to/detector.geom
```
**Solution**: Verify `geometry.geom_file` path is correct

**Issue**: `OOM (Out of Memory)`
```
Ray ObjectStoreFullError: ...
```
**Solution**: Decrease `max_pending_tasks` or increase Ray object store size

**Issue**: `Writer faster than pipeline`
```
Queue empty, waiting... (repeatedly)
```
**Solution**: Normal if pipeline hasn't started yet. Otherwise check pipeline is producing to Q2.

---

## Future Enhancements

### Phase 1: Core Implementation (Current)
- ✅ Basic writer with separate config
- ✅ Ray task-based peak finding
- ✅ Actor-based file writing
- ✅ Backpressure control

### Phase 2: Robustness
- [ ] Configuration validator tool
- [ ] Better error messages for config mismatches
- [ ] Automatic queue topology discovery (eliminate manual config duplication)
- [ ] Health monitoring endpoint

### Phase 3: Optimization
- [ ] Adaptive `num_cpu_workers` based on CPU utilization
- [ ] Smart buffering based on downstream I/O speed
- [ ] Compression for CXI files

### Phase 4: Production Features
- [ ] Integration with run management system
- [ ] Metadata injection (beamline conditions, etc.)
- [ ] Multi-run output (separate directories per run)

---

## References

- **Technical design**: `PLAN-Q2-CXI-WRITER-v2.md`
- **Ray best practices**: `RAY-BEST-PRACTICES-REVIEW.md`
- **Peak finding algorithm**: `KNOWLEDGE-FIND-PEAKPOS.md`
- **Pipeline config**: `examples/configs/peaknet-socket-profile-673m-with-output.yaml`
- **ShardedQueue**: `peaknet_pipeline_ray/utils/queue.py`

---

## Summary

**Key Decisions**:
1. ✅ **Proper Python package** (`cxi-pipeline-ray`) - professional, maintainable infrastructure
2. ✅ **Separate from pipeline repo** - clean separation of inference vs post-processing
3. ✅ **Fully configurable** - no hard-coded values, everything driven by YAML config
4. ✅ **Explicit coordination settings** - Ray/queue settings duplicated in both configs
5. ✅ **Import shared utilities** - reuse `ShardedQueueManager` from `peaknet-pipeline-ray`
6. ✅ **Validation tooling** - built-in config validation mode
7. ✅ **Test configs in `$TEST_DIR`** - package provides examples, users customize in test dir

**Package vs Testing**:
- **Package**: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/cxi-pipeline-ray`
  - Installable, version-controlled, production-ready
  - Example configs in `examples/configs/`
- **Testing**: `$TEST_DIR`
  - Test-specific configs: `cxi_writer_test.yaml`
  - Validation scripts
  - Temporary output directories

**Next Steps**:
1. Create package structure in `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/cxi-pipeline-ray`
2. Implement `pyproject.toml` with dependencies
3. Implement core components (peak finding, file writer, coordinator)
4. Implement CLI with config loading
5. Create example config in `examples/configs/`
6. Create test config in `$TEST_DIR`
7. Test with small dataset
8. Profile and tune performance
