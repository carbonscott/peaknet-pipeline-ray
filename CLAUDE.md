/sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml## Data processing pipeline

As shown in the JPG illustration (`externals/PeakNet-Pipeline-Scratchpad.jpg`),
the entire pipeline consists of 5 stages:
- R stage: Data producer that is controlled by a proprietary MPI based process
- Q1 stage: Queue that will be filled with data pushed from the R stage through
  the socket (S), waiting to be consumed by the next P stage
- P stage: the actual PeakNet inference process
- Q2 stage: Queue that will be filled with output from the inference, and wait
  to be consumed by the postprocessor in the next stage
- W stage: The post processing stage that will eventually proudces files on disk

## Project Overview

- **Goal**: Make a package to run machine leanring model inference for a
  specific model called peaknet at scale with streaming data source
- **Approach**: Pipelining
  - Stage 1: getting input data from Ray's object store (there will be a
    separate process streaming data into the Ray's object store, which is out of
    the scope of this repo)
  - Stage 2: compute with double buffering based on the input data in Ray's
    object store
  - Stage 3: output results to Ray's object store (there will be a separate post
    processing based on these results in Ray's object store, but it is out of
    the scope to discuss the post processing)

NOTE: Stage 1, 2, and 3 are concepts in the PeakNet pipeline only.  They are
only components within the process `Q1->P->Q2`.

### Key directory

```
export TEST_DIR="/sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml"
export STREAMER_DIR="/sdf/data/lcls/ds/prj/prjcwang31/results/software/lclstreamer"
export PIPELINE_DEV_DIR="/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray"
```

### Pipeline Stage Progress

#### **R to S**

- One version of this process exists in `$STREAMER_DIR`.
- Pulling from socket (I have launched it, and please ask me if you need help
  launching it, because it should be launched in a special way that I don't have
  time to explain right now):
  ```
  cd $TEST_DIR && python psana_pull_script_inspect.py
  ```
- Push to socket:
  ```
  cd $STREAMER_DIR && pixi run --environment psana1 mpirun -n 8 lclstreamer --config examples/lclstreamer-psana1-to-sdfada.yaml
  ```

#### **S to Q1**

- An exact working version exists now, but may not be perfect.  I still suspect
  there's bottleneck in serving the data.
- A simulated S to Q1 exists in `$PIPELINE_DEV_DIR/peaknet_pipeline_ray/core/peaknet_ray_data_producer.py`, where the socket data source is replaced by a random tensor data source.
- When you launch command like
  ```
  peaknet-pipeline --config examples/configs/peaknet.yaml --max-actors 4 --total-samples 10240 --verbose
  ```
  it calls the simulated data source under the hood.

#### **Q1 to P**

- It exists in `$PIPELINE_DEV_DIR`.
- When you launch command like
  ```
  peaknet-pipeline --config examples/configs/peaknet.yaml --max-actors 4 --total-samples 10240 --verbose
  ```
  the data ingestion process Q1 to P should work under the hood.

#### **P**

- It exists in `$PIPELINE_DEV_DIR`
- It's the double buffered pipeline.

#### **P to Q2**

- It exists in `$PIPELINE_DEV_DIR`
- It should be part of the double buffered pipeline's device to host (D2H) process.

#### **Q2 to W**

- The data writer does NOT exist yet.
- **Not a priority** right now.

### Past works

- **peaknet-ray**
  - Path: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/inference-pipeline-examples/peaknet-ray`
  - An example of model inference pipeline with double buffering.
  - Battery included - nsys profiling integration ready, data producer ready, model
    consumers ready, customization supported by hydra.
  - The only problem is that this is not a package.  I want to create a package
    so that I can just launch the double buffered pipeline with a single
    command (this is doable in setup.py or maybe in toml too, but you know what
    to do).
- **peaknet**
  - Path: `/sdf/home/c/cwang31/codes/peaknet`
  - The peaknet library
- **lclstream**
  - Path: `/sdf/data/lcls/ds/prj/prjcwang31/results/software/lclstreamer`
  - A library to function as the data producer in R stage.

### Where to develop the package

- `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray`

### Handling permission

Ask me now to add all relevant directories to working directories.

### Ray Documentation

Path: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/ray/doc/`

When programming in Ray, it's always good to research this directory before you
implement anything.

If you want to be careful about the patterns and anti-patterns of using Ray,
they are discussed in
`/sdf/data/lcls/ds/prj/prjcwang31/results/codes/ray/doc/source/ray-core/patterns`.
There's a summary of these patterns and anti-patterns in
`/sdf/data/lcls/ds/prj/prjcwang31/results/learn-ray/PATTERNS.md` if you find it
helpful.

Ray APIs can be confusing.  Please look up the documentation before making your
decisions.

### nsys profiling tips

Start with small sized data (like 1x512x512) as described in `$TEST_DIR/peaknet-random-profile.yaml`.

```
CUDA_VISIBLE_DEVICES=0,1 peaknet-pipeline --config $TEST_DIR/peaknet-random-profile.yaml --max-actors 2 --total-samples 20480 --verbose
CUDA_VISIBLE_DEVICES=0,1 peaknet-pipeline --config $TEST_DIR/peaknet-random-profile.yaml --max-actors 2 --total-samples 20480 --verbose --compile-mode reduce-overhead
```
