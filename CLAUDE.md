## Data processing pipeline

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

Ray APIs can be confusing.  Please look up the documentation before making your
decisions.
