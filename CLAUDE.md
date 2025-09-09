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

- The entire architecture that includes data source and post processing is
  described in my hand-written scatch notes in
  `PeakNet-Pipeline-Scratchpad.jpg`.  However, this package should focus on the
  `Q1->P->Q2` process if you can understand my handwriting.

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
  - Path: /sdf/home/c/cwang31/codes/peaknet
  - The peaknet library

### Where to develop the package

- `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray`

### Handling permission

Ask me now to add all relevant directories to working directories.

### Ray Documentation

Path: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/ray/doc/`

Ray APIs can be confusing.  Please look up the documentation before making your
decisions.
