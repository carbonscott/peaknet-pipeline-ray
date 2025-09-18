## Key directory

```
export TEST_DIR="/sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml"
export STREAMER_DIR="/sdf/data/lcls/ds/prj/prjcwang31/results/software/lclstreamer"
export PIPELINE_DEV_DIR="/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray"
```


## Pipeline Stage Progress

### **R to S**

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

### **S to Q1**

- An exact working version does NOT exist yet.
- A simulated S to Q1 exists in `$PIPELINE_DEV_DIR/peaknet_pipeline_ray/core/peaknet_ray_data_producer.py`, where the socket data source is replaced by a random tensor data source.
- When you launch command like
  ```
  peaknet-pipeline --config examples/configs/peaknet.yaml --max-actors 4 --total-samples 10240 --verbose
  ```
  it calls the simulated data source under the hood.

- **A priority** to find a good way to repurpose or adapt the pull script in
  `TEST_DIR/psana_pull_script_inspect.py` into our pipeline data source codes in
  `$PIPELINE_DEV_DIR` so that we truly enable socket data source, while keeping
  the random data source for testing purposes.

### **Q1 to P**

- It exists in `$PIPELINE_DEV_DIR`.
- When you launch command like
  ```
  peaknet-pipeline --config examples/configs/peaknet.yaml --max-actors 4 --total-samples 10240 --verbose
  ```
  the data ingestion process Q1 to P should work under the hood.

### **P**

- It exists in `$PIPELINE_DEV_DIR`
- It's the double buffered pipeline.

### **P to Q2**

- It exists in `$PIPELINE_DEV_DIR`
- It should be part of the double buffered pipeline's device to host (D2H) process.

### **Q2 to W**

- The data writer does NOT exist yet.
- **Not a priority** right now.
