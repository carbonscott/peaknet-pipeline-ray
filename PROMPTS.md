I feel mixed precision was not indeed used, could you double check?

---

Read STATUS.md.

I see `warmup_samples: 0` in $TEST_DIR/peaknet-socket.yaml.  I forgot what does
it mean in socket mode (not random).  If I set it to 0, what will happen?  If I
set it to 500, what will happen?  Also, I hope warm up examples are generated,
instead of consuming actual data.  Let's say these 500 examples can be generated
for warm-up, and the warm-up can even happen before the first actual data are
streamed in.  Is our current implementation doing this?

---

Hmm... it may not, you can basically generate (B, C, H, W) but reuse this batch
500 times (user can specify this number, maybe we can still call it
warmup_samples) so it doesn't have to spend time on generating warm up samples
all the time.

---

I have a suspicion that the socket based processing is facing a data starving
problem. I wanted to use the built-in data producer in this repo to Go through
the torch, compile warm-up as well as in front stage, and I want to understand
whether they could still process properly by profiling the whole process.  Un
unfortunately I don't seem to have much notes about running the building data
producer. It's there are information about using it in the very early stage of
the development so you can probably explore the timeline using get directly and
find a clue on how to use it.  I can share with you. What's the problem? I'm
seeing right now. It'll be attached below.

<text>
(StreamingDataProducer pid=3764258) 2025-09-25 00:18:23,117 - INFO - Producer 1: Generated 80/160 batches (1280 samples) at 28.7 samples/s [repeated 8x across cluster]
(PeakNetPipelineActorWithProfiling pid=3762015) [DEBUG] Batch 58: output tensor device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=3762015) [DEBUG] GPU memory after inference: 1.44 GB [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=3762015) [DEBUG] Batch 58: PeakNet forward pass completed on GPU [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=3762015) [DEBUG] Batch 58: input tensor device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=3762015) [DEBUG] Batch 58: model device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=3762015) [DEBUG] GPU memory before inference: 1.44 GB [repeated 9x across cluster]
(StreamingDataProducer pid=3764258) 2025-09-25 00:18:28,777 - INFO - Producer 1: Generated 96/160 batches (1536 samples) at 30.5 samples/s [repeated 8x across cluster]
(PeakNetPipelineActorWithProfiling pid=3762010) [DEBUG] Batch 62: output tensor device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=3762010) [DEBUG] GPU memory after inference: 1.44 GB [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=3762010) [DEBUG] Batch 62: PeakNet forward pass completed on GPU [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=3762010) [DEBUG] Batch 62: input tensor device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=3762010) [DEBUG] Batch 62: model device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=3762010) [DEBUG] GPU memory before inference: 1.44 GB [repeated 9x across cluster]

💥 Streaming pipeline failed: Task was killed due to the node running low on memory.
Memory on the node (IP: 172.24.49.143, ID: 8515c7e1e7b932197658c55aa4b9ec1332499950c4c95bd3804e7e08) where the task (actor ID: 4f48435abed4c32e8891fa7f0d000000, name=StreamingDataProducer.__init__, pid=3764266, memory used=23.49GB
) was running was 718.15GB / 755.24GB (0.950882), which exceeds the memory usage threshold of 0.95. Ray killed this worker (ID: 58a9081c65758054d39fc0ab5aa0242c932be2cc0b64673ef0fb1ce5) because it was the most recently scheduled t
ask; to see more information about memory usage on this node, use `ray logs raylet.out -ip 172.24.49.143`. To see the logs of the worker, use `ray logs worker-58a9081c65758054d39fc0ab5aa0242c932be2cc0b64673ef0fb1ce5*out -ip 172.24
.49.143. Top 10 memory users:
PID     MEM(GB) COMMAND
3691053 40.50   ray::RayQueue
3691055 40.50   ray::RayQueue
3346494 40.50   ray::RayQueue
3346507 28.31   ray::RayQueue
3346536 27.42   ray::RayQueue
3346532 27.41   ray::RayQueue
3346533 27.40   ray::RayQueue
3346534 27.16   ray::RayQueue
3346526 27.15   ray::RayQueue
3764262 23.94   ray::StreamingDataProducer.stream_batches_to_queue
Refer to the documentation on how to address the out of memory issue: https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html. Consider provisioning more memory on this node or reducing task parallelism by reques
ting more CPUs per task. Set max_restarts and max_task_retries to enable retry when the task crashes due to OOM. To adjust the kill threshold, set the environment variable `RAY_memory_usage_threshold` when starting Ray. To disable
 worker killing, set the environment variable `RAY_memory_monitor_refresh_ms` to zero.
Traceback (most recent call last):
  File "/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray/peaknet_pipeline_ray/pipeline.py", line 923, in run_streaming_pipeline
    performance = self._run_streaming_workflow(actors, enable_output_queue)
  File "/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray/peaknet_pipeline_ray/pipeline.py", line 1125, in _run_streaming_workflow
    remaining_results = ray.get(remaining_tasks)
  File "/sdf/scratch/users/c/cwang31/miniconda2/pytorch-2.6/lib/python3.13/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/sdf/scratch/users/c/cwang31/miniconda2/pytorch-2.6/lib/python3.13/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
  File "/sdf/scratch/users/c/cwang31/miniconda2/pytorch-2.6/lib/python3.13/site-packages/ray/_private/worker.py", line 2882, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
                                  ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/sdf/scratch/users/c/cwang31/miniconda2/pytorch-2.6/lib/python3.13/site-packages/ray/_private/worker.py", line 970, in get_objects
    raise value
ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory.
Memory on the node (IP: 172.24.49.143, ID: 8515c7e1e7b932197658c55aa4b9ec1332499950c4c95bd3804e7e08) where the task (actor ID: 4f48435abed4c32e8891fa7f0d000000, name=StreamingDataProducer.__init__, pid=3764266, memory used=23.49GB
) was running was 718.15GB / 755.24GB (0.950882), which exceeds the memory usage threshold of 0.95. Ray killed this worker (ID: 58a9081c65758054d39fc0ab5aa0242c932be2cc0b64673ef0fb1ce5) because it was the most recently scheduled t
ask; to see more information about memory usage on this node, use `ray logs raylet.out -ip 172.24.49.143`. To see the logs of the worker, use `ray logs worker-58a9081c65758054d39fc0ab5aa0242c932be2cc0b64673ef0fb1ce5*out -ip 172.24
.49.143. Top 10 memory users:
PID     MEM(GB) COMMAND
3691053 40.50   ray::RayQueue
3691055 40.50   ray::RayQueue
3346494 40.50   ray::RayQueue
3346507 28.31   ray::RayQueue
3346536 27.42   ray::RayQueue
3346532 27.41   ray::RayQueue
3346533 27.40   ray::RayQueue
3346534 27.16   ray::RayQueue
3346526 27.15   ray::RayQueue
3764262 23.94   ray::StreamingDataProducer.stream_batches_to_queue
Refer to the documentation on how to address the out of memory issue: https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html. Consider provisioning more memory on this node or reducing task parallelism by reques
ting more CPUs per task. Set max_restarts and max_task_retries to enable retry when the task crashes due to OOM. To adjust the kill threshold, set the environment variable `RAY_memory_usage_threshold` when starting Ray. To disable
 worker killing, set the environment variable `RAY_memory_monitor_refresh_ms` to zero.
</text>

by running `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 peaknet-pipeline --config peaknet-random.yaml --max-actors 2 --verbose --compile-mode reduce-overhead`

---

I have a very strong evidence suggesting that when torch Compile is in use the
Anset provider failed to capture Any GPU activities in contrast, I can clearly
see GPu activities when I was not using torch compile.  Something doesn't add
up.  I have been running the profiling very long, and still no GPU activities
are captured.  I am guessing the code implementation is right, it's probably
some stupid thing like some configs are not corrected passed into the code.
Please investigate!  The config I used to run these profiling is in $TEST_DIR
think

---

Why am I seeing PeakNetPipelineActor is initialized after the warm-up?
Shouldn't the motto be warmed up and then just passing to the Actor?

<text>
(PeakNetPipelineActorWithProfiling pid=263049) [DEBUG] Batch 97: model device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=263049) [DEBUG] GPU memory before inference: 1.44 GB [repeated 10x across cluster]
(PeakNetPipelineActorWithProfiling pid=263050) 2025-09-25 12:11:40,155 - INFO - Warmup completed: 100 iterations processed
(PeakNetPipelineActorWithProfiling pid=263050) 2025-09-25 12:11:40,155 - INFO - ✅ PeakNetPipelineActor initialized successfully on GPU 0
(PeakNetPipelineActorWithProfiling pid=263050) 2025-09-25 12:11:40,155 - INFO - Model: peaknet_config=True, compile_mode=reduce-overhead, warmup_samples=100
(PeakNetPipelineActorWithProfiling pid=263050) 2025-09-25 12:11:40,156 - INFO - Shapes: input_shape=(1, 1920, 1920), output_shape=(2, 1920, 1920)
(PeakNetPipelineActorWithProfiling pid=263050) 2025-09-25 12:11:40,197 - INFO - Actor 0: Starting streaming from queue
(PeakNetPipelineActorWithProfiling pid=263050) /sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray/peaknet_pipeline_ray/config/data_structures.py:103: UserWarning: The given NumPy array is not writable, and PyTorch
 does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of war
ning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:203.)
(PeakNetPipelineActorWithProfiling pid=263050)   tensor = torch.from_numpy(numpy_array)      # Zero-copy view
(StreamingDataProducer pid=3981824) 2025-09-25 12:11:35,886 - INFO - Producer 1: Generated 320/640 batches (5120 samples) at 50.0 samples/s
(PeakNetPipelineActorWithProfiling pid=263050) [DEBUG] Batch 4: output tensor device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=263050) [DEBUG] GPU memory after inference: 1.44 GB [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=263050) [DEBUG] Batch 4: PeakNet forward pass completed on GPU [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=263050) [DEBUG] Batch 4: input tensor device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=263050) [DEBUG] Batch 4: model device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=263050) [DEBUG] GPU memory before inference: 1.44 GB [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=263049) [DEBUG] Batch 6: output tensor device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=263049) [DEBUG] GPU memory after inference: 1.44 GB [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=263049) [DEBUG] Batch 6: PeakNet forward pass completed on GPU [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=263049) [DEBUG] Batch 6: input tensor device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=263049) [DEBUG] Batch 6: model device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActorWithProfiling pid=263049) [DEBUG] GPU memory before inference: 1.44 GB [repeated 9x across cluster]
(StreamingDataProducer pid=3981823) 2025-09-25 12:11:52,598 - INFO - Producer 0: Generated 384/640 batches (6144 samples) at 51.6 samples/s
(PeakNetPipelineActorWithProfiling pid=263049) 2025-09-25 12:11:42,860 - INFO - Warmup completed: 100 iterations processed
(PeakNetPipelineActorWithProfiling pid=263049) 2025-09-25 12:11:42,861 - INFO - ✅ PeakNetPipelineActor initialized successfully on GPU 0
(PeakNetPipelineActorWithProfiling pid=263049) 2025-09-25 12:11:42,861 - INFO - Model: peaknet_config=True, compile_mode=reduce-overhead, warmup_samples=100
(PeakNetPipelineActorWithProfiling pid=263049) 2025-09-25 12:11:42,861 - INFO - Shapes: input_shape=(1, 1920, 1920), output_shape=(2, 1920, 1920)
(PeakNetPipelineActorWithProfiling pid=263049) 2025-09-25 12:11:42,912 - INFO - Actor 0: Starting streaming from queue
(PeakNetPipelineActorWithProfiling pid=263049) /sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray/peaknet_pipeline_ray/config/data_structures.py:103: UserWarning: The given NumPy array is not writable, and PyTorch
 does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of war
ning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:203.)
</text>

---

I vaguely remember the cuda capturing was working at commit
ebb780e589b71932ffa9a8b3703f58c55a2d047b, could we check out that version?  I
would like to run it again and test it out.

---

I like that you use the following command for testing. It's not using a lot of
examples while being able to capture the actual GPU activities I like that.
`CUDA_VISIBLE_DEVICES=0,1 peaknet-pipeline --config /sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml/peaknet-random.yaml --max-actors 1 --total-samples 160 --verbose --compile-mode reduce-overhead`

What if I tell you unfortunately the so-called new fix isn't working but I can
tell you also what is working was the commit version I mentioned before.  This
makes me wonder something goes horribly wrong between these two commitments.
Practically speaking, we could completely revert back to that commit and move on
from there, but wouldn't it being intellectually interesting for you to
understand what truly the problem is.  you have all the tools you can check the
Ray docs you can search the Internet you have all the source code and you know
where the working directory is you know other tools.  Thank you.  think hard

---

Actually, which version is more similar to the peaknet-ray version?

---

Let's be pragmatic!  Let's revrt back to the commit I mentioned before.  It's
also mentioned in LESSONS_01.md, though, these lessons are not necessarily
correct.  Future development will go from there.

---


Read STATUS.md, ignore the problem section in the end.

I don't think the `warmup_samples` is passed into the pipeline, it's not
working when I ran 
`CUDA_VISIBLE_DEVICES=0,1 peaknet-pipeline --config /sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml/peaknet-random.yaml --max-actors 1 --total-samples 160 --verbose --compile-mode reduce-overhead`

---

It's confusing to use warmup_samples to define the number of warmup runs,
because the total number of warmup iterations would depend on the batch size.  I
am guessing a better approach is to use warmup_iterations.  What do you think?

---

peaknet-random.yaml and peaknet-socket.yaml have subtle difference.  I want to
enable true streaming based processing when `total_samples` is null.  I guess,
if I can directly specify it as int('inf'), which I know don't exists in python,
but there should be something similar to it.  I would just do that.  So I don't
need to change anything in the code and it will process until all streamed-in
data are consumed.  (I suppose the coordinate will be able to terminate the
pipeline).  Is my assessment fair?  think hard

---

could you explain a bit more on why the logic in 
<text>
  if runtime.total_samples is not None:
      # Calculate required batches to reach total_samples
      total_batches_needed = (runtime.total_samples + runtime.batch_size - 1) // runtime.batch_size
      batches_per_producer = max(1, total_batches_needed // runtime.num_producers)
      # ...
  else:
      batches_per_producer = runtime.batches_per_producer  # Falls back to config value
</text>
means the streaming mode is implemented?  If total_samples is None,
`batches_per_producer = runtime.batches_per_producer`, how does this achieve
the streaming mode?  think hard

---

Thanks for the assessment.  I feel the fix shouldn't be hard, right?  I feel if
the data source type is socket, then total_samples shouldn't play a role at all.
think hard

---

I'm getting <text>   Waiting for producers to finish...
(LightweightSocketProducer pid=926041) 2025-09-25 22:14:08,529 - INFO - LightweightSocketProducer 0 initialized: socket=tcp://sdfada008:12321
(LightweightSocketProducer pid=926041) 2025-09-25 22:14:08,533 - INFO - Producer 0: Starting lightweight streaming from tcp://sdfada008:12321
(LightweightSocketProducer pid=926041) 2025-09-25 22:14:08,536 - INFO - Producer 0: Listening on tcp://sdfada008:12321
(PeakNetPipelineActorWithProfiling pid=1622413) 2025-09-25 22:14:14,335 - INFO - === Initializing PeakNetPipelineActor ===
(PeakNetPipelineActorWithProfiling pid=1622413) 2025-09-25 22:14:14,339 - INFO - Using Ray-assigned GPU device 0 (physical GPU 0)
(PeakNetPipelineActorWithProfiling pid=1622413) 2025-09-25 22:14:14,339 - INFO - ✅ Actor GPU assignment complete - using CUDA device 0
(PeakNetPipelineActorWithProfiling pid=1622413) 2025-09-25 22:14:14,339 - INFO - CUDA_VISIBLE_DEVICES: 0
(PeakNetPipelineActorWithProfiling pid=1622413) Creating PeakNet model from native configuration
(PeakNetPipelineActorWithProfiling pid=1622413) Model image_size: 512
(PeakNetPipelineActorWithProfiling pid=1622413) Model num_channels: 1
(PeakNetPipelineActorWithProfiling pid=1622413) Model num_classes: 2
(raylet) It looks like you're creating a detached actor in an anonymous namespace. In order to access this actor in the future, you will need to explicitly connect to this namespace with ray.init(namespace="e56f228a-609c-42f3-a15b-a64eb35dfe6f", ...) [repeated 7x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)
(PeakNetPipelineActorWithProfiling pid=1622413) 2025-09-25 22:14:14,754 - INFO - ✅ CUDA context established on device 0
(PeakNetPipelineActorWithProfiling pid=1622413) 2025-09-25 22:14:14,755 - INFO - Actor GPU: NVIDIA L40S (45596 MB)
(PeakNetPipelineActorWithProfiling pid=1622413) 2025-09-25 22:14:14,756 - INFO - Actor CPU affinity: 0-11,24-59,72-95
(PeakNetPipelineActorWithProfiling pid=1622414) PeakNet model created: 33.2M parameters
(PeakNetPipelineActorWithProfiling pid=1622414) Backbone: ConvNextV2 [96, 192, 384, 768]
(PeakNetPipelineActorWithProfiling pid=1622414) BiFPN: 2 blocks, 256 features
(PeakNetPipelineActorWithProfiling pid=1622414) Input size: 512×512
(PeakNetPipelineActorWithProfiling pid=1622414) ✓ Got num_classes from model config: 2
(PeakNetPipelineActorWithProfiling pid=1622414) ⚠ PeakNet model on CPU: cpu
(PeakNetPipelineActorWithProfiling pid=1622413) [DEBUG] Pipeline buffer validation:
(PeakNetPipelineActorWithProfiling pid=1622413) [DEBUG]   Buffer A: input cuda:0, output cuda:0
(PeakNetPipelineActorWithProfiling pid=1622413) [DEBUG]   Buffer B: input cuda:0, output cuda:0
(PeakNetPipelineActorWithProfiling pid=1622413) ✓ All pipeline buffers verified on GPU 0
(PeakNetPipelineActorWithProfiling pid=1622413) [DEBUG] Pipeline buffer memory: 0.094 GB (0.031 GB input + 0.062 GB output)
(PeakNetPipelineActorWithProfiling pid=1622413) 2025-09-25 22:14:15,150 - INFO - PeakNet mode: input_shape=(1, 512, 512), output_shape=(2, 512, 512)
(PeakNetPipelineActorWithProfiling pid=1622413) 2025-09-25 22:14:15,150 - INFO - Created autocast context: dtype=bfloat16
(PeakNetPipelineActorWithProfiling pid=1622413) 2025-09-25 22:14:15,171 - INFO - Running model warmup with 500 iterations...
(PeakNetPipelineActorWithProfiling pid=1622413) 2025-09-25 22:14:15,375 - INFO - Warmup: Processing 500 iterations of batch size 16
(PeakNetPipelineActorWithProfiling pid=1622413) 2025-09-25 22:14:15,376 -
WARNING - Warmup failed: The size of tensor a (512) must match the size of
tensor b (1691) at non-singleton dimension 2, continuing without warmup</text>
by running CUDA_VISIBLE_DEVICES=0,1 peaknet-pipeline --config
peaknet-socket-profile.yaml --max-actors 2 --verbose.  What was going on?  think
hard

---

I see warmup is performed even when torch compile is not in use.  Let's fix it now.

---

Fundamentally, where data source is random or socket, my double buffered inference pipeline
implemntation and execution is the same, right?  I saw profiling results like
`socket-profile.png` in socket mode.  I don't think there is explicit host device
synchronization.  Is it a sign of input data starving?  If so, do you think it's
the data producer (in lclstreamer)'s issue, or issues in Q1 to P stage? think hard

---

I saw a cudastreamsynchronize bewtween warmup and the actual inference on real
data.  Could you investigate on why?  I thought there is supposed to be a host
device synchronization, not cuda stream synchroniztion.  Maybe the cuda stream
synchronization is built into pytorch's compile?  I don't know.  think hard

---

Could you investigate one detail?  Where does the hdf5 unpacking happen?  Is it
from S to Q1, or is it from Q1 to P?  I suspect it's the latter.

---

There might be multiple solutions coded in the repo at once.  Which one was used
in peaknet-pipeline (defined in setup.py)?  Then, track down hdf5 unpacking.
think

---

Now, let's explore the option of moving the unpacking to "S to Q1".  I suspect
the latency of unpacking hdf5 is not negligible.  think hard

---

Actually, before implementing a solution, please investigate another details.
If the source_type is random (the producer might be streaming_producer), does
the producer produces a HDF5 and then got unpacked during "Q1 to P"?  think

---

Does that mean the "Q1 to P" process can process two scenarios - (1) fetch HDF5
from Ray's object store; (2) fetch PipelineInput object from Ray's object store?
could you confirm?  think

---

Beautiful!  I guess it's not super hard to move HDF5 unpacking to "S to Q1",
right?  You don't really need to modify the "Q1 to P" process since it is
ready to process three data sources.  think hard

---

I am a bit concerned that the streaming data source idea might not be working at
all.  When the model is being compiled, the GPU utilization is 100%.  But when
the compilation is done, and actual data processing `S->Q1->P` seems to have
almost 0% GPU utilization.  What might be going on?  Please investigate.  

The command I was running is 

```
CUDA_VISIBLE_DEVICES=0,1 peaknet-pipeline --config peaknet-socket-profile.yaml --max-actors 2 --verbose  --compile-mode reduce-overhead
```

think hard

---

Is it possible that S gets one (C, H, W) image at a time instead of (B, C, H, W)
as it does now.  I feel the 

---

I suspect we need many fetchers in `S to Q1` stage.  Could you investigate how
many fetches we have right now?  What would happen if I increase `num_producers`
to more than 1 like 10?  think hard

---

Please educate me on your new implementation.  why is raw_buffer_size needed?
what is its relation with queue controlled by queue_num_shards.  

---

The GPU only gets occasionally busy when I use socket as the source type (in
`$TEST_DIR/peaknet-socket-profile.yaml`), whereas it always gets busy when using
random as the source type (`$TEST_DIR/peaknet-random-profile.yaml`).  Is HDF5
itself the bottleneck (totally, my speculation).  Or are the pipelines 
actually different?  think hard

---

I have been moving where to unpack HDF5 in the overarching pipeline (not just
the ML double buffered inference pipeline, sorry for confusion) after commit
`1767603621f1d7f9c5726345b497ace4179c7c49`.  I don't get much progress.  And,
now I am suspecting it's actually because unpacking HDF5 itself is the
bottleneck.  Now, I'm a bit concerns these changes are pre-mature.

If I decide to do something on the producer side (lclstreamer), do you suggest I
completely use the version at commit `1767603621f1d7f9c5726345b497ace4179c7c49`.
What options do I have?
