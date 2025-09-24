I'm trying to make design decisions on how to take an experimental direcotry
like `peaknet-ray` and make it a package.

**A few concerns**
- The post processing might require not only the image data that the pipeline
  will actually need to process, but also metadata like photon energy.  It feels
  like photon energy data should just be a pass-through since the peaknet model
  itself doesn't need it.  I'm thinking maybe it's a good idea to define a data
  schema in python data classes or pydantic that should enable such pass
  through?
- In high performance setting, is it really a good idea to have two queues as I
  described in the architecture shown in the JPG image.  This will affect how I
  think about the stage 1 and stage3 of the pipeline.
- At the core of the peaknet model itself, it only cares about the actual image
  tensor input.  However, the pipeline has more configurations required.  It is
  maybe a good idea to review the hydra configuration already implemented in the
  peaknet-ray project, so that we can understand how to design the pipeline as a
  standalone package.

---

We are talking about building a software package.  A few thoughts below:

- Hydra config might be great for managing many experiments, but isn't really a
  good choice for managing the user configuration for a package.  What do you
  think?  Argparse might still be a better option?  Along with data schema
  defined in pydantic?  I think maybe dataclasses is good enough since I control
  what goes into the pipeline (the data source).
  - Thereby, no need to worry about what hydra configuration files you need to
    keep.

Please plan it again and make according changes to the PACKAGE_DESIGN.md

---

No need to hard code presets, these can be a config in the examples directory
(just create it if it doesn't exist in the repo yet).

The DataConfig is good for the model in the pipeline itself.  However, the
downstream processing (which isn't present in this pipeline and neither in peaknet-ray repo
yet) needs additional information (like photon energy in float) to work, so
some information require pass-through the pipeline, meaning they need to be
when the pipeline puts the results to Ray's object store for the downstream
process to take care of it.  We need a new design.

---

I have created the first package version of the peaknet inference pipeline in
ray.  It's obviously inspired by the projects in `peaknet-ray`, could you help
me test if it can repeat things like `python run_peaknet_streaming_pipeline.py experiment=peaknet_test_run
  streaming.runtime.max_actors=9 streaming.runtime.total_samples=20480` as in
  the `peaknet-ray` directory?  I should have already launch the Ray head node.

---

lol, okay, the plan is written in PACKAGE_DESIGN.md, so please review it and
then implement the rest

---

I ran `python -m peaknet_pipeline_ray.cli.main --config
examples/configs/peaknet_with_model.yaml --max-actors 2 --total-samples 10240
--verbose`, but there's still no GPU activities recorded.  Are you sure you are
using the right configs?  Or are you really following the example in
peaknet-ray?  I suspect it's not really running inference on GPUs for whatever
reasons.

---

I don't like the solution you just proposed.  The package itself should NOT rely
on hydra config.  Please rethink the configuration from scratch for the model
part of the pipeline!

---

I'm running the tests right now.  However, let's plan the first git commit.  I
think CLAUDE.md, is okay to be in the project.  README is a bit over-selling,
like the wording "production quality".  We are just starting!  Be humble.  I
like minimal no BS README, right now, the most important message to deliver in
the README is to tell users how to run it with nsys profiling enabled.

Also, make sure you only consider the yaml config that has been actually used in
the test.

Plan the commit.

---

I also have a problem with
<text>
model:
  yaml_path: null
  weights_path: null  # No weights = randomly initialized model (but REAL GPU computation!)

  # Simplified PeakNet configuration - no complex nested structures
  peaknet_config:
    model:
      # Basic model parameters
      image_size: 1920
      num_channels: 1
      num_classes: 2

      # Backbone configuration
      backbone_hidden_sizes: [96, 192, 384, 768]
      backbone_depths: [3, 3, 9, 3]

      # BiFPN configuration
      bifpn_num_blocks: 2
      bifpn_num_features: 256

      # Segmentation head configuration
      seg_out_channels: 256

      # Other settings
      from_scratch: false
</text>
in `examples/configs/peaknet.yaml`.

Since peaknet already has its config directly available in the config, what's
the value of having yaml_path?  Isn't it supposed to point to a peaknet config?
If so, could you remove it since it doesn't provide actual values.

Also, I have a problem with the name of `process_batch_from_refs`, though, this
is what it was in the peaknet-ray repo.  Could you come up with a better name
that makes more sense while not having a conflict with other parts of the
package?

Also, I think `peaknet_pipeline_ray/core/peaknet_ray_data_producer.py` is
supposed to be... how can I say?  For testing purposes, because, in reality, the
data source is external to this package.  The purpose of the data producer is to
enable testing even when the actual data source isn't available.

---

formatting-wise, I don't like empty lines with only whitespaces, could you
remove them?  Just make it like `^$` in regex sense.

---

could you explain to me the role of each python file under
peaknet_pipeline_ray/core?  For example, I'm very confused as to why we have
both `peaknet_pipeline.py` and `peaknet_ray_pipeline_actor.py`.  There are other
things like this example too.  Please investigate.


---

you probably know that our goal of the package peaknet-pipeline-ray is be a
central part of a data processing pipeline.  As shown in the JPG illustration
(`externals/PeakNet-Pipeline-Scratchpad.jpg`), the entire pipeline consists of 5
stages:
- R stage: Data producer that is controlled by a proprietary MPI based process
- Q1 stage: Queue that will be filled with data pushed from the R stage through
  the socket (S), waiting to be consumed by the next P stage
- P stage: the actual PeakNet inference process
- Q2 stage: Queue that will be filled with output from the inference, and wait
  to be consumed by the postprocessor in the next stage
- W stage: The post processing stage that will eventually proudces files on
  disk


Now, I'm debating how to structure the entire overarching pipeline (it will need
to live in a different repo) and what remains to be implemented in the current
repo.

The R stage will be handled by an existing software, which will push data to a
socket (S).

I have a dummy pull script that can function partially for the Q1 stage without
the Queue part yet.  This script is in
`/sdf/data/lcls/ds/prj/prjcwang31/results/software/lclstreamer/examples/psana_pull_script.py`.
When I say "partially", I mean it will launch a pull server and data will indeed
be pulled when the data producer is running, but it doesn't put the pullled data
into Ray's object store.  A key decision needs to be made here as to if we
should define a scope within the Ray object store as to whether we simply need a
name for Q1 or we need a dedicated ray namespace.  As you may have thought,
later, Q2 is a different Queue, so it's important to differentiate these two
queues.

Another question is that should I implement `Q1->P` and `P->Q2` inside
peaknet-pipeline-ray, or it should live in the overarching pipeline repo (which
doesn't exist yet, but I will create it if we decide that is the best move).  My
thoguht is that `P->Q2` might have to be done within peaknet-pipeline-ray,
because I don't intend to make this package to be general purpose.  It should
function as a key part in this overarching data processing pipeline, not
standalone.  Okay, I made the package standalone, but this is for software
maintainence.  Now, with the similar logic, I feel `Q1->P` should be part of the
package (peaknet-pipeline-ray) too.  The overarching pipeline repo can just be
an orchestrating package that put all functions together.  What's your thoughts?

---

I have problems with 
<text>
  1. Q1 Component: Extend/replace RayDataProducerManager to:
    - Connect to the socket (using lclstreamer pattern)
    - Parse HDF5 data from socket messages
    - Push parsed data to Ray's object store with appropriate namespacing
</text>

**One minor question**

What's the benefit of haivng a name like `q1_input_{timestamp}_{batch_id}`, is
it complicating things?  Think about the function of the queue, every time a
`ray.get` from whichever worker, it will get a unique batch, correct?  So I
don't understand what value a fine granularity like `_{timestamp}_{batch_id}`
will bring to the table.  I think the main concern I had before was to be able
to differentiate Q1 (input queue) and Q2 (outupt queue).  That's why I was
asking if we should do it using queue name or ray namespace.

**One major question**

I think the current script
`/sdf/data/lcls/ds/prj/prjcwang31/results/software/lclstreamer/examples/psana_pull_script.py`
has already handled `S->Q1` partially (the pull part), and some work needs to be
done in the orchestration repo to actually enable the full `S->Q1` process.
Let's discuss it later.

Now, are you saying `Q1->P` has already been implemented, at least partially?  I
think it's important to know what data will live in Q1 and Q2 so that you can
identify gaps between the goal and our current state.
- Q1
  - PeakNetPipielineInput
    - Tensor data, with the shape of (C, H, W), C is typically 1.  Though, I
      don't like hard code, especially, H and W can be vairables.  Once the
      model is initialized, (C, H, W) will be determined.
    - Metadata, float
- Q2
  - PeakNetPipelineOutput
    - Tensor data, with the shape of (C, H, W), C is typically 2, but again, I
      don't want to hard code it.  These value will be determined after the
      PeakNet model is initialized.
    - Metadata, float, basically identical to the metadat in the
      PeakNetPipelineInput, so it's fundamentally a pass-through for downstream
      processing purposes.

Could you adjust your pipeline data producer to facilitate this development?
Because I don't have `S->Q1` available right now, we have to fake these data.

Let's make sure we can complete the process of `Q1->P`, `P` (which I believe is
mostly done in this repo), and `P->Q2`.

---

Your assessments trigger a few more questions.

Would be using a data container like PipelineInput or PipelineOutput in a
streaming fashion be really efficient?  Maybe it should be things like 

- PipelineInput
  - ray.ObjectRef to underlying tensor data
  - ray.ObjectRef to underlying float

similarly goes for `PipelineOutput`.

hmm... maybe your propose is better.  Let's discuss.

---

Is the following design really the right one?

<code>
  @dataclass
  class PipelineInput:
      image_data_ref: ray.ObjectRef    # -> torch.Tensor (gets zero-copy via numpy conversion)
      metadata: Dict[str, float]       # Small data, can be embedded directly
      batch_id: str

      @classmethod
      def create_and_store(cls, image_tensor: torch.Tensor, metadata: Dict[str, float], batch_id: str):
          # Convert to numpy for zero-copy optimization
          if image_tensor.is_cuda:
              image_tensor = image_tensor.cpu()

          # Store tensor separately for zero-copy access
          tensor_ref = ray.put(image_tensor.numpy())  # Optimal path for plasma store

          pipeline_input = cls(
              image_data_ref=tensor_ref,
              metadata=metadata,
              batch_id=batch_id
          )

          return ray.put(pipeline_input)

      def get_image_tensor(self) -> torch.Tensor:
          """Get tensor with zero-copy from object store."""
          numpy_array = ray.get(self.image_data_ref)  # Zero-copy
          return torch.from_numpy(numpy_array)        # Zero-copy view
</code>

We can safely assume that `S->Q1` will only operate on tensors or numpy arrays
on cpus.  Maybe we should assume the use of numpy arrays throughout the
overarching pipeline except within stage P since it's on GPU.

Since nested ray reference can be complicated, it's a good idea to have methods
to handle saving data to the Ray object store and getting data out of the Ray
object store.  Maybe I just don't know how the code works.  I thought
image_data_ref is needed both during the saving process.  And what does 
<code>
          pipeline_input = cls(
              image_data_ref=tensor_ref,
              metadata=metadata,
              batch_id=batch_id
          )

          return ray.put(pipeline_input)
</code>
do?  It returns to whom?

---

Is batch_id really important?  Please note that the class of PipelineInput
doesn't have to do too much.  The Queue should be able to handle who is the next
data point (due to the FIFO mechanism).  Would that influence the design?

---

I think the idea is sound.  The main problem is that `S->Q1` and `Q1->P` are two
processes that run independently, meaning you can interrupt `S->Q1`, but then
you can resume it.  These two processes just need to agree on which Queue to
interact with - `S->Q1` only cares about putting data into the queue and `Q1->P`
only cares about getting data out of the queue.  This might influence the
design.

---

We are getting there.  Instead of creating Q1Queue, considering we are going to
have Q2Queue, I believe some abstraction here would be truly useful.  Unless, it
messes up with the types.  What do you think?

---

I'm back, and let's continue where we left off.  I left a note in DISCUSSION.md.

---

I thought you won't actually need Q1Queue and Q2Queue, these will be just two
instances of RayQueue, is the inheritence pattern really helpful?  how much
helpful?

---

Let's use the simpler approach for the queue.  Also, you created your to do
really quickly, shall we discuss what to do before you make a todo?

---

- Let's implement the queue system and test it first.  I can launch the head node
so you don't have to launch the head node over and over again.  Okay, it's
launched.

- Yes, and I think peaknet-pipeline-ray should have RayQueue as a utility.  Other
  packages that might need it can import it from peaknet-pipeline-ray.

- You might have a better judgement call, but I think most of the codes
  transfer, and you just have to deal with the new PipelineInput/PipelineOutput
  containers.
- For this package, the MVP example would be to run the process `S->Q1`,
  and `Q1->P->Q2`.  I know we don't have the actual socket yet, and you don't
  need it for testing MVP.  You just need to have a data source that can put
  data into `Q1`, so `(fake data source)->Q1`.  Again, head node has already
  been launched.  You can test the entire pipeline within this node (I guess,
  the same infrastructure will scale up to multiple nodes).

---

Hold on, could you explain why the coordination is file based?  I thought with a
queue name, things will work just fine without files.  Please explain.

---

Hmm... look!  FIFO should be native to the data structure.  And imagine if I'm a
Ray actor or worker, I really don't care which batch I'm going to get from the
input Queue, because all I should care is when I call `ray.get`, a new piece of
data will be delivered to me, so that I can process them.  Do you understand?
Also, one implementation I had ages ago for this is 
<code>
import ray
from collections import deque

@ray.remote
class Queue:
    def __init__(self, maxsize=100):
        self.items = deque(maxlen=maxsize)

    def put(self, item):
        try:
            if len(self.items) < self.items.maxlen:
                self.items.append(item)
                return True
            return False
        except Exception as e:
            print(f"Error in put: {e}")
            return False

    def get(self):
        try:
            return self.items.popleft() if self.items else None
        except Exception as e:
            print(f"Error in get: {e}")
            return None

    def size(self):
        try:
            return len(self.items)
        except Exception as e:
            print(f"Error in size: {e}")
            return 0

def create_queue(queue_name="shared_queue", ray_namespace="default", maxsize=100):
    try:
        return Queue.options(name=queue_name, namespace=ray_namespace, lifetime="detached").remote(maxsize=maxsize)
    except Exception as e:
        print(f"Error creating queue '{queue_name}' in namespace '{ray_namespace}': {e}")
        return None
</code>

I'm not saying you should use this.  I only want to you to learn from this and
see if you can come up with your own that works for our case.

---

So actor manages the queue, you are saying!  Okay, let's work on this queue
implementation and test it.

Does it scale up to mulitple nodes?  If an actor only lives in one node?  Or
actor actually lives in the entire Ray cluster?

---

Instead of sharding it later, I think we can take a crack now at designing a
scalable queue.  What configurations should it have?  Like how many shards, so
the throughput is not hinged on one python actor.

---

I'm not sure if auto-scaling is what we need.  Look, we will always manually set
up the shard number.  You can have a default to maybe just one.  Also, how is
it supposed to run in multi-nodes (e.g. two nodes) but all managed by Ray?  I'm
just thinking about orchestration.  So we decided earlier that the queue will
be managed by the package `peaknet-pipeline-ray` as a utility.  Even it runs in
two different processes: `S->Q1` and `Q1->P`, the configuration for the queue
should be consistent.

As for the strategy, I think round robin is good enough.  Let's not make it
complicated for now.

---

It's also possible that process `Q1->P` processes can be running on two nodes
too, like B1 and B2 nodes.  I'm not saying this is the most efficient way to
handle this, but you know, this is to give you an example on the flexibility the
queue needs to have.

---

Let's think about this carefully.  The Queue should only worry about get and put
methods, and have a queue name that communicates the message on which queue to
get or put.  The orchestration of shards should be handled at a higher level,
and we don't have to worry about it at the Queue level.

---

I don't see why we need 
<code>
def create_queue(queue_name: str, num_shards: int = 1, maxsize_per_shard: int = 1000) -> ShardedQueueManager:
    """Convenience function to create a sharded queue manager.
····
    Args:
        queue_name: Name for the queue.
        num_shards: Number of shards (default=1 for simple queue).
        maxsize_per_shard: Maximum size per shard.
········
    Returns:
        ShardedQueueManager instance.
    """
    return ShardedQueueManager(queue_name, num_shards, maxsize_per_shard)
</code>

The user code just needs to run ShardedQueueManager(queue_name, num_shards,
maxsize_per_shard).  You might have to update all downstream codes that uses it.
I guess most of them are test files.

---

Regarding the serialization issue, could you look up Ray Docs again to see if
you have addressed it correctly?

---

Please think carefully what files should be checked into git.  Like
peaknet_pipeline_ray/utils/, peaknet_pipeline_ray/core/README.md,
peaknet_pipeline_ray/config/data_structures.py, and CLAUDE.md.  We have another
deep question to think about.

Now, it's a good time to remind ourselves that peaknet-pipeline-ray functions as
a standalone package.  RayDataProducerManager was used to test the pipeline.  It
just happens that in order to test the pipeline itself, I need a data source,
and that's why I had RayDataProducerManager.  Now, we actaully will use the
peaknet pipeline in the overarching pipeline.  Thereby, I believe, a data
producer manager shouldn't be part of `./peaknet_pipeline_ray/pipeline.py`.
And, it should be able to use ShardedQueueManager just as easily.  Do you know
what I mean?  I don't have a good solution right now, so I appreciate your
inputs on this.

---

And make a tag that up to this commit, it's about testing only the pipeline with
generated data.

I think instead of having a test harness.  Let's fully support this new queue
based pipeline.  And the tests should be built to support such pipeline.  This
way, even RayDataProducerManager can still be re-purposed to generate data and
then put into Q1.  Does it make sense?  Are most of the part aleady done?


---


The original implementation in the tagged version `v0.1-no-queue-version`
has already demonstrated the double buffering approach.  The only thing that it
lacks is a nice data packaging like PipelineInput/PipelineOupout that the
current implementation has.  However, you and I both might realize that the
such data packaging is not very nice for the ML inference pipieline because it
likes to process a batch of data at once, and the such packaging makes it harder
(you have to modify the actual pipeline codes).  And if we think about it again,
the data packaging we have in the current version only brings the benefit of a
metadata pass through, which can be done in other simpler means.  Please
evaluate if my assessment is fair.

Now, having said that, completely going back to `v0.1-no-queue-version` might
not be the best optional because you lose all the Queue implementation that you
might not want to re-implement.  The benefit is you will be able to see how the
pipeline used to look like.  Though, that implementation has the producer
built into the implementaiton, which I don't like.

Maybe, a better approach is starting off a branch now (maybe call it
simple_queue_pipeline), and then focusing on addressing the data packaging
issue while still allowing passing through.  Because right now, even the batch
size in ``examples/configs/peaknet.yaml isn't really working.

Now, I want to hear your thoughts.

---

Now, we are talking.  The second issue is the data producer should simulte the
behavior of a streaming data producer like the following one except now we are
using Ray not sockets.

<code>
import torch
import time
import argparse
import io
import numpy as np
from pynng import Push0
import threading

def tensor_to_bytes(tensor):
    """Convert a PyTorch tensor to bytes for transmission.

    This version skips numpy.save() and directly gets raw bytes.
    """
    # Ensure tensor is on CPU and in the right format
    tensor_cpu = tensor.cpu()
    tensor_np = tensor_cpu.numpy()

    # Get raw bytes (more efficient than numpy.save)
    tensor_bytes = tensor_np.tobytes()

    return tensor_bytes

def dummy_data_pusher_thread(
    address,
    C=1,
    H=1920,
    W=1920,
    total_size=10000,
    push_interval=0.001,
    continuous=False,
    stream_id=0
):
    """
    Push random tensors to the specified address.

    Args:
        address (str): Address to push data to (URL format).
        C (int): Number of channels.
        H (int): Height of the tensor.
        W (int): Width of the tensor.
        total_size (int): Total number of samples to push.
        push_interval (float): Interval between pushes in seconds.
        continuous (bool): Whether to continuously push data.
        stream_id (int): Identifier for this stream.
    """
    with Push0(listen=address) as sock:
        print(f"Stream {stream_id}: Listening at {address}, pushing data with shape ({C}, {H}, {W})")

        # Counter for continuous mode
        counter = 0

        while True:
            for i in range(total_size):
                # In continuous mode, use a counter to ensure unique indices
                if continuous:
                    global_idx = counter
                    counter += 1
                else:
                    global_idx = i

                # Generate a random tensor
                tensor = torch.randn(C, H, W)

                # Convert tensor to bytes (directly, without numpy.save)
                data = tensor_to_bytes(tensor)

                # Prepare metadata (include shape information for tensor reconstruction)
                metadata = {
                    'index': global_idx,
                    'shape': (C, H, W),
                    'total_size': total_size,
                    'stream_id': stream_id
                }
                metadata_bytes = str(metadata).encode('utf-8')

                # Push metadata header and data payload
                sock.send(metadata_bytes + b'\n' + data)

                # Wait for the specified interval
                if push_inverval > 0: time.sleep(push_interval)

                if (i + 1) % 1000 == 0:
                    print(f"Stream {stream_id}: Pushed {i + 1} samples in current cycle")

            # If not continuous, break after one cycle
            if not continuous:
                print(f"Stream {stream_id}: Done pushing {total_size} samples")
                break

            print(f"Stream {stream_id}: Completed one cycle of {total_size} samples, continuing...")

def dummy_data_pusher(
    address,
    C=1,
    H=1920,
    W=1920,
    total_size=10000,
    push_interval=0.001,
    continuous=False,
    num_streams=1
):
    """
    Push random tensors to the specified address(es).

    Args:
        address (str): Base address to push data to (URL format).
        C (int): Number of channels.
        H (int): Height of the tensor.
        W (int): Width of the tensor.
        total_size (int): Total number of samples to push.
        push_interval (float): Interval between pushes in seconds.
        continuous (bool): Whether to continuously push data.
        num_streams (int): Number of push streams to create.
    """
    threads = []

    for i in range(num_streams):
        # Create address for this stream
        if num_streams > 1:
            # Parse the base address to create a new one with offset port
            if address.startswith('tcp://') and ':' in address.split('//')[1]:
                # Format: tcp://host:port
                host_port = address.split('//')[1]
                host, port = host_port.rsplit(':', 1)
                try:
                    new_port = int(port) + i
                    stream_address = f"tcp://{host}:{new_port}"
                except ValueError:
                    # Fallback if port can't be parsed
                    stream_address = f"{address}_{i}"
            else:
                # Non-TCP or format without port
                stream_address = f"{address}_{i}"
        else:
            stream_address = address

        # Create and start thread for this stream
        thread = threading.Thread(
            target=dummy_data_pusher_thread,
            args=(
                stream_address,
                C,
                H,
                W,
                total_size,
                push_interval,
                continuous,
                i
            )
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)

    # In continuous mode, we don't want to block the main thread waiting
    if continuous:
        try:
            while True:
                time.sleep(1)  # Keep the main thread alive
        except KeyboardInterrupt:
            print("Interrupted by user, shutting down...")
    else:
        # In non-continuous mode, wait for all threads to complete
        for thread in threads:
            thread.join()

def main():
    parser = argparse.ArgumentParser(description='Push dummy data for PyTorch training.')
    parser.add_argument('--address', default='tcp://127.0.0.1:5555', help='Address to push data to (URL format).')
    parser.add_argument('--C', type=int, default=3, help='Number of channels.')
    parser.add_argument('--H', type=int, default=224, help='Height of the tensor.')
    parser.add_argument('--W', type=int, default=224, help='Width of the tensor.')
    parser.add_argument('--total-size', type=int, default=10000, help='Total number of samples to push.')
    parser.add_argument('--push-interval', type=float, default=0.001, help='Interval between pushes in seconds.')
    parser.add_argument('--continuous', action='store_true', help='Continuously push data (loop through indices).')
    parser.add_argument('--num-streams', type=int, default=1, help='Number of push streams to create.')

    args = parser.parse_args()

    dummy_data_pusher(
        args.address,
        args.C,
        args.H,
        args.W,
        args.total_size,
        args.push_interval,
        args.continuous,
        args.num_streams
    )

if __name__ == '__main__':
    main()
</code>

Could you make a plan to address both issues?

---

Is there a benefit of using Dict for batch data?  For ML inference, the batch
will be processed at once so it doesn't really matter it's a Dict or List.  Now,
what matters is how the pipeline handles both tensor data and metadata.  With
the concern of metadata, pipeline just expects a tensor with the shape of (B, C,
H, W), which could be assembled at the data producer stage, or between `Q1` and
`P`.  Though, if it's done in the data producer stage, the batch size
configuration should be consistent for the pipeline and the producer.  Now,
metadata can simply be a List with the underlying order matching the orders in
the (B, C, H, W) tensor.  Even if you want packaging like you propose now with
BatchData and BatchOutput, you just unpack them to get the real tensor data,
input them to the pipeline and get the output, and the metadata just got copied
over to the BatchOutput.  Is this a good design?  Again, the key question is
where BatchData is packaged?  From the producer or between `Q1` and `P`.

---

Run `python -m peaknet_pipeline_ray.cli.main --config
examples/configs/peaknet.yaml --max-actors 4 --total-samples 20480 --verbose
--enable-profiling` and monitor the process until the end of it (it will take a
few minutes).  I don't really see GPU being busy (you can use `watch -n 1
nvidia-smi` for a while from time to time).  I think something is wrong.

---

I worked on some streaming based ML model inference code implmenetation in
peaknet_pipeline_ray.  Now, please evaluate the quality of the implmentation
from the following perspectives:
- Is the double buffering done correctly?
- Is the double buffering processing a batch of data, or one data point at a
  time?
- Is the producer actually feeding data into a shared queue?
- Is the consumer actually fetching data from the shared queue?
- Is the double buffering pipeline actually processing such batch of data from
  the queue?

A harder question is, how different the pipeline is compared with the
implmentation in commit `8a327dd6ad5d4e9cdcb8c8adb9b40b425921c5f2`.

---

You really add a lot of timing related logging.  I don't believe in these
because (1) Ray is fundamentally async; (2) GPU compute is fundamentally async;
I will use profilers (both Ray dashboard and nsys profilers) to understand
timing related information.  You really don't need to add them into the logging.

---

My understanding of this coordinator based synchronization for the overall
pipeline is that 
- Producer will inform (getting registered) the coordinator that it's done, but
  doesn't terminate right away.
- Pipeline acotr will keep processing next batch if the queue still has next
  batch to process.
- When the pipeline actor gets nothing, or the queue becomes empty, it asks the
  coordinator if it should be done with the processing or still wait because not
  every producer has finished producing.
- If the queue is empty and all producer is done with producing, then a system
  wide shut down signal can be issued by the coordinator, which will shutdown
  the producers and pipeline actors, and eventually itself.

I think the current code might not be doing it exactly, but is my idea better
what has been offered now?

---

You might have also break the origianl double buffered pipeline (you can refer
to commit 8a327dd6ad5d4e9cdcb8c8adb9b40b425921c5f2 to see how the pipeline used
to work).  Look at the profiling results in broken_pipeline.png, you can see a
clear cudastreamsynchronize that is preventing it.  I think you might break it
when adjusting the peaknet pipeline that used to work in processing a finite
number of inputs to now essentially becoming in theory infinitely many inputs
until the queue is empty and all producers are done with their jobs.  Please
check if it's true and propose a fix if so.

---

I think you have gone down a path that is too deeply wrong.  Let's work on a new
plan.  Go back to the commit that is associated with the tag
"v0.1-no-queue-version".  The double buffered pipeline should still work there.
Though, the queue might or might not be implemented yet.  Now, I basically want
the same coordination mechansim that we have discussed, while still preserving
the original pipeline execpt you might have to change it to a "while"
conditional loop logic because the previous processing logic assumes you are
processing a finite number of batches, which doesn't requires coordination for
the synchronization.  You can commit the key python files and unstaged changes
for now in the current branch, but let's branch off the implementation from the
commit associated with the tag "v0.1-no-queue-version".  And see how it works.
Write down your new plan carefully in PLAN_STREAM.md.

---

DONE with queue based processing

---

TODO:

(raylet) It looks like you're creating a detached actor in an anonymous namespace. In order to access this actor in the future, you will need to explicitly connect to this namespace with ray.init(namespace="3560c1b0-fbf0-4de9-b73d-533862029e4b", ...)

Instead of making it anonymous, why don't we address this warning!

---

considering there needs to be an agreement with the socket pusher on how to
unpack the data streamed in, could you look up `$STREAMER_DIR` and the pull
script `$TEST_DIR/psana_pull_script_inspect.py` to understand if you can make
the data unpacking configurable (like file format, e.g. hdf5, but could be
others in the future, and if the format has hierarchies like in hdf5, how to
unpack it specificially?)?  such configuration can go into
`$TEST_DIR/peaknet.yaml`?  Or should it be separated?

---

I remember the high level socket library nng runs puller as a server, right?
You should launch the puller side (consumer) before running the pusher.  Please
write a temp but concise readme TEST_STREAM_README.md about how to run tests.

---

Let's do it together.

If I ran 
```
cd $TEST_DIR
peaknet-pipeline --config peaknet.yaml --max-actors 4 --total-samples 10240 --verbose
```

I got two information.

```
(PeakNetPipelineActor pid=464957) ✓ All pipeline buffers verified on GPU 0 [repeated 3x across cluster]
(PeakNetPipelineActor pid=464957) [DEBUG] Pipeline buffer memory: 1.318 GB (0.439 GB input + 0.879 GB output) [repeated 3x across cluster]
(PeakNetPipelineActor pid=464957) [DEBUG] Batch 3: input tensor device: cuda:0 [repeated 15x across cluster]
(PeakNetPipelineActor pid=464957) [DEBUG] Batch 3: model device: cuda:0 [repeated 15x across cluster]
(PeakNetPipelineActor pid=464957) [DEBUG] GPU memory before inference: 1.45 GB [repeated 15x across cluster]
(PeakNetPipelineActor pid=464959) [DEBUG] Batch 4: output tensor device: cuda:0 [repeated 16x across cluster]
(PeakNetPipelineActor pid=464959) [DEBUG] GPU memory after inference: 1.89 GB [repeated 16x across cluster]
(PeakNetPipelineActor pid=464959) [DEBUG] Batch 4: PeakNet forward pass completed on GPU [repeated 16x across cluster]
(PeakNetPipelineActor pid=464959) [DEBUG] Batch 6: input tensor device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActor pid=464959) [DEBUG] Batch 6: model device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActor pid=464959) [DEBUG] GPU memory before inference: 1.45 GB [repeated 9x across cluster]                                                                                        (PeakNetPipelineActor pid=464954) [DEBUG] Batch 6: output tensor device: cuda:0 [repeated 11x across cluster]
(PeakNetPipelineActor pid=464954) [DEBUG] GPU memory after inference: 1.89 GB [repeated 11x across cluster]
(PeakNetPipelineActor pid=464954) [DEBUG] Batch 6: PeakNet forward pass completed on GPU [repeated 11x across cluster]
(PeakNetPipelineActor pid=464954) [DEBUG] Batch 8: input tensor device: cuda:0 [repeated 11x across cluster]
(PeakNetPipelineActor pid=464954) [DEBUG] Batch 8: model device: cuda:0 [repeated 11x across cluster]
(PeakNetPipelineActor pid=464954) [DEBUG] GPU memory before inference: 1.45 GB [repeated 11x across cluster]
(PeakNetPipelineActor pid=464959) [DEBUG] Batch 9: output tensor device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActor pid=464959) [DEBUG] GPU memory after inference: 1.89 GB [repeated 9x across cluster]
(PeakNetPipelineActor pid=464959) [DEBUG] Batch 9: PeakNet forward pass completed on GPU [repeated 9x across cluster]
(PeakNetPipelineActor pid=464959) [DEBUG] Batch 11: input tensor device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActor pid=464959) [DEBUG] Batch 11: model device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActor pid=464959) [DEBUG] GPU memory before inference: 1.45 GB [repeated 9x across cluster]
   ⚠️  Actor health check failed: Get timed out: some object(s) not ready.
   Continuing anyway...

🌊 Starting Streaming Workflow
   Producers: 1
   Actors: 4
   Batches per producer: 640
   Expected total: 640 batches, 10240 samples
   Input shape: [1, 1920, 1920]
   Inter-batch delay: 0.01s

📡 Step 1: Creating Streaming Coordinator
📦 Step 2: Creating Queue Infrastructure
(raylet) It looks like you're creating a detached actor in an anonymous namespace. In order to access this actor in the future, you will need to explicitly connect to this namespace with ray.init
(namespace="0d38d4dc-b1b4-47c8-b671-350de7676522", ...)
   Q1 (input): 4 shards, 100 items/shard                                                                                                                                                           🏭 Step 3: Launching Streaming Producers (socket)
   Socket: sdfada015:12321
   HDF5 fields: ['detector_data', 'timestamp', 'photon_wavelength', 'random']
2025-09-18 12:09:46,115 - INFO - Created 1 socket HDF5 producers
🎭 Step 4: Launching Streaming Pipeline Actors
⏳ Step 5: Streaming Processing (Real-time)
   📊 Producers generating data...
   ⚡ Actors processing continuously...
   🔄 Double buffering preserved - no per-batch sync!

   Waiting for producers to finish...
(PeakNetPipelineActor pid=464957) [DEBUG] Batch 11: output tensor device: cuda:0 [repeated 11x across cluster]
(PeakNetPipelineActor pid=464957) [DEBUG] GPU memory after inference: 1.89 GB [repeated 11x across cluster]
(PeakNetPipelineActor pid=464957) [DEBUG] Batch 11: PeakNet forward pass completed on GPU [repeated 11x across cluster]
(PeakNetPipelineActor pid=464957) [DEBUG] Batch 13: input tensor device: cuda:0 [repeated 11x across cluster]
(PeakNetPipelineActor pid=464957) [DEBUG] Batch 13: model device: cuda:0 [repeated 11x across cluster]
(PeakNetPipelineActor pid=464957) [DEBUG] GPU memory before inference: 1.45 GB [repeated 11x across cluster]
(SocketHDF5Producer pid=464950) 2025-09-18 12:09:50,886 - INFO - SocketHDF5Producer 0 initialized: socket=tcp://sdfada015:12321, batch_size=16, fields=['detector_data', 'timestamp', 'photon_wavel
ength', 'random']
(SocketHDF5Producer pid=464950) 2025-09-18 12:09:50,890 - INFO - Producer 0: Starting to stream 640 batches from tcp://sdfada015:12321
(SocketHDF5Producer pid=464950) 2025-09-18 12:09:50,892 - ERROR - Synchronous dial failed; attempting asynchronous now
(SocketHDF5Producer pid=464950) Traceback (most recent call last):
(SocketHDF5Producer pid=464950)   File "/sdf/scratch/users/c/cwang31/miniconda2/pytorch-2.6/lib/python3.13/site-packages/pynng/nng.py", line 391, in dial
(SocketHDF5Producer pid=464950)     return self.dial(address, block=True)
(SocketHDF5Producer pid=464950)            ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
(SocketHDF5Producer pid=464950)   File "/sdf/scratch/users/c/cwang31/miniconda2/pytorch-2.6/lib/python3.13/site-packages/pynng/nng.py", line 388, in dial
(SocketHDF5Producer pid=464950)     return self._dial(address, flags=0)
(SocketHDF5Producer pid=464950)            ~~~~~~~~~~^^^^^^^^^^^^^^^^^^
(SocketHDF5Producer pid=464950)   File "/sdf/scratch/users/c/cwang31/miniconda2/pytorch-2.6/lib/python3.13/site-packages/pynng/nng.py", line 407, in _dial
(SocketHDF5Producer pid=464950)     check_err(ret)
(SocketHDF5Producer pid=464950)     ~~~~~~~~~^^^^^
(SocketHDF5Producer pid=464950)   File "/sdf/scratch/users/c/cwang31/miniconda2/pytorch-2.6/lib/python3.13/site-packages/pynng/exceptions.py", line 202, in check_err
(SocketHDF5Producer pid=464950)     raise exc(string, err)
(SocketHDF5Producer pid=464950) pynng.exceptions.ConnectionRefused: Connection refused
(SocketHDF5Producer pid=464950) 2025-09-18 12:09:50,894 - INFO - Producer 0: Connected to tcp://sdfada015:12321
(PeakNetPipelineActor pid=464957) 2025-09-18 12:09:20,778 - INFO - === Initializing PeakNetPipelineActor === [repeated 3x across cluster]
(PeakNetPipelineActor pid=464957) 2025-09-18 12:09:20,786 - INFO - Using Ray-assigned GPU device 0 (physical GPU 3) [repeated 3x across cluster]
(PeakNetPipelineActor pid=464957) 2025-09-18 12:09:20,786 - INFO - ✅ Actor GPU assignment complete - using CUDA device 0 [repeated 3x across cluster]
(PeakNetPipelineActor pid=464957) 2025-09-18 12:09:20,786 - INFO - CUDA_VISIBLE_DEVICES: 3 [repeated 3x across cluster]
(PeakNetPipelineActor pid=464957) 2025-09-18 12:09:21,078 - INFO - ✅ CUDA context established on device 0 [repeated 3x across cluster]
(PeakNetPipelineActor pid=464957) 2025-09-18 12:09:21,079 - INFO - Actor GPU: NVIDIA L40S (45596 MB) [repeated 3x across cluster]
(PeakNetPipelineActor pid=464957) 2025-09-18 12:09:21,079 - INFO - Actor CPU affinity: 0-11,24-59,72-95 [repeated 3x across cluster]
(PeakNetPipelineActor pid=464957) 2025-09-18 12:09:21,467 - INFO - PeakNet mode: input_shape=(1, 1920, 1920), output_shape=(2, 1920, 1920) [repeated 3x across cluster]
(PeakNetPipelineActor pid=464957) 2025-09-18 12:09:21,714 - INFO - Running model warmup with 500 samples... [repeated 3x across cluster]
(PeakNetPipelineActor pid=464954) 2025-09-18 12:09:21,969 - INFO - Warmup: Processing 32 batches of size 16 [repeated 3x across cluster]
(raylet) It looks like you're creating a detached actor in an anonymous namespace. In order to access this actor in the future, you will need to explicitly connect to this namespace with ray.init
(namespace="0d38d4dc-b1b4-47c8-b671-350de7676522", ...) [repeated 3x across cluster]
(PeakNetPipelineActor pid=464959) [DEBUG] Batch 14: output tensor device: cuda:0 [repeated 9x across cluster]
(PeakNetPipelineActor pid=464959) [DEBUG] GPU memory after inference: 1.89 GB [repeated 9x across cluster]
```

I see some inference is running anyway?  Is it because there's a warm-up stage?  Please confirm.

Also, I thought when the pusher has not started, the puller should wait, but it errors out?

---

Let's review STATUS.md and focus on our priority again.

I see you have
`$PIPELINE_DEV_DIR/peaknet_pipeline_ray/core/socket_hdf5_producer.py` now, but
is this producer actually waiting for data from a pusher that I will launch
manually else where in a completely different node?  The current behavior seems
to suggest the code is defaulting back to some fake or random data because the
inference pipeline still runs after warm-up, in fact, it should wait and only
run when it indeed sees data pushed from the socket by another problem that I
mentioned earlier (that I will launch myself).  Also, ideally, the warm-up
should start after the first data fetched from the sokect to understand the
data's actual shape, so called post warm-up.  The shape config in the current
peaknet.yaml is for the random data source.  However, please also provide a
shape option under the socket data source config in the peaknet.yaml that can
allow the warm-up (let's call it pre warm-up) to start even before actual data
pushed to the socket by the pusher unless the value of the data shape is null
then it should do post warm-up.

---

I will profile it to understand if the socket blocking behavior is damaging.

Now, please look into this error.

```
📦 Step 2: Creating Queue Infrastructure
(raylet) It looks like you're creating a detached actor in an anonymous namespace. In order to access this actor in the future, you will need to explicitly connect to this namespace with ray.init(namespace="6c681878-0828-4960-986a-0b7b6a2976b5", ...)
   Q1 (input): 4 shards, 100 items/shard
🏭 Step 3: Launching Streaming Producers (socket)

💥 Streaming pipeline failed: expected 'except' or 'finally' block (socket_hdf5_producer.py, line 184)
Traceback (most recent call last):
  File "/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray/peaknet_pipeline_ray/pipeline.py", line 657, in run_streaming_pipeline
    performance = self._run_streaming_workflow(actors, enable_output_queue)
  File "/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray/peaknet_pipeline_ray/pipeline.py", line 775, in _run_streaming_workflow
    producers = self._create_data_producers(runtime, data)
  File "/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray/peaknet_pipeline_ray/pipeline.py", line 928, in _create_data_producers
    from .core.socket_hdf5_producer import create_socket_hdf5_producers
  File "/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray/peaknet_pipeline_ray/core/socket_hdf5_producer.py", line 184
    def _extract_shape_from_data(self, raw_data: bytes) -> Optional[Tuple[int, int, int]]:
SyntaxError: expected 'except' or 'finally' block
```

---

Did you also update $TEST_DIR/peaknet.yaml?  There used to be two fields where
shape is mentioned.

```
data:
  shape: [1, 1920, 1920]  # Full resolution as requested (channels inferred from shape[0])

# NEW: Data source configuration for socket streaming
data_source:
  source_type: socket     # Use "random" for synthetic data, "socket" for LCLStreamer

  # Socket connection settings
  socket_hostname: sdfada015  # Change to actual hostname (e.g., sdfada015)
  socket_port: 12321          # Default LCLStreamer port
  socket_timeout: 30.0        # Socket receive timeout in seconds
  socket_retry_attempts: 5    # Number of connection retry attempts

  # WARMUP CONTROL: Shape configuration determines warmup timing
  # Option 1 - PRE-WARMUP: Specify shape to warmup before any socket data arrives
  # shape: [1, 1920, 1920]     # Use this for immediate warmup with known shape

  # Option 2 - POST-WARMUP: Set to null to warmup after shape detection from first data
  shape: null               # Use this to detect shape from first socket data packet

  # NOTE: Pre-warmup is faster startup but requires knowing the data shape in advance.
  #       Post-warmup is more flexible but waits for first data to determine shape.
```

I believe we should only keep one.  Is that right?

---

I feel it's not a concise design in <text>
data:
  shape: [1, 1920, 1920]  # Full resolution as requested (channels inferred from shape[0])

transforms:
  # Shape conversion from socket data (H, W) to model input (C, H, W)
  add_channel_dimension: true   # Convert (H, W) -> (C, H, W)
  num_channels: 1              # Number of channels to add (usually 1 for detector data)
  channel_dim: 1               # Position to insert channel dimension (after batch)

  # Padding to normalize size differences
  pad_to_target: true          # Pad from 1691x1691 to 1920x1920
  pad_style: center            # Center padding for detector images

# NEW: Data source configuration for socket streaming
data_source:
  source_type: socket     # Use "random" for synthetic data, "socket" for LCLStreamer

  # Socket connection settings
  socket_hostname: sdfada015  # Change to actual hostname (e.g., sdfada015)
  socket_port: 12321          # Default LCLStreamer port
  socket_timeout: 300.0        # Socket receive timeout in seconds
  socket_retry_attempts: 5    # Number of connection retry attempts

  # Explicit shape for socket data (must match actual detector data from LCLStreamer)
  shape: [1, 1691, 1691]  # C, H, W - actual detector shape with channel
  dimension</text>

Look, the padding transformation doesn't really case about what the initial size
is, you can check it yourself.  Thereby, what's the use of <text>shape: [1,
1691, 1691]</text>?

---

I want to simplify this process.  Let's just trust the users know what they are
doing from the pusher side.  If the pusher produces 16 examples in a batch, then
the puller along with the inference pipeline will fetch this batch (no need to
unpack it to 16 individual examples, even the transformation works on a batch of
examples already).  Could we do this instead?

---

Read STATUS.md, and understand what our priority is.  I can tell you (or
checking the git log yourself) that we have already implemented a solution that
is trying to get us to the priority.  Now, one thing is that I really need is to
profile the inference pipeline using nsys (which is also implemented).  OK,
normally when we use nsys, We can control sea and program that is it is
profiling and we will still be able to get the profiling results for now since
this nsys profiler is handled Directly inside Ray I'm not sure if I'm doing
something wrong but now when I control see the program you know I have one right
actor for each machine learning inference pipeline I control C I see nothing
there's no profiling results. Could you really look into our implementation and
also looking to Ray docs And further look up online can I still use the
profiler in Ray That is subject to a control C operation.  Another reason I'm
asking this is the testing data source is actually quite large. I can't really
wait for the end of the data source.  So I have to control it out. I guess what
I'm really asking is making control C a proper operation that this pipeline
needs so that you know when the pipeline actually exits normally the profiler
will generate a profiling result.

---

I included externals/zoom-in.png that shows the ML inference pipeline has good
h2d/d2h and compute overlap.  So I don't see any major problems in the pipeline
itself yet.  However, I also included externals/zoom-out.png to show you the
problem - the pipeline has GAPS.  My judgement is that these gaps are not caused
by host device synchronization, but not enough data pushed to device to process.
I have increased the queue size in a hope to mitigate this, but it doesn't seem
to help.  Could you help me diagnose it?  You don't have to run codes, and I can
run it for you because it's a bit hard to run these.

I don't want to jump to conclusions too soon.  Let's focus on narrowing down the
problem first by running some experiments and just observe what happens.  A few
hypothesis to test:
- Queue too small
- Pusher worker too small
- the S to Q1 process implemented right now doesn't encourage pre-fetching and
  loading into memory for H2D later

I believe the ML inference pipeline itself is okay, but not sure about the
overarching pipeline (especially, now considering the data are streamed in
through sockets and then put into Ray object store for all pipeline actors to
consume)

---

my pusher code shows <text>[22:28:30] ERROR    Unable to connect to the URL
tcp://sdfada001:12321 due to the following error: Connection refused</text>.
Are you sure your current implementation has a pull server running?  You can
find clues from /sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml/psana_pull_script_inspect.py

---

Please understand our progress status now.  I'm going to implement preprocessing
directly in the producer.  Do you think the current code implementation requires
change, or it's a matter of changing the configuration?

---

If this is the case, could you simply take the preprocessing out completely from
the codebase and the config in
`/sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml/peaknet-socket.yaml`.
It's a good lesson to learn, but I believe the production version doesn't need
it, thus we should clean up the repo.  But for educational purposes, before we
revert it back to no pre-processing from Q1 to P, let's create a branch and tag
it as exp-preprocess-Q1-to-P.  What do you think?
