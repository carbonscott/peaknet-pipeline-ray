"""Ray-based queue implementation for distributed pipeline coordination.

IMPORTANT: Ray ObjectRef Auto-Dereferencing Behavior
====================================================

Ray has specific rules for when ObjectRefs are automatically dereferenced:

1. TOP-LEVEL ARGUMENTS: Ray auto-dereferences ObjectRefs passed directly as arguments
   Example: actor.method.remote(obj_ref) → obj_ref gets resolved to actual data

2. NESTED ARGUMENTS: Ray does NOT auto-dereference ObjectRefs inside containers  
   Example: actor.method.remote([obj_ref]) → obj_ref remains as ObjectRef

3. OBJECT ATTRIBUTES: Ray does NOT auto-dereference ObjectRefs inside custom objects
   Example: actor.method.remote(dataclass_with_objref_field) → ObjectRef preserved

Our queue system leverages behavior #3 - PipelineInput objects containing ObjectRefs
are passed as top-level arguments, but the ObjectRefs inside remain preserved, giving
us optimal memory efficiency without API complexity.

See: https://docs.ray.io/en/latest/ray-core/objects.html#passing-object-arguments
"""

import ray
import time
from collections import deque
from typing import Any, Optional, List


@ray.remote
class RayQueue:
    """A simple FIFO queue implemented as a Ray actor.
    
    This actor manages a single deque and provides thread-safe access
    across multiple Ray processes and nodes. The queue supports named,
    detached actors for persistence across process restarts.
    """
    
    def __init__(self, maxsize: int = 1000):
        """Initialize the queue with a maximum size.
        
        Args:
            maxsize: Maximum number of items the queue can hold.
        """
        self.items = deque(maxlen=maxsize)
        self.maxsize = maxsize

    def put(self, item: Any) -> bool:
        """Put an item into the queue.
        
        Args:
            item: The item to put into the queue (typically a Ray ObjectRef).
            
        Returns:
            True if item was successfully added, False if queue is full.
        """
        try:
            if len(self.items) < self.maxsize:
                self.items.append(item)
                return True
            return False
        except Exception as e:
            print(f"Error in put: {e}")
            return False

    def get(self) -> Optional[Any]:
        """Get an item from the queue.
        
        Returns:
            The next item from the queue, or None if queue is empty.
        """
        try:
            return self.items.popleft() if self.items else None
        except Exception as e:
            print(f"Error in get: {e}")
            return None

    def size(self) -> int:
        """Get the current size of the queue.
        
        Returns:
            Number of items currently in the queue.
        """
        try:
            return len(self.items)
        except Exception as e:
            print(f"Error in size: {e}")
            return 0

    def is_empty(self) -> bool:
        """Check if the queue is empty.
        
        Returns:
            True if queue is empty, False otherwise.
        """
        return len(self.items) == 0

    def is_full(self) -> bool:
        """Check if the queue is full.
        
        Returns:
            True if queue is at maximum capacity, False otherwise.
        """
        return len(self.items) >= self.maxsize


class ShardedQueueManager:
    """Manages multiple RayQueue actors for distributed, scalable queueing.
    
    This class orchestrates multiple RayQueue shards to provide higher
    throughput than a single queue actor. It handles round-robin putting
    and polling gets across all shards transparently.
    """
    
    def __init__(self, base_name: str, num_shards: int = 1, maxsize_per_shard: int = 1000):
        """Initialize the sharded queue manager.
        
        Args:
            base_name: Base name for the queue (shards will be named base_name_shard_0, etc.)
            num_shards: Number of queue shards to create/manage.
            maxsize_per_shard: Maximum size for each individual shard.
        """
        self.base_name = base_name
        self.num_shards = num_shards
        self.maxsize_per_shard = maxsize_per_shard
        self.shard_names = [f"{base_name}_shard_{i}" for i in range(num_shards)]
        
        # Round-robin counter for puts
        self._put_counter = 0
        
        # Initialize or connect to shard actors
        self._setup_shards()

    def _setup_shards(self):
        """Create or connect to the Ray actor shards."""
        self.shard_actors = []
        
        for shard_name in self.shard_names:
            try:
                # Try to get existing actor
                actor = ray.get_actor(shard_name)
                self.shard_actors.append(actor)
            except ValueError:
                # Actor doesn't exist, create it
                actor = RayQueue.options(
                    name=shard_name, 
                    lifetime="detached"
                ).remote(maxsize=self.maxsize_per_shard)
                self.shard_actors.append(actor)

    def put(self, item: Any) -> bool:
        """Put an item into one of the shards using round-robin.
        
        Ray ObjectRef Behavior Note:
        ---------------------------
        When passing PipelineInput objects containing ObjectRefs to queue actors,
        Ray does NOT auto-dereference the ObjectRefs because they are attributes
        of custom objects. This preserves memory efficiency - only the ObjectRef
        (pointer) is stored, not the actual array data.
        
        Args:
            item: The item to put (typically PipelineInput with ObjectRefs).
            
        Returns:
            True if item was successfully added, False if all shards are full.
        """
        # Try each shard starting from the next round-robin position
        start_shard = self._put_counter % self.num_shards
        
        for attempt in range(self.num_shards):
            shard_id = (start_shard + attempt) % self.num_shards
            success = ray.get(self.shard_actors[shard_id].put.remote(item))
            
            if success:
                self._put_counter = shard_id + 1  # Next put starts from next shard
                return True
        
        # All shards are full
        return False

    def get(self, timeout: float = 0.0) -> Optional[Any]:
        """Get an item from any available shard.
        
        Args:
            timeout: Maximum time to wait for an item (0.0 = no waiting).
            
        Returns:
            The next available item, or None if timeout expires.
        """
        start_time = time.time()
        
        while True:
            # Try all shards in round-robin order
            for shard_actor in self.shard_actors:
                item = ray.get(shard_actor.get.remote())
                if item is not None:
                    return item
            
            # Check timeout
            if timeout > 0 and (time.time() - start_time) >= timeout:
                break
                
            # Brief sleep before retrying if no timeout specified
            if timeout == 0:
                break
            time.sleep(0.01)
        
        return None

    def size(self) -> int:
        """Get the total size across all shards.
        
        Returns:
            Total number of items across all shards.
        """
        sizes = ray.get([actor.size.remote() for actor in self.shard_actors])
        return sum(sizes)

    def is_empty(self) -> bool:
        """Check if all shards are empty.
        
        Returns:
            True if all shards are empty, False otherwise.
        """
        return self.size() == 0

    def get_shard_info(self) -> List[dict]:
        """Get information about each shard for debugging/monitoring.
        
        Returns:
            List of dictionaries with shard information.
        """
        shard_info = []
        sizes = ray.get([actor.size.remote() for actor in self.shard_actors])
        
        for i, (name, size) in enumerate(zip(self.shard_names, sizes)):
            shard_info.append({
                'shard_id': i,
                'name': name,
                'size': size,
                'maxsize': self.maxsize_per_shard
            })
        
        return shard_info

