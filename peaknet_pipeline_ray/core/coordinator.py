"""Lightweight coordination actor for streaming pipeline termination.

This module provides a simple, efficient coordination mechanism for managing
when streaming pipeline actors should terminate. The coordinator tracks when
all data producers have finished and provides a query-based approach for
actors to determine when they can safely shutdown.

Key design principles:
1. Lightweight: Minimal CPU usage, no storage of large data
2. Query-based: Actors ask when to shutdown, coordinator doesn't push
3. No timeouts: Uses producer completion state, not arbitrary time limits
4. Thread-safe: Ray actor provides natural thread safety
"""

import ray
import logging
from typing import Set, Dict, Any
import time

logging.basicConfig(level=logging.INFO)


@ray.remote(num_cpus=0.1)  # Lightweight coordination actor
class StreamingCoordinator:
    """Centralized coordinator for intelligent streaming pipeline termination.
    
    This coordinator manages the lifecycle of a streaming pipeline by tracking
    when data producers finish and providing a query interface for processing
    actors to determine when they should terminate.
    
    State machine:
    INITIALIZED -> RUNNING -> PRODUCERS_FINISHED -> COMPLETED
    """
    
    def __init__(self):
        """Initialize the coordinator."""
        self.producers_finished = set()  # Set of finished producer IDs
        self.actors_finished = set()     # Set of finished actor IDs  
        self.expected_producers = 0
        self.expected_actors = 0
        self.state = "INITIALIZED"
        self.start_time = time.time()
        
        logging.info("StreamingCoordinator initialized")
    
    def set_expected_counts(self, producers: int, actors: int) -> None:
        """Set the expected number of producers and actors.
        
        Args:
            producers: Number of data producers that will register completion
            actors: Number of processing actors that will participate
        """
        self.expected_producers = producers
        self.expected_actors = actors
        self.state = "RUNNING"
        
        logging.info(f"Coordinator expecting {producers} producers and {actors} actors")
    
    def register_producer_finished(self, producer_id: str) -> None:
        """Register that a producer has finished generating data.
        
        Args:
            producer_id: Unique identifier for the producer
        """
        self.producers_finished.add(producer_id)
        
        logging.info(
            f"Producer {producer_id} finished "
            f"({len(self.producers_finished)}/{self.expected_producers})"
        )
        
        # Update state when all producers are done
        if len(self.producers_finished) >= self.expected_producers:
            self.state = "PRODUCERS_FINISHED"
            elapsed = time.time() - self.start_time
            logging.info(f"All {self.expected_producers} producers finished in {elapsed:.2f}s")
    
    def all_producers_finished(self) -> bool:
        """Check if all expected producers have finished.
        
        Returns:
            True if all producers have registered completion
        """
        return len(self.producers_finished) >= self.expected_producers
    
    def should_actor_shutdown(self, queue_empty: bool = True) -> bool:
        """Core method: Determine if an actor should shutdown.
        
        This is the primary interface for processing actors to query whether
        they should terminate. The logic is:
        1. If queue is not empty -> keep processing
        2. If producers still working -> keep waiting  
        3. If all producers done AND queue empty -> safe to shutdown
        
        Args:
            queue_empty: Whether the actor's input queue is empty
            
        Returns:
            True if the actor should shutdown, False if it should continue
        """
        # If queue has data, keep processing
        if not queue_empty:
            return False
        
        # If producers still working, keep waiting for more data
        if self.state != "PRODUCERS_FINISHED":
            return False
        
        # All producers done AND queue empty = safe to shutdown
        return True
    
    def register_actor_finished(self, actor_id: str) -> None:
        """Register that an actor has finished processing.
        
        Args:
            actor_id: Unique identifier for the actor
        """
        self.actors_finished.add(actor_id)
        
        logging.info(
            f"Actor {actor_id} finished "
            f"({len(self.actors_finished)}/{self.expected_actors})"
        )
        
        # Update state when all actors are done
        if len(self.actors_finished) >= self.expected_actors:
            self.state = "COMPLETED"
            elapsed = time.time() - self.start_time
            logging.info(f"Pipeline completed in {elapsed:.2f}s - all actors finished")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current coordinator state for monitoring and debugging.
        
        Returns:
            Dictionary with current state information
        """
        return {
            'state': self.state,
            'producers_finished': len(self.producers_finished),
            'expected_producers': self.expected_producers,
            'actors_finished': len(self.actors_finished), 
            'expected_actors': self.expected_actors,
            'all_producers_finished': self.all_producers_finished(),
            'uptime_seconds': time.time() - self.start_time,
            'finished_producer_ids': list(self.producers_finished),
            'finished_actor_ids': list(self.actors_finished)
        }
    
    def is_completed(self) -> bool:
        """Check if the entire pipeline has completed.
        
        Returns:
            True if all producers and actors have finished
        """
        return self.state == "COMPLETED"


def create_streaming_coordinator(expected_producers: int, expected_actors: int) -> ray.actor.ActorHandle:
    """Convenience function to create and configure a streaming coordinator.
    
    Args:
        expected_producers: Number of data producers
        expected_actors: Number of processing actors
        
    Returns:
        Ray actor handle for the coordinator
    """
    coordinator = StreamingCoordinator.remote()
    ray.get(coordinator.set_expected_counts.remote(expected_producers, expected_actors))
    return coordinator


if __name__ == "__main__":
    # Simple test of the coordinator
    import time
    
    if not ray.is_initialized():
        ray.init()
    
    print("Testing StreamingCoordinator...")
    
    # Create coordinator
    coordinator = create_streaming_coordinator(expected_producers=2, expected_actors=2)
    
    # Check initial state
    state = ray.get(coordinator.get_state.remote())
    print(f"Initial state: {state}")
    
    # Simulate producers finishing
    ray.get(coordinator.register_producer_finished.remote("producer_0"))
    ray.get(coordinator.register_producer_finished.remote("producer_1"))
    
    # Check if actors should shutdown
    should_shutdown = ray.get(coordinator.should_actor_shutdown.remote(queue_empty=True))
    print(f"Should actors shutdown? {should_shutdown}")
    
    # Register actors finishing
    ray.get(coordinator.register_actor_finished.remote("actor_0"))
    ray.get(coordinator.register_actor_finished.remote("actor_1"))
    
    # Final state
    final_state = ray.get(coordinator.get_state.remote())
    print(f"Final state: {final_state}")
    
    print("✅ StreamingCoordinator test passed!")