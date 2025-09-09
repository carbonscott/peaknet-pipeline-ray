#!/usr/bin/env python3
"""
Data Producer for PipelineInput structures with metadata pass-through.

Generates PipelineInput objects with random image data and mock metadata,
putting them into Ray's object store for pipeline actor consumption.
"""

import ray
import torch
import numpy as np
import time
import logging
from typing import List, Tuple, Dict, Any
import uuid

from ..config import PipelineInput

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_mock_metadata(batch_id: str, sample_idx: int) -> Dict[str, Any]:
    """Generate mock metadata for testing purposes."""
    return {
        'photon_energy': np.random.uniform(8000.0, 12000.0),  # eV
        'timestamp': time.time(),
        'run_id': f"run_{batch_id[:8]}",  
        'sample_idx': sample_idx,
        'detector_distance': np.random.uniform(100.0, 200.0),  # mm
        'pulse_energy': np.random.uniform(0.1, 1.0),  # mJ
        'beam_center_x': np.random.uniform(256, 512),
        'beam_center_y': np.random.uniform(256, 512)
    }


@ray.remote
def generate_pipeline_input_batch(
    batch_size: int,
    tensor_shape: Tuple[int, int, int],
    batch_id: int,
    pin_memory: bool = True,
    deterministic: bool = False
) -> List[ray.ObjectRef]:
    """
    Generate a batch of PipelineInput objects and put them in Ray object store.
    
    Args:
        batch_size: Number of PipelineInput objects to generate in this batch
        tensor_shape: Shape of each image tensor (C, H, W)
        batch_id: Unique identifier for this batch
        pin_memory: Whether to pin memory for faster GPU transfers
        deterministic: Use fixed seed for reproducible results
        
    Returns:
        List of Ray object references to the generated PipelineInput objects
    """
    if deterministic:
        np.random.seed(42 + batch_id)
        torch.manual_seed(42 + batch_id)
    
    batch_uuid = str(uuid.uuid4())
    logging.info(f"Producer generating batch {batch_id} ({batch_uuid}) with {batch_size} samples")
    
    # Generate PipelineInput objects
    pipeline_inputs = []
    for i in range(batch_size):
        # Generate image data
        image_data = torch.randn(*tensor_shape)
        if pin_memory and torch.cuda.is_available():
            image_data = image_data.pin_memory()
        
        # Generate mock metadata
        metadata = generate_mock_metadata(batch_uuid, i)
        
        # Create PipelineInput object
        pipeline_input = PipelineInput(
            image_data=image_data,
            metadata=metadata,
            batch_id=f"{batch_uuid}_{i}"
        )
        
        pipeline_inputs.append(pipeline_input)
    
    # Put PipelineInput objects into Ray object store
    object_refs = []
    for pipeline_input in pipeline_inputs:
        obj_ref = ray.put(pipeline_input)
        object_refs.append(obj_ref)
    
    logging.info(f"Producer batch {batch_id} complete: {len(object_refs)} PipelineInputs in object store")
    return object_refs


@ray.remote
def generate_streaming_pipeline_inputs(
    num_batches: int,
    batch_size: int, 
    tensor_shape: Tuple[int, int, int],
    producer_id: int,
    inter_batch_delay: float = 0.0,
    pin_memory: bool = True,
    deterministic: bool = False
) -> List[List[ray.ObjectRef]]:
    """
    Generate multiple batches of PipelineInputs with optional delays to simulate streaming.
    
    Args:
        num_batches: Number of batches to generate
        batch_size: Number of PipelineInputs per batch
        tensor_shape: Shape of each image tensor (C, H, W)  
        producer_id: Unique identifier for this producer
        inter_batch_delay: Delay between batches in seconds
        pin_memory: Whether to pin memory for faster GPU transfers
        deterministic: Use fixed seed for reproducible results
        
    Returns:
        List of batches, each batch being a list of object references to PipelineInputs
    """
    if deterministic:
        np.random.seed(1000 + producer_id)
        torch.manual_seed(1000 + producer_id)
    
    logging.info(f"Streaming producer {producer_id} starting: {num_batches} batches")
    
    all_batches = []
    
    for batch_idx in range(num_batches):
        start_time = time.time()
        
        # Generate batch with unique ID combining producer and batch
        batch_id = producer_id * 1000 + batch_idx
        
        # Generate PipelineInput batch
        batch_refs = ray.get(generate_pipeline_input_batch.remote(
            batch_size=batch_size,
            tensor_shape=tensor_shape,
            batch_id=batch_id,
            pin_memory=pin_memory,
            deterministic=deterministic
        ))
        
        all_batches.append(batch_refs)
        
        # Inter-batch delay for streaming simulation
        if inter_batch_delay > 0 and batch_idx < num_batches - 1:
            elapsed = time.time() - start_time
            sleep_time = max(0, inter_batch_delay - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        if (batch_idx + 1) % 10 == 0:
            logging.info(f"Producer {producer_id} completed {batch_idx + 1}/{num_batches} batches")
    
    logging.info(f"Streaming producer {producer_id} complete: {len(all_batches)} batches generated")
    return all_batches


class PipelineInputProducerManager:
    """
    Manager for multiple data producers that generate PipelineInput objects.
    """
    
    def __init__(self, num_producers: int = 4):
        self.num_producers = num_producers
        self.producer_futures = []
        
    def start_producers(
        self,
        batches_per_producer: int,
        batch_size: int,
        tensor_shape: Tuple[int, int, int],
        inter_batch_delay: float = 0.1,
        pin_memory: bool = True,
        deterministic: bool = False
    ) -> List[ray.ObjectRef]:
        """
        Start multiple streaming producers.
        
        Returns:
            List of futures from producer tasks
        """
        logging.info(f"Starting {self.num_producers} PipelineInput producers")
        logging.info(f"  Batches per producer: {batches_per_producer}")
        logging.info(f"  Batch size: {batch_size}")
        logging.info(f"  Tensor shape: {tensor_shape}")
        logging.info(f"  Total samples: {self.num_producers * batches_per_producer * batch_size}")
        
        self.producer_futures = []
        
        for producer_id in range(self.num_producers):
            future = generate_streaming_pipeline_inputs.remote(
                num_batches=batches_per_producer,
                batch_size=batch_size,
                tensor_shape=tensor_shape,
                producer_id=producer_id,
                inter_batch_delay=inter_batch_delay,
                pin_memory=pin_memory,
                deterministic=deterministic
            )
            self.producer_futures.append(future)
            
        return self.producer_futures
        
    def get_all_batches(self) -> List[List[ray.ObjectRef]]:
        """
        Wait for all producers to complete and collect all batches.
        
        Returns:
            Flattened list of all batches from all producers
        """
        logging.info("Waiting for all producers to complete...")
        
        all_producer_batches = ray.get(self.producer_futures)
        
        # Flatten the nested structure
        all_batches = []
        for producer_batches in all_producer_batches:
            all_batches.extend(producer_batches)
            
        logging.info(f"Collected {len(all_batches)} total batches from all producers")
        return all_batches