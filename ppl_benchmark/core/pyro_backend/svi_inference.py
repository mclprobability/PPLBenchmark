"""
SVI inference module for Pyro backend.

This module provides functions for performing SVI (Stochastic Variational Inference) 
using the Pyro framework. It counts forward and backward calls for benchmarking purposes.
"""

import time
import torch
import pyro
import pyro.infer
from pyro.optim import Adam
from ppl_benchmark.models.base_model import BenchmarkPyroModule
from ppl_benchmark.core.results import BenchmarkResult, SVIResult


def svi(svi_params: dict, model: BenchmarkPyroModule, guide, *model_args):
    """
    Perform SVI inference with call counting.
    
    This function performs SVI inference using Pyro and counts forward and 
    backward calls for benchmarking purposes.

    Note: SVI particles are not vectorized due to current limitations in the call counting implementation.
    
    Args:
        svi_params (dict): Parameters for the SVI algorithm.
        model (BenchmarkPyroModule): The model to perform inference on.
        guide: The guide/variational distribution.
        *model_args: Arguments to pass to the model.
        
    Returns:
        BenchmarkResult: The benchmark results including execution time and call counts.
        
    Raises:
        ValueError: If model arguments are not torch Tensors.
    """
    nr_iterations = svi_params.get("svi_iterations", 100)
    learning_rate = svi_params.get("learning_rate", 0.01)
    num_particles = svi_params.get("num_particles", 1)
    # Ensure Pyro param store is clean
    pyro.clear_param_store()

    # Ensure data are torch tensors
    for arg in model_args:
        if not isinstance(arg, torch.Tensor):
            raise ValueError("Model arguments must be torch Tensors")
   

    # SVI setup (Pyro)
    optimizer = Adam({"lr": learning_rate})
    # TODO: particle vectorization not yet supported with the current counter implementation
    loss = pyro.infer.Trace_ELBO(num_particles=num_particles, vectorize_particles=False)
    svi = pyro.infer.SVI(model, guide, optimizer, loss=loss)

    losses = []
    start_timer = time.perf_counter()

    # Training loop (Pyro uses step instead of init/update)
    for step in range(nr_iterations):
        loss = svi.step(*model_args)
        if step % 200 == 0:
            print(f"Step {step}, loss = {loss:.4f}")
        losses.append(loss)

    stop_timer = time.perf_counter()
    execution_time = stop_timer - start_timer

    # Extract learned parameters from Pyro param store
    param_state = pyro.get_param_store().get_state()
    # convert tensors to numpy for readability
    optimized_parameters = {}
    for name, val in param_state.items():
        if isinstance(val, torch.Tensor):
            optimized_parameters[name] = val.detach().cpu().numpy()
        else:
            optimized_parameters[name] = val

    svi_result = SVIResult(guide=guide, svi_state=None, losses=losses)
    benchmark_result = BenchmarkResult(execution_time=execution_time, forward_calls=model.forward_counter.get(), 
                                       backward_calls=model.backward_counter.get(), svi_result=svi_result, mcmc_result=None)

    return benchmark_result
