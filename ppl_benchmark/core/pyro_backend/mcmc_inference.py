"""
MCMC inference module for Pyro backend.

This module provides functions for performing MCMC inference using the Pyro framework.
It counts forward and backward calls for benchmarking purposes.
"""

import time
from pyroapi import pyro
import torch
from ppl_benchmark.models.base_model import BenchmarkPyroModule
from pyro.infer.mcmc import MCMC

from ppl_benchmark.core.results import BenchmarkResult, MCMCResult


def mcmc(kernel, mcmc_params: dict, model: BenchmarkPyroModule, *model_args):
    """
    Perform MCMC inference with call counting.
    
    This function performs MCMC inference using Pyro and counts forward and 
    backward calls for benchmarking purposes.
    
    Args:
        kernel: The MCMC kernel to use.
        mcmc_params (dict): Parameters for the MCMC algorithm.
        model (BenchmarkPyroModule): The model to perform inference on.
        *model_args: Arguments to pass to the model.

    Returns:
        BenchmarkResult: The benchmark results including execution time and call counts.
        
    Raises:
        ValueError: If model arguments are not torch Tensors.
    """
    num_samples = mcmc_params.get("mcmc_samples", 300)
    warmup_steps = mcmc_params.get("mcmc_warmup_steps", 200)
    num_chains = mcmc_params.get("num_chains", 1)

    # Ensure Pyro param store is clean
    pyro.clear_param_store()

    # Ensure data are torch tensors
    for arg in model_args:
        if not isinstance(arg, torch.Tensor):
            raise ValueError("Model arguments must be torch Tensors")

    mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains) 

    start_time = time.perf_counter()
    mcmc.run(*model_args) 
    execution_time = time.perf_counter() - start_time

    posterior_samples = mcmc.get_samples()
    mcmc_result = MCMCResult(posterior_samples=posterior_samples)
    benchmark_result = BenchmarkResult(execution_time=execution_time, forward_calls=model.forward_counter.get(), 
                                       backward_calls=model.backward_counter.get(), svi_result=None, mcmc_result=mcmc_result)

    return benchmark_result
