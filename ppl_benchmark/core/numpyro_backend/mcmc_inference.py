"""
MCMC inference module for NumPyro backend.

This module provides functions for performing MCMC inference using the NumPyro framework.
It includes both a deprecated function and a newer function that counts forward and 
backward calls for benchmarking purposes.
"""

import time
import jax
import numpyro
from numpyro.infer import MCMC
from numpyro.infer.util import initialize_model
from numpyro.infer import NUTS, MCMC

from ppl_benchmark.core.numpyro_backend import numpyro_utils
from ppl_benchmark.core.results import BenchmarkResult, MCMCResult


def mcmc(kernel, mcmc_params: dict, model, *model_args):
    """
    Deprecated MCMC function.
    
    This function is deprecated and should not be used. Use mcmc_counted() instead.
    
    Args:
        kernel: The MCMC kernel to use.
        mcmc_params (dict): Parameters for the MCMC algorithm.
        model: The model to perform inference on.
        *model_args: Arguments to pass to the model.
        
    Raises:
        NotImplementedError: Always raised as this function is deprecated.
    """
    raise NotImplementedError("Deprecated in favor of mcmc_counted(). Do not use this function anymore")
    num_samples = mcmc_params.get("num_samples", 300)
    warmup_steps = mcmc_params.get("warmup_steps", 200)

    mcmc = MCMC(kernel, num_samples=num_samples, num_warmup=warmup_steps) 

    rng_key = jax.random.PRNGKey(0)
    start_time = time.perf_counter()
    mcmc.run(rng_key, *model_args) 
    execution_time = time.perf_counter() - start_time

    print(f"MCMC Executed in {execution_time:0.8f} seconds")

    posterior_samples = mcmc.get_samples()
    mcmc_result = MCMCResult(posterior_samples=posterior_samples)
    benchmark_result = BenchmarkResult(execution_time=execution_time, forward_calls=None, 
                                       backward_calls=None, 
                                       svi_result=None, mcmc_result=mcmc_result)
    return benchmark_result


def mcmc_counted(kernel, mcmc_params: dict, model, *model_args):
    """
    Perform MCMC inference with call counting.
    
    This function performs MCMC inference using NumPyro and counts forward and 
    backward calls for benchmarking purposes.
    
    Args:
        kernel: The MCMC kernel to use. It must have a potential_fn attribute.
        mcmc_params (dict): Parameters for the MCMC algorithm.
        model: The model to perform inference on. Expects a class with the model implemented in a 'model' method.
        *model_args: Arguments to pass to the model.
        
    Returns:
        BenchmarkResult: The benchmark results including execution time and call counts.
    """
    
    num_samples = mcmc_params.get("mcmc_samples", 300)
    warmup_steps = mcmc_params.get("mcmc_warmup_steps", 200)
    num_chains = mcmc_params.get("num_chains", 1)

    rng_key = jax.random.PRNGKey(0)

    # ---- initialize model ----
    model_info = initialize_model(
        rng_key,
        model.model,
        model_args=model_args,
        dynamic_args=False,
        validate_grad=True,
    )

    _, potential_fn, postprocess_fn, model_trace = model_info
    # ---- run MCMC ----
    # When passing a direct potential_fn to NUTS, MCMC.init() needs init_params consistent with
    # the potential_fn inputs. initialize_model returned useful init info; we can extract the init
    # "z" from model_info.param_info (ParamInfo) for init params.
    init_params_info = model_info.param_info
    z_init = init_params_info.z

    # Create wrapped potential + private counters
    counted_pot, counters = numpyro_utils.make_counted_potential(potential_fn)

    # Set up MCMC
    # todo: the kernel must provide a potential_fn attribute. 
    kernel = kernel.__class__(potential_fn=counted_pot)
    # kernel = NUTS(potential_fn=counted_pot)
    # kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=warmup_steps, num_samples=num_samples, num_chains=num_chains)

    # Replicate the initial parameters across chains
    def replicate_chain(x):
        return jax.numpy.repeat(jax.numpy.expand_dims(x, axis=0), mcmc.num_chains, axis=0)
    if mcmc.num_chains > 1:
        z_init = jax.tree.map(replicate_chain, z_init)

    # Run sampling 
    _, rng_key = jax.random.split(rng_key)
    start_time = time.perf_counter()
    mcmc.run(rng_key, init_params=z_init)
    execution_time = time.perf_counter() - start_time

    posterior_samples = mcmc.get_samples()
    mcmc_result = MCMCResult(posterior_samples=posterior_samples)
    benchmark_result = BenchmarkResult(execution_time=execution_time, forward_calls=counters["fwd"], 
                                       backward_calls=counters["bwd"], 
                                       svi_result=None, mcmc_result=mcmc_result)
    return benchmark_result
