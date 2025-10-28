"""
Benchmark module for probabilistic programming frameworks.

This module provides functions to benchmark models using different
probabilistic programming frameworks (Pyro and NumPyro) and inference
routines (SVI and MCMC).
"""

import os

import numpyro
import pyro

from ppl_benchmark.config import Config, Framework, InferenceRoutine, Model
from ppl_benchmark.core.pyro_backend import (
    svi_inference as pyro_svi,
    mcmc_inference as pyro_mcmc,
)
from ppl_benchmark.core.numpyro_backend import (
    svi_inference as numpyro_svi,
    mcmc_inference as numpyro_mcmc,
)
from ppl_benchmark.core.results import BenchmarkResult
import os
import unittest

import ppl_benchmark.models.bayesian_regression as breg_model


def benchmark_model(
    inference_framework: Framework,
    inference_routine: InferenceRoutine,
    inference_method_param_dict,
    model,
    *model_args,
    guide=None,
    kernel=None,
) -> BenchmarkResult:
    """
    Benchmark a model using the specified framework and inference routine.

    This function runs inference on a model using either Pyro or NumPyro framework
    with either Stochastic Variational Inference (SVI) or Markov Chain Monte Carlo (MCMC)
    inference routine, and returns benchmark results.

    Args:
        inference_framework (Framework): The probabilistic programming framework to use
            (Framework.PYRO or Framework.NUMPYRO).
        inference_routine (InferenceRoutine): The inference routine to use
            (InferenceRoutine.SVI or InferenceRoutine.MCMC).
        inference_method_param_dict (dict): Parameters for the inference method.
        model: The model to benchmark.
        *model_args: Arguments to pass to the model.
        guide: The guide/variational distribution for SVI (default: None).
        kernel: The kernel for MCMC (default: None).

    Returns:
        BenchmarkResult: The benchmark results including execution time, number of model executions (forward counts), and number of gradient computations.

    Raises:
        NotImplementedError: If the specified framework or inference routine is not implemented.
        ValueError: If the benchmark result is None.
    """
    benchmark_result = None
    if inference_framework == Framework.PYRO:
        if inference_routine == InferenceRoutine.SVI:
            benchmark_result = pyro_svi.svi(
                inference_method_param_dict, model, guide, *model_args
            )
        elif inference_routine == InferenceRoutine.MCMC:
            benchmark_result = pyro_mcmc.mcmc(
                kernel, inference_method_param_dict, model, *model_args
            )
        else:
            raise NotImplementedError(
                f"Inference routine {inference_routine} in framework {inference_framework} not implemented"
            )
    elif inference_framework == Framework.NUMPYRO:
        if inference_routine == InferenceRoutine.SVI:
            benchmark_result = numpyro_svi.svi(
                inference_method_param_dict, model, guide, *model_args
            )
        elif inference_routine == InferenceRoutine.MCMC:
            benchmark_result = numpyro_mcmc.mcmc_counted(
                kernel, inference_method_param_dict, model, *model_args
            )
        else:
            raise NotImplementedError(
                f"Inference routine {inference_routine} in framework {inference_framework} not implemented"
            )
    else:
        raise NotImplementedError(
            f"Inference framework {inference_framework} not implemented"
        )

    if benchmark_result is None:
        raise ValueError("Benchmark result is None, something went wrong")
    return benchmark_result


def eval_benchmark_result(benchmark_result: BenchmarkResult):
    """
    Evaluate and print benchmark results.

    This function prints the benchmark results including execution time,
    forward and backward call counts, and details of SVI or MCMC results.

    Args:
        benchmark_result (BenchmarkResult): The benchmark results to evaluate.
    """
    print("Benchmark Result:")
    print(f"Inference runtime: {benchmark_result.execution_time:0.3f} s")
    print(f"Forward calls: {benchmark_result.forward_calls}")
    print(f"Backward calls: {benchmark_result.backward_calls}")
    if benchmark_result.svi_result is not None:
        print("SVI Result:")
        print("TODO: add SVI result details here")
    if benchmark_result.mcmc_result is not None:
        print("MCMC Result:")
        print("TODO: add MCMC result details here")
