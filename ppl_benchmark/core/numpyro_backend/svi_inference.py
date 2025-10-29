import time
from datetime import datetime
import jax
from numpyro.optim import Adam
from numpyro.infer import SVI, Trace_ELBO

from ppl_benchmark.core.results import BenchmarkResult, SVIResult

def svi(svi_params: dict, model, guide, *model_args) -> BenchmarkResult:
    """ Perform SVI inference, time the inference routine and count forward and backward calls.

    Args:
        svi_params (dict): Parameters for the SVI algorithm.
        model (class): A numpyro model class with a 'model' method.
        guide (Any): A numpyro guide/variational distribution.
        *model_args: Arguments to pass to the model.

    Returns:
        BenchmarkResult: The benchmark results including execution time and call counts.
    """
    nr_iterations = svi_params.get("svi_iterations", 100)
    learning_rate = svi_params.get("learning_rate", 0.01)
    num_particles = svi_params.get("num_particles", 1)

    # SVI setup
    optimizer = Adam(step_size=learning_rate)
    loss = Trace_ELBO(num_particles=num_particles, vectorize_particles=False)
    svi = SVI(model.model, guide, optimizer, loss=loss)

    rng_key = jax.random.PRNGKey(0)
    svi_state = svi.init(rng_key, *model_args)

    losses = []

    # Training loop
    start_timer = time.perf_counter()
    update_fn = jax.jit(svi.update)
    for step in range(nr_iterations):
        # svi_state, loss = svi.update(svi_state, *model_args)
        svi_state, loss = update_fn(svi_state, *model_args)
        if step % 200 == 0:
            print(f"Step {step}, loss = {loss:.4f}")
        losses.append(loss)
    stop_timer = time.perf_counter()
    execution_time = stop_timer - start_timer
    
    if svi_state.mutable_state and "forward_counter" in svi_state.mutable_state:
        forward_count = svi_state.mutable_state["forward_counter"]["value"]
    else:
        forward_count = None

    if svi_state.mutable_state and "backward_counter" in svi_state.mutable_state:
        backward_count = svi_state.mutable_state["backward_counter"]["value"]
    else:
        try:
            backward_count = model.backward_counter["value"]
        except Exception as e:
            print("Model has no attribute 'backward_counter'")
            print(e)
            backward_count = None

    svi_result = SVIResult(guide=None, svi_state=svi_state, losses=losses)
    benchmark_result = BenchmarkResult( execution_time=execution_time, forward_calls=forward_count, backward_calls=backward_count, 
                                       svi_result=svi_result, mcmc_result=None)
    return benchmark_result