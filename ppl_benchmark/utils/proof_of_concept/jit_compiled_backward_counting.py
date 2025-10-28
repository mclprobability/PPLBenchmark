# single_file_backward_counter.py
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS
from numpyro.optim import Adam
import numpy as np

# -----------------------
# Counter (no globals)
# -----------------------
class BackwardCounter:
    """Simple python-side counter object owned by the user."""
    def __init__(self):
        self.backward_calls = 0

    def inc(self, x=None):
        # id_tap passes an array; ignore its value and increment counter.
        self.backward_calls += 1
        # Return None is fine for id_tap handlers; we don't need a return value.
        return None

# -----------------------
# Factory to create track(x) bound to a callback
# -----------------------
def make_track_grad(tap_fn):
    """
    Returns a function `track(x)` such that:
    - forward: track(x) == x
    - backward: when the adjoint runs (VJP), tap_fn is called (via host_callback.id_tap).
    tap_fn is a Python callable (e.g., counter.inc) and is NOT traced by JAX.
    """
    @jax.custom_vjp
    def track(x):
        return x

    def fwd(x):
        # forward returns output and residuals (we don't need residuals)
        return x, None

    def bwd(_, g):
        # This function runs during the backward pass; we call id_tap to run
        # the python-side tap_fn with the cotangent g as argument.
        # Using id_tap inside the VJP ensures the callback runs during gradient computation.
        jax.debug.callback(tap_fn)
        # return cotangent for inputs (same as identity)
        return (g,)

    track.defvjp(fwd, bwd)
    return track

# -----------------------
# Model + Guide definitions
# -----------------------
def make_model_and_guide(track):
    """
    Returns model(data) and guide(data) that both make use of `track`.
    `track` should be a function track(x) -> x where calling it causes
    the backward callback to be called during adjoint.
    """

    def model(data):
        mu = numpyro.sample("mu", dist.Normal(0., 5.))
        sigma = numpyro.sample("sigma", dist.LogNormal(0., 1.))
        with numpyro.plate("data", data.shape[0]):
            numpyro.sample("obs", dist.Normal(mu, sigma), obs=data)

        # ensure this model's computational graph depends on latent vars
        # but doesn't change log_prob: add a zero factor that uses track(...)
        # we create a small expression involving mu and sigma and pass it through track,
        # then add it as a factor (log-probability additive). Because the value is zero,
        # it doesn't change target density, but it ensures the gradient w.r.t latent
        # variables flows through track so its backward VJP runs under MCMC.
        dummy = (mu * 0.0) + (sigma * 0.0)  # still depends on mu & sigma
        # Use factor to include dummy in potential energy graph (but with zero weight)
        numpyro.factor("grad_hook_in_model", track(dummy) * 0.0)

    guide = numpyro.infer.autoguide.AutoDiagonalNormal(model)

    return model, guide

# -----------------------
# Example usage: SVI and MCMC runs
# -----------------------
def run_example():
    # Create a counter object (user-owned, not global)
    counter = BackwardCounter()

    # Make a track function bound to counter.inc (no globals)
    track = make_track_grad(counter.inc)

    # Build model & guide with that track
    model, guide = make_model_and_guide(track)

    # Create synthetic data
    true_mu = 1.5
    true_sigma = 0.8
    rng = np.random.default_rng(0)
    data = jnp.array(rng.normal(true_mu, true_sigma, size=200))

    # -----------------------
    # Run SVI
    # -----------------------
    print("=== Running SVI ===")
    optimizer = Adam(step_size=0.01)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    svi_state = svi.init(jax.random.PRNGKey(0), data)

    # Run a few steps of SVI to cause gradient computations
    update_fn = jax.jit(svi.update)
    for i in range(150):
        svi_state, loss = update_fn(svi_state, data)
        if (i + 1) % 50 == 0:
            print(f"SVI iter {i+1}, loss = {loss:.3f}")

    print("After SVI: counter.backward_calls =", counter.backward_calls)
    # The counter should be > 0 because SVI uses gradients (backward) and our VJP tapped into them.

    # -----------------------
    # Run MCMC (NUTS)
    # -----------------------
    print("\n=== Running MCMC (NUTS) ===")
    # reset PRNG for MCMC
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=200, num_samples=300, num_chains=1)
    mcmc.run(jax.random.PRNGKey(1), data=data)

    # After MCMC, the HMC/NUTS kernel computed gradients of the potential energy many times;
    # our track() VJP should have been invoked during those gradients and incremented the counter.
    print("After MCMC: counter.backward_calls =", counter.backward_calls)

    # Inspect posterior summary
    print("\nMCMC summary (first 5 samples of mu):")
    samples = mcmc.get_samples()
    print(samples["mu"][:5])

    return counter, svi_state, mcmc

if __name__ == "__main__":
    counter, svi_state, mcmc = run_example()
