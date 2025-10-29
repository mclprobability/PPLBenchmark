# Model Implementation Guidelines

This directory contains the probabilistic models used in the benchmark suite. To ensure consistent and measurable performance across different probabilistic programming languages (PPLs), all models must adhere to the following implementation guidelines.

---

## 🏗️ Core Requirements

All models in this directory must provide implementations for **both** the **Pyro** and **NumPyro** frameworks.

### 1. Pyro Models

Pyro models are designed to use `torch.nn.Module` conventions and the `PyroModule` class, leveraging PyTorch's automatic differentiation (autograd) system.

* **Base Class:** All Pyro models **must** be derived from `BenchmarkPyroModule` (defined in `base_model.py`).
    ```python
    from ppl_benchmark.models.base_model import BenchmarkPyroModule

    class MyModelPyro(BenchmarkPyroModule):
        # ...
    ```

* **Benchmark Call:** To enable the benchmarking of forward and backward passes during inference, you **must** call the parent's `forward` method (`super().forward(<var_requiring_gradient>)`) on the tensor whose gradient computation you want to track. This call inserts a custom PyTorch function (`ForwardCounterFunction`) into the computation graph that increments the model's internal counters on both the forward and backward passes.

    ```python
    def forward(self, x, y=None):
        # ... your model logic to compute mean ...
        mean = self.linear(x).squeeze(-1)

        # 🎯 Required: Call super().forward() on the variable that is
        # part of the gradient computation for the observation likelihood.
        mean = super().forward(mean)

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean
    ```

---

### 2. NumPyro Models

NumPyro models are based on JAX, which favors functional programming.

* **Functional Design:** NumPyro models **should** be implemented in a functional style. The only exception is the **`backward_counter`**, which must be tracked as mutable state outside the core model function (e.g., as a field on the class or dataclass) to track backward passes across iterations, as the forward counter is reset in MCMC.

* **Benchmark Integration:** To enable benchmarking of the forward and backward passes for SVI, you **must** integrate the counting mechanism using JAX's custom vector-Jacobian product (`jax.custom_vjp`).

    **Required Steps:**
    1.  Initialize the **`forward_counter`** as JAX mutable state inside the model (this state is tracked for SVI):
        ```python
        forward_counter = numpyro.primitives.mutable("forward_counter", {"value": 0})
        ```
    2.  Define and apply a `counted_identity` function using `jax.custom_vjp` within the model.
    3.  The `counted_fwd` definition must increment the `forward_counter`.
    4.  The `counted_bwd` definition must update the external **`backward_counter`** via a host callback (`jax.experimental.io_callback`).
    5.  Wrap the variable requiring gradient tracking (e.g., the mean of the likelihood distribution) with this function:
        ```python
        mean = counted_identity(mean)
        ```

    ***Example Snippet from `bayesian_regression.py`:***
    ```python
    @dataclass(frozen=True)
    class BayesianRegressionNumPyroFunctional:
        # ... (other fields) ...
        backward_counter = {"value":0} # Mutable state for backward count

        def model(self, x, y=None):
            # ... (sample priors) ...

            # 1. Initialize forward counter as JAX mutable state
            forward_counter = numpyro.primitives.mutable("forward_counter", {"value": 0})

            # 2. Define custom VJP function
            @jax.custom_vjp
            def counted_identity(x):
                return x

            def counted_fwd(x):
                forward_counter["value"] += 1 # Forward count increment
                return x, None
            
            def update_callback():
                self.backward_counter["value"] +=1 # Backward count increment
                return None
            
            def counted_bwd(_, g):
                jax.experimental.io_callback(update_callback, None)
                return (g,)
            
            counted_identity.defvjp(counted_fwd, counted_bwd)

            mean = jnp.dot(x, linear_weight.T) + linear_bias
            
            # 5. Apply the custom VJP here
            mean = counted_identity(mean) 

            # ... (sample likelihood) ...
            return numpyro.sample("obs", distnp.Normal(mean, sigma), obs=y)
    ```
This structure ensures that the core computational steps of the model are tracked consistently across both Pyro (PyTorch-based) and NumPyro (JAX-based) implementations, providing accurate metrics for the benchmark.