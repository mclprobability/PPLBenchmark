import jax
import jax.numpy as jnp
from functools import partial

    """
    Showcase how to implement a custom gradient in jax that updates its states over iterations. 
    This could be implemented in numpyro, to truely enable gradient-profiling methods. 
    A successful implementation requires changes in mcmc.run() and svi.update(), since the gradients are retrieved there.
    Atm, this is considered too much effort to be implemented. 
    Keep this file as inspiration, if the functionality shall be deeper integrated in the future. 
    """

def with_running_grad(fun):
    """
    A wrapper that adds a "running gradient" VJP to a function.

    Args:
      fun: The arbitrary differentiable function to wrap. It should take one
           JAX array and return a scalar or JAX array.

    Returns:
      A new function with a custom VJP. This function takes (state, primals)
      and returns (output, new_state).
    """

    # Create the function with the custom VJP decorator
    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    def fun_with_state(fun_static, state, x):
        # The forward pass simply computes the original function.
        # The state is passed through transparently for now.
        return fun_static(x)

    # --- Define the forward and backward passes for the custom VJP ---

    # 1. Forward Pass
    def fun_fwd(fun_static, state, x):
        # Execute the original function to get the output
        y = fun_static(x)
        # The 'residuals' are values needed by the backward pass.
        # We need the original input `x`, the current `state`, and the function `fun` itself.
        residuals = (state, x)
        # The forward pass must return the output and the residuals.
        return y, residuals

    # 2. Backward Pass
    def fun_bwd(fun_static, residuals, g):
        # Unpack the residuals
        state, x = residuals

        # g is the incoming cotangent (gradient from the next layer up)

        # Compute the VJP of the original function to get the gradient for this step
        _primals_out, vjp_fun = jax.vjp(fun_static, x)
        (grad_x,) = vjp_fun(g)

        # This is the core logic: update the state with the new gradient
        new_state = state + grad_x

        # The backward pass must return a tuple of gradients corresponding to the
        # inputs of the forward pass (`fun_static`, `state`, `x`).
        # - Gradient for `fun_static` is None (it's non-differentiable).
        # - Gradient for `state` is the *updated* state. This is the trick
        #   that threads the accumulated sum back out of the gradient computation.
        # - Gradient for `x` is the standard VJP result.
        return (new_state, grad_x)

    # Register the forward and backward functions with the custom VJP
    fun_with_state.defvjp(fun_fwd, fun_bwd)

    # Return a version of the function where the static `fun` argument is fixed.
    return partial(fun_with_state, fun)


# 1. Define an arbitrary differentiable function
def my_function(x):
    return jnp.sum(jnp.sin(x))


# 2. Create the version with our custom VJP
running_grad_sin = with_running_grad(my_function)


# 3. Define a loss function that uses it
#    Note how it takes state and returns the updated state.
def loss_fn(state, x):
    # running_grad_sin returns the function output, but we don't need it for the loss.
    # We only care about the state management during the backward pass.
    # We'll just return the original function's output as our loss.
    y = running_grad_sin(state, x)
    return y, state  # Return dummy state for has_aux=True


# 4. Use jax.value_and_grad
#    We want the gradient with respect to both state (arg 0) and x (arg 1).
#    `has_aux=True` tells JAX that our loss_fn returns a tuple of (output, auxiliary_data).
grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)

# --- Simulation ---
key = jax.random.PRNGKey(42)
x = jax.random.normal(key, (3,))

# Initialize the running gradient sum to zeros
# It must have the same shape as `x`
running_sum = jnp.zeros_like(x)

print(f"Initial input x: {x}")
print(f"Initial running sum: {running_sum}\n")

# Run the backward pass multiple times
for i in range(3):
    print(f"--- Iteration {i+1} ---")

    # The standard gradient of sin(x) is cos(x)
    # For jnp.sum(jnp.sin(x)), the VJP is just jnp.cos(x)
    expected_grad = jnp.cos(x)

    # Calculate the gradients
    (value, _), (grad_sum, grad_x) = grad_fn(running_sum, x)

    print(f"Function output value: {value:.4f}")
    print(f"Gradient w.r.t. x (grad_x): {grad_x}")
    print(f"Expected current grad:      {expected_grad}")
    print(f"Updated running sum (grad_sum): {grad_sum}\n")

    # Update the state for the next iteration
    running_sum = grad_sum
