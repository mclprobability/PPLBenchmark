"""
Utility functions for NumPyro backend.

This module provides utility functions for the NumPyro backend, 
particularly for creating counted potential functions that track 
forward and backward calls.
"""

import jax
from jax import debug


def make_counted_potential(potential_fn):
    """
    Factory function that builds a counted potential function and counters.
    
    This function creates a wrapped potential function that counts forward 
    and backward calls, which is useful for benchmarking purposes.
    
    Args:
        potential_fn: The original potential function to wrap.
        
    Returns:
        tuple: A tuple containing:
            - wrapped_potential_fn: The wrapped potential function that counts calls.
            - counters_dict: A dictionary with 'fwd' and 'bwd' keys tracking 
              forward and backward calls respectively.
    """

    # local mutable counters (Python side)
    counters = {"fwd": 0, "bwd": 0}

    # callback functions capture counters by closure
    def tap_fwd(_arg, _transforms=None):
        counters["fwd"] += 1

    def tap_bwd(_arg, _transforms=None):
        counters["bwd"] += 1

    @jax.custom_vjp
    def counted_potential(z):
        """
        Counted potential function with custom VJP.
        
        Args:
            z: Input to the potential function.
            
        Returns:
            The result of the potential function.
        """
        debug.callback(tap_fwd, None)
        return potential_fn(z)

    def counted_fwd(z):
        val = potential_fn(z)
        debug.callback(tap_fwd, None)
        return val, (z,)

    def counted_bwd(res, g):
        (z_res,) = res
        _, vjp_fun = jax.vjp(potential_fn, z_res)
        (z_cot,) = vjp_fun(g)
        debug.callback(tap_bwd, None)
        return (z_cot,)

    counted_potential.defvjp(counted_fwd, counted_bwd)
    return counted_potential, counters
