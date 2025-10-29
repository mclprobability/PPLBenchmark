import jax.experimental
import jax.numpy as jnp
from matplotlib.pylab import f
import numpyro
import numpyro.distributions as distnp

import numpyro.primitives
from torch import nn
import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.nn import PyroSample
import jax
from dataclasses import dataclass
from ppl_benchmark.core.counter import Counter
from ppl_benchmark.models import base_model 

# Bayesian Linear Regression model in numpyro


@dataclass(frozen=True)
class BayesianRegressionNumPyroFunctional:
    in_features: int = 3
    out_features: int = 1
    w_loc: float = 0.0
    w_scale: float = 1.0
    b_loc: float = 0.0
    b_scale: float = 10.0
    s_scale: float = 2.0
    backward_counter = {"value":0} # not really frozen and not pure functional... 


    def model(self, x, y=None):
        linear_weight = numpyro.sample(
            "linear_weight",
            distnp.Normal(self.w_loc, self.w_scale).expand([self.out_features, self.in_features]).to_event(2),
        )
        linear_bias = numpyro.sample("linear_bias", distnp.Normal(self.b_loc, self.b_scale).expand([self.out_features]).to_event(1))
        sigma = numpyro.sample("sigma", distnp.HalfNormal(self.s_scale))
        mean = jnp.dot(x, linear_weight.T) + linear_bias

        # define forward counter as mutable state. this is the standard and preferred way to track a state during SVI inference. 
        # This state is reset for every iteration in MCMC, as the concept of a "running state" does not make sense in MCMC sampling. 
        # Hence, we can only use it for SVI. 
        forward_counter = numpyro.primitives.mutable("forward_counter", {"value": 0})

        # custom function to count forward and backward calls. Must be defined inside the model to enable access to the counters. 
        # The counters must not be passed as arguments, as this would break jax's functional data flow. 
        @jax.custom_vjp
        def counted_identity(x):
            return x

        def counted_fwd(x):
            forward_counter["value"] += 1
            return x, None
        
        def update_callback():
            self.backward_counter["value"] +=1
            return None
        
        def counted_bwd(_, g):
            # this callback is executed on the host and allows side effects
            jax.experimental.io_callback(update_callback, None)
            return (g,)
        counted_identity.defvjp(counted_fwd, counted_bwd)

        mean = counted_identity(mean)

        # Match the squeeze(-1) behavior from the Pyro example if out_features=1
        if self.out_features == 1:
            mean = jnp.squeeze(mean, axis=-1)
            # We also need to adapt Y if it was passed (N, 1) and we squeezed mean to (N,)
            if y is not None and y.ndim > 1 and y.shape[-1] == 1:
                y = jnp.squeeze(y, axis=-1)
    
        with numpyro.plate("data", x.shape[0]):
            return numpyro.sample("obs", distnp.Normal(mean, sigma), obs=y)
    

# Bayesian Linear Regression model in pyro
class BayesianRegressionPyro(base_model.BenchmarkPyroModule):
    def __init__(self, 
                 in_features = 3, 
                 out_features = 1,
                 w_loc = 0., 
                 w_scale = 1.,
                 b_loc = 0.,
                 b_scale = 10.,
                 s_scale = 2.):
        super().__init__()
        self.linear = PyroModule[torch.nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(w_loc, w_scale).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(b_loc, b_scale).expand([out_features]).to_event(1))
        self.sigma = PyroSample(dist.HalfNormal(s_scale))

    
    def forward(self, x, y=None):
        sigma = self.sigma

        mean = self.linear(x).squeeze(-1)
        # call the parent forward to trigger benchmark counting
        mean = super().forward(mean)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)

        return mean