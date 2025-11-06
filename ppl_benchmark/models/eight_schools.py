# a  new problem

# 8 schools model 

import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import ppl_benchmark.core.pyro_backend.mcmc_inference as pnuts

import jax.numpy as jnp
import numpyro
import numpyro.distributions as distnp
import ppl_benchmark.core.numpyro_backend.mcmc_inference as nmcmc
from ppl_benchmark.core.counter import Counter
from ppl_benchmark.models.base_model import BenchmarkPyroModule




class EightSchoolsPyro(BenchmarkPyroModule):
    def __init__(self, J=8, mu_loc=0., mu_scale=10., tau_loc=0., tau_scale=10.):

        super().__init__()
        self.J = J

        # priors
        self.mu = PyroSample(dist.Normal(mu_loc, mu_scale))
        self.tau = PyroSample(dist.HalfCauchy(scale=tau_scale))        

    def forward(self, y, sigma):
        mu = self.mu
        tau = self.tau

        with pyro.plate("schools", self.J):
            theta = pyro.sample("theta", dist.Normal(mu, tau))
            # how often is this called? we execute the plate J-times
            theta = super().forward(theta)
            obs = pyro.sample("obs", dist.Normal(theta, sigma), obs=y)

        return theta 




class EightSchoolsNumPyro:
    def __init__(self, J=8, mu_loc=0., mu_scale=10., tau_loc=0., tau_scale=10.):
        self.J = J
        self.mu = distnp.Normal(mu_loc, mu_scale)
        self.tau = distnp.HalfCauchy(tau_scale)
        self.forward_counter = Counter()

    def __call__(self, y, sigma):
        self.forward_counter.inc()
        mu = numpyro.sample("mu", self.mu)
        tau = numpyro.sample("tau", self.tau)
        with numpyro.plate('schools', self.J):
            # todo: CFi: does this count J-times (as the plate "is executed J-times")?
            theta = nmcmc.counted_identity(numpyro.sample('theta', distnp.Normal(mu, tau)))

            numpyro.sample('obs', distnp.Normal(theta, sigma), obs=y)

        return theta
