import jax
import numpyro
import numpyro.primitives
from pyro.nn import PyroModule
import torch

from ppl_benchmark.core.counter import Counter

class ForwardCounterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, forward_counter: Counter, backward_counter: Counter):
        forward_counter.increment()
        #print("fc", forward_counter.get())
        ctx.backward_counter = backward_counter
        return input

    @staticmethod
    def backward(ctx, grad_output):
        ctx.backward_counter.increment()
        return grad_output.clone(), None, None

class BenchmarkPyroModule(PyroModule):

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self.forward_counter = Counter()
        self.backward_counter = Counter()

    def forward(self, x):
        return  ForwardCounterFunction.apply(x, self.forward_counter, self.backward_counter)
    
