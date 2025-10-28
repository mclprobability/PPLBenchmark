"""
inference_results.py
--------------------
Small container dataclasses used as the canonical return types for inference helpers
(SVI / MCMC) in this project.

INSTRUCTIONS FOR AI AGENTS
- When updating inference helper functions (SVI / MCMC), prefer returning these
  dataclasses instead of bare tuples. Update all callsites within the same change
  so runtime behavior remains correct.
- For the current task the project maintainer explicitly does not require
  backward-compatibility for tuple-returning helpers; caller updates are expected
  to be made in follow-up changes.

Use these dataclasses as a small, stable contract for inference outputs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional



@dataclass
class SVIResult:
    """Container for results returned by SVI-style inference helpers.

    Fields are intentionally permissive (typing.Any) because values come from
    multiple probabilistic backends (Pyro, NumPyro, etc.). Keep field names
    stable; additional backend-specific data can go into ``extra``.
    """

    guide: Optional[Any] = None # variational posterior parameters for Pyro
    svi_state: Optional[Any] = None # variational posterior parameters for NumPyro
    losses: Optional[List[float]] = None


@dataclass
class MCMCResult:
    """Container for results returned by MCMC-style inference helpers.

    - ``posterior_samples``: backend samples object (dict/Mapping-like). Keep as
      Any for compatibility across frameworks.
    - ``extra``: optional map for backend-specific metadata (e.g. per-sample
      counters, step-size info, etc.).
    """

    posterior_samples: Any
   

@dataclass
class BenchmarkResult:
    """Base class for inference result containers.
        This class contains the metrics that we are interested in and additionally the inference routine results, if any.
    """
    execution_time: Optional[float] = None
    forward_calls: Optional[int] = None
    backward_calls: Optional[int] = None

    svi_result: Optional[SVIResult] = None
    mcmc_result: Optional[MCMCResult] = None

__all__ = ["SVIResult", "MCMCResult"]
