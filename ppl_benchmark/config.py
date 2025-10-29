from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any

from ppl_benchmark.utils.configuration import load_yaml_config, CONFIG_PATH


class Model(Enum):
    BAYESIAN_LINEAR_REG = "BayesianLinearReg"
    EIGHT_SCHOOLS = "8Schools"


class Framework(Enum):
    PYRO = "Pyro"
    NUMPYRO = "NumPyro"


class InferenceRoutine(Enum):
    SVI = "Svi"
    MCMC = "Mcmc"


def _str_to_enum(enum_cls: Enum, value: Any):
    if isinstance(value, enum_cls):
        return value
    if value is None:
        return None
    # accept either member name or member value
    for member in enum_cls:
        if isinstance(value, str) and (value == member.name or value == member.value):
            return member
    raise ValueError(f"Cannot convert {value!r} to {enum_cls}")


@dataclass
class Config:
    # experiment choices
    e_model: Model = Model.BAYESIAN_LINEAR_REG
    e_framework: Framework = Framework.PYRO
    # renamed from e_interference_routine -> e_inference_routine
    e_inference_routine: InferenceRoutine = InferenceRoutine.SVI

    # SVI inference params
    svi_iterations: int = 200
    learning_rate: float = 0.01
    num_particles: int = 1

    # MCMC inference params
    mcmc_samples: int = 100
    mcmc_warmup_steps: int = 20
    num_chains: int = 1
    
    # paths and runtime
    seed: int = 1
    smoke_test_env_var: str = "CI"

    @classmethod
    def load_from_yaml(cls, filename: str = "base/ppl_benchmark.yml") -> "Config":
        # load yaml values (if present) and merge into defaults
        cfg = cls()
        raw = load_yaml_config(configfile=filename, location=CONFIG_PATH)
        if not raw:
            return cfg

        # map simple scalar values
        if "e_model" in raw:
            cfg.e_model = _str_to_enum(Model, raw.get("e_model"))
        if "e_framework" in raw:
            cfg.e_framework = _str_to_enum(Framework, raw.get("e_framework"))
        if "e_inference_routine" in raw:
            cfg.e_inference_routine = _str_to_enum(InferenceRoutine, raw.get("e_inference_routine"))


        for k, v in raw.items():
            setattr(cfg, k, v)


        return cfg


__all__ = ["Config", "Model", "Framework", "InferenceRoutine"]
