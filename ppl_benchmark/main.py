from dataclasses import asdict
import os
from pathlib import Path
import numpyro
from parameterized import parameterized
import pyro

from ppl_benchmark.config import Config, Model, Framework, InferenceRoutine
from ppl_benchmark.core import benchmark as bm
import ppl_benchmark.utils.data_preparation as md
import ppl_benchmark.models.bayesian_regression as breg_model

from ppl_benchmark.utils import PROJECT_ROOT, CONFIG, log, update_yaml_config, load_yaml_config

def main(*args, **kwargs):
    """"
    Demonstration of an example model definition and benchmarking run on all supported PPLs and inference frameworks.
    This function is similar to the unit test in tests/ppl_benchmark/test_bayesian_linear_regression.py.
    """

    # load the configuration from yaml files
    cfg = load_yaml_config("base/ppl_benchmark.yml")
    update_yaml_config(cfg, configfile="local/ppl_benchmark.yml")
    
    # prepare data 
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "data" / "0_raw"
    input_path = input_dir / "rugged_data.csv"
    df = md.transform_data(input_path, nrows=10) # prepare and limit data set

    # define all framework and inference routine combinations for which a benchmark should be run
    ALL_FRAMEWORKS = [Framework.PYRO, Framework.NUMPYRO]
    ALL_ROUTINES = [InferenceRoutine.SVI, InferenceRoutine.MCMC]
    PARAMS = []
    for f in ALL_FRAMEWORKS:
        for r in ALL_ROUTINES:
            PARAMS.append((f, r))

    for framework, inference_routine in PARAMS:
        print(f"Framework: {framework}, Inference routine: {inference_routine}")

        # prepare data and model for the respective framework. 
        match framework:
            case Framework.PYRO:
                x_data, y_data = md.data_interaction(df)
                model = breg_model.BayesianRegressionPyro(**cfg.get("priors") if hasattr(cfg, 'priors') else {})

            case Framework.NUMPYRO:
                x_data, y_data = md.data_interaction_np(df)
                model = breg_model.BayesianRegressionNumPyroFunctional(**cfg.priors if hasattr(cfg, 'priors') else {})

            case _:
                raise NotImplementedError(f"Framework {framework} not implemented in test")
        model_args = (x_data, y_data)

        # build the guide or kernel depending on framework and inference routine
        guide = None
        kernel = None
        if framework == Framework.PYRO:
            if inference_routine == InferenceRoutine.MCMC:
                kernel = pyro.infer.mcmc.NUTS(model, adapt_step_size=True)
            elif inference_routine == InferenceRoutine.SVI:
                guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
            else:
                raise NotImplementedError(f"Inference routine {inference_routine} in framework {framework} not implemented")    
        elif framework == Framework.NUMPYRO:
            if inference_routine == InferenceRoutine.MCMC:
                kernel = numpyro.infer.NUTS(model, adapt_step_size=True)
            elif inference_routine == InferenceRoutine.SVI:
                guide = numpyro.infer.autoguide.AutoDiagonalNormal(model.model)
            else:
                raise NotImplementedError(f"Inference routine {inference_routine} in framework {framework} not implemented")
        else:
            raise NotImplementedError(f"Framework {framework} not implemented in test")

        # run the benchmark (core functionality)
        result = bm.benchmark_model(framework, inference_routine, 
                                    cfg, model, *model_args, 
                                    guide=guide, kernel=kernel, 
                                    )
        # print the benchmark metrics. Add visualization or result saving as needed.
        bm.eval_benchmark_result(result)