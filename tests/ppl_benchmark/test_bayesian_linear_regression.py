from dataclasses import asdict
import os
import unittest
import jax
import numpyro
from parameterized import parameterized
import pyro

from ppl_benchmark.config import Config, Model, Framework, InferenceRoutine
from ppl_benchmark.core import benchmark as bm
import ppl_benchmark.utils.data_preparation as md
import ppl_benchmark.models.bayesian_regression as breg_model


ALL_FRAMEWORKS = [Framework.PYRO, Framework.NUMPYRO]
ALL_ROUTINES = [InferenceRoutine.SVI, InferenceRoutine.MCMC]

PARAMS = []
for f in ALL_FRAMEWORKS:
    for r in ALL_ROUTINES:
        PARAMS.append((f, r))


class TestBayesianLinearRegression(unittest.TestCase):

    @parameterized.expand(PARAMS)
    def test_bayesian_linear_regression(self, framework, inference_routine):
        """Run Bayesian Linear Regression pipeline for each
        (framework, inference routine) combination without invoking
        the top-level __main__ entrypoint.

        The test constructs a small Config and uses core.benchmark.benchmark_model
        to dispatch to the implemented inference helpers. Success is defined
        as completing without raising an exception.
        """
        # force CI/smoke behaviour used elsewhere in the suite
        os.environ["CI"] = "1"

        print(f"Framework: {framework}, Inference routine: {inference_routine}")
        # lightweight config
        cfg = Config()
        cfg.e_model = Model.BAYESIAN_LINEAR_REG
        cfg.e_framework = framework
        cfg.e_inference_routine = inference_routine
        cfg.svi_iterations = 20
        cfg.mcmc_samples = 15
        cfg.mcmc_warmup_steps = 5
        cfg.rows_nr = 10

        # prepare data similar to __main__.py
        # tests are located in tests/ppl_benchmark/, project root is two levels up
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        input_dir = os.path.join(project_root, "data", "0_raw")
        input_path = os.path.join(input_dir, "rugged_data.csv")
        df = md.transform_data(input_path, cfg.rows_nr)

        # prepare data for pyro and numpyro
        match framework:
            case Framework.PYRO:
                x_data, y_data = md.data_interaction(df)
                model = breg_model.BayesianRegressionPyro(**cfg.priors if hasattr(cfg, 'priors') else {})

            case Framework.NUMPYRO:
                x_data, y_data = md.data_interaction_np(df)
                model = breg_model.BayesianRegressionNumPyroFunctional(**cfg.priors if hasattr(cfg, 'priors') else {})

            case _:
                raise NotImplementedError(f"Framework {framework} not implemented in test")
        model_args = (x_data, y_data)

        # build model and model args compatible with benchmark.benchmark_model
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

        result = bm.benchmark_model(framework, inference_routine, 
                                    asdict(cfg), model, *model_args, 
                                    guide=guide, kernel=kernel, 
                                    )
        # print a small summary to help debugging in CI logs
        bm.eval_benchmark_result(result)

        # assert that all fields of BenchmarkResult are set
        self.assertIsNotNone(result.execution_time)
        self.assertIsNotNone(result.forward_calls)
        self.assertIsNotNone(result.backward_calls)
        self.assertGreater(result.forward_calls, 0.)
        self.assertGreater(result.backward_calls, 0.)
        if inference_routine == InferenceRoutine.SVI:
            self.assertIsNotNone(result.svi_result)
            self.assertIsNone(result.mcmc_result)
        elif inference_routine == InferenceRoutine.MCMC:
            self.assertIsNotNone(result.mcmc_result)
            self.assertIsNone(result.svi_result)
        else:  
            raise NotImplementedError(f"Inference routine {inference_routine} in framework {framework} not implemented")