from pyro.infer import Predictive
import torch
from pyro.infer import Predictive
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def compute_summary(samples):
    """
    Counts mean, std, 5% i 95% percentile for every variable in samples
    samples: dict with posteriori samples
    """
    site_stats = {}
    for k, v in samples.items():
        # Accept both torch tensors and numpy arrays
        if torch.is_tensor(v):
            arr = v.detach().cpu().numpy()
        else:
            arr = np.array(v)

        # compute statistics using numpy
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        p5 = np.percentile(arr, 5, axis=0)
        p95 = np.percentile(arr, 95, axis=0)

        # convert back to torch tensors for downstream compatibility
        site_stats[k] = {
            "mean" : torch.tensor(mean) if torch.is_tensor(v) else mean,
            "std": torch.tensor(std) if torch.is_tensor(v) else std,
            "5%": torch.tensor(p5) if torch.is_tensor(v) else p5,
            "95%": torch.tensor(p95) if torch.is_tensor(v) else p95,
        }

    return site_stats


def generate_predictive_samples(model, x_data, num_samples, guide=None, posterior_samples=None, return_sites=("linear.weight", "obs", "_RETURN")):

    if guide is None and posterior_samples is None:
        raise ValueError("Either guide (SVI) or posterior_samples (MCMC) must be provided.")

    if posterior_samples is not None:
        predictive = Predictive(model, posterior_samples=posterior_samples, return_sites=return_sites)
    else:
        predictive = Predictive(model, guide=guide, num_samples=num_samples, return_sites=return_sites)

    samples = predictive(x_data)
    pred_summary = compute_summary(samples)
    return samples, pred_summary

# def prepare_predictions_df(x_data, y_data, pred_summary):
# 
#     # DataFrame with predictions, 90% CI and real values
# 
#     mu = pred_summary["_RETURN"]
#     y = pred_summary["obs"]
#     predictions = pd.DataFrame({
#         "cont_africa": x_data[:, 0].cpu().numpy(),
#         "x_data": x_data[:, 1].cpu().numpy(),
#         "mu_mean": mu["mean"].cpu().numpy(),
#         "mu_perc_5": mu["5%"].cpu().numpy(),
#         "mu_perc_95": mu["95%"].cpu().numpy(),
#         "y_mean": y["mean"].cpu().numpy(),
#         "y_perc_5": y["5%"].cpu().numpy(),
#         "y_perc_95": y["95%"].cpu().numpy(),
#         "true_gdp": y_data.cpu().numpy(),
#     })
#     return predictions


