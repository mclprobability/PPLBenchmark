import matplotlib.pyplot as plt
from numpyro.infer import Predictive
import jax.numpy as jnp
import jax
import os

import matplotlib.pyplot as plt
from numpyro.infer import Predictive
import jax.numpy as jnp
import jax
import os
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from numpyro.infer import Predictive
import jax.numpy as jnp
import jax
import os
import pandas as pd
import numpy as np
import ppl_benchmark.visualisation.plot_data_visualization as vis



import jax.numpy as jnp



def prepare_predictions_df_numpyro(x_data, y_data, pred_samples, ci=90):

    def summary_stats(samples, axis=0):
        mean = jnp.mean(samples, axis=axis)
        lower = jnp.percentile(samples, (100 - ci) / 2, axis=axis)
        upper = jnp.percentile(samples, 100 - (100 - ci) / 2, axis=axis)
        return mean, lower, upper

    mu_mean, mu_perc_5, mu_perc_95 = summary_stats(pred_samples["mu"])
    y_mean, y_perc_5, y_perc_95 = summary_stats(pred_samples["obs"])

    df = pd.DataFrame({
        "cont_africa": np.array(x_data[:, 0]).astype(int),
        "x_data": np.array(x_data[:, 1]),
        "mu_mean": np.array(mu_mean),
        "mu_perc_5": np.array(mu_perc_5),
        "mu_perc_95": np.array(mu_perc_95),
        "y_mean": np.array(y_mean),
        "y_perc_5": np.array(y_perc_5),
        "y_perc_95": np.array(y_perc_95),
        "true_gdp": np.array(y_data),
    })
    return df



# ========================
#  MCMC
# ========================
def predict_and_plot_mcmc(model, posterior_samples, x_data, y_data, res_path, data_size, timestamp, ci=90):
    predictive = Predictive(model, posterior_samples, return_sites=["obs", "mu"])
    rng_key = jax.random.PRNGKey(1)
    pred_samples = predictive(rng_key, x=x_data)

    df_preds = prepare_predictions_df_numpyro(x_data, y_data, pred_samples, ci)

    #vis.plot_regression_ci(df_preds, res_path, data_size, timestamp)        
    #vis.plot_posterior_predictive(df_preds, res_path, data_size, timestamp)   #, method_name="mcmc"
    return df_preds


# ========================
# SVI
# ========================


def predict_and_plot_svi(model, params, guide, x_data, y_data, res_path, data_size, timestamp, num_samples, ci=90):
    
    rng_key = jax.random.PRNGKey(1)
    
    predictive = Predictive(model, guide=guide, params=params, num_samples=num_samples)


    pred_samples = predictive(rng_key, x=x_data)
    

    df_preds = prepare_predictions_df_numpyro(x_data, y_data, pred_samples, ci)

    #vis.plot_regression_ci(df_preds, res_path, data_size, timestamp)

    #vis.plot_posterior_predictive(df_preds, res_path, data_size, timestamp)
    
    return df_preds