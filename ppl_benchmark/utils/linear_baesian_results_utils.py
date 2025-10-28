import torch
import pyro
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp


def extract_params(guide=None, posterior_samples=None):
    if guide is not None:
        guide.requires_grad_(False)

        # all parameters
        params = pyro.get_param_store()

        loc = params["AutoDiagonalNormal.loc"].detach()
        scale = params["AutoDiagonalNormal.scale"].detach()

        # tensor with 5 parameters 
        weight_loc = loc[:3].numpy()
        bias_loc = loc[3].item()
        sigma_loc = loc[4].item()

        weight_scale = scale[:3].numpy()
        bias_scale = scale[3].item()
        sigma_scale = scale[4].item()
    
    elif posterior_samples is not None: 
        # todo: this is not a location but a mean - lets name it appropriately
        weight_loc = posterior_samples['linear.weight'].mean(0).squeeze(0).detach().cpu().numpy()
        bias_loc   = posterior_samples['linear.bias'].mean().item()
        sigma_loc  = posterior_samples['sigma'].mean().item()

        weight_scale = posterior_samples['linear.weight'].std(0).squeeze(0).detach().cpu().numpy()
        bias_scale   = posterior_samples['linear.bias'].std().item()
        sigma_scale  = posterior_samples['sigma'].std().item()


    return {
        "weight_loc": weight_loc,
        "weight_scale": weight_scale,
        "bias_loc": bias_loc,
        "bias_scale": bias_scale,
        "sigma_loc": sigma_loc,
        "sigma_scale": sigma_scale
    }

def extract_params_numpyro(guide_params=None):

    weight_loc = np.array(guide_params['linear.weight_auto_loc'])
    weight_scale = np.array(guide_params['linear.weight_auto_scale'])

    bias_loc = float(guide_params['linear.bias_auto_loc'])
    bias_scale = float(guide_params['linear.bias_auto_scale'])

    sigma_loc = float(guide_params['sigma_auto_loc'])
    sigma_scale = float(guide_params['sigma_auto_scale'])

    return {
        "weight_loc": weight_loc,
        "weight_scale": weight_scale,
        "bias_loc": bias_loc,
        "bias_scale": bias_scale,
        "sigma_loc": sigma_loc,
        "sigma_scale": sigma_scale
    }


def save_results_to_csv(filepath, nrows, loss, avg_loss, loc, scale):
    row = {
        "nrows": nrows,
        "loss": loss,
        "avg_loss": avg_loss,
        "w1": loc[0],
        "w1_scale": scale[0],
        "w2": loc[1],
        "w2_scale": scale[1],
        "w3": loc[2],
        "w3_scale": scale[2],
        "bias": loc[3],
        "bias_scale": scale[3],
        "sigma": loc[4],
        "sigma_scale": scale[4]
    }

    file_exists = os.path.exists(filepath)
    df = pd.DataFrame([row])
    df.to_csv(filepath, mode='a', index=False, header=not file_exists)


def prior_distribution_results(r_path, i_path,  nrows,  priors,  forward_calls, backward_calls, e_time, time_stamp, loss=None, avg_loss=None, iterations_nr=None, num_samples_mcmc = None, warmup_steps = None, step_size = None):


    def filter_none(d):
        return {k: v for k, v in d.items() if v is not None}
    
    r_row = filter_none({
        "time_stamp": time_stamp,
        "nrows": nrows,
        "final_loss": loss,
        "avg_loss_per_data": avg_loss,
        "forward_calls": forward_calls,
        "backward_calls": backward_calls,
        "training_loop_execution_time[s]": e_time

    })

    i_row = filter_none({
        "time_stamp": time_stamp,
        "nrows": nrows,
        "iterations_nr": iterations_nr,
        "weight_dist": [priors['w_m'], priors['w_s']],
        "bias_dist": [priors['b_m'], priors['b_s']],
        "scale_dist": [priors['s_m'], priors['s_s']],
        **({"num_samples_mcmc": num_samples_mcmc,
        "warmup_steps": warmup_steps,} if num_samples_mcmc is not None else {})
    })

    

    r_folder = os.path.dirname(r_path)
    if r_folder and not os.path.exists(r_folder):
        os.makedirs(r_folder, exist_ok=True)

    file_exists = os.path.exists(r_path)
    df = pd.DataFrame([r_row])
    df.to_csv(r_path, mode='a', index=False, header=not file_exists)

    i_folder = os.path.dirname(i_path)
    if i_folder and not os.path.exists(i_folder):
        os.makedirs(i_folder, exist_ok=True)

    file_exists = os.path.exists(i_path)
    df = pd.DataFrame([i_row])
    df.to_csv(i_path, mode='a', index=False, header=not file_exists)

def save_o_row(o_path, nrows, time_stamp, iterations_nr, params):

    o_row = {
        "time_stamp": time_stamp,
        "nrows": nrows,
        "iterations_nr": iterations_nr,
        "w1": params['weight_loc'][0],
        "w2": params['weight_loc'][1],
        "w3": params['weight_loc'][2],
        "w1_scale": params['weight_scale'][0],
        "w2_scale": params['weight_scale'][1],
        "w3_scale": params['weight_scale'][2],
        "bias": params['bias_loc'],
        "bias_scale": params['bias_scale'],
        "sigma": params['sigma_loc'],
        "sigma_scale": params['sigma_scale']
    }

    o_folder = os.path.dirname(o_path)
    if o_folder and not os.path.exists(o_folder):
        os.makedirs(o_folder, exist_ok=True)

    file_exists = os.path.exists(o_path)
    df = pd.DataFrame([o_row])
    df.to_csv(o_path, mode='a', index=False, header=not file_exists)
    print(f"Optimized parameters saved to {o_path}")


def save_posterior_samples_both(file_path, posterior_samples, framework):

    folder = os.path.dirname(file_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    num_samples = list(posterior_samples.values())[0].shape[0]
    data = []

    for i in range(num_samples):
        row = {"sample_idx": i}

        # Obsługa bias
        if framework.lower() == "pyro":
            bias_val = posterior_samples['linear.bias'][i].detach().cpu().item()
            weight = posterior_samples['linear.weight'][i].detach().cpu().numpy().flatten()
            sigma_val = posterior_samples['sigma'][i].detach().cpu().item()
        elif framework.lower() == "numpyro":
            bias_val = np.array(posterior_samples['linear.bias'][i])
            if bias_val.ndim > 0:
                bias_val = bias_val.item()
            else:
                bias_val = float(bias_val)

            weight = np.array(posterior_samples['linear.weight'][i]).flatten()

            sigma_val = np.array(posterior_samples['sigma'][i])
            if sigma_val.ndim > 0:
                sigma_val = sigma_val.item()
            else:
                sigma_val = float(sigma_val)
        else:
            raise ValueError("Framework must be 'pyro' or 'numpyro'")

        row["linear.bias"] = bias_val
        for j in range(weight.shape[0]):
            row[f"linear.weight_{j}"] = weight[j]
        row["sigma"] = sigma_val

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Posterior samples saved to {file_path}")
