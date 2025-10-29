import numpy as np
import pandas as pd
import torch

import jax.numpy as jnp

# data collection and preparation

# nrows to split the data
def transform_data(data_csv, nrows=None):
    data = pd.read_csv(data_csv, encoding="ISO-8859-1")
    
    # choosing columns
    df = data[["cont_africa", "rugged", "rgdppc_2000"]].copy()
    
    # remove African Nations (cont_africa == 1)
    df = df[df["cont_africa"] == 0]
    
    df = df[np.isfinite(df["rgdppc_2000"])]
    
    if nrows is not None:
        df = df.head(nrows)
    
    # log 
    df["rgdppc_2000"] = np.log(df["rgdppc_2000"])
    
    return df

def data_interaction(df):
    df["cont_africa_x_rugged"] = df["cont_africa"] * df["rugged"]
    data = torch.tensor(df[["cont_africa", "rugged", "cont_africa_x_rugged", "rgdppc_2000"]].values, dtype=torch.float)
    x_data, y_data = data[:, :-1], data[:, -1]
    return x_data, y_data




def data_interaction_np(df):
    df["cont_africa_x_rugged"] = df["cont_africa"] * df["rugged"]
    data = df[["cont_africa", "rugged", "cont_africa_x_rugged", "rgdppc_2000"]].to_numpy(dtype=float)
    data = jnp.array(data)
    x_data, y_data = data[:, :-1], data[:, -1]
    return x_data, y_data


def eight_schools_data(data_csv, nrows=None):
    data = pd.read_csv(data_csv, encoding="ISO-8859-1")
    df = data[["school", "improvement", "stderr"]]
    if nrows is not None:
        df = df.head(nrows)
    df = df[np.isfinite(df["improvement"])]
    df = df[np.isfinite(df["stderr"])]
    
    return df


def eight_schools_data_conversion(df, framework):
    if framework == "NumPyro":
        y = jnp.array(df["improvement"].values, dtype=jnp.float32)
        sigma = jnp.array(df["stderr"].values, dtype=jnp.float32)
    elif framework == "Pyro":
        y = torch.tensor(df["improvement"].values, dtype=torch.float32)
        sigma = torch.tensor(df["stderr"].values, dtype=torch.float32)

    return y, sigma








