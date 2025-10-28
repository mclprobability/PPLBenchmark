
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyro
import torch
import sys
import os

# the directory of the current script
base_dir = os.path.dirname(__file__)  # directory where this script is
output_dir = os.path.join(base_dir, "plots")

def visualize_plot(df, path, nrows, timestamp):
    fig, ax = plt.subplots(figsize=(12, 6), sharey=True)

    sns.scatterplot(x=df["rugged"], y=df["rgdppc_2000"], ax=ax)
    ax.set(
        xlabel="x",
        ylabel="y",
        title="Data plot"
    )
    


    plot_folder = os.path.join(path, "plots/data_plot")
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder, exist_ok=True)



    data_plot_path = os.path.join(plot_folder, f"data_plot{nrows}_timestamp{timestamp}.png")
    plt.savefig(data_plot_path)
    plt.close()




def bayesian_loss_plot(losses, plot_path, nrows, iterations, timestamp):
           
    plot_folder = os.path.join(plot_path, "plots/bayesian_loss_plot")
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder, exist_ok=True)

    # plot drawing
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO Loss")
    plt.title("Bayesian linear regression loss plot")
    plt.grid(True)



    loss_plot_path = os.path.join(plot_folder, f"loss_plot_data{nrows}_iterations{iterations}_timestamp{timestamp}.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Plot saved as {loss_plot_path}")



def plot_regression_ci(predictions, plot_path, nrows, timestamp, iterations=None):
    
               
    plot_folder = os.path.join(plot_path, "plots/bayesian_reg_uncertainty")
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder, exist_ok=True)

    preds_sorted = predictions.sort_values(by="x_data")
    fig, ax = plt.subplots(figsize=(8, 6))

    
    ax.plot(preds_sorted["x_data"], preds_sorted["mu_mean"])
    ax.fill_between(preds_sorted["x_data"], preds_sorted["mu_perc_5"], preds_sorted["mu_perc_95"], alpha=0.5)
    ax.plot(preds_sorted["x_data"], preds_sorted["true_gdp"], "o")
    ax.set(xlabel="x", ylabel="y", title="Regression line 90% CI")

    if iterations is not None:
        reg_plot_path = os.path.join(plot_folder, f"bayesian_reg_uncertainty{nrows}_iterations{iterations}_timestamp{timestamp}.png")
    else: 
        reg_plot_path = os.path.join(plot_folder, f"bayesian_reg_uncertainty{nrows}_timestamp{timestamp}.png")
    plt.savefig(reg_plot_path)
    plt.close()
    print(f"Plot saved as {reg_plot_path}")




def plot_posterior_predictive(predictions, plot_path, nrows, timestamp, iterations=None):
                   
    plot_folder = os.path.join(plot_path, "plots/bayesian_reg_prediction")
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder, exist_ok=True)
    
    preds_sorted = predictions.sort_values(by="x_data")
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(preds_sorted["x_data"], preds_sorted["y_mean"])
    ax.fill_between(preds_sorted["x_data"], preds_sorted["y_perc_5"], preds_sorted["y_perc_95"], alpha=0.5)
    ax.plot(preds_sorted["x_data"], preds_sorted["true_gdp"], "o")
    ax.set(xlabel="x", ylabel="y", title="Posterior predictive distribution with 90% CI")


    if iterations is not None:
        reg_plot_path = os.path.join(plot_folder, f"bayesian_reg_bayesian_reg_prediction{nrows}_iterations{iterations}_timestamp{timestamp}.png")
    else: 
        reg_plot_path = os.path.join(plot_folder, f"bayesian_reg_bayesian_reg_prediction{nrows}_timestamp{timestamp}.png")
    
    plt.savefig(reg_plot_path)
    plt.close()
    print(f"Plot saved as {reg_plot_path}")




# for linear regression
def regression_fit_plot(df, x_data, linear_reg_model, path):
        fit = df.copy()
        fit["mean"] = linear_reg_model(x_data).detach().cpu().numpy()

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
        african_nations = fit[fit["cont_africa"] == 1]
        non_african_nations = fit[fit["cont_africa"] == 0]
        fig.suptitle("Regression Fit", fontsize=16)
        ax[0].plot(non_african_nations["x_data"], non_african_nations["rgdppc_2000"], "o")
        ax[0].plot(non_african_nations["x_data"], non_african_nations["mean"], linewidth=2)
        ax[0].set(xlabel="Terrain Ruggedness Index",
                ylabel="log GDP (2000)",
                title="Non African Nations")
        ax[1].plot(african_nations["x_data"], african_nations["rgdppc_2000"], "o")
        ax[1].plot(african_nations["x_data"], african_nations["mean"], linewidth=2)
        ax[1].set(xlabel="Terrain Ruggedness Index",
                ylabel="log GDP (2000)",
                title="African Nations");

        save_dir = os.path.join(output_dir, "regression")
        os.makedirs(save_dir, exist_ok=True)

        existing = [f for f in os.listdir(save_dir) if f.startswith("regression_plot") and f.endswith(".png")]
        next_id = len(existing) + 1

        r_plot_path = os.path.join(save_dir, f"regression_plot{next_id:03d}.png")
        plt.savefig(r_plot_path)
        plt.close(fig)
        print(f"Plot saved as {r_plot_path}")


