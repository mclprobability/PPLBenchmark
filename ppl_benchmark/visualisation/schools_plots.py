import pandas as pd
import matplotlib.pyplot as plt
import os



def plot_eight_schools(y_data, sigma, df, res_path, timestamp, data_size):

    schools = df["school"].values  

    fig, ax = plt.subplots()
    ax.bar(schools, y_data, yerr=sigma, capsize=5)

    ax.set_title("8 Schools treatment effects")
    ax.set_xlabel("School")
    ax.set_ylabel("Improvement")

    fig.set_size_inches(10, 8)

    plot_folder = os.path.join(res_path, "plots/data_plot_8schools")
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder, exist_ok=True)

    data_plot_path = os.path.join(plot_folder, f"data_plot_8s_{data_size}_timestamp{timestamp}.png")
    plt.savefig(data_plot_path)
