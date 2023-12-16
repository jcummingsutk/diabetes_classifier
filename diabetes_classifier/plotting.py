import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_binary_col(data: pd.DataFrame, col: str) -> plt.Figure:
    unique_values = list(data[col].unique())
    data1 = data[data[col] == np.max(unique_values)]
    data0 = data[data[col] == np.min(unique_values)]
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    sns.countplot(data, x=col, hue="Diabetes", ax=ax1)

    ax2 = fig.add_subplot(gs[1, 0])
    explode = [0.2, 0]
    perc_diab_given_0 = np.sum(data0["Diabetes"].to_numpy()) / data0.shape[0] * 100
    percents = [perc_diab_given_0, 100 - perc_diab_given_0]
    ax2.pie(
        percents,
        labels=["Diabetes", "No Diabetes"],
        explode=explode,
        autopct="%.0f%%",
    )
    ax2.set_title(f"{col}=0")

    ax3 = fig.add_subplot(gs[1, 1])
    perc_diab_given_1 = np.sum(data1["Diabetes"].to_numpy()) / data1.shape[0] * 100
    percents = [perc_diab_given_1, 100 - perc_diab_given_1]
    ax3.pie(
        percents,
        labels=["Diabetes", "No Diabetes"],
        explode=explode,
        autopct="%.0f%%",
    )
    ax3.set_title(f"{col}=1")
    return fig


def plot_numerical_cols(data: pd.DataFrame, col: str) -> plt.Figure:
    xs = []
    fracs_with_diabetes = []
    num_samples = []
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 1)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, :])
    for entry in np.sort(data[col].unique()):
        num_with_diabetes = np.sum(data.loc[data[col] == entry, "Diabetes"])
        total = data.loc[data[col] == entry].shape[0]
        frac_diabetes = num_with_diabetes / total
        num_samples.append(total)
        xs.append(entry)
        fracs_with_diabetes.append(frac_diabetes)
    ax0.set_ylabel("Number of Samples")
    ax1.set_xlabel(col)
    ax1.set_ylabel("Fraction With Diabetes")
    sns.scatterplot(x=xs, y=fracs_with_diabetes, ax=ax1)
    sns.scatterplot(x=xs, y=num_samples, ax=ax0)
    return fig
