import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def graph_violins(hour_counts, hour_blocks = np.arange(24).reshape(3, -1), fig_size = (12, 5), grid_size = 100):
    fig = plt.figure(figsize = fig_size)

    # Get the hours for each date from the time index. 
    hours = hour_counts.index.get_level_values('hour')

    n_plots = len(hour_blocks)
    if n_plots < 2:
        axes = [plt.gca()]
    else:
        axes = fig.subplots(n_plots, 1)
        axes = axes.reshape(-1)

    for ax, plot_hours in zip(axes, hour_blocks): 
        in_plot = [hour in plot_hours for hour in hours] 
        plot_data = hour_counts.loc[in_plot, ['count']]
        sns.violinplot(x = plot_data.index.get_level_values('hour'), y = np.log10(plot_data['count']), 
                       ax = ax, gridsize = grid_size, scale = 'width')
        ax.set_ylabel('Log10 Hourly Pickup Count')
        ax.set_xlabel('Hour (0 is Midnight)')
    plt.tight_layout()

    return fig, axes

def graph_distribution_by_hour(hour_counts, hour_blocks = np.arange(24).reshape(4, -1), #[np.arange(-5, 2)%24, np.arange(2, 7),
                               #np.arange(7, 13), np.arange(13, 24 - 5)], fig_size = (12, 10)):
                               fig_size = (12, 5), grid_size = 100):

    fig = plt.figure(figsize = fig_size)

    # Get the hours for each date from the time index. 
    hours = hour_counts.index.get_level_values('hour')

    n_plots = len(hour_blocks)
    if n_plots < 2:
        axes = [plt.gca()]
    else:
        axes = fig.subplots(int(n_plots / 2 + 0.8), 2)
        axes = axes.reshape(-1)

    for ax, plot_hours in zip(axes, hour_blocks): 
        for hour_i, plot_hour in enumerate(plot_hours):
            color = plt.get_cmap('winter')(hour_i / len(plot_hours))
            in_hour_pickups = hour_counts.loc[hours == plot_hour, 'count']
            sns.kdeplot(np.log10(in_hour_pickups), color = color, ax = ax, gridsize = grid_size)
        ax.legend(['hr ' + str(hr) for hr in plot_hours], loc = (1.1, 0.0))
        ax.set_xlabel('Log10 Hourly Pickup Count')
        ax.set_ylabel('Probability Distribution')
    plt.tight_layout()

    return fig, axes
