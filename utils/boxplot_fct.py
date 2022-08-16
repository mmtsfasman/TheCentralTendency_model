import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def boxplot_fct(list_of_data, filename, labels=None, colors=None, xlabel="prior reliance (H)", ylabel="subject-wise difference", title="", figsize=(10,10), ylim=None):

    if not colors:
        colors = sns.color_palette('husl', n_colors=len(list_of_data)) #plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig = plt.figure(figsize=figsize)
    parameters = {'xtick.labelsize': 24,
                  'ytick.labelsize': 24,
                  'axes.labelsize': 24,
                  'axes.titlesize': 30}
    plt.rcParams.update(parameters)
    ax = fig.add_subplot(111)
    for n, data in enumerate(list_of_data):
        # plt.plot(np.tile(n, (len(data),)), data, '*')
        mean_value = np.mean(data) # np.percentile(data, 50)
        upper_quantile = np.percentile(data, 90)
        lower_quantile = np.percentile(data, 10)
        lower_std_dev = np.mean(data) - np.sqrt(np.var(data))
        upper_std_dev = np.mean(data) + np.sqrt(np.var(data))
        ax.boxplot([lower_std_dev, lower_quantile, mean_value, upper_quantile, upper_std_dev], positions=[n+1])
        ax.scatter(np.tile(n+1, (len(data),)), data, color=colors[n])
    if labels:
        ax.set_xticklabels(labels)
    if ylim:
        plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
