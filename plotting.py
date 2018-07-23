import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="darkgrid")
sns.set_palette("tab10")

### Plotting
def plot_accuracies_per_tag(title, **kwargs):
    if kwargs is None:
        print("Need accuracies to plot")
        return

    classes = set(sum([list(kwargs[exp].keys()) for exp in kwargs], []))
    classes = list(classes)
    classes.remove('__OVERALL__')
    classes = ['Overall'] + sorted(classes)

    experiments = {}
    for exp in kwargs:
        experiments[exp] = {k: kwargs[exp][k] for k in kwargs[exp] if k != '__OVERALL__'}
        experiments[exp]['Overall'] = kwargs[exp]['__OVERALL__']

    df = pd.DataFrame([{
            'experiment': exp,
            **experiments[exp]
        } for exp in experiments])

    df = pd.melt(df, id_vars="experiment", var_name="type", value_name="accuracy")

    fig, ax = plt.subplots(1,1, figsize=(16,6))
    plt.sca(ax)
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylim((0.5, 1.05))
    sns.factorplot(ax=ax, x='type', y='accuracy', hue='experiment', 
                    data=df, kind='bar', order=classes, legend_out=True)
    
    handles, labels = ax.get_legend_handles_labels()
    l = ax.legend(handles=handles, labels=labels, title=title, frameon=True, fancybox=True, framealpha=0.8, facecolor="#FFFFFF")

    return fig

def plot_distributedness(title, top_neurons_per_tag):
    fig, ax = plt.subplots(1,1, figsize=(16,6))

    tags = sorted(top_neurons_per_tag.keys())
    number_of_neurons = [len(top_neurons_per_tag[tag]) for tag in tags]
    ax = sns.barplot(x=tags, y=number_of_neurons, color="yellowgreen")
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylabel("Number of Important Neurons")
    ax.set_title(title)

def plot_accuracies(title, overall_acc,
        top_10_acc, random_10_acc, bottom_10_acc,
        top_15_acc, random_15_acc, bottom_15_acc,
        top_20_acc, random_20_acc, bottom_20_acc):
    exps = ["Full", " ",
        "Top 10%", "Random 10%", "Bottom 10%", "  ",
        "Top 15%", "Random 15%", "Bottom 15%", "   ",
        "Top 20%", "Random 20%", "Bottom 20%"]

    concise_exps = ["", "Full", " ",
        "  ", "10% Neurons", "   ", "    ",
        "     ", "15% Neurons", "      ", "       ",
        "        ", "20% Neurons", "         ", "          "]

    accs = np.array([0, overall_acc, 0,
        top_10_acc, random_10_acc, bottom_10_acc, 0, 
        top_15_acc, random_15_acc, bottom_15_acc, 0,
        top_20_acc, random_20_acc, bottom_20_acc, 0])

    min_value = np.min(accs[accs>0.01])

    colors = [(0,0,0), 'limegreen', (0,0,0),
            'dodgerblue', 'gold', 'orangered', 'g',
            'dodgerblue', 'gold', 'orangered', 'g',
            'dodgerblue', 'gold', 'orangered', 'g', (0,0,0)]
    ax = sns.barplot(x=concise_exps, y=accs, palette=colors)

    ax.set_ylim((max(min_value-0.1,0), 1.04))

    ax.set_title(title)
    handles = [ax.patches[1], ax.patches[3], ax.patches[4], ax.patches[5]]
    labels = ["All Neurons", "Top Neurons", "Random Neurons", "Bottom Neurons"]
    ax.legend(handles=handles, labels=labels, frameon=True, fancybox=True, ncol=2, framealpha=0.8, facecolor="#FFFFFF")
