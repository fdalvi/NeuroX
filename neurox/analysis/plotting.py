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
    
    
def plot_bar_graph_layerwise_mean_weights(weights) :
    y = np.arange(0,13)
    weights = np.mean(np.abs(weights), axis = 0)
    weights = np.reshape(weights, (9984,1))
    weights_layerwise = np.split(weights, 13, axis=0)
    s = np.zeros((13,))
    for i in y :
        s[i] = np.sum(weights_layerwise[i], axis=0)
    plt.rcParams["figure.figsize"] = (5,5)
    plt.bar(y+1, s)
    plt.xlabel('Layers')
    plt.ylabel('Sum of weights')
    plt.xticks(y+1)
    plt.show()

def plot_weight_distribution(model_name, data, label, **kwargs) :
    label2idx = data["label2idx"]
    if (label=='total') :
        if (model_name=='SGL') : 
            model_weights = kwargs['weights']
            acc = kwargs['accuracies']
            l = []
            for i in label2idx :
                if (acc[i]['Train'] < 0.5) :
                    l.append(label2idx[i])
                    print(i+ " label deleted")
            model_weights = np.delete(model_weights, l, axis = 0)
            print(len(model_weights))
        elif (model_name=='NeuroX') :
            model_weights = get_weights(kwargs['model'], model_name)
            acc = kwargs['accuracies']
            l = []
            for i in label2idx :
                if (acc[i] < 0.5) :
                    l.append(label2idx[i])
                    print(i+ " label deleted")
            model_weights = np.delete(model_weights, l, axis = 0)
            print(len(model_weights))

        model_weights_1 = np.mean(np.abs(model_weights), axis = 0)
        y = np.arange(0,13)
        model_weights_3 = np.zeros((9984,))
        for i in y :
            k = np.arange((i*768),((i+1)*768))
            q = np.mean(model_weights_1[k])
            for p in k :
                model_weights_3[p] = q
            
        
        plt.rcParams["figure.figsize"] = (12,18)
        fig, ax = plt.subplots(4)
        ax[0].plot(model_weights_1)
        ax[0].set(title = 'Mean distribution of weights')
        ax[1].plot(model_weights_3.transpose())
        ax[1].set(title ='Mean weight per layer')
        ax[2].plot(sorted(model_weights_1))
        ax[2].set(title = 'Sorted weights')
        plt.show()
        sort_idx = np.argsort(model_weights)[::-1]
        print("\nNeurons with most weight : "+str(sort_idx[:10]))
    else :
        if (model_name=='SGL') : 
            model_weights = kwargs['weights']
        elif (model_name=='NeuroX') :
            model_weights = get_weights(kwargs['model'], model_name)
        
        
        plt.rcParams["figure.figsize"] = (10,12)
        fig, ax = plt.subplots(3)
        ax[0].plot(model_weights[label2idx[label]])
        ax[0].set(title = 'Signed weights for label '+label, xlabel = 'Neurons', ylabel = 'Neuron weights')
        ax[1].plot(np.abs(model_weights[label2idx[label]]))
        ax[1].set(title ='Absolute weights for label '+label, xlabel = 'Neurons', ylabel = 'Neuron weights')
        ax[2].plot(sorted(np.abs(model_weights[label2idx[label]])))
        ax[2].set(title ='Sorted weights for label '+label)
        fig.tight_layout(pad = 2.0)
        plt.show()  
        
def plot_weight_distribution_per_layer(model_name, data, label, layer, **kwargs) : 
    label2idx = data["label2idx"]
    if (model_name=='SGL') :
        model_weights = kwargs['weights']
    elif (model_name=='NeuroX') :
        model_weights = get_weights(kwargs['model'], model_name)
   
        
    if (label=='total') : 
        weights = np.mean(np.abs(model_weights), axis = 0)
    else :
        weights = np.abs(model_weights[label2idx[label]])
        print("For label "+label)
        
    weights = np.reshape(weights, (9984,1))
    weights_layerwise = np.split(weights, 13, axis=0)
        
    if (layer==0) : 
        plt.rcParams["figure.figsize"] = (15,15)
        fig, ax = plt.subplots(4,4)
        for i in range(0,13) :
            ax[int(i/4), int(i%4)].plot(weights_layerwise[i])
            ax[int(i/4), int(i%4)].set(title = 'Layer '+str(i+1)+' weights')
            mean = weights_layerwise[i].mean()
            sparsity_mask = weights_layerwise[i] > 1e-10 * mean
            print("\nFeatures selected from Layer "+str(i+1)+" : "+str(sparsity_mask.sum()))
            print("Spiky Neuron : "+str(np.argmax(weights_layerwise[i]))) 
        fig.tight_layout(pad= 2.0)
        plt.show()
            
    else : 
        plt.rcParams["figure.figsize"] = (5,5)
        mean = weights_layerwise[layer-1].mean()
        sparsity_mask = weights_layerwise[layer-1] > 1e-10 * mean
        print("Features selected from Layer "+str(layer)+" : "+str(sparsity_mask.sum()))
        plt.plot(weights_layerwise[layer-1])
        plt.title('Layer '+str(layer)+' weights')
        plt.show()
        
def sparsity(model_name, data, **kwargs) :
    label2idx = data["label2idx"]
    if (model_name=='SGL') :
        model_weights = kwargs['weights']
    elif (model_name=='NeuroX') :
        model_weights = get_weights(kwargs['model'], model_name)
   

    weights = np.abs(model_weights)
    scaler = MinMaxScaler()
    n_weights = scaler.fit_transform(weights.T).T
    weights = np.mean(n_weights, axis = 0)
    weight = np.reshape(weights, (9984,1))
    weights_layerwise = np.split(weight, 13, axis=0)
    sparsity =  np.zeros((13,))   
    total = int(0)
    for i in range(0,13) :
        sparsity_mask = weights_layerwise[i] > 0.005
        print("\nFeatures selected from Layer "+str(i+1)+" : "+str(sparsity_mask.sum()))
        print("Spiky Neuron : "+str(np.argmax(weights_layerwise[i]))) 
        sparsity[i] = sparsity_mask.sum()
        total = int(total + sparsity[i])
    p = np.arange(1,14)
    plt.rcParams["figure.figsize"] = (5,5)
    plt.plot(p,sparsity)
    plt.xticks(np.arange(1,14))
    plt.ylabel("No. of activated neurons")
    plt.xlabel("Layers")
    plt.title("Sparsity trend overall")

    plt.show()
    print("\n Selected features = "+str(total))
    
def neurons_per_label(data, labels, weights) :
    n = np.zeros((len(labels),))
    weight = np.abs(weights)
    scaler = MinMaxScaler()
    n_weights = scaler.fit_transform(weight.T).T
    for i,p in zip(labels,range(0,len(labels))): 
        label_weight = n_weights[data['label2idx'][i]]
        sparsity_mask = label_weight > 0.005 
        n[p] = sparsity_mask.sum()
    plt.rcParams["figure.figsize"] = (5,5)
    plt.bar(labels, n)
    plt.xlabel('Labels')
    plt.ylabel('Number of neurons')
    plt.xticks(rotation=70)
    plt.show()
    
    