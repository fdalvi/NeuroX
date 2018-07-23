import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import torch
import torch.nn as nn

from torch.autograd import Variable
from tqdm import tqdm, tqdm_notebook, tnrange

sns.set(style="darkgrid")
sns.set_palette("tab10")

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        out = self.linear(x)
        return out

## Data Preprocessing
def keep_specific_neurons(X, neuron_list):
    return X[:, neuron_list]

## Regularizers
def l1_penalty(var):
    return torch.abs(var).sum()

def l2_penalty(var):
    return torch.sqrt(torch.pow(var, 2).sum())

## Train helpers
def batch_generator(X, y, batch_size=32):
    start_idx = 0
    while start_idx < X.shape[0]:
        yield X[start_idx:start_idx+batch_size], y[start_idx:start_idx+batch_size]
        start_idx = start_idx + batch_size

## Training and evaluation
def train_logreg_model(X_train, y_train, 
                        lambda_l1=None, lambda_l2=None, 
                        num_epochs=10, batch_size=32):
    if (lambda_l1 is None or
        lambda_l2 is None):
        print("Please provide regularizer weights")
        return

    print("Creating model...")
    num_classes = len(set(y_train))
    print("Number of training instances:", X_train.shape[0])
    print("Number of classes:", num_classes)

    model = LogisticRegression(X_train.shape[1], num_classes)

    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters())

    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train)

    for epoch in range(num_epochs):
        num_tokens = 0
        avg_loss = 0
        for inputs, labels in tqdm_notebook(batch_generator(X_tensor, y_tensor, batch_size=batch_size), desc = 'epoch [%d/%d]'%(epoch+1, num_epochs)):
            num_tokens += inputs.shape[0]
            inputs = Variable(inputs)
            labels = Variable(labels)
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            weights = list(model.parameters())[0]
            
            loss = criterion(outputs, labels) #+ lambda_l1 * l1_penalty(weights) + lambda_l2 * l2_penalty(weights)
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()

        print ('Epoch: [%d/%d], Loss: %.4f' 
               % (epoch+1, num_epochs, avg_loss/num_tokens))

    return model

def evaluate_model(model, X, y, idx_to_class=None):
    # Test the Model
    correct = 0
    wrong = 0

    class_correct = {}
    class_wrong = {}

    for inputs, labels in tqdm_notebook(batch_generator(torch.from_numpy(X), torch.from_numpy(y)), desc = 'Evaluating'):
        inputs = Variable(inputs)
        labels = Variable(labels)

        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        
        for i in range(0, len(predicted)):
            idx = labels[i].item()
            if idx_to_class:
                key = idx_to_class[idx]
            else:
                key = idx

            if predicted[i] == idx:
                class_correct[key] = class_correct.get(key, 0) + 1
            else:
                class_wrong[key] = class_wrong.get(key, 0) + 1
                    
        correct += (predicted == labels.data).sum()
        wrong += (predicted != labels.data).sum()

    correct = correct.item()
    wrong = wrong.item()

    assert(correct == sum([class_correct[k] for k in class_correct]))
    assert(wrong == sum([class_wrong[k] for k in class_wrong]))
        
    print ('Number of correctly predicted instances: ', correct)
    print ('Number of incorrectly predicted instances: ', wrong)
    print('Accuracy of the model: %0.2f %%' % (100 * correct / (correct+wrong)))
    
    class_accuracies = {}
    class_accuracies['__OVERALL__'] = correct/(correct+wrong)

    for i in idx_to_class:
        c = idx_to_class[i]
        total = class_correct.get(c, 0) + class_wrong.get(c, 0)
        if total == 0:
            class_accuracies[c] = 0
        else:
            class_accuracies[c] = class_correct.get(c, 0)/total
    return class_accuracies

### Neuron selection

# returns set of all top neurons, as well as top neurons per
# class based on the percentage of mass covered in the weight
# distribution
# Distributed tasks will have more top neurons than focused ones
def get_top_neurons(model, percentage, class_to_idx):
    weights = np.abs(list(model.parameters())[0].data.numpy())
    top_neurons = {}
    for c in class_to_idx:
        total_mass = np.sum(weights[class_to_idx[c], :])
        sort_idx = np.argsort(weights[class_to_idx[c], :])[::-1]
        cum_sums = np.cumsum(weights[class_to_idx[c], sort_idx])
        top_neurons[c] = sort_idx[np.where(cum_sums < total_mass * percentage)[0]]

    top_neurons_union = set()
    for k in top_neurons:
        for t_n in top_neurons[k]:
            top_neurons_union.add(t_n)

    return np.array(list(top_neurons_union)), top_neurons

# returns set of all top neurons, as well as top neurons per
# class based on the threshold
# Distributed tasks will have more top neurons than focused ones
def get_top_neurons_hard_threshold(model, threshold, class_to_idx):
    weights = np.abs(list(model.parameters())[0].data.numpy())
    top_neurons = {}
    for c in class_to_idx:
        top_neurons[c] = np.where(weights[class_to_idx[c], :] > np.max(weights[class_to_idx[c], :])/threshold)[0]

    top_neurons_union = set()
    for k in top_neurons:
        for t_n in top_neurons[k]:
            top_neurons_union.add(t_n)

    return np.array(list(top_neurons_union)), top_neurons

# returns set of all bottom neurons, as well as bottom neurons per
# class based on the percentage of total neurons
# The percentage is (almost) evenly divided within each class
# The equal division leads to slighty fewer total neurons because
# of overlap between classes
def get_bottom_neurons(model, percentage, class_to_idx):
    weights = np.abs(list(model.parameters())[0].data.numpy())

    class_percentage = percentage/len(class_to_idx)

    bottom_neurons = {}
    for c in class_to_idx:
        idx = np.argsort(weights[class_to_idx[c], :])

        # randomly round to maintain closer total number
        num_elements = class_percentage * weights.shape[1]
        num_elements = num_elements + (random.random() < (num_elements - int(num_elements)))

        bottom_neurons[c] = idx[:int(num_elements)]

    bottom_neurons_union = set()
    for k in bottom_neurons:
        for t_n in bottom_neurons[k]:
            bottom_neurons_union.add(t_n)

    return np.array(list(bottom_neurons_union)), bottom_neurons

# returns set of random neurons, based on a percentage
def get_random_neurons(model, percentage):
    weights = np.abs(list(model.parameters())[0].data.numpy())

    mask = np.random.random((weights.shape[1],))
    idx = np.where(mask<=percentage)[0]

    return idx

# ordering is the global ordering of neurons
# cutoffs is how groups of neurons were added to the ordering
#   - ordering within each chunk is _random_
#   - increase the search_stride to get smaller chunks, and consequently
#       better orderings
def get_neuron_ordering(model, class_to_idx, search_stride=100):
    num_neurons = list(model.parameters())[0].data.numpy().shape[1]
    neuron_orderings = [get_top_neurons(model, p/search_stride, class_to_idx)[0] for p in tqdm_notebook(range(search_stride+1))]

    considered_neurons = set()
    ordering = []
    cutoffs = []
    for local_ordering in neuron_orderings:
        local_ordering = list(local_ordering)
        new_neurons = set(local_ordering).difference(considered_neurons)
        ordering = ordering + list(new_neurons)
        considered_neurons = considered_neurons.union(new_neurons)

        cutoffs.append(len(ordering))
        
    return ordering, cutoffs

def get_neuron_ordering_granular(model, class_to_idx, granularity=50, search_stride=100):
    num_neurons = list(model.parameters())[0].data.numpy().shape[1]
    neuron_orderings = [get_top_neurons(model, p/search_stride, class_to_idx)[0] for p in tqdm_notebook(range(search_stride+1))]

    sliding_idx = 0
    considered_neurons = set()
    ordering = []
    cutoffs = []
    for i in range(0, num_neurons+1, granularity):
        while len(neuron_orderings[sliding_idx]) < i:
            sliding_idx = sliding_idx+1
        new_neurons = set(neuron_orderings[sliding_idx]).difference(considered_neurons)
        if len(new_neurons) != 0:
            ordering = ordering + list(new_neurons)
            considered_neurons = considered_neurons.union(new_neurons)

            cutoffs.append(len(ordering))
        
    return ordering, cutoffs



# returns a view with only selected neurons
def filter_activations_keep_neurons(neurons_to_keep, X):
    return X[:, neurons_to_keep]

# returns a view with all but selected neurons
def filter_activations_remove_neurons(neurons_to_remove, X):
    neurons_to_keep = np.arange(X.shape[1])
    neurons_to_keep[neurons_to_remove] = -1
    neurons_to_keep = np.where(neurons_to_keep != -1)[0]
    return X[:, neurons_to_keep]

# returns a new matrix of all zeros except selected neurons
def zero_out_activations_keep_neurons(neurons_to_keep, X):
    _X = np.zeros_like(X)

    _X[:, neurons_to_keep] = X[:, neurons_to_keep]

    return _X

# returns a new matrix of with all selected neurons as zeros
def zero_out_activations_remove_neurons(neurons_to_remove, X):
    _X = np.copy(X)

    _X[:, neurons_to_remove] = 0

    return _X

### Stats
def print_overall_stats(all_results):
    model = all_results['model']
    num_neurons = list(model.parameters())[0].data.numpy().shape[1]
    print("Overall accuracy: %0.02f%%"%(100*all_results['original_accs']['__OVERALL__']))

    print("")
    print("Global results")
    print("10% Neurons")
    print("\tKeep Top accuracy: %0.02f%%"%(100*all_results['global_results']['10%']['keep_top_accs']['__OVERALL__']))
    print("\tKeep Random accuracy: %0.02f%%"%(100*all_results['global_results']['10%']['keep_random_accs']['__OVERALL__']))
    print("\tKeep Bottom accuracy: %0.02f%%"%(100*all_results['global_results']['10%']['keep_bottom_accs']['__OVERALL__']))
    print("15% Neurons")
    print("\tKeep Top accuracy: %0.02f%%"%(100*all_results['global_results']['15%']['keep_top_accs']['__OVERALL__']))
    print("\tKeep Random accuracy: %0.02f%%"%(100*all_results['global_results']['15%']['keep_random_accs']['__OVERALL__']))
    print("\tKeep Bottom accuracy: %0.02f%%"%(100*all_results['global_results']['15%']['keep_bottom_accs']['__OVERALL__']))
    print("20% Neurons")
    print("\tKeep Top accuracy: %0.02f%%"%(100*all_results['global_results']['20%']['keep_top_accs']['__OVERALL__']))
    print("\tKeep Random accuracy: %0.02f%%"%(100*all_results['global_results']['20%']['keep_random_accs']['__OVERALL__']))
    print("\tKeep Bottom accuracy: %0.02f%%"%(100*all_results['global_results']['20%']['keep_bottom_accs']['__OVERALL__']))
    print("")
    print("Full order of neurons:")
    print(all_results['global_results']['ordering'])
    
    print("--------------------")
    print("")
    print("Local results")
    for idx, percentage in enumerate(all_results['local_results']['percentages']):
        print("Weight Mass percentage: %d%%"%(percentage*100))
        _, top_neurons, top_neurons_per_tag = all_results['local_results']['local_top_neurons'][idx]
        print("Percentage of all neurons: %0.0f%%"%(100*len(top_neurons)/num_neurons))
        print("Top Neurons:", sorted(top_neurons))
        print("")
        print("Top neurons per tag:")
        for tag in top_neurons_per_tag:
            print("\t" + tag + ":", sorted(top_neurons_per_tag[tag]))
            print("")

def print_machine_stats(all_results):
    model = all_results['model']
    num_neurons = list(model.parameters())[0].data.numpy().shape[1]
    print("Filtering out:")
    print("%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%s"%(
        100*all_results['original_accs']['__OVERALL__'],
        100*all_results['global_results']['10%']['keep_top_accs']['__OVERALL__'],
        100*all_results['global_results']['10%']['keep_random_accs']['__OVERALL__'],
        100*all_results['global_results']['10%']['keep_bottom_accs']['__OVERALL__'],
        100*all_results['global_results']['15%']['keep_top_accs']['__OVERALL__'],
        100*all_results['global_results']['15%']['keep_random_accs']['__OVERALL__'],
        100*all_results['global_results']['15%']['keep_bottom_accs']['__OVERALL__'],
        100*all_results['global_results']['20%']['keep_top_accs']['__OVERALL__'],
        100*all_results['global_results']['20%']['keep_random_accs']['__OVERALL__'],
        100*all_results['global_results']['20%']['keep_bottom_accs']['__OVERALL__'],
        str(all_results['global_results']['ordering'][:300])))
    print("\nZero out:")
    print("%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f"%(
        100*all_results['original_accs']['__OVERALL__'],
        100*all_results['global_results']['10%']['zero_out_top_accs']['__OVERALL__'],
        100*all_results['global_results']['10%']['zero_out_random_accs']['__OVERALL__'],
        100*all_results['global_results']['10%']['zero_out_bottom_accs']['__OVERALL__'],
        100*all_results['global_results']['15%']['zero_out_top_accs']['__OVERALL__'],
        100*all_results['global_results']['15%']['zero_out_random_accs']['__OVERALL__'],
        100*all_results['global_results']['15%']['zero_out_bottom_accs']['__OVERALL__'],
        100*all_results['global_results']['20%']['zero_out_top_accs']['__OVERALL__'],
        100*all_results['global_results']['20%']['zero_out_random_accs']['__OVERALL__'],
        100*all_results['global_results']['20%']['zero_out_bottom_accs']['__OVERALL__'],
        ))

    for idx, percentage in enumerate(all_results['local_results']['percentages']):
        print("\nLocal %d%%:"%(percentage*100))
        top_neurons = all_results['local_results']['local_top_neurons'][idx][1]
        top_neurons_per_tag = all_results['local_results']['local_top_neurons'][idx][2]
        top_neurons_per_tag_list = {k: list(v) for k,v in top_neurons_per_tag.items()}
        print("%0.2f%%\t%s\t%s"%(100*len(top_neurons)/num_neurons, str(sorted(top_neurons)), str(top_neurons_per_tag_list)))

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
