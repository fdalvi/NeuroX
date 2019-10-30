import math
import numpy as np
import torch
import torch.nn as nn

from imblearn.under_sampling import RandomUnderSampler
from torch.autograd import Variable

from . import metrics


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if isnotebook():
    from tqdm import tqdm_notebook as progressbar
else:
    from tqdm import tqdm as progressbar


class LinearNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearNet, self).__init__()
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
        yield X[start_idx : start_idx + batch_size], y[
            start_idx : start_idx + batch_size
        ]
        start_idx = start_idx + batch_size


## Training and evaluation
def train_logreg_model(
    X_train,
    y_train,
    lambda_l1=None,
    lambda_l2=None,
    num_epochs=10,
    batch_size=32,
    learning_rate=0.001,
):
    return train_model(
        X_train,
        y_train,
        model_type="classification",
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


def train_linear_regression_model(
    X_train,
    y_train,
    lambda_l1=None,
    lambda_l2=None,
    num_epochs=10,
    batch_size=32,
    learning_rate=0.001,
):
    return train_model(
        X_train,
        y_train,
        model_type="regression",
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


def train_model(
    X_train,
    y_train,
    model_type,
    lambda_l1=None,
    lambda_l2=None,
    num_epochs=10,
    batch_size=32,
    learning_rate=0.001,
):
    print("Training %s model" % (model_type))
    # Check if we can use GPU's for training
    use_gpu = torch.cuda.is_available()

    if lambda_l1 is None or lambda_l2 is None:
        print("Please provide regularizer weights")
        return

    print("Creating model...")
    if model_type == "classification":
        num_classes = len(set(y_train))
        assert (
            num_classes > 1
        ), "Classification problem must have more than one target class"
    else:
        num_classes = 1
    print("Number of training instances:", X_train.shape[0])
    if model_type == "classification":
        print("Number of classes:", num_classes)

    model = LinearNet(X_train.shape[1], num_classes)
    if use_gpu:
        model = model.cuda()

    if model_type == "classification":
        criterion = nn.CrossEntropyLoss()
    elif model_type == "regression":
        criterion = nn.MSELoss()
    else:
        assert (
            model_type == "classification" or model_type == "regression"
        ), "Invalid model type"

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train)

    for epoch in range(num_epochs):
        num_tokens = 0
        avg_loss = 0
        for inputs, labels in progressbar(
            batch_generator(X_tensor, y_tensor, batch_size=batch_size),
            desc="epoch [%d/%d]" % (epoch + 1, num_epochs),
        ):
            num_tokens += inputs.shape[0]
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            inputs = Variable(inputs)
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            if model_type == "regression":
                outputs = outputs.squeeze()
            weights = list(model.parameters())[0]

            loss = (
                criterion(outputs, labels)
                + lambda_l1 * l1_penalty(weights)
                + lambda_l2 * l2_penalty(weights)
            )
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

        print(
            "Epoch: [%d/%d], Loss: %.4f"
            % (epoch + 1, num_epochs, avg_loss / num_tokens)
        )

    return model


def evaluate_model(
    model,
    X,
    y,
    idx_to_class=None,
    return_predictions=False,
    source_tokens=None,
    batch_size=32,
    metric="accuracy",
):
    # Check if we can use GPU's for training
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model = model.cuda()

    # Test the Model
    y_pred = []

    def source_generator():
        for s in source_tokens:
            for t in s:
                yield t

    src_words = source_generator()

    if return_predictions:
        predictions = []
    else:
        src_word = -1

    for inputs, labels in progressbar(
        batch_generator(
            torch.from_numpy(X), torch.from_numpy(y), batch_size=batch_size
        ),
        desc="Evaluating",
    ):
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs = Variable(inputs)
        labels = Variable(labels)

        outputs = model(inputs)

        if outputs.data.shape[1] == 1:
            # Regression
            predicted = outputs.data
        else:
            # Classification
            _, predicted = torch.max(outputs.data, 1)

        for i in range(0, len(predicted)):
            if source_tokens:
                src_word = next(src_words)
            else:
                src_word = src_word + 1

            idx = labels[i].item()
            if idx_to_class:
                key = idx_to_class[idx]
            else:
                key = idx

            y_pred.append(predicted[i])

            if return_predictions:
                predictions.append((src_word, key, predicted[i] == idx))

    y_pred = np.array(y_pred)

    result = metrics.compute_score(y_pred, y, metric)

    print("Score (%s) of the model: %0.2f" % (metric, result))

    class_scores = {}
    class_scores["__OVERALL__"] = result

    if idx_to_class:
        for i in idx_to_class:
            class_name = idx_to_class[i]
            class_instances_idx = np.where(y == i)[0]
            y_pred_filtered = y_pred[class_instances_idx]
            y_filtered = y[class_instances_idx]
            total = y_filtered.shape
            if total == 0:
                class_scores[class_name] = 0
            else:
                class_scores[class_name] = metrics.compute_score(
                    y_pred_filtered, y_filtered, metric
                )

    if return_predictions:
        return class_scores, predictions
    return class_scores


# multiclass utils
def tok2idx(toks):
    uniq_toks = set().union(*toks)
    return {p: idx for idx, p in enumerate(uniq_toks)}


def idx2tok(srcidx):
    return {v: k for k, v in srcidx.items()}


def count_target_words(tokens):
    return sum([len(t) for t in tokens["target"]])


def create_tensors(
    tokens, activations, task_specific_tag, mappings=None, model_type="classification"
):
    assert (
        model_type == "classification" or model_type == "regression"
    ), "Invalid model type"
    num_tokens = count_target_words(tokens)
    print("Number of tokens: ", num_tokens)

    num_neurons = activations[0].shape[1]

    source_tokens = tokens["source"]
    target_tokens = tokens["target"]

    ####### creating pos and source to index and reverse
    if mappings is not None:
        if model_type == "classification":
            label2idx, idx2label, src2idx, idx2src = mappings
        else:
            src2idx, idx2src = mappings
    else:
        if model_type == "classification":
            label2idx = tok2idx(target_tokens)
            idx2label = idx2tok(label2idx)
        src2idx = tok2idx(source_tokens)
        idx2src = idx2tok(src2idx)

    print("length of source dictionary: ", len(src2idx))
    if model_type == "classification":
        print("length of target dictionary: ", len(label2idx))

    X = np.zeros((num_tokens, num_neurons), dtype=np.float32)
    if model_type=="classification":
        y = np.zeros((num_tokens,), dtype=np.int)
    else:
        y = np.zeros((num_tokens,), dtype=np.float32)

    example_set = set()

    idx = 0
    for instance_idx, instance in enumerate(target_tokens):
        for token_idx, _ in enumerate(instance):
            if idx < num_tokens:
                X[idx] = activations[instance_idx][token_idx, :]

            example_set.add(source_tokens[instance_idx][token_idx])
            if model_type == "classification":
                if (
                    mappings is not None
                    and target_tokens[instance_idx][token_idx] not in label2idx
                ):
                    y[idx] = label2idx[task_specific_tag]
                else:
                    y[idx] = label2idx[target_tokens[instance_idx][token_idx]]
            elif model_type == "regression":
                y[idx] = float(target_tokens[instance_idx][token_idx])

            idx += 1

    print(idx)
    print("Total instances: %d" % (num_tokens))
    print(list(example_set)[:20])

    if model_type == "classification":
        return X, y, (label2idx, idx2label, src2idx, idx2src)
    return X, y, (src2idx, idx2src)


def filter_activations_by_layers(
    train_activations, test_activations, filter_layers, rnn_size, num_layers, is_brnn
):
    _layers = filter_layers.split(",")

    layer_prefixes = ["f"]
    if is_brnn:
        layer_prefixes = ["f", "b"]

    # FILTER settings
    layers = list(
        range(1, num_layers + 1)
    )  # choose which layers you need the activations
    filtered_train_activations = None
    filtered_test_activations = None

    layers_idx = []
    for brnn_idx, b in enumerate(layer_prefixes):
        for l in layers:
            if "%s%d" % (b, l) in _layers:
                start_idx = brnn_idx * (num_layers * rnn_size) + (l - 1) * rnn_size
                end_idx = brnn_idx * (num_layers * rnn_size) + (l) * rnn_size

                print(
                    "Including neurons from %s%d(#%d to #%d)"
                    % (b, l, start_idx, end_idx)
                )
                layers_idx.append(np.arange(start_idx, end_idx))
    layers_idx = np.concatenate(layers_idx)

    filtered_train_activations = [a[:, layers_idx] for a in train_activations]
    filtered_test_activations = [a[:, layers_idx] for a in test_activations]

    return filtered_train_activations, filtered_test_activations


### Neuron selection

# returns set of all top neurons, as well as top neurons per
# class based on the percentage of mass covered in the weight
# distribution
# Distributed tasks will have more top neurons than focused ones
def get_top_neurons(model, percentage, class_to_idx):
    weights = list(model.parameters())[0].data.cpu()
    weights = np.abs(weights.numpy())
    top_neurons = {}
    for c in class_to_idx:
        total_mass = np.sum(weights[class_to_idx[c], :])
        sort_idx = np.argsort(weights[class_to_idx[c], :])[::-1]
        cum_sums = np.cumsum(weights[class_to_idx[c], sort_idx])
        unselected_neurons = np.where(cum_sums >= total_mass * percentage)[0]
        if unselected_neurons.shape[0] == 0:
            selected_neurons = np.arange(cum_sums.shape[0])
        else:
            selected_neurons = np.arange(unselected_neurons[0] + 1)
        top_neurons[c] = sort_idx[selected_neurons]

    top_neurons_union = set()
    for k in top_neurons:
        for t_n in top_neurons[k]:
            top_neurons_union.add(t_n)

    return np.array(list(top_neurons_union)), top_neurons


# returns set of all top neurons, as well as top neurons per
# class based on the threshold
# Distributed tasks will have more top neurons than focused ones
def get_top_neurons_hard_threshold(model, threshold, class_to_idx):
    weights = list(model.parameters())[0].data.cpu()
    weights = np.abs(weights.numpy())
    top_neurons = {}
    for c in class_to_idx:
        top_neurons[c] = np.where(
            weights[class_to_idx[c], :]
            > np.max(weights[class_to_idx[c], :]) / threshold
        )[0]

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
    weights = list(model.parameters())[0].data.cpu()
    weights = np.abs(weights.numpy())

    bottom_neurons = {}
    for c in class_to_idx:
        total_mass = np.sum(weights[class_to_idx[c], :])
        sort_idx = np.argsort(weights[class_to_idx[c], :])
        cum_sums = np.cumsum(weights[class_to_idx[c], sort_idx])
        unselected_neurons = np.where(cum_sums >= total_mass * percentage)[0]
        if unselected_neurons.shape[0] == 0:
            selected_neurons = np.arange(cum_sums.shape[0])
        else:
            selected_neurons = np.arange(unselected_neurons[0] + 1)
        bottom_neurons[c] = sort_idx[selected_neurons]

    bottom_neurons_union = set()
    for k in bottom_neurons:
        for t_n in bottom_neurons[k]:
            bottom_neurons_union.add(t_n)

    return np.array(list(bottom_neurons_union)), bottom_neurons


# Returns num_bottom_neurons bottom neurons from the global ordering
def get_fixed_number_of_bottom_neurons(model, num_bottom_neurons, class_to_idx):
    ordering = get_neuron_ordering(model, class_to_idx)

    return ordering[-num_bottom_neurons:]


# returns set of random neurons, based on a percentage
def get_random_neurons(model, percentage):
    weights = list(model.parameters())[0].data.cpu()
    weights = np.abs(weights.numpy())

    mask = np.random.random((weights.shape[1],))
    idx = np.where(mask <= percentage)[0]

    return idx


# ordering is the global ordering of neurons
# cutoffs is how groups of neurons were added to the ordering
#   - ordering within each chunk is _random_
#   - increase the search_stride to get smaller chunks, and consequently
#       better orderings
def get_neuron_ordering(model, class_to_idx, search_stride=100):
    neuron_orderings = [
        get_top_neurons(model, p / search_stride, class_to_idx)[0]
        for p in progressbar(range(search_stride + 1))
    ]

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


def get_neuron_ordering_granular(
    model, class_to_idx, granularity=50, search_stride=100
):
    weights = list(model.parameters())[0].data.cpu()
    num_neurons = weights.numpy().shape[1]
    neuron_orderings = [
        get_top_neurons(model, p / search_stride, class_to_idx)[0]
        for p in progressbar(range(search_stride + 1))
    ]

    sliding_idx = 0
    considered_neurons = set()
    ordering = []
    cutoffs = []
    for i in range(0, num_neurons + 1, granularity):
        while len(neuron_orderings[sliding_idx]) < i:
            sliding_idx = sliding_idx + 1
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
    model = all_results["model"]
    weights = list(model.parameters())[0].data.cpu()
    num_neurons = weights.numpy().shape[1]
    print(
        "Overall accuracy: %0.02f%%"
        % (100 * all_results["original_accs"]["__OVERALL__"])
    )

    print("")
    print("Global results")
    print("10% Neurons")
    print(
        "\tKeep Top accuracy: %0.02f%%"
        % (100 * all_results["global_results"]["10%"]["keep_top_accs"]["__OVERALL__"])
    )
    print(
        "\tKeep Random accuracy: %0.02f%%"
        % (
            100
            * all_results["global_results"]["10%"]["keep_random_accs"]["__OVERALL__"]
        )
    )
    print(
        "\tKeep Bottom accuracy: %0.02f%%"
        % (
            100
            * all_results["global_results"]["10%"]["keep_bottom_accs"]["__OVERALL__"]
        )
    )
    print("15% Neurons")
    print(
        "\tKeep Top accuracy: %0.02f%%"
        % (100 * all_results["global_results"]["15%"]["keep_top_accs"]["__OVERALL__"])
    )
    print(
        "\tKeep Random accuracy: %0.02f%%"
        % (
            100
            * all_results["global_results"]["15%"]["keep_random_accs"]["__OVERALL__"]
        )
    )
    print(
        "\tKeep Bottom accuracy: %0.02f%%"
        % (
            100
            * all_results["global_results"]["15%"]["keep_bottom_accs"]["__OVERALL__"]
        )
    )
    print("20% Neurons")
    print(
        "\tKeep Top accuracy: %0.02f%%"
        % (100 * all_results["global_results"]["20%"]["keep_top_accs"]["__OVERALL__"])
    )
    print(
        "\tKeep Random accuracy: %0.02f%%"
        % (
            100
            * all_results["global_results"]["20%"]["keep_random_accs"]["__OVERALL__"]
        )
    )
    print(
        "\tKeep Bottom accuracy: %0.02f%%"
        % (
            100
            * all_results["global_results"]["20%"]["keep_bottom_accs"]["__OVERALL__"]
        )
    )
    print("")
    print("Full order of neurons:")
    print(all_results["global_results"]["ordering"])

    print("--------------------")
    print("")
    print("Local results")
    for idx, percentage in enumerate(all_results["local_results"]["percentages"]):
        print("Weight Mass percentage: %d%%" % (percentage * 100))
        _, top_neurons, top_neurons_per_tag = all_results["local_results"][
            "local_top_neurons"
        ][idx]
        print(
            "Percentage of all neurons: %0.0f%%"
            % (100 * len(top_neurons) / num_neurons)
        )
        print("Top Neurons:", sorted(top_neurons))
        print("")
        print("Top neurons per tag:")
        for tag in top_neurons_per_tag:
            print("\t" + tag + ":", sorted(top_neurons_per_tag[tag]))
            print("")


def print_machine_stats(all_results):
    model = all_results["model"]
    weights = list(model.parameters())[0].data.cpu()
    num_neurons = weights.numpy().shape[1]
    print("Filtering out:")
    print(
        "%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%s"
        % (
            100 * all_results["original_accs"]["__OVERALL__"],
            100 * all_results["global_results"]["10%"]["keep_top_accs"]["__OVERALL__"],
            100
            * all_results["global_results"]["10%"]["keep_random_accs"]["__OVERALL__"],
            100
            * all_results["global_results"]["10%"]["keep_bottom_accs"]["__OVERALL__"],
            100 * all_results["global_results"]["15%"]["keep_top_accs"]["__OVERALL__"],
            100
            * all_results["global_results"]["15%"]["keep_random_accs"]["__OVERALL__"],
            100
            * all_results["global_results"]["15%"]["keep_bottom_accs"]["__OVERALL__"],
            100 * all_results["global_results"]["20%"]["keep_top_accs"]["__OVERALL__"],
            100
            * all_results["global_results"]["20%"]["keep_random_accs"]["__OVERALL__"],
            100
            * all_results["global_results"]["20%"]["keep_bottom_accs"]["__OVERALL__"],
            str(all_results["global_results"]["ordering"][:300]),
        )
    )
    print("\nZero out:")
    print(
        "%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f"
        % (
            100 * all_results["original_accs"]["__OVERALL__"],
            100
            * all_results["global_results"]["10%"]["zero_out_top_accs"]["__OVERALL__"],
            100
            * all_results["global_results"]["10%"]["zero_out_random_accs"][
                "__OVERALL__"
            ],
            100
            * all_results["global_results"]["10%"]["zero_out_bottom_accs"][
                "__OVERALL__"
            ],
            100
            * all_results["global_results"]["15%"]["zero_out_top_accs"]["__OVERALL__"],
            100
            * all_results["global_results"]["15%"]["zero_out_random_accs"][
                "__OVERALL__"
            ],
            100
            * all_results["global_results"]["15%"]["zero_out_bottom_accs"][
                "__OVERALL__"
            ],
            100
            * all_results["global_results"]["20%"]["zero_out_top_accs"]["__OVERALL__"],
            100
            * all_results["global_results"]["20%"]["zero_out_random_accs"][
                "__OVERALL__"
            ],
            100
            * all_results["global_results"]["20%"]["zero_out_bottom_accs"][
                "__OVERALL__"
            ],
        )
    )

    for idx, percentage in enumerate(all_results["local_results"]["percentages"]):
        print("\nLocal %d%%:" % (percentage * 100))
        top_neurons = all_results["local_results"]["local_top_neurons"][idx][1]
        top_neurons_per_tag = all_results["local_results"]["local_top_neurons"][idx][2]
        top_neurons_per_tag_list = {k: list(v) for k, v in top_neurons_per_tag.items()}
        print(
            "%0.2f%%\t%s\t%s"
            % (
                100 * len(top_neurons) / num_neurons,
                str(sorted(top_neurons)),
                str(top_neurons_per_tag_list),
            )
        )


### Data Balancing functions
def balance_binary_class_data(X, y):
    rus = RandomUnderSampler()
    X_res, y_res = rus.fit_resample(X, y)

    return X_res, y_res


# Returns a balanced X,y pair
#  If num_required_instances is not provided, all classes are sampled to
#   match the minority class
#  If num_required_instances is provided, classes are sampled proportionally
#   Note: returned number of instances may not always be eqial to num_required_instances
#   because of rounding proportions
def balance_multi_class_data(X, y, num_required_instances=None):
    if num_required_instances:
        total = y.shape[0]
        unique, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(unique, counts))
        num_instances_per_class = {
            key: math.ceil(count / total * num_required_instances)
            for key, count in class_counts.items()
        }
        print(num_instances_per_class)
        rus = RandomUnderSampler(sampling_strategy=num_instances_per_class)
    else:
        rus = RandomUnderSampler()

    X_res, y_res = rus.fit_resample(X, y)

    return X_res, y_res
