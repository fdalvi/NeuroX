import math
import numpy as np

from imblearn.under_sampling import RandomUnderSampler

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


def get_progress_bar():
    if isnotebook():
        from tqdm import tqdm_notebook as progressbar
    else:
        from tqdm import tqdm as progressbar
        
    return progressbar

## Train helpers
def batch_generator(X, y, batch_size=32):
    start_idx = 0
    while start_idx < X.shape[0]:
        yield X[start_idx : start_idx + batch_size], y[
            start_idx : start_idx + batch_size
        ]
        start_idx = start_idx + batch_size

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


def save_model(model, mappings):
    """
    Saves a model and its associated mappings as a pkl object
    """
    pass