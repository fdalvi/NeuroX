import math
import numpy as np

from imblearn.under_sampling import RandomUnderSampler

def isnotebook():
    """
    Utility function to detect if the code being run is within a jupyter
    notebook. Useful to change progress indicators for example.

    Returns
    -------
    isnotebook : bool
        True if the function is being called inside a notebook, False otherwise.
    """
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
    """
    Utility function to get a progress bar depending on the environment the code
    is running in. A normal text-based progress bar is returned in normal
    shells, and a notebook widget-based progress bar is returned in jupyter
    notebooks.

    Returns
    -------
    progressbar : function
        The appropriate progressbar from the tqdm library.

    """
    if isnotebook():
        from tqdm import tqdm_notebook as progressbar
    else:
        from tqdm import tqdm as progressbar

    return progressbar

def batch_generator(X, y, batch_size=32):
    """
    Generator function to generate batches of data for training/evaluation.

    This function takes two tensors representing the activations and labels
    respectively, and yields batches of parallel data. The last batch may
    contain fewer than ``batch_size`` elements.

    Parameters
    ----------
    X : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually the
        output of ``interpretation.utils.create_tensors``
    y : numpy.ndarray
        Numpy Vector of size [``NUM_TOKENS``] with class labels for each input
        token. For classification, 0-indexed class labels for each input token
        are expected. For regression, a real value per input token is expected.
        Usually the output of ``interpretation.utils.create_tensors``
    batch_size : int, optional
        Number of samples to return in each call. Defaults to 32.

    Yields
    ------
    X_batch : numpy.ndarray
        Numpy Matrix of size [``batch_size`` x ``NUM_NEURONS``]. The final batch
        may have fewer elements than the requested ``batch_size``
    y_batch : numpy.ndarray
        Numpy Vector of size [``batch_size``]. The final batch may have fewer
        elements than the requested ``batch_size``
    """
    start_idx = 0
    while start_idx < X.shape[0]:
        yield X[start_idx : start_idx + batch_size], y[
            start_idx : start_idx + batch_size
        ]
        start_idx = start_idx + batch_size

def tok2idx(tokens):
    """
    Utility function to generate unique indices for a set of tokens.

    Parameters
    ----------
    tokens : list of lists
        List of sentences, where each sentence is a list of tokens. Usually
        returned from ``data.loader.load_data``

    Returns
    -------
    tok2idx_mapping : dict
        A dictionary with tokens as keys and a unique index for each token as
        values
    """
    uniq_tokens = set().union(*tokens)
    return {p: idx for idx, p in enumerate(uniq_tokens)}


def idx2tok(srcidx):
    """
    Utility function to an inverse mapping from a ``tok2idx`` mapping.

    Parameters
    ----------
    tok2idx_mapping : dict
        Token to index mapping, usually the output for
        ``interpretation.utils.tok2idx``.

    Returns
    -------
    idx2tok : dict
        A dictionary with unique indices as keys and their associated tokens as
        values
    """
    return {v: k for k, v in srcidx.items()}


def count_target_words(tokens):
    """
    Utility function to count the total number of tokens in a dataset.

    Parameters
    ----------
    tokens : list of lists
        List of sentences, where each sentence is a list of tokens. Usually
        returned from ``data.loader.load_data``

    Returns
    -------
    count : int
        Total number of tokens in the given ``tokens`` structure
    """
    return sum([len(t) for t in tokens["target"]])


def create_tensors(
    tokens, activations, task_specific_tag, mappings=None, task_type="classification"
):
    """
    Method to pre-process loaded datasets into tensors that can be used to train
    probes and perform analyis on. The input tokens are represented as list of
    sentences, where each sentence is a list of tokens. Each token also has
    an associated label. All tokens from all sentences are flattened into one
    dimension in the returned tensors. The returned tensors will thus have
    ``total_num_tokens`` rows.

    Parameters
    ----------
    tokens : list of lists
        List of sentences, where each sentence is a list of tokens. Usually
        returned from ``data.loader.load_data``
    activations : list of numpy.ndarray
        List of *sentence representations*, where each *sentence representation*
        is a numpy matrix of shape
        ``[num tokens in sentence x concatenated representation size]``. Usually
        retured from ``data.loader.load_activations``
    task_specific_tag : str
        Label to assign tokens with unseen labels. This is particularly useful
        if some labels are never seen during train, but are present in the dev
        or test set. This is usually set to the majority class in the task.
    mappings : list of dicts
        List of four python dicts: ``label2idx``, ``idx2label``, ``src2idx`` and
        ``idx2src`` for classification tasks. List of two dicts  ``src2idx`` and
        ``idx2src`` for regression tasks. Each dict represents either the
        mapping from class labels to indices and source tokens to indices or
        vice versa. Usually returned from a previous call to ``create_tensors``.
    task_type : str
        Either "classification" or "regression", indicate the kind of task that
        is being probed.

    Returns
    -------
    X : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]
    y : numpy.ndarray
        Numpy vector of size [``NUM_TOKENS``]
    mappings : list of dicts
        List of four python dicts: ``label2idx``, ``idx2label``, ``src2idx`` and
        ``idx2src`` for classification tasks. List of two dicts  ``src2idx`` and
        ``idx2src`` for regression tasks. Each dict represents either the
        mapping from class labels to indices and source tokens to indices or
        vice versa.

    Notes
    -----
    - ``mappings`` should be created exactly once, and should be reused for subsequent calls
    - For example, ``mappings`` can be created on train data, and the passed during the call for dev and test data.

    """
    assert (
        task_type == "classification" or task_type == "regression"
    ), "Invalid model type"
    num_tokens = count_target_words(tokens)
    print("Number of tokens: ", num_tokens)

    num_neurons = activations[0].shape[1]

    source_tokens = tokens["source"]
    target_tokens = tokens["target"]

    ####### creating pos and source to index and reverse
    if mappings is not None:
        if task_type == "classification":
            label2idx, idx2label, src2idx, idx2src = mappings
        else:
            src2idx, idx2src = mappings
    else:
        if task_type == "classification":
            label2idx = tok2idx(target_tokens)
            idx2label = idx2tok(label2idx)
        src2idx = tok2idx(source_tokens)
        idx2src = idx2tok(src2idx)

    print("length of source dictionary: ", len(src2idx))
    if task_type == "classification":
        print("length of target dictionary: ", len(label2idx))

    X = np.zeros((num_tokens, num_neurons), dtype=np.float32)
    if task_type=="classification":
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
            if task_type == "classification":
                if (
                    mappings is not None
                    and target_tokens[instance_idx][token_idx] not in label2idx
                ):
                    y[idx] = label2idx[task_specific_tag]
                else:
                    y[idx] = label2idx[target_tokens[instance_idx][token_idx]]
            elif task_type == "regression":
                y[idx] = float(target_tokens[instance_idx][token_idx])

            idx += 1

    print(idx)
    print("Total instances: %d" % (num_tokens))
    print(list(example_set)[:20])

    if task_type == "classification":
        return X, y, (label2idx, idx2label, src2idx, idx2src)
    return X, y, (src2idx, idx2src)


################################## Statictics ##################################
def print_overall_stats(all_results):
    """
    Method to pretty print overall results.

    .. warning::
        This method was primarily written to process results from internal
        scripts and pipelines.

    Parameters
    ----------
    all_results : dict
        Dictionary containing the probe, overall scores, scores from selected
        neurons, neuron ordering and neuron selections at various percentages

    """
    probe = all_results["probe"]
    weights = list(probe.parameters())[0].data.cpu()
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
    """
    Method to print overall results in tsv format.

    .. warning::
        This method was primarily written to process results from internal
        scripts and pipelines.

    Parameters
    ----------
    all_results : dict
        Dictionary containing the probe, overall scores, scores from selected
        neurons, neuron ordering and neuron selections at various percentages

    """

    probe = all_results["probe"]
    weights = list(probe.parameters())[0].data.cpu()
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


################################ Data Balancing ################################
def balance_binary_class_data(X, y):
    """
    Method to balance binary class data.

    The majority class is under-sampled randomly to match the minority class in
    it's size.

    Parameters
    ----------
    X : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually
        returned from ``interpretation.utils.create_tensors``
    y : numpy.ndarray
        Numpy vector of size [``NUM_TOKENS``]. Usually returned from
        ``interpretation.utils.create_tensors``

    Returns
    -------
    X_balanced : numpy.ndarray
        Numpy matrix of size [``NUM_BALANCED_TOKENS`` x ``NUM_NEURONS``]
    y_balanced : numpy.ndarray
        Numpy vector of size [``NUM_BALANCED_TOKENS``]

    """
    rus = RandomUnderSampler()
    X_res, y_res = rus.fit_resample(X, y)

    return X_res, y_res


def balance_multi_class_data(X, y, num_required_instances=None):
    """
    Method to balance multi class data.

    All classes are under-sampled randomly to match the minority class in
    their size. If ``num_required_instances`` is provided, all classes are
    sampled proportionally so that the total number of selected examples is
    approximately ``num_required_instances`` (because of rounding proportions).

    Parameters
    ----------
    X : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually
        returned from ``interpretation.utils.create_tensors``
    y : numpy.ndarray
        Numpy vector of size [``NUM_TOKENS``]. Usually returned from
        ``interpretation.utils.create_tensors``
    num_required_instances : int, optional
        Total number of required instances. All classes are sampled
        proportionally.

    Returns
    -------
    X_balanced : numpy.ndarray
        Numpy matrix of size [``NUM_BALANCED_TOKENS`` x ``NUM_NEURONS``]
    y_balanced : numpy.ndarray
        Numpy vector of size [``NUM_BALANCED_TOKENS``]

    """
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

def load_probe(probe_path):
    """
    Loads a probe and its associated mappings from probe_path

    .. warning::
        This method is currently not implemented.

    Parameters
    ----------
    probe_path : str
        Path to a pkl object saved by interpretation.utils.save_probe

    Returns
    -------
    probe : interpretation.linear_probe.LinearProbe
        Trained probe model
    mappings : list of dicts
        List of four python dicts: ``label2idx``, ``idx2label``, ``src2idx`` and
        ``idx2src`` for classification tasks. List of two dicts  ``src2idx`` and
        ``idx2src`` for regression tasks. Each dict represents either the
        mapping from class labels to indices and source tokens to indices or
        vice versa.

    """
    pass

def save_probe(probe_path, probe, mappings):
    """
    Saves a model and its associated mappings as a pkl object at probe_path

    .. warning::
        This method is currently not implemented.

    Parameters
    ----------
    probe_path : str
        Path to save a pkl object
    probe : interpretation.linear_probe.LinearProbe
        Trained probe model
    mappings : list of dicts
        List of four python dicts: ``label2idx``, ``idx2label``, ``src2idx`` and
        ``idx2src`` for classification tasks. List of two dicts  ``src2idx`` and
        ``idx2src`` for regression tasks. Each dict represents either the
        mapping from class labels to indices and source tokens to indices or
        vice versa.

    """
    pass