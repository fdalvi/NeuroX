"""Module for layer and neuron level linear-probe based analysis.

This module contains functions to train, evaluate and use a linear probe for
both layer-wise and neuron-wise analysis.

.. seealso::
        `Dalvi, Fahim, et al. "What is one grain of sand in the desert? analyzing individual neurons in deep nlp models." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33. No. 01. 2019. <https://ojs.aaai.org/index.php/AAAI/article/view/4592>`_
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from . import metrics
from . import utils

class LinearProbe(nn.Module):
    """Torch model for linear probe"""
    def __init__(self, input_size, num_classes):
        """Initialize a linear model"""
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        """Run a forward pass on the model"""
        out = self.linear(x)
        return out

################################# Regularizers #################################
def l1_penalty(var):
    """
    L1/Lasso regularization penalty

    Parameters
    ----------
    var : torch.Variable
        Torch variable representing the weight matrix over which the penalty
        should be computed

    Returns
    -------
    penalty : torch.Variable
        Torch variable containing the penalty as a single floating point value

    """
    return torch.abs(var).sum()

def l2_penalty(var):
    """
    L2/Ridge regularization penalty.

    Parameters
    ----------
    var : torch.Variable
        Torch variable representing the weight matrix over which the penalty
        should be computed

    Returns
    -------
    penalty : torch.Variable
        Torch variable containing the penalty as a single floating point value

    Notes
    -----
    The penalty is derived from the L2-norm, which has a square root. The exact
    optimization can also be done without the square root, but this makes no
    difference in the actual output of the optimization because of the scaling
    factor used along with the penalty.

    """
    return torch.sqrt(torch.pow(var, 2).sum())

############################ Training and Evaluation ###########################
def _train_probe(
    X_train,
    y_train,
    task_type,
    lambda_l1=0,
    lambda_l2=0,
    num_epochs=10,
    batch_size=32,
    learning_rate=0.001,
):
    """
    Internal helper method to train a linear probe.

    This method is used internally for both classification and regression based
    tasks in order to train probes for them. A logistic regression model
    is trained with Cross Entropy loss for classification tasks and a linear
    regression model is trained with MSE loss for regression tasks. The
    optimizer used is Adam with default ``torch.optim`` hyperparameters.

    Parameters
    ----------
    X_train : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually the
        output of ``interpretation.utils.create_tensors``
    y_train : numpy.ndarray
        Numpy Vector of size [``NUM_TOKENS``] with class labels for each input
        token. For classification, 0-indexed class labels for each input token
        are expected. For regression, a real value per input token is expected.
        Usually the output of ``interpretation.utils.create_tensors``.
    task_type : str
        Either "classification" or "regression", indicate the kind of task that
        is being probed.
    lambda_l1 : float, optional
        L1 Penalty weight in the overall loss. Defaults to 0, i.e. no L1
        regularization
    lambda_l2 : float, optional
        L2 Penalty weight in the overall loss. Defaults to 0, i.e. no L2
        regularization
    num_epochs : int, optional
        Number of epochs to train the linear model for. Defaults to 10
    batch_size : int, optional
        Batch size for the input to the linear model. Defaults to 32
    learning_rate : float, optional
        Learning rate for optimizing the linear model. 

    Returns
    -------
    probe : interpretation.linear_probe.LinearProbe
        Trained probe for the given task.

    """
    progressbar = utils.get_progress_bar()
    print("Training %s probe" % (task_type))
    # Check if we can use GPU's for training
    use_gpu = torch.cuda.is_available()

    if lambda_l1 is None or lambda_l2 is None:
        print("Please provide regularizer weights")
        return

    print("Creating model...")
    if task_type == "classification":
        num_classes = len(set(y_train))
        assert (
            num_classes > 1
        ), "Classification problem must have more than one target class"
    else:
        num_classes = 1
    print("Number of training instances:", X_train.shape[0])
    if task_type == "classification":
        print("Number of classes:", num_classes)

    probe = LinearProbe(X_train.shape[1], num_classes)
    if use_gpu:
        probe = probe.cuda()

    if task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    elif task_type == "regression":
        criterion = nn.MSELoss()
    else:
        assert (
            task_type == "classification" or task_type == "regression"
        ), "Invalid task type"

    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train)

    for epoch in range(num_epochs):
        num_tokens = 0
        avg_loss = 0
        for inputs, labels in progressbar(
            utils.batch_generator(X_tensor, y_tensor, batch_size=batch_size),
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
            outputs = probe(inputs)
            if task_type == "regression":
                outputs = outputs.squeeze()
            weights = list(probe.parameters())[0]

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

    return probe

def train_logistic_regression_probe(
    X_train,
    y_train,
    lambda_l1=0,
    lambda_l2=0,
    num_epochs=10,
    batch_size=32,
    learning_rate=0.001,
):
    """
    Train a logistic regression probe.

    This method trains a linear classifier that can be used as a probe to perform
    neuron analysis. Use this method when the task that is being probed for is a
    classification task. A logistic regression model is trained with Cross
    Entropy loss. The optimizer used is Adam with default ``torch.optim``
    package hyperparameters.

    Parameters
    ----------
    X_train : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually the
        output of ``interpretation.utils.create_tensors``. ``dtype`` of the
        matrix must be ``np.float32``
    y_train : numpy.ndarray
        Numpy Vector with 0-indexed class labels for each input token. The size
        of the vector must be [``NUM_TOKENS``].  Usually the output of
        ``interpretation.utils.create_tensors``. Assumes that class labels are
        continuous from ``0`` to ``NUM_CLASSES-1``. ``dtype`` of the
        matrix must be ``np.int``
    lambda_l1 : float, optional
        L1 Penalty weight in the overall loss. Defaults to 0, i.e. no L1
        regularization
    lambda_l2 : float, optional
        L2 Penalty weight in the overall loss. Defaults to 0, i.e. no L2
        regularization
    num_epochs : int, optional
        Number of epochs to train the linear model for. Defaults to 10
    batch_size : int, optional
        Batch size for the input to the linear model. Defaults to 32
    learning_rate : float, optional
        Learning rate for optimizing the linear model. 

    Returns
    -------
    probe : interpretation.linear_probe.LinearProbe
        Trained probe for the given task.

    """
    return _train_probe(
        X_train,
        y_train,
        task_type="classification",
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


def train_linear_regression_probe(
    X_train,
    y_train,
    lambda_l1=None,
    lambda_l2=None,
    num_epochs=10,
    batch_size=32,
    learning_rate=0.001,
):
    """
    Train a linear regression probe.

    This method trains a linear classifier that can be used as a probe to perform
    neuron analysis. Use this method when the task that is being probed for is a
    regression task. A linear regression model is trained with MSE loss. The
    optimizer used is Adam with default ``torch.optim`` package hyperparameters.

    Parameters
    ----------
    X_train : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually the
        output of ``interpretation.utils.create_tensors``. ``dtype`` of the
        matrix must be ``np.float32``
    y_train : numpy.ndarray
        Numpy Vector with real-valued labels for each input token. The size
        of the vector must be [``NUM_TOKENS``].  Usually the output of
        ``interpretation.utils.create_tensors``. ``dtype`` of the
        matrix must be ``np.float32``
    lambda_l1 : float, optional
        L1 Penalty weight in the overall loss. Defaults to 0, i.e. no L1
        regularization
    lambda_l2 : float, optional
        L2 Penalty weight in the overall loss. Defaults to 0, i.e. no L2
        regularization
    num_epochs : int, optional
        Number of epochs to train the linear model for. Defaults to 10
    batch_size : int, optional
        Batch size for the input to the linear model. Defaults to 32
    learning_rate : float, optional
        Learning rate for optimizing the linear model. 

    Returns
    -------
    probe : interpretation.linear_probe.LinearProbe
        Trained probe for the given task.

    """
    return _train_probe(
        X_train,
        y_train,
        model_type="regression",
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


def evaluate_probe(
    probe,
    X,
    y,
    idx_to_class=None,
    return_predictions=False,
    source_tokens=None,
    batch_size=32,
    metric="accuracy",
):
    """
    Evaluates a trained probe.

    This method evaluates a trained probe on the given data, and supports
    several standard metrics.

    Parameters
    ----------
    probe : interpretation.linear_probe.LinearProbe
        Trained probe model
    X : numpy.ndarray
        Numpy Matrix of size [``NUM_TOKENS`` x ``NUM_NEURONS``]. Usually the
        output of ``interpretation.utils.create_tensors``. ``dtype`` of the
        matrix must be ``np.float32``
    y : numpy.ndarray
        Numpy Vector of size [``NUM_TOKENS``] with class labels for each input
        token. For classification, 0-indexed class labels for each input token
        are expected. For regression, a real value per input token is expected.
        Usually the output of ``interpretation.utils.create_tensors``
    idx_to_class : dict, optional
        Class index to name mapping. Usually returned by
        ``interpretation.utils.create_tensors``. If this mapping is provided,
        per-class metrics are also computed. Defaults to None.
    return_predictions : bool, optional
        If set to True, actual predictions are also returned along with scores
        for further use. Defaults to False.
    source_tokens : list of lists, optional
        List of all sentences, where each is a list of the tokens in that
        sentence. Usually returned by ``data.loader.load_data``. If provided and
        ``return_predictions`` is True, each prediction will be paired with its
        original token. Defaults to None.
    batch_size : int, optional
        Batch size for the input to the model. Defaults to 32
    metrics : str, optional
        Metric to use for evaluation scores. For supported metrics see
        ``interpretation.metrics``

    Returns
    -------
    scores : dict
        The overall score on the given data with the key ``__OVERALL__``. If
        ``idx_to_class`` mapping is provided, additional keys representing each
        class and their associated scores are also part of the dictionary.
    predictions : list of 3-tuples, optional
        If ``return_predictions`` is set to True, this list will contain a
        3-tuple for every input sample, representing 
        ``(source_token, predicted_class, was_predicted_correctly)``

    """
    progressbar = utils.get_progress_bar()

    # Check if we can use GPU's for evaluation
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        probe = probe.cuda()

    # Test the Model
    y_pred = []

    def source_generator():
        for s in source_tokens:
            for t in s:
                yield t

    src_words = source_generator()

    if return_predictions:
        predictions = []
        src_word = -1

    for inputs, labels in progressbar(
        utils.batch_generator(
            torch.from_numpy(X), torch.from_numpy(y), batch_size=batch_size
        ),
        desc="Evaluating",
    ):
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs = Variable(inputs)
        labels = Variable(labels)

        outputs = probe(inputs)

        if outputs.data.shape[1] == 1:
            # Regression
            predicted = outputs.data
        else:
            # Classification
            _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().numpy()

        for i in range(0, len(predicted)):
            idx = predicted[i]
            if idx_to_class:
                key = idx_to_class[idx]
            else:
                key = idx

            y_pred.append(predicted[i])

            if return_predictions:
                if source_tokens:
                    src_word = next(src_words)
                else:
                    src_word = src_word + 1
                predictions.append((src_word, key, labels[i].item() == idx))

    y_pred = np.array(y_pred)

    result = metrics.compute_score(y_pred, y, metric)

    print("Score (%s) of the probe: %0.2f" % (metric, result))

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

############################### Neuron Selection ###############################
def get_top_neurons(probe, percentage, class_to_idx):
    """
    Get top neurons from a trained probe.

    This method returns the set of all top neurons based on the given percentage.
    It also returns top neurons per class. All neurons (sorted by weight in
    ascending order) that account for ``percentage`` of the total weight mass
    are returned. See the given reference for the compcomplete selection algorithm
    description.

    .. seealso::
        `Dalvi, Fahim, et al. "What is one grain of sand in the desert? analyzing individual neurons in deep nlp models." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33. No. 01. 2019. <https://ojs.aaai.org/index.php/AAAI/article/view/4592>`_

    .. note::
        Absolute weight values are used for selection, instead of raw signed
        values

    Parameters
    ----------
    probe : interpretation.linear_probe.LinearProbe
        Trained probe model
    percentage : float
        Real number between 0 and 1, with 0 representing no weight mass and 1
        representing the entire weight mass, i.e. all neurons.
    class_to_idx : dict
        Class to class index mapping. Usually returned by
        ``interpretation.utils.create_tensors``.

    Returns
    -------
    overall_top_neurons : numpy.ndarray
        Numpy array with all top neurons
    top_neurons : dict
        Dictionary with top neurons for every class, with the class name as the
        key and ``numpy.ndarray`` of top neurons (for that class) as the value.

    Notes
    -----
    - One can expect distributed tasks to have more top neurons than focused tasks
    - One can also expect complex tasks to have more top neurons than simpler tasks

    """
    weights = list(probe.parameters())[0].data.cpu()
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


def get_top_neurons_hard_threshold(probe, fraction, class_to_idx):
    """
    Get top neurons from a trained probe based on the maximum weight.

    This method returns the set of all top neurons based on the given threshold.
    All neurons that have a weight above ``threshold * max_weight`` are
    considered as top neurons. It also returns top neurons per class. 

    .. note::
        Absolute weight values are used for selection, instead of raw signed
        values

    Parameters
    ----------
    probe : interpretation.linear_probe.LinearProbe
        Trained probe model
    fraction : float
        Fraction of maximum weight per class to use for selection
    class_to_idx : dict
        Class to class index mapping. Usually returned by
        ``interpretation.utils.create_tensors``.

    Returns
    -------
    overall_top_neurons : numpy.ndarray
        Numpy array with all top neurons
    top_neurons : dict
        Dictionary with top neurons for every class, with the class name as the
        key and ``numpy.ndarray`` of top neurons (for that class) as the value.

    """
    weights = list(probe.parameters())[0].data.cpu()
    weights = np.abs(weights.numpy())
    top_neurons = {}
    for c in class_to_idx:
        top_neurons[c] = np.where(
            weights[class_to_idx[c], :]
            > np.max(weights[class_to_idx[c], :]) / fraction
        )[0]

    top_neurons_union = set()
    for k in top_neurons:
        for t_n in top_neurons[k]:
            top_neurons_union.add(t_n)

    return np.array(list(top_neurons_union)), top_neurons


def get_bottom_neurons(probe, percentage, class_to_idx):
    """
    Get bottom neurons from a trained probe.

    Analogous to ``interpretation.linear_probe.get_top_neurons``. This method
    returns the set of all bottom neurons based on the given percentage.
    It also returns bottom neurons per class. All neurons (sorted by weight
    in ascending order) that account for ``percentage`` of the total weight mass
    are returned. See the given reference for the complete selection algorithm
    description.

    .. seealso::
        `Dalvi, Fahim, et al. "What is one grain of sand in the desert? analyzing individual neurons in deep nlp models." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33. No. 01. 2019. <https://ojs.aaai.org/index.php/AAAI/article/view/4592>`_

    .. note::
        Absolute weight values are used for selection, instead of raw signed
        values

    Parameters
    ----------
    probe : interpretation.linear_probe.LinearProbe
        Trained probe model
    percentage : float
        Real number between 0 and 1, with 0 representing no weight mass and 1
        representing the entire weight mass, i.e. all neurons.
    class_to_idx : dict
        Class to class index mapping. Usually returned by
        ``interpretation.utils.create_tensors``.

    Returns
    -------
    overall_bottom_neurons : numpy.ndarray
        Numpy array with all bottom neurons
    bottom_neurons : dict
        Dictionary with bottom neurons for every class, with the class name as the
        key and ``numpy.ndarray`` of bottom neurons (for that class) as the value.

    """
    weights = list(probe.parameters())[0].data.cpu()
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


def get_random_neurons(probe, probability):
    """
    Get random neurons from a trained probe.

    This method returns a random set of neurons based on the probability. Each
    neuron is either discarded or included based on a uniform random variable's
    value (included if its less than probability, discarded otherwise)

    Parameters
    ----------
    probe : interpretation.linear_probe.LinearProbe
        Trained probe model
    probability : float
        Real number between 0 and 1, with 0 representing no selection and 1
        representing selection of all neurons.

    Returns
    -------
    random_neurons : numpy.ndarray
        Numpy array with random neurons

    """
    weights = list(probe.parameters())[0].data.cpu()
    weights = np.abs(weights.numpy())

    mask = np.random.random((weights.shape[1],))
    idx = np.where(mask <= probability)[0]

    return idx


def get_neuron_ordering(probe, class_to_idx, search_stride=100):
    """
    Get global ordering of neurons from a trained probe.

    This method returns the global ordering of neurons in a model based on
    the given probe's weight values. Top neurons are computed at increasing
    percentages of the weight mass and then accumulated in-order. See given
    reference for a complete description of the selection algorithm.

    For example, if the neuron list at 1% weight mass is [#2, #52, #134], and 
    at 2% weight mass is [#2, #4, #52, #123, #130, #134, #567], the returned
    ordering will be [#2, #52, #134, #4, #123, #130, #567]. 
    Within each percentage, the ordering of neurons is arbitrary. In this case,
    the importance of #2, #52 and #134 is not necessarily in that order.
    The cutoffs between each percentage selection are also returned. Increasing
    the ``search_stride`` will decrease the distance between each cutoff, making
    the overall ordering more accurate.

    .. seealso::
        `Dalvi, Fahim, et al. "What is one grain of sand in the desert? analyzing individual neurons in deep nlp models." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33. No. 01. 2019. <https://ojs.aaai.org/index.php/AAAI/article/view/4592>`_

    .. note::
        Absolute weight values are used for selection, instead of raw signed
        values


    Parameters
    ----------
    probe : interpretation.linear_probe.LinearProbe
        Trained probe model
    class_to_idx : dict
        Class to class index mapping. Usually returned by
        ``interpretation.utils.create_tensors``.
    search_stride : int, optional
        Defines how many pieces the percent weight mass selection is divided
        into. Higher leads to more a accurate ordering. Defaults to 100.

    Returns
    -------
    global_neuron_ordering : numpy.ndarray
        Numpy array of size ``NUM_NEURONS`` with neurons in decreasing order
        of importance.
    cutoffs : list
        Indices where each percentage selection begins. All neurons between two
        cutoff values are arbitrarily ordered.

    """
    progressbar = utils.get_progress_bar()

    neuron_orderings = [
        get_top_neurons(probe, p / search_stride, class_to_idx)[0]
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
    probe, class_to_idx, granularity=50, search_stride=100
):
    """
    Get global ordering of neurons from a trained probe.

    This method is an alternative to
    ``interpretation.linear_probe.get_neuron_ordering``. It works very similarly
    to that method, except that instead of adding the neurons from each
    percentage selection, neurons are added in chunks of ``granularity``
    neurons.

    .. seealso::
        `Dalvi, Fahim, et al. "What is one grain of sand in the desert? analyzing individual neurons in deep nlp models." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33. No. 01. 2019. <https://ojs.aaai.org/index.php/AAAI/article/view/4592>`_

    .. note::
        Absolute weight values are used for selection, instead of raw signed
        values


    Parameters
    ----------
    probe : interpretation.linear_probe.LinearProbe
        Trained probe model
    class_to_idx : dict
        Class to class index mapping. Usually returned by
        ``interpretation.utils.create_tensors``.
    granularity : int, optional
        Approximate number of neurons in each chunk of selection. Defaults to
        50.
    search_stride : int, optional
        Defines how many pieces the percent weight mass selection is divided
        into. Higher leads to more a accurate ordering. Defaults to 100.

    Returns
    -------
    global_neuron_ordering : numpy.ndarray
        Numpy array of size ``NUM_NEURONS`` with neurons in decreasing order
        of importance.
    cutoffs : list
        Indices where each chunk of selection begins. Each chunk will contain
        approximately ``granularity`` neurons. All neurons between two
        cutoff values (i.e. a chunk) are arbitrarily ordered.

    """
    progressbar = utils.get_progress_bar()

    weights = list(probe.parameters())[0].data.cpu()
    num_neurons = weights.numpy().shape[1]
    neuron_orderings = [
        get_top_neurons(probe, p / search_stride, class_to_idx)[0]
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

# Returns num_bottom_neurons bottom neurons from the global ordering
def get_fixed_number_of_bottom_neurons(probe, num_bottom_neurons, class_to_idx):
    """
    Get global bottom neurons.

    This method returns a fixed number of bottoms neurons from the global
    ordering computed using ``interpretation.linear_probe.get_neuron_ordering``.

    .. note::
        Absolute weight values are used for selection, instead of raw signed
        values


    Parameters
    ----------
    probe : interpretation.linear_probe.LinearProbe
        Trained probe model
    num_bottom_neurons : int
        Number of bottom neurons for selection
    class_to_idx : dict
        Class to class index mapping. Usually returned by
        ``interpretation.utils.create_tensors``.

    Returns
    -------
    global_bottom_neurons : numpy.ndarray
        Numpy array of size ``num_bottom_neurons`` with bottom neurons using the
        global ordering

    """
    ordering, _ = get_neuron_ordering(probe, class_to_idx)

    return ordering[-num_bottom_neurons:]