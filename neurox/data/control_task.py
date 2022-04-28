from collections import Counter

import numpy as np


def create_sequence_labeling_dataset(
    train_tokens,
    dev_source=None,
    test_source=None,
    case_sensitive=True,
    sample_from="same",
):
    """
    Method that prepares labels for a control task, as defined in §2.1 of `Hewitt and Liang (2019) <https://aclanthology.org/D19-1275.pdf>`

    Target classes are selected randomly for each token type in the datasets.
    The number of control task classes is the same as the number of classes in ``train_tokens['target']``. The distribution of control task labels can be specified.

    Parameters
    ----------
    train_tokens : dict
        Dictionary containing two lists of lists representing the training set, ``source`` and ``target``. As produced by :func:`dataloader. <mymodule.MyClass.foo>`
    dev_source : list, optional
        List containing the ``source`` tokens from the development set, as produced by ``dev_tokens['source']``
    test_source : list, optional
        List containing the ``source`` tokens from the test set, as produced by ``test_tokens['source']``
    case_sensitive: bool, optional
        defaults to True. Sets whether the token comparison (for assigning the control task labels) is case-sensitive
        or case-insensitive.
    sample_from : str, optional
        defaults to 'same'. The distribution from which control task labels are sampled.
        'same': Labels are sampled from the same distribution as the main task labels.
        'uniform': Labels are sampled from a uniform distribution.

    Returns
    -------
    control_task_tokens : list
        A list with either one, two or three elements - depending on whether
        control task labels for only the train, or also dev and test set should be created.
        Each element of the list is a dictionary containing two lists, ``source`` and ``target``.
        The ``source`` list is the same as from the ``tokens`` input.
        The ``target`` list is the list of control task labels.
    """

    # compute label stats in task training data
    labels_flat = [l for sublist in train_tokens["target"] for l in sublist]
    label_freqs = Counter(labels_flat)
    ct_labels = list(range(len(label_freqs)))
    ct_label_distr = [v / sum(label_freqs.values()) for v in label_freqs.values()]
    if sample_from == "uniform":
        ct_label_distr = [1 / len(ct_label_distr) for i in ct_labels]

    # create control task labels
    word_types_to_ct_label = dict()
    datasets = [train_tokens["source"]]
    if dev_source is not None:
        datasets.append(dev_source)
    if test_source is not None:
        datasets.append(test_source)
    result = []
    for source_dataset in datasets:
        ct_target = []
        for sent in source_dataset:
            ct_labels_for_sent = []
            for tok in sent:
                tok = tok if case_sensitive else tok.lower()
                if tok in word_types_to_ct_label:
                    ct_labels_for_sent.append(word_types_to_ct_label[tok])
                else:
                    label_for_tok = np.random.choice(ct_labels, p=ct_label_distr)
                    ct_labels_for_sent.append(label_for_tok)
                    word_types_to_ct_label[tok] = label_for_tok
            ct_target.append(ct_labels_for_sent)
        assert len(source_dataset) == len(ct_target)
        assert all([len(s) == len(t) for s, t in zip(source_dataset, ct_target)])
        result.append({"source": source_dataset, "target": ct_target})
    return result
