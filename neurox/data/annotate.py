"""Given a list of sentences, their activations and a pattern, create a binary labeled dataset based on the pattern
where pattern can be a regular expression, a list of words and a function. 
For example, one can create a binary dataset of years vs. not-years (2004 vs. this) by specifying the regular expression
that matches the pattern of year. 
The program will extract positive class examples based on the provided filter and will consider rest of the examples as
negative class examples. The output of the program is a word file, a label file and an activation file.

"""
import collections
import numpy as np
import argparse
import sys
from typing import Pattern

import neurox.data.loader as data_loader
import neurox.data.utils as data_utils

def _create_binary_data(tokens, activations, binary_filter, balance_data=False):
    """
    Given a list of tokens, their activations and a binary_filter, create the binary labeled data. A binary filter can be
    a set, regex or a function. The regex option expects output of re.compile.
    
    Parameters
    ----------
    tokens : dictionary
        A dictonary of sentences with their dummy labels. The format is the output of the data_loader.load_data function
    activations: list
        A list of sentence-wise activations 
    binary_filter: a set of words or a regex object or a function
    
    Returns
    -------
    annotated_dataset : tuple of (tokens, labels and activations)
        A list of selected positive and negative class words, their labels (the
        string "positive" for examples matching the filter and "negative" for
        others) and their activations
    
    Example
    -------
    _create_binary_data(tokens, activations, re.compile(r'^\w\w$')) select words of two characters only as a positive class
    _create_binary_data(tokens, activations, {'is', 'can'}) select occrrences of 'is' and 'can' as a positive class
    """
    
    filter_fn = None
    if isinstance(binary_filter, set):
        filter_fn = lambda x: x in binary_filter
    elif isinstance(binary_filter, Pattern):
        filter_fn = lambda x: binary_filter.match(x)
    elif isinstance(binary_filter, collections.Callable):
        filter_fn = binary_filter
    else:
        raise NotImplementedError("ERROR: does not belong to any configuration")
        
    positive_class_words = []
    positive_class_activations = []
    negative_class_words = []    
    negative_class_activations = []
    
    print ("Creating binary dataset ...")
    for s_idx, sentences in enumerate(tokens['source']):
        for w_idx, word in enumerate(sentences):
            if filter_fn(word):
                positive_class_words.append(word)
                positive_class_activations.append(activations[s_idx][w_idx])
            else:
                negative_class_words.append(word)
                negative_class_activations.append(activations[s_idx][w_idx])

    if len(negative_class_words) == 0 or len(positive_class_words) == 0:
        raise ValueError(
            "Positive or Negative class examples are zero"
        )
    elif len(negative_class_words) < len(positive_class_words):
        print ("WARNING: the negative class examples are less than the positive class examples")
        print ("Postive class examples: ", len(positive_class_words), "Negative class examples: ", len(negative_class_words))

    if balance_data:
        negative_class_words, negative_class_activations = data_utils._balance_negative_class(negative_class_words, negative_class_activations, len(positive_class_words))
    
    print ("Number of Positive examples: ", len(positive_class_words))
    
    labels = (['positive']*len(positive_class_words)) + ['negative']*len(negative_class_words)
    
    return positive_class_words+negative_class_words, labels, positive_class_activations+negative_class_activations



def annotate_data(source_path, activations_path, binary_filter, output_prefix, output_type="hdf5", decompose_layers=False, filter_layers=None):
    """
    Given a set of sentences, per word activations, a binary_filter and output_prefix, creates binary data and save it to the disk. 
    A binary filter can be a set of words, a regex object or a function
    
    Parameters
    ----------
    source_path : text file with one sentence per line 
    activations: list
        A list of sentence-wise activations 
    binary_filter: a set of words or a regex object or a function
    output_prefix: prefix of the output files that will be saved as the output of this script
    
    Returns
    -------
    Saves a word file, a binary label file and their activations
    
    Example
    -------
    annotate_data(source_path, activations_path, re.compile(r'^\w\w$')) select words of two characters only as a positive class
    annotate_data(source_path, activations_path, {'is', 'can'}) select occrrences of 'is' and 'can' as a positive class
    """

    activations, num_layers = data_loader.load_activations(activations_path)

    # giving source_path instead of labels since labels will be generated later
    tokens = data_loader.load_data(source_path, source_path, activations, max_sent_l=512)

    words, labels, activations = _create_binary_data(tokens, activations, binary_filter)
    activations = [np.swapaxes(a.reshape((a.shape[1], num_layers, -1)), 0, 1) for a in activations]
    data_utils.save_files(words, labels, activations, output_prefix, output_type, decompose_layers,
    filter_layers)
