"""Given a list of sentence and their activations, create a binary labeled dataset based on a
regular expression, a list of words and a fuction. 
For example, one can create a binary dataset of years vs. not-years (2004 vs. this) by specifying the regular expression
that matches the pattern of year. 
The program will extract positive examples based on the provided filter and will randomly extract negative examples
of the same size as that of the positive examples. The output of the program is a word file, a label file and an activation file.

Author: Hassan Sajjad
Last Modified: 4 July, 2021
Last Modified: 21 February, 2022
"""

import re
import collections
import numpy as np
import argparse
import sys

from neurox.data.writer import ActivationsWriter
import neurox.data.loader as data_loader

def _create_binary_data(tokens, activations, binary_filter, num_layers, output_file, output_type="autodetect", decompose_layers=False,
    filter_layers=None,):
    """
    Given a set of sentences and per word activations, create a binary dataset using the provided pattern of the
    positive class. A pattern can be a set of words, a regex object or a function
    
    Parameters
    ----------
    tokens : dictionary
        A dictonary of sentences with their dummy labels. The format is the output of the data_loader.load_data function
    activations: list
        A list of sentence-wise activations 
    binary_filter: a set of words or a regex object or a function
    num_layers: number of layers
    
    Returns
    -------
    Saves a word file, a binary label file and their activations
    
    Example
    -------
    create_binary_data(tokens, activations, re.compile(r'^\w\w$'), 12) select words of two characters only as a positive class
    create_binary_data(tokens, activations, {'is', 'can'}, 12) select occrrences of 'is' and 'can' as a positive class
    """
    
    filter_fn = None
    if isinstance(binary_filter, set):
        filter_fn = lambda x: x in binary_filter
    elif isinstance(binary_filter, re.Pattern):
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
                positive_class_activations.append(activations[s_idx][w_idx].reshape((num_layers, 1, -1)))
            else:
                negative_class_words.append(word)
                negative_class_activations.append(activations[s_idx][w_idx].reshape((num_layers, 1, -1)))

    negative_class_words, negative_class_activations = _balance_negative_class(negative_class_words, negative_class_activations, len(positive_class_words))    
    
    print ("Number of Positive examples: ", len(positive_class_words))
    
    labels = (['positive']*len(positive_class_words)) + ['negative']*len(positive_class_words)
    
    save_files(positive_class_words+negative_class_words, labels, positive_class_activations+negative_class_activations, output_file, output_type, decompose_layers,
    filter_layers)
    
    
def _balance_negative_class(words, activations, positive_class_size):
    """
    Helper function to shuffle the negative class instances and select the number of instances equal to 
    the positive class
    
    Parameters
    ----------
    words: list
        A list of words
    activations: list
        A list of word-wise activations 
    positive_class_size: number of words to select
    
    Returns
    -------
    A list of words and activations equal in size to the passed length
    
    """    
    print ("Balancing Negative class ...")
    indices = list(range((len(words))))
    np.random.shuffle(indices)

    swords = []
    sactivations = []
    for i in range(positive_class_size):
        swords.append(words[indices[i]])
        sactivations.append(activations[indices[i]])
    
    return swords, sactivations
    
def save_files(words, labels, activations, output_file, output_type, decompose_layers, filter_layers):
    
    """
    Save word and label file in text format and activations in hdf5 format
    
    Parameters
    ----------
    words: list
        A list of words
    labels: list
        A list of labels for every word
    activations: list
        A list of word-wise activations 
    positive_class_size: number of words to select
    
    Returns
    -------
    Save word, label and activation files
    
    """ 
    print("Preparing output files")

    with open(output_file+".word", "w") as file:
        print(*words, sep = "\n", file = file)
        file.close()

    with open(output_file+".label", "w") as file:
        print(*labels, sep = "\n", file = file)
        file.close()
    
    writer = ActivationsWriter.get_writer(output_file, filetype=output_type, decompose_layers=decompose_layers, filter_layers=filter_layers)

    for word_idx, word in enumerate(words):
        writer.write_activations(word_idx, [word], activations[word_idx])

    writer.close()


def annotate_data(sentences, activations)

    activations, num_layers = data_loader.load_activations(activations, 768)
    # giving sentences instead of labels since labels will be generated later
    tokens = data_loader.load_data(sentences, sentences, activations, 512)

    _create_binary_data(tokens, activations, binary_filter, num_layers, output_file, output_type="autodetect", decompose_layers=False,
    filter_layers=None,)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="Text file path with one sentene per line")
    parser.add_argument("--activations", help="Activation file")
    parser.add_argument("--binary_filter", help="A set, pattern or function based on which binary data will be created")
    parser.add_argument("--out_file", help="Prefix of output file")
    parser.add_argument("--output_type", help="Type of output activation file, json or hdf5")
    parser.add_argument("--decompose_layers", help="A boolean value to specify if a layers need to be save separately")
    parser.add_argument("--filter_layers", help="Specify a layer if activations of only a particular layer needs to be saved")

    annotate_data(arg.input_file,
                 arg.activations,
                 arg.binary_filter,
                 arg.output_file,
                 arg.output_type,
                 arg.decompose_layers,
                 arg.filter_layers)
    

    if __name__ == "__main__":
    main()