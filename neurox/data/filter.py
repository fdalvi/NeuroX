"""Create a binary word-level data set using a set of positive examples, or a regular expression or a function
Author: Hassan Sajjad
Last Modified: 4 July, 2021
"""

import re
import collections
import numpy as np

def create_binary_data(tokens, activations, binary_filter, num_layers, outputfile):
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
    Saves a word file, a binay label file and their activations
    
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
    
    save_files(positive_class_words+negative_class_words, labels, positive_class_activations+negative_class_activations, outputfile)
    
    
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
    
def save_files(words, labels, activations, outputfile):
    
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
    
    with open(outputfile+".word", "w") as file:
        print(*words, sep = "\n", file = file)
        file.close()

    with open(outputfile+".label", "w") as file:
        print(*labels, sep = "\n", file = file)
        file.close()
    
    h5py_path = outputfile+".hdf5"
    with h5py.File(h5py_path, "w") as output_file:
        sentence_to_index = {}
        for word_idx, word in enumerate(words):
            output_file.create_dataset(
                str(word_idx),
                activations[word_idx].shape,
                dtype="float32",
                data=activations[word_idx],
            )
            # TODO: Replace with better implementation with list of indices
            final_sentence = word
            counter = 1
            while final_sentence in sentence_to_index:
                counter += 1
                final_sentence = f"{sentence} (Occurrence {counter})"
            sentence = final_sentence
            sentence_to_index[sentence] = str(word_idx)

        sentence_index_dataset = output_file.create_dataset(
            "sentence_to_index", (1,), dtype=h5py.special_dtype(vlen=str)
        )
        sentence_index_dataset[0] = json.dumps(sentence_to_index)
    output_file.close()
    print ("Saved .word, .label. and .hdf5 files with prefix -> ", outputfile)

