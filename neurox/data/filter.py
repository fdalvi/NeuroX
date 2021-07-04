#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import codecs
import h5py
import json
import re
import collections

"""Create a binary word-level data set using a set of positive examples, or a regular expression or a function
Author: Hassan Sajjad
Last Modified: 4 July, 2021
"""

def create_concept_data(tokens, activations, concept_filter):
    """
    Given a set of sentences and per word activations, create a binary dataset using the provided pattern of the
    positive class. A pattern can be a set of words, a regex object or a function
    
    Parameters
    ----------
    tokens : dictionary
        A dictonary of sentences with their dummy labels. The format is the output of the data_loader.load_data function
    activations: list
        A list of sentence-wise activations 
    concept_filter: a set of words or a regex object or a function
    
    Returns
    -------
    Saves a word file, a binay label file and their activations
    
    Example
    -------
    create_concept_data(tokens, activations, re.compile(r'^\w\w$')) select words of two characters only as a positive class
    create_concept_data(tokens, activations, {'is', 'can'}) select occrrences of 'is' and 'can' as a positive class
    """
    
    filter_fn = None
    if isinstance(concept_filter, set):
        filter_fn = lambda x: x in concept_filter
    elif isinstance(concept_filter, re.Pattern):
        filter_fn = lambda x: concept_filter.match(x)
    elif isinstance(concept_filter, collections.Callable):
        filter_fn = concept_filter
    else:
        print ("ERROR: doest not belong to any configuration")
        return
        
    positive_class_words = []
    positive_class_activations = []
    negative_class_words = []    
    negative_class_activations = []
    
    print ("Creating concept dataset ...")
    for sidx, sentences in enumerate(tokens['source']):
        for widx, word in enumerate(sentences):
            if filter_fn(word):
                positive_class_words.append(word)
                positive_class_activations.append(activations[sidx][widx].reshape((num_layers, 1, -1)))
            else:
                negative_class_words.append(word)
                negative_class_activations.append(activations[sidx][widx].reshape((num_layers, 1, -1)))

    negative_class_words, negative_class_activations = _balance_negative_class(negative_class_words, negative_class_activations, len(positive_class_words))    
    
    print ("Number of Positive examples: ", len(positive_class_words))
    
    labels = (['positive']*len(positive_class_words)) + ['negative']*len(positive_class_words)
    
    save_files(positive_class_words+negative_class_words, labels, positive_class_activations+negative_class_activations)
    
    
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
    
def save_files(words, labels, activations):
    
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
    
    with codecs.open("output.word", "w", "utf-8-sig") as file:
        print(*words, sep = "\n", file = file)
        file.close()

    with codecs.open("output.label", "w", "utf-8-sig") as file:
        print(*labels, sep = "\n", file = file)
        file.close()
    
    h5py_path = "output.hdf5"
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
    print ("Saved files at output.word, output.label and output.hdf5")

