import numpy as np
from neurox.data.writer import ActivationsWriter

def _balance_negative_class(words, activations, positive_class_size):
    """
    Helper function to shuffle the negative class instances and select the number of instances equal to 
    the positive class. If negative class examples are fewer than positive class examples, it does not 
    perform any balancing and return the data as received
    
    Parameters
    ----------
    words: list
        A list of words
    activations: list
        A list of word-wise activations 
    positive_class_size: number of words to select
    
    Returns
    -------
    A list of words and activations equal or less than in size to the passed length
    
    """

    if len(words) <= positive_class_size:
        print ("No need of balancing the data. Negative class is equal or smaller in size to the positive class")
        return words, activations

    print ("Balancing Negative class ...")
    indices = list(range((len(words))))
    np.random.shuffle(indices)

    swords = []
    sactivations = []
    for i in range(positive_class_size):
        swords.append(words[indices[i]])
        sactivations.append(activations[indices[i]])
    
    return swords, sactivations


def save_files(words, labels, activations, output_prefix, output_type="hdf5", decompose_layers=False, filter_layers=None):
    
    """
    Save word and label files in the text format and activations in the specified format (default hdf5 format)
    
    Parameters
    ----------
    words: list
        A list of words
    labels: list
        A list of labels for every word
    activations: list
        A list of word-wise activations 
    output_prefix: string
        Specify prefix of the output files
    
    Returns
    -------
    Save word, label and activation files
    
    """ 

    with open(output_prefix+".word", "w") as file:
        print(*words, sep = "\n", file = file)
        file.close()

    with open(output_prefix+".label", "w") as file:
        print(*labels, sep = "\n", file = file)
        file.close()
    
    writer = ActivationsWriter.get_writer(f"{output_prefix}.{output_type}", filetype=output_type, decompose_layers=decompose_layers, filter_layers=filter_layers)

    for word_idx, word in enumerate(words):
        writer.write_activations(word_idx, [word], activations[word_idx])

    writer.close()

