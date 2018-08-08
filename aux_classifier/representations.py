import numpy as np
from tqdm import tqdm

def bpe_get_avg_activations(tokens, activations):
    all_activations = []
    num_neurons = activations[0].size(1)

    for i in range(0, len(tokens['source_aux'])):
        sourceIndex = 0 
        thisBPE = ""
        source = tokens['source'][i]
        source_aux = tokens['source_aux'][i]
        num_words = len(source)
        new_activations = np.zeros((num_words, num_neurons))
    
        word_boundaries = []
        
        for j in range(0, len(tokens['source_aux'][i])):
            currSourceWord = tokens['source'][i][sourceIndex]
            thisBPE = thisBPE + tokens['source_aux'][i][j]
        
            if (thisBPE != currSourceWord):
                thisBPE = thisBPE[:-2]
            else:
                word_boundaries.append(j)
                sourceIndex = sourceIndex+1
                thisBPE = ""
    
        assert(len(word_boundaries) == num_words)
    
        prev_idx = 0
        for word_idx, boundary in enumerate(word_boundaries):
            avg_vector = np.average(activations[i][prev_idx:boundary+1, :], axis=0)
            new_activations[word_idx, :] = avg_vector
            prev_idx = boundary+1
    
        all_activations.append(new_activations)
    
    return all_activations

def bpe_get_last_activations(tokens, activations, is_brnn=True):
    all_activations = []
    num_neurons = activations[0].size(1)

    for i in range(0, len(tokens['source_aux'])):
        sourceIndex = 0 
        thisBPE = ""
        source = tokens['source'][i]
        source_aux = tokens['source_aux'][i]
        num_words = len(source)
        new_activations = np.zeros((num_words, num_neurons))
    
        word_boundaries = []
        
        for j in range(0, len(tokens['source_aux'][i])):
            currSourceWord = tokens['source'][i][sourceIndex]
            thisBPE = thisBPE + tokens['source_aux'][i][j]
        
            if (thisBPE != currSourceWord):
                thisBPE = thisBPE[:-2]
            else:
                word_boundaries.append(j)
                sourceIndex = sourceIndex+1
                thisBPE = ""
    
        assert(len(word_boundaries) == num_words)
        
        rnn_boundary = int(num_neurons/2)
        if not is_brnn:
            rnn_boundary = num_neurons
    
        prev_idx = 0
        for word_idx, boundary in enumerate(word_boundaries):
            # 0 - num_neurons/2: Forward
            # num_neurons/2 - : Backward
            new_activations[word_idx, :rnn_boundary] = activations[i][boundary, :rnn_boundary]
            if is_brnn:
                new_activations[word_idx, rnn_boundary:] = activations[i][prev_idx, rnn_boundary:]
            prev_idx = boundary+1
    
        all_activations.append(new_activations)
    
    return all_activations

def char_get_avg_activations(tokens, activations):
    
    all_activations = []
    num_neurons = activations[0].size(1)

    for i in tqdm(range(0, len(tokens['source_aux']))):
        sourceIndex = 0 
        thisChar = ""
        source = tokens['source'][i]
        source_aux = tokens['source_aux'][i]
        num_words = len(source)
        new_activations = np.zeros((num_words, num_neurons))
    
        word_boundaries = []
        
        for word_idx, word in enumerate(tokens['source'][i]):
            if word_idx == 0:
                word_boundaries.append(len(word)-1)
            else:
                word_boundaries.append(len(word) + 1 + word_boundaries[-1])
        
        if (len(word_boundaries) != num_words):
            print(i, len(word_boundaries), num_words)
        assert(len(word_boundaries) == num_words)
        assert tokens['source_aux'][i].count('_')+1-tokens['source'][i].count('_') == num_words, \
            "Number of words dont match! (line: %d, source: %d, aux: %d)\n%s\n%s"%(i+1, num_words, tokens['source_aux'][i].count('_')+1,
                " ".join(tokens['source'][i]), " ".join(tokens['source_aux'][i]))

        prev_idx = 0
        for word_idx, boundary in enumerate(word_boundaries):
            avg_vector = np.average(activations[i][prev_idx:boundary+1, :], axis=0)
            new_activations[word_idx, :] = avg_vector
            prev_idx = boundary+2
    
        all_activations.append(new_activations)
    
    return all_activations

def char_get_last_activations(tokens, activations, is_brnn=True):
    all_activations = []
    num_neurons = activations[0].size(1)

    for i in tqdm(range(0, len(tokens['source_aux']))):
        sourceIndex = 0 
        thisChar = ""
        source = tokens['source'][i]
        source_aux = tokens['source_aux'][i]
        num_words = len(source)
        new_activations = np.zeros((num_words, num_neurons))
    
        word_boundaries = []
        
        for word_idx, word in enumerate(tokens['source'][i]):
            if word_idx == 0:
                word_boundaries.append(len(word)-1)
            else:
                word_boundaries.append(len(word) + 1 + word_boundaries[-1])
        
        if (len(word_boundaries) != num_words):
            print(i, len(word_boundaries), num_words)
        assert(len(word_boundaries) == num_words)
        assert tokens['source_aux'][i].count('_')+1-tokens['source'][i].count('_') == num_words, \
            "Number of words dont match! (line: %d, source: %d, aux: %d)\n%s\n%s"%(i+1, num_words, tokens['source_aux'][i].count('_')+1,
                " ".join(tokens['source'][i]), " ".join(tokens['source_aux'][i]))
        
        rnn_boundary = int(num_neurons/2)
        if not is_brnn:
            rnn_boundary = num_neurons
        
        prev_idx = 0
        
        for word_idx, boundary in enumerate(word_boundaries):
            # 0 - num_neurons/2: Forward
            # num_neurons/2 - : Backward
            new_activations[word_idx, :rnn_boundary] = activations[i][boundary, :rnn_boundary]
            if is_brnn:
                new_activations[word_idx, rnn_boundary:] = activations[i][prev_idx, rnn_boundary:]
            prev_idx = boundary+1
    
        all_activations.append(new_activations)
    
    return all_activations