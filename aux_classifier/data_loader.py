import numpy as np
import torch

from torch.utils.serialization import load_lua

def load_activations(activations_path):
    file_ext = activations_path.split('.')[-1]

    activations = None

    # Load activations based on type
    # Also ensure everything is on the CPU
    #   as activations may have been saved as CUDA variables
    if file_ext == "t7":
        print("Loading seq2seq-attn activations from %s..." % (activations_path))
        activations = load_lua(activations_path)['encodings']
        activations = [a.cpu() for a in activations]
    elif file_ext == "pt":
        print("Loading OpenNMT-py activations from %s..." % (activations_path))
        activations = torch.load(activations_path)
        activations = [torch.stack([torch.cat(token) for token in sentence]).cpu() for sentence in activations]

    return activations


def load_aux_data(source_path, labels_path, source_aux_path, activations, max_sent_l):
    tokens = {
        'source_aux': [],
        'source': [],
        'target': []
    }

    skipped_lines = set()
    with open(source_aux_path) as fp:
        for line_idx, line in enumerate(fp):
            line_tokens = line.strip().split()
            if len(line_tokens) > max_sent_l:
                print("Skipping line #%d because of length (aux)"%(line_idx))
                skipped_lines.add(line_idx)
            tokens['source_aux'].append(line_tokens)
    with open(source_path) as fp:
        for line_idx, line in enumerate(fp):
            line_tokens = line.strip().split()
            if len(line_tokens) > max_sent_l:
                print("Skipping line #%d because of length (source)"%(line_idx))
                skipped_lines.add(line_idx)
            tokens['source'].append(line_tokens)

    with open(labels_path) as fp:
        for line_idx, line in enumerate(fp):
            line_tokens = line.strip().split()
            if len(line_tokens) > max_sent_l:
                print("Skipping line #%d because of length (label)"%(line_idx))
                skipped_lines.add(line_idx)
            tokens['target'].append(line_tokens)

    assert len(tokens['source_aux']) == len(tokens['source']) and len(tokens['source_aux']) == len(tokens['target']), \
        "Number of lines do not match (source: %d, aux: %d, target: %d)!"%(len(tokens['source']), len(tokens['source_aux']), len(tokens['target']))

    for num_deleted, line_idx in enumerate(sorted(skipped_lines)):
        del(tokens['source_aux'][line_idx])
        del(tokens['source'][line_idx])
        del(tokens['target'][line_idx])

    # Check if all data is well formed (whether we have activations + labels for each and every word)
    invalid_activation_idx = []
    for idx, activation in enumerate(activations):
        if activation.shape[0] == len(tokens['source_aux'][idx]) and len(tokens['source'][idx]) == len(tokens['target'][idx]):
            pass
        else:
            invalid_activation_idx.append(idx)
            print ("Skipping line: ", idx, "A: %d, aux: %d, src: %d, tgt: %s"%(activation.shape[0], len(tokens['source_aux'][idx]), len(tokens['source'][idx]), len(tokens['target'][idx])))
    
    for num_deleted, idx in enumerate(invalid_activation_idx):
        print("Deleting line %d: %d activations, %s aux, %d source, %d target"%
            (idx - num_deleted, 
                activations[idx - num_deleted].shape[0], 
                len(tokens['source_aux'][idx - num_deleted]),
                len(tokens['source'][idx - num_deleted]), 
                len(tokens['target'][idx - num_deleted])
            )
        )
        del(activations[idx - num_deleted])
        del(tokens['source_aux'][idx - num_deleted])
        del(tokens['source'][idx - num_deleted])
        del(tokens['target'][idx - num_deleted])
        
    for idx, activation in enumerate(activations):
        assert(activation.shape[0] == len(tokens['source_aux'][idx]))
        assert(len(tokens['source'][idx]) == len(tokens['target'][idx]))
    
    return tokens

def load_data(source_path, labels_path, activations, max_sent_l, sentence_classification=False):
    tokens = {
        'source': [],
        'target': []
    }

    with open(source_path) as fp:
        for line in fp:
            line_tokens = line.strip().split()
            if len(line_tokens) > max_sent_l:
                continue
            tokens['source'].append(line_tokens)

    with open(labels_path) as fp:
        for line in fp:
            line_tokens = line.strip().split()
            if len(line_tokens) > max_sent_l:
                continue
            tokens['target'].append(line_tokens)

    assert len(tokens['source']) == len(tokens['target']), \
        "Number of lines do not match (source: %d, target: %d)!"%(len(tokens['source']), len(tokens['target']))

    # Check if all data is well formed (whether we have activations + labels for each and every word)
    invalid_activation_idx = []
    for idx, activation in enumerate(activations):
        if activation.shape[0] == len(tokens['source'][idx]) \
            and (sentence_classification or activation.shape[0] == len(tokens['target'][idx])):
            pass
        else:
            invalid_activation_idx.append(idx)
            print ("Skipping line: ", idx)
            print ("A: %d, S: %d, T: %d" % (activation.shape[0], len(tokens['source'][idx]), len(tokens['target'][idx])))

            # assert len(invalid_activation_idx) < 100, "Too many mismatches - your paths are probably incorrect!"

    
    for num_deleted, idx in enumerate(invalid_activation_idx):
        print("Deleting line %d: %d activations, %d source, %d target"%
            (idx - num_deleted, 
                activations[idx - num_deleted].shape[0], 
                len(tokens['source'][idx - num_deleted]), 
                len(tokens['target'][idx - num_deleted])
            )
        )
        del(activations[idx - num_deleted])
        del(tokens['source'][idx - num_deleted])
        del(tokens['target'][idx - num_deleted])
        
    for idx, activation in enumerate(activations):
        assert(activation.shape[0] == len(tokens['source'][idx]))
        if not sentence_classification:
            assert(activation.shape[0] == len(tokens['target'][idx]))
    
    return tokens