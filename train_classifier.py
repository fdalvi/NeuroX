# coding: utf-8

import argparse
import codecs
import dill as pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re

from itertools import product as p
from torch.utils.serialization import load_lua
from tqdm import tqdm, tqdm_notebook, tnrange

# Import lib
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import aux_classifier.utils as utils
import aux_classifier.representations as repr

def main():
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('--train-source', dest='train_source', required=True,
                    help='Location of train source file')
    parser.add_argument('--train-aux-source', dest='train_aux_source',
                    help='Location of aux train source file (BPE/CHAR)')
    parser.add_argument('--train-labels', dest='train_labels', required=True,
                    help='Location of train source labels')
    parser.add_argument('--train-activations', dest='train_activations', required=True,
                    help='Location of train source activations')

    parser.add_argument('--test-source', dest='test_source', required=True,
                    help='Location of test source file')
    parser.add_argument('--test-aux-source', dest='test_aux_source',
                    help='Location of aux test source file (BPE/CHAR)')
    parser.add_argument('--test-labels', dest='test_labels', required=True,
                    help='Location of test source labels')
    parser.add_argument('--test-activations', dest='test_activations', required=True,
                    help='Location of test source activations')
    
    parser.add_argument('--exp-type', dest='exp_type', 
                    choices=['word', 'charcnn', 'bpe_avg', 'bpe_last', 'char_avg', 'char_last'],
                    default='word', required=True,
                    help='Type of experiment')

    parser.add_argument('--task-specific-tag', dest='task_specific_tag', 
                    required=True, help='Tag incase test has unknown tags')

    parser.add_argument('--max-sent-l', dest='max_sent_l', type=int,
                    default=250, help='Tag incase test has unknown tags')

    parser.add_argument('--output-dir', dest='output_dir', 
                    required=True, help='Location to save all results')

    parser.add_argument('--filter-layers', dest='filter_layers', default=None,
                    type=str, help='Use specific layers for training. Format: f1,b1,f2,b2')

    args = parser.parse_args()

    print("Creating output directory...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Constants
    NUM_EPOCHS = 10
    BATCH_SIZE = 128
    BRNN = 2

    print("Loading activations...")
    train_activations = load_lua(args.train_activations)['encodings']
    test_activations = load_lua(args.test_activations)['encodings']
    print("Number of train sentences: %d"%(len(train_activations)))
    print("Number of test sentences: %d"%(len(test_activations)))

    def load_aux_data(source_path, labels_path, source_aux_path, activations):
        tokens = {
            'source_aux': [],
            'source': [],
            'target': []
        }

        with open(source_aux_path) as fp:
            for line in fp:
                line_tokens = line.strip().split()
                if len(line_tokens) > args.max_sent_l:
                    continue
                tokens['source_aux'].append(line_tokens)
        with open(source_path) as fp:
            for line in fp:
                line_tokens = line.strip().split()
                if len(line_tokens) > args.max_sent_l:
                    continue
                tokens['source'].append(line_tokens)

        with open(labels_path) as fp:
            for line in fp:
                line_tokens = line.strip().split()
                if len(line_tokens) > args.max_sent_l:
                    continue
                tokens['target'].append(line_tokens)

        # Check if all data is well formed (whether we have activations + labels for each and every word)
        invalid_activation_idx = []
        for idx, activation in enumerate(activations):
            if activation.shape[0] == len(tokens['source_aux'][idx]) and len(tokens['source'][idx]) == len(tokens['target'][idx]):
                pass
            else:
                invalid_activation_idx.append(idx)
                print ("Skipping line: ", idx, "A: %d, aux: %d, src: %d, tgt: %s"%(activation.shape[0], len(tokens['source_aux'][idx]), len(tokens['source'][idx]), len(tokens['target'][idx])))
        
        for num_deleted, idx in enumerate(invalid_activation_idx):
            print("Deleting line %d: %d activations, %d source, %d target"%
                (idx - num_deleted, 
                    activations[idx - num_deleted].shape[0], 
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

    def load_data(source_path, labels_path, activations):
        tokens = {
            'source': [],
            'target': []
        }

        with open(source_path) as fp:
            for line in fp:
                line_tokens = line.strip().split()
                if len(line_tokens) > args.max_sent_l:
                    continue
                tokens['source'].append(line_tokens)

        with open(labels_path) as fp:
            for line in fp:
                line_tokens = line.strip().split()
                if len(line_tokens) > args.max_sent_l:
                    continue
                tokens['target'].append(line_tokens)

        # Check if all data is well formed (whether we have activations + labels for each and every word)
        invalid_activation_idx = []
        for idx, activation in enumerate(activations):
            if activation.shape[0] == len(tokens['source'][idx]) and activation.shape[0] == len(tokens['target'][idx]):
                pass
            else:
                invalid_activation_idx.append(idx)
                print ("Skipping line: ", idx)
        
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
            assert(activation.shape[0] == len(tokens['target'][idx]))
        
        return tokens

    if args.exp_type == 'word' or args.exp_type == 'charcnn':
        train_tokens = load_data(args.train_source, args.train_labels, train_activations)
        test_tokens = load_data(args.test_source, args.test_labels, test_activations)
    else:
        train_tokens = load_aux_data(args.train_source, args.train_labels, args.train_aux_source, train_activations)
        test_tokens = load_aux_data(args.test_source, args.test_labels, args.test_aux_source, test_activations)

    NUM_TOKENS = sum([len(t) for t in train_tokens['target']])
    print('Number of total train tokens: %d'%(NUM_TOKENS))

    if args.exp_type != 'word' and args.exp_type != 'charcnn':
        NUM_SOURCE_AUX_TOKENS = sum([len(t) for t in train_tokens['source_aux']])
        print('Number of AUX source words: %d'%(NUM_SOURCE_AUX_TOKENS)) 

    NUM_SOURCE_TOKENS = sum([len(t) for t in train_tokens['source']])
    print('Number of source words: %d'%(NUM_SOURCE_TOKENS)) 

    NUM_NEURONS = train_activations[0].shape[1]
    print('Number of neurons: %d'%(NUM_NEURONS))

    if args.exp_type == 'bpe_avg':
        train_activations = repr.bpe_get_avg_activations(train_tokens, train_activations)
        test_activations = repr.bpe_get_avg_activations(test_tokens, test_activations)
    elif args.exp_type == 'bpe_last':
        train_activations = repr.bpe_get_last_activations(train_tokens, train_activations, is_brnn=(BRNN == 2))
        test_activations = repr.bpe_get_last_activations(test_tokens, test_activations, is_brnn=(BRNN == 2))
    elif args.exp_type == 'char_avg':
        train_activations = repr.char_get_avg_activations(train_tokens, train_activations)
        test_activations = repr.char_get_avg_activations(test_tokens, test_activations)
    elif args.exp_type == 'char_last':
        train_activations = repr.char_get_last_activations(train_tokens, train_activations, is_brnn=(BRNN == 2))
        test_activations = repr.char_get_last_activations(test_tokens, test_activations, is_brnn=(BRNN == 2))

    # Filtering
    if args.filter_layers:
        _layers = args.filter_layers.split(',')

        RNN_SIZE = 500
        NUM_LAYERS = 2

        # FILTER settings
        layers = [1, 2] # choose which layers you need the activations
        filtered_train_activations = None
        filtered_test_activations = None

        layers_idx = []
        for brnn_idx, b in enumerate(['f','b']):
            for l in layers:
                if "%s%d"%(b, l) in _layers:
                    start_idx = brnn_idx * (NUM_LAYERS*RNN_SIZE) + (l-1) * RNN_SIZE
                    end_idx = brnn_idx * (NUM_LAYERS*RNN_SIZE) + (l) * RNN_SIZE

                    print("Including neurons from %s%d(#%d to #%d)"%(b, l, start_idx, end_idx))
                    layers_idx.append(np.arange(start_idx, end_idx))
        layers_idx = np.concatenate(layers_idx)

        filtered_train_activations = [a[:, layers_idx] for a in train_activations]
        filtered_test_activations = [a[:, layers_idx] for a in test_activations]

        train_activations = filtered_train_activations
        test_activations = filtered_test_activations

    # multiclass utils
    def src2idx(toks):
        uniq_toks = set().union(*toks)
        return {p: idx for idx, p in enumerate(uniq_toks)}

    def idx2src(srcidx):
        return {v: k for k, v in srcidx.items()}
    # -----------------------------------------------------


    def count_target_words(tokens):
        return sum([len(t) for t in tokens["target"]])
        
        
    def create_data(tokens, activations, mappings=None):

        num_tokens = count_target_words(tokens)
        print ("Number of tokens: ", num_tokens)
        
        num_neurons = activations[0].shape[1]
        
        source_tokens = tokens["source"]
        target_tokens = tokens["target"]

        ####### creating pos and source to index and reverse
        if mappings is not None:
            pos_idx, idx_pos, src_idx, idx_src = mappings
        else:
            pos_idx = src2idx(target_tokens)
            idx_pos= idx2src(pos_idx)
            src_idx = src2idx(source_tokens)
            idx_src = idx2src(src_idx)
        
        print ("length of POS dictionary: ", len(pos_idx))
        print ("length of source dictionary: ", len(src_idx))
        #######
        #
        
        X = np.zeros((num_tokens, num_neurons), dtype=np.float32)
        y = np.zeros((num_tokens,), dtype=np.int)

        example_set = set()
        
        idx = 0
        for instance_idx, instance in enumerate(target_tokens):
            for token_idx, token in enumerate(instance):
                if idx < num_tokens:
                    X[idx] = activations[instance_idx][token_idx, :]
                
                example_set.add(source_tokens[instance_idx][token_idx])
                if mappings is not None and target_tokens[instance_idx][token_idx] not in pos_idx:
                    y[idx] = pos_idx[args.task_specific_tag]
                else:
                    y[idx] = pos_idx[target_tokens[instance_idx][token_idx]]

                idx += 1

        print (idx)
        print("Total instances: %d"%(num_tokens))
        print(list(example_set)[:20])
        
        return X, y, (pos_idx, idx_pos, src_idx, idx_src)

    print("Creating train tensors...")
    X, y, mappings = create_data(train_tokens, train_activations)
    print (X.shape)
    print (y.shape)

    print("Creating test tensors...")
    X_test, y_test, mappings = create_data(test_tokens, test_activations, mappings)

    label2idx, idx2label, src2idx, idx2src = mappings

    print(np.sum(X))
    print(np.sum(y))

    print("Building model...")
    model = utils.train_logreg_model(X, y, lambda_l1=0.00001, lambda_l2=0.00001, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    train_accuracies = utils.evaluate_model(model, X, y, idx2label)
    test_accuracies, predictions = utils.evaluate_model(model, X_test, y_test, idx2label, return_predictions=True, source_tokens=test_tokens['source'])

    print("Calculating statistics...")
    label_counts = {}
    for s in train_tokens['target']:
        for t in s:
            label_counts[t] = label_counts.get(t, 0) + 1

    token_train_counts = {}
    for s in train_tokens['source']:
        for t in s:
            token_train_counts[t] = token_train_counts.get(t, 0) + 1

    token_counts = {}
    for s in test_tokens['source']:
        for t in s:
            token_counts[t] = token_train_counts.get(t, 0)

    print("Saving everything...")
    with open(os.path.join(args.output_dir, "model.pkl"), "wb") as fp:
        pickle.dump({
            'model': model,
            'label2idx': label2idx,
            'idx2label': idx2label,
            'src2idx': src2idx,
            'idx2src': idx2src
            }, fp)
    
    with open(os.path.join(args.output_dir, "train_accuracies.json"), "w") as fp:
        json.dump(train_accuracies, fp)

    with open(os.path.join(args.output_dir, "test_accuracies.json"), "w") as fp:
        json.dump(test_accuracies, fp)

    with open(os.path.join(args.output_dir, "label_counts.json"), "w") as fp:
        json.dump(label_counts, fp)

    with open(os.path.join(args.output_dir, "token_counts.json"), "w") as fp:
        json.dump(token_counts, fp)

    with open(os.path.join(args.output_dir, "test_predictions.json"), "w") as fp:
        json.dump(predictions, fp)

if __name__ == '__main__':
    main()



