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
from tqdm import tqdm, tqdm_notebook, tnrange

# Import lib
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import aux_classifier.utils as utils
import aux_classifier.representations as repr
import aux_classifier.data_loader as data_loader

def load_data_and_train(train_source, train_aux_source, train_labels, train_activations,
                        test_source, test_aux_source, test_labels, test_activations,
                        exp_type, task_specific_tag, max_sent_l, n_epochs, batch_size,
                        is_brnn, filter_layers, ignore_start_token, num_neurons_per_layer, lambda_l1, lambda_l2):
    print("Loading activations...")
    train_activations, NUM_LAYERS = data_loader.load_activations(train_activations, num_neurons_per_layer, is_brnn=is_brnn)
    test_activations, _ = data_loader.load_activations(test_activations, num_neurons_per_layer, is_brnn=is_brnn)
    print("Number of train sentences: %d"%(len(train_activations)))
    print("Number of test sentences: %d"%(len(test_activations)))

    if exp_type == 'word' or exp_type == 'charcnn' or exp_type == 'sent_last':
        train_tokens = data_loader.load_data(train_source, train_labels, train_activations, max_sent_l, ignore_start_token=ignore_start_token, sentence_classification=(exp_type == 'sent_last'))
        test_tokens = data_loader.load_data(test_source, test_labels, test_activations, max_sent_l, ignore_start_token=ignore_start_token, sentence_classification=(exp_type == 'sent_last'))
    else:
        # TODO: Implement sentence classification for subword systems
        train_tokens = data_loader.load_aux_data(train_source, train_labels, train_aux_source, train_activations, max_sent_l, ignore_start_token=ignore_start_token)
        test_tokens = data_loader.load_aux_data(test_source, test_labels, test_aux_source, test_activations, max_sent_l, ignore_start_token=ignore_start_token)

    NUM_TOKENS = sum([len(t) for t in train_tokens['target']])
    print('Number of total train tokens: %d'%(NUM_TOKENS))

    if exp_type != 'word' and exp_type != 'charcnn' and exp_type != 'sent_last':
        NUM_SOURCE_AUX_TOKENS = sum([len(t) for t in train_tokens['source_aux']])
        print('Number of AUX source words: %d'%(NUM_SOURCE_AUX_TOKENS)) 

    NUM_SOURCE_TOKENS = sum([len(t) for t in train_tokens['source']])
    print('Number of source words: %d'%(NUM_SOURCE_TOKENS)) 

    NUM_NEURONS = train_activations[0].shape[1]
    print('Number of neurons: %d'%(NUM_NEURONS))
    print("Number of layers: %d" % (NUM_LAYERS))

    if exp_type == 'bpe_avg':
        train_activations = repr.bpe_get_avg_activations(train_tokens, train_activations)
        test_activations = repr.bpe_get_avg_activations(test_tokens, test_activations)
    elif exp_type == 'bpe_last':
        train_activations = repr.bpe_get_last_activations(train_tokens, train_activations, is_brnn=is_brnn)
        test_activations = repr.bpe_get_last_activations(test_tokens, test_activations, is_brnn=is_brnn)
    elif exp_type == 'char_avg':
        train_activations = repr.char_get_avg_activations(train_tokens, train_activations)
        test_activations = repr.char_get_avg_activations(test_tokens, test_activations)
    elif exp_type == 'char_last':
        train_activations = repr.char_get_last_activations(train_tokens, train_activations, is_brnn=is_brnn)
        test_activations = repr.char_get_last_activations(test_tokens, test_activations, is_brnn=is_brnn)
    elif exp_type == 'sent_last':
        train_activations = repr.sent_get_last_activations(train_tokens, train_activations, is_brnn=is_brnn)
        test_activations = repr.sent_get_last_activations(test_tokens, test_activations, is_brnn=is_brnn)
        train_tokens['source'] = [['sent_%d' % (i)] for i in range(len(train_activations))]
        test_tokens['source'] = [['sent_%d' % (i)] for i in range(len(test_activations))]

    # Filtering
    if filter_layers:
        train_activations, test_activations = utils.filter_activations_by_layers(train_activations, test_activations, filter_layers, num_neurons_per_layer, NUM_LAYERS, is_brnn)
        print("Filtered number of neurons: %d" % (train_activations[0].shape[1]))

    print("Creating train tensors...")
    X, y, mappings = utils.create_tensors(train_tokens, train_activations, task_specific_tag)
    print (X.shape)
    print (y.shape)

    print("Creating test tensors...")
    X_test, y_test, mappings = utils.create_tensors(test_tokens, test_activations, task_specific_tag, mappings)

    label2idx, idx2label, src2idx, idx2src = mappings

    print("Building model...")
    model = utils.train_logreg_model(X, y, lambda_l1=lambda_l1, lambda_l2=lambda_l2, num_epochs=n_epochs, batch_size=batch_size)
    train_accuracies = utils.evaluate_model(model, X, y, idx2label)
    test_accuracies, predictions = utils.evaluate_model(model, X_test, y_test, idx2label, return_predictions=True, source_tokens=test_tokens['source'])

    return model, label2idx, idx2label, src2idx, idx2src, train_accuracies, test_accuracies, predictions, train_tokens, test_tokens

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
                    choices=['word', 'charcnn', 'bpe_avg', 'bpe_last', 'char_avg', 'char_last', 'sent_last'],
                    default='word', required=True,
                    help='Type of experiment')

    parser.add_argument('--task-specific-tag', dest='task_specific_tag', 
                    required=True, help='Tag incase test has unknown tags')

    parser.add_argument('--max-sent-l', dest='max_sent_l', type=int,
                    default=250, help='Maximum sentence length')
    parser.add_argument('--is-unidirectional', dest='is_brnn', action='store_false',
                    default=True, help='Include this flag if original model is unidirectional, \
                                or if the representations are from the decoder side')
    parser.add_argument('--num-neurons-per-layer', dest='num_neurons_per_layer', type=int,
                    default=500, help='Number of neurons in each layer')

    parser.add_argument('--output-dir', dest='output_dir', 
                    required=True, help='Location to save all results')

    parser.add_argument('--filter-layers', dest='filter_layers', default=None,
                    type=str, help='Use specific layers for training. Format: f1,b1,f2,b2')

    parser.add_argument('--ignore-start-token', dest='ignore_start_token', default=False,
                    action='store_true', help='Ignore the first token of every sentence')

    parser.add_argument('--lambda1', dest='lambda_l1', default=0.00001,
                    type=float, help='Regularizer weight L1')
    parser.add_argument('--lambda2', dest='lambda_l2', default=0.00001,
                    type=float, help='Regularizer weight L2')

    args = parser.parse_args()

    if args.train_activations.split('.')[-1] == "pt" and args.filter_layers:
        _layers = [x.strip() for x in args.filter_layers.strip().split(',')]
        for l in _layers:
            if l[0] == 'f':
                assert "b%s"%(l[1]) in _layers, "OpenNMT-py does not support separating forward and backward layers. Please specify both fX,bX for layer X."


    print("Creating output directory...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Constants
    NUM_EPOCHS = 10
    BATCH_SIZE = 512

    result = load_data_and_train(args.train_source, args.train_aux_source, args.train_labels, args.train_activations,
                        args.test_source, args.test_aux_source, args.test_labels, args.test_activations,
                        args.exp_type, args.task_specific_tag, args.max_sent_l, NUM_EPOCHS, BATCH_SIZE,
                        args.is_brnn, args.filter_layers, args.ignore_start_token, args.num_neurons_per_layer,
                        args.lambda_l1, args.lambda_l2)

    model, label2idx, idx2label, src2idx, idx2src, train_accuracies, test_accuracies, test_predictions, train_tokens, test_tokens = result

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
        json.dump(test_predictions, fp)

if __name__ == '__main__':
    main()



