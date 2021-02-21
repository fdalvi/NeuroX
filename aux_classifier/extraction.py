## Script that takes input sentences, extracts activations [all layers] for all words
# in a given vocab, aggregates activations over subwords, and saves all tokens with
# their activations [N=occurrence of the word] in an hdf5 structure for efficient
# retrieval
#
# HDF5 structure:
# - tokens
#    - token_1
#        - 0 -> [13 * 768] matrix
#        - 1 -> [13 * 768] matrix
#    - token_2
#        - 0 -> [13 * 768] matrix
#        - 1 -> [13 * 768] matrix
#        - 2 -> [13 * 768] matrix
#        - 3 -> [13 * 768] matrix
#
# In the above case, `token_1` occurs 2 times in the dataset, and `token_2` occurs
# 4 times. We have 13 layers from BERT and 768 dimensions from each layer
#
# Author: Fahim Dalvi
# Last Modified: 2 March, 2020
# Last Modified: 9 September, 2020
# Last Modified: 15 September, 2020
# Last Modified: 1 February, 2020

import argparse
import collections
import json
import sys

import numpy as np
import torch
import h5py

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

## Globals
tokenization_counts = {}
MAX_SEQ_LEN = 512


def get_model_and_tokenizer(model_desc, device="cpu", random_weights=False):
    """
    Automatically get the appropriate ``transformers`` model and tokenizer based
    on the model description

    Parameters
    ----------
    model_desc : str
        Model description; can either be a model name like ``bert-base-uncased``
        or a path to a trained model
    
    device : str, optional
        Device to load the model on, cpu or gpu. Default is cpu.

    random_weights : bool, optional
        Whether the weights of the model should be randomized. Useful for analyses
        where one needs an untrained model.

    Returns
    -------
    model : transformers model
        An instance of one of the transformers.modeling classes
    tokenizer : transformers tokenizer
        An instance of one of the transformers.tokenization classes
    """
    model = AutoModel.from_pretrained(model_desc, output_hidden_states=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_desc)

    if random_weights:
        print("Randomizing weights")
        model.init_weights()

    return model, tokenizer


def aggregate_repr(state, start, end, aggregation):
    """
    Function that aggregates activations/embeddings over a span of subword tokens.
    This function will usually be called once per word. For example, if we had the sentence:
        "This is an example"
    Tokenized by BPE:
        "this is an ex @@am @@ple"

    The function should be called 4 times:
        aggregate_repr(state, 0, 0, aggregation)
        aggregate_repr(state, 1, 1, aggregation)
        aggregate_repr(state, 2, 2, aggregation)
        aggregate_repr(state, 3, 5, aggregation)
    
    Parameters
    ----------
    state : numpy.ndarray
        Matrix of size [ NUM_LAYERS x NUM_SUBWORD_TOKENS_IN_SENT x LAYER_DIM]
    start : int
        Index of the first subword of the word being processed
    end : int
        Index of the last subword of the word being processed
    aggregation : {'first', 'last', 'average'}
        Aggregation method for combining subword activations

    Returns
    -------
    word_vector : numpy. ndarray
        Matrix of size [NUM_LAYERS x LAYER_DIM]
    """
    if aggregation == "first":
        return state[:, start, :]
    elif aggregation == "last":
        return state[:, end, :]
    elif aggregation == "average":
        return np.average(state[:, start : end + 1, :], axis=1)


# this follows the HuggingFace API for transformers
def get_sentence_repr(
    sentence,
    model,
    tokenizer,
    device="cpu",
    include_embeddings=False,
    aggregation="last",
):
    """
    Get representations for one sentence
    """

    special_tokens = [
        x for x in tokenizer.all_special_tokens if x != tokenizer.unk_token
    ]
    special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)

    original_tokens = sentence.split(" ")
    # Add a letter and space before each word since some tokenizers are space sensitive
    tmp_tokens = [
        "a" + " " + x if x_idx != 0 else x for x_idx, x in enumerate(original_tokens)
    ]
    assert len(original_tokens) == len(tmp_tokens)

    with torch.no_grad():
        # Get tokenization counts if not already available
        for token_idx, token in enumerate(tmp_tokens):
            tok_ids = [
                x for x in tokenizer.encode(token) if x not in special_tokens_ids
            ]
            if token_idx != 0:
                # Ignore the first token (added letter)
                tok_ids = tok_ids[1:]

            if token in tokenization_counts:
                assert tokenization_counts[token] == len(
                    tok_ids
                ), "Got different tokenization for already processed word"
            else:
                tokenization_counts[token] = len(tok_ids)
        ids = tokenizer.encode(sentence, truncation=True, max_length=MAX_SEQ_LEN)
        input_ids = torch.tensor([ids]).to(device)
        # Hugging Face format: list of torch.FloatTensor of shape (batch_size, sequence_length, hidden_size) (hidden_states at output of each layer plus initial embedding outputs)
        all_hidden_states = model(input_ids)[-1]
        # convert to format required for contexteval: numpy array of shape (num_layers, sequence_length, representation_dim)
        if include_embeddings:
            all_hidden_states = [
                hidden_states[0].cpu().numpy() for hidden_states in all_hidden_states
            ]
        else:
            all_hidden_states = [
                hidden_states[0].cpu().numpy()
                for hidden_states in all_hidden_states[:-1]
            ]
        all_hidden_states = np.array(all_hidden_states)

    print('Sentence         : "%s"' % (sentence))
    print("Original    (%03d): %s" % (len(original_tokens), original_tokens))
    print(
        "Tokenized   (%03d): %s"
        % (
            len(tokenizer.convert_ids_to_tokens(ids)),
            tokenizer.convert_ids_to_tokens(ids),
        )
    )

    ids_without_special_tokens = [x for x in ids if x not in special_tokens_ids]
    segmented_tokens = tokenizer.convert_ids_to_tokens(ids_without_special_tokens)

    counter = 0
    detokenized = []
    final_hidden_states = np.zeros(
        (all_hidden_states.shape[0], len(original_tokens), all_hidden_states.shape[2])
    )

    for token_idx, token in enumerate(tmp_tokens):
        current_word_start_idx = counter
        current_word_end_idx = counter + tokenization_counts[token]
        final_hidden_states[:, len(detokenized), :] = aggregate_repr(
            all_hidden_states,
            current_word_start_idx,
            current_word_end_idx - 1,
            aggregation,
        )
        detokenized.append(
            "".join(segmented_tokens[current_word_start_idx:current_word_end_idx])
        )

        counter += tokenization_counts[token]

    print("Detokenized (%03d): %s" % (len(detokenized), detokenized))
    print("Counter: %d" % (counter))

    if len(ids) >= 512:
        print("[WARNING] Input truncated because of length, skipping check")
    else:
        assert counter == len(ids_without_special_tokens)
        assert len(detokenized) == len(original_tokens)
    print("===================================================================")

    return final_hidden_states, detokenized


def extract_representations(
    model_desc,
    input_corpus,
    output_file,
    device="cpu",
    aggregation="last",
    output_type="json",
    random_weights=False,
    ignore_embeddings=False,
):
    print("Loading model")
    model, tokenizer = get_model_and_tokenizer(
        model_desc, device=device, random_weights=random_weights
    )

    print("Reading input corpus")

    def corpus_generator(input_corpus_path):
        with open(input_corpus_path, "r") as fp:
            for line in fp:
                yield line.strip()
            return

    print("Preparing output file")
    if output_type == "hdf5":
        if not output_file.endswith(".hdf5"):
            print(
                "[WARNING] Output filename (%s) does not end with .hdf5, but output file type is hdf5."
                % (output_file)
            )
        output_file = h5py.File(output_file, "w")
        sentence_to_index = {}
    elif output_type == "json":
        if not output_file.endswith(".json"):
            print(
                "[WARNING] Output filename (%s) does not end with .json, but output file type is json."
                % (output_file)
            )
        output_file = open(output_file, "w", encoding="utf-8")

    print("Extracting representations from model")
    for sentence_idx, sentence in enumerate(corpus_generator(input_corpus)):
        hidden_states, extracted_words = get_sentence_repr(
            sentence,
            model,
            tokenizer,
            device=device,
            include_embeddings=(not ignore_embeddings),
            aggregation=aggregation,
        )

        print("Hidden states: ", hidden_states.shape)
        print("# Extracted words: ", len(extracted_words))

        if output_type == "hdf5":
            output_file.create_dataset(
                str(sentence_idx), hidden_states.shape, dtype="float32", data=hidden_states
            )
            sentence_to_index[sentence] = str(sentence_idx)
        elif output_type == "json":
            output_json = collections.OrderedDict()
            output_json["linex_index"] = sentence_idx
            all_out_features = []

            for word_idx, extracted_word in enumerate(extracted_words):
                all_layers = []
                for layer_idx in range(hidden_states.shape[0]):
                    layers = collections.OrderedDict()
                    layers["index"] = layer_idx
                    layers["values"] = [
                        round(x.item(), 8)
                        for x in hidden_states[layer_idx, word_idx, :]
                    ]
                    all_layers.append(layers)
                out_features = collections.OrderedDict()
                out_features["token"] = extracted_word
                out_features["layers"] = all_layers
                all_out_features.append(out_features)
            output_json["features"] = all_out_features
            output_file.write(json.dumps(output_json) + "\n")

    if output_type == "hdf5":
        sentence_index_dataset = output_file.create_dataset(
            "sentence_to_index", (1,), dtype=h5py.special_dtype(vlen=str)
        )
        sentence_index_dataset[0] = json.dumps(sentence_to_index)

    output_file.close()


HDF5_SPECIAL_TOKENS = {".": "__DOT__", "/": "__SLASH__"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_desc", help="Name of model")
    parser.add_argument(
        "input_corpus", help="Text file path with one sentence per line"
    )
    parser.add_argument(
        "output_file",
        help="Output file path where extracted representations will be stored",
    )
    parser.add_argument(
        "--aggregation",
        help="first, last or average aggregation for word representation in the case of subword segmentation",
        default="last",
    )
    parser.add_argument(
        "--output-type",
        choices=["hdf5", "json"],
        default="json",
        help="Output format of the extracted representations",
    )
    parser.add_argument("--decompose_layers", action="store_true")
    parser.add_argument("--disable_cuda", action="store_true")
    parser.add_argument("--ignore_embeddings", action="store_true")
    parser.add_argument(
        "--random_weights",
        action="store_true",
        help="generate representations from randomly initialized model",
    )
    args = parser.parse_args()

    assert args.aggregation in [
        "average",
        "first",
        "last",
    ], "Invalid aggregation option, please specify first, average or last."

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    extract_representations(
        args.model_desc,
        args.input_corpus,
        args.output_file,
        device=device,
        aggregation=args.aggregation,
        output_type=args.output_type,
        random_weights=args.random_weights,
        ignore_embeddings=args.ignore_embeddings,
    )


if __name__ == "__main__":
    main()
