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

import argparse
import collections
import json
import sys

import numpy as np
import torch
import h5py

sys.path.append("/export/work/static_embedding/software/transformers/src/")

from tqdm import tqdm
from transformers import (
    XLNetTokenizer,
    XLNetModel,
    GPT2Tokenizer,
    GPT2Model,
    XLMTokenizer,
    XLMModel,
    BertTokenizer,
    BertModel,
    RobertaTokenizer,
    RobertaModel,
    DistilBertTokenizer,
    DistilBertModel,
)


def get_model_and_tokenizer(
    model_name, device="cpu", random_weights=False, model_path=None
):
    """
    model_path: if given, initialize from path instead of official repo
    """

    init_model = model_name
    if model_path:
        print("Initializing model from local path:", model_path)
        init_model = model_path

    if model_name.startswith("xlnet"):
        model = XLNetModel.from_pretrained(init_model, output_hidden_states=True).to(
            device
        )
        tokenizer = XLNetTokenizer.from_pretrained(init_model)
        sep = u"▁"
    elif model_name.startswith("gpt2"):
        model = GPT2Model.from_pretrained(init_model, output_hidden_states=True).to(
            device
        )
        tokenizer = GPT2Tokenizer.from_pretrained(init_model)
        sep = "Ġ"
    elif model_name.startswith("xlm"):
        model = XLMModel.from_pretrained(init_model, output_hidden_states=True).to(
            device
        )
        tokenizer = XLMTokenizer.from_pretrained(init_model)
        sep = "</w>"
    elif model_name.startswith("bert"):
        model = BertModel.from_pretrained(init_model, output_hidden_states=True).to(
            device
        )
        tokenizer = BertTokenizer.from_pretrained(init_model)
        sep = "##"
    elif model_name.startswith("distilbert"):
        model = DistilBertModel.from_pretrained(
            init_model, output_hidden_states=True
        ).to(device)
        tokenizer = DistilBertTokenizer.from_pretrained(init_model)
        sep = "##"
    elif model_name.startswith("roberta"):
        model = RobertaModel.from_pretrained(model_name, output_hidden_states=True).to(
            device
        )
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        sep = "Ġ"
    else:
        print("Unrecognized model name:", model_name)
        sys.exit()

    if random_weights:
        print("Randomizing weights")
        model.init_weights()

    return model, tokenizer, sep


# aggregate_repr
# Function that aggregates activations/embeddings over a span of subword tokens
#
# Parameters:
#  state: Matrix of size [ NUM_LAYERS x NUM_SUBWORD_TOKENS_IN_SENT x LAYER_DIM]
#  start: index of the first subword of the word being processed
#  end: index of the last subword of the word being processed
#  aggregation: aggregation method
#
# Returns:
#  word_vector: Matrix of size [NUM_LAYERS x LAYER_DIM]
#
# This function will be called once per word. For example, if we had the sentence:
#   "This is an example"
# Tokenized by BPE:
#   "this is an ex @@am @@ple"
#
# The function will be called 4 times:
#   aggregate_repr(state, 0, 0, aggregation)
#   aggregate_repr(state, 1, 1, aggregation)
#   aggregate_repr(state, 2, 2, aggregation)
#   aggregate_repr(state, 3, 5, aggregation)
def aggregate_repr(state, start, end, aggregation):
    if aggregation == "first":
        return state[:, start, :]
    elif aggregation == "last":
        return state[:, end, :]
    elif aggregation == "average":
        return np.average(state[:, start : end + 1, :], axis=1)


# this follows the HuggingFace API for pytorch-transformers
def get_sentence_repr(
    sentence,
    model,
    tokenizer,
    sep,
    model_name,
    filter_vocab,
    device="cpu",
    include_embeddings=False,
    aggregation="last",
    tokenization_aggregation="average",
    filter_words=["[CLS]", "[PAD]", "[SEP]"],
):
    """
    Get representations for one sentence
    """

    with torch.no_grad():
        ids = tokenizer.encode(sentence, truncation=True, max_length=512)
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

    print("Original   : ", sentence.split(" "))
    print("Tokenized  : ", tokenizer.convert_ids_to_tokens(ids))

    special_tokens = tokenizer.all_special_tokens
    detokenized = []
    extracted_words = []
    # For each word, take the representation of its last sub-word
    segmented_tokens = tokenizer.convert_ids_to_tokens(ids)
    assert (
        len(segmented_tokens) == all_hidden_states.shape[1]
    ), "incompatible tokens and states"

    final_hidden_states = np.zeros(
        (all_hidden_states.shape[0], len(segmented_tokens), all_hidden_states.shape[2])
    )

    if (
        model_name.startswith("gpt2")
        or model_name.startswith("xlnet")
        or model_name.startswith("roberta")
    ):
        # if next token is a new word, take current token's representation
        # TODO: Fix for word vocab check
        if segmented_tokens[0].startswith(sep):
            current_word = segmented_tokens[0][len(sep) :]
        else:
            current_word = segmented_tokens[0]

        current_word_start_idx = 0
        current_word_end_idx = 0

        for i in range(len(segmented_tokens) - 1):
            if (
                segmented_tokens[i + 1].startswith(sep)
                or segmented_tokens[i] in special_tokens
                or segmented_tokens[i + 1] in special_tokens
            ):
                current_word_end_idx = i
                detokenized.append(current_word)
                if (filter_vocab is None or current_word.lower() in filter_vocab) and (current_word not in special_tokens):
                    final_hidden_states[:, len(extracted_words), :] = aggregate_repr(
                        all_hidden_states,
                        current_word_start_idx,
                        current_word_end_idx,
                        aggregation,
                    )
                    extracted_words.append(current_word)
                if segmented_tokens[i + 1].startswith(sep):
                    current_word = segmented_tokens[i + 1][len(sep) :]
                else:
                    current_word = segmented_tokens[i + 1]
                current_word_start_idx = i + 1
            else:
                current_word += segmented_tokens[i + 1]
        detokenized.append(current_word)
        if (filter_vocab is None or current_word.lower() in filter_vocab) and (current_word not in special_tokens):
            current_word_end_idx = len(segmented_tokens) - 1
            final_hidden_states[:, len(extracted_words), :] = aggregate_repr(
                all_hidden_states,
                current_word_start_idx,
                current_word_end_idx,
                aggregation,
            )
            extracted_words.append(current_word)
    elif model_name.startswith("bert") or model_name.startswith("distilbert"):
        # if next token is not a continuation, take current token's representation
        if segmented_tokens[0].startswith(sep):
            current_word = segmented_tokens[0][len(sep) :]
        else:
            current_word = segmented_tokens[0]
        current_word_start_idx = 0
        current_word_end_idx = 0

        for i in range(len(segmented_tokens) - 1):
            if (
                not segmented_tokens[i + 1].startswith(sep)
                or segmented_tokens[i] in special_tokens
                or segmented_tokens[i + 1] in special_tokens
            ):
                current_word_end_idx = i
                detokenized.append(current_word)
                if (filter_vocab is None or current_word.lower() in filter_vocab) and (current_word not in special_tokens):
                    final_hidden_states[:, len(extracted_words), :] = aggregate_repr(
                        all_hidden_states,
                        current_word_start_idx,
                        current_word_end_idx,
                        aggregation,
                    )
                    extracted_words.append(current_word)
                current_word = segmented_tokens[i + 1]
                current_word_start_idx = i + 1
            else:
                if segmented_tokens[i + 1].startswith(sep):
                    current_word += segmented_tokens[i + 1][len(sep) :]
                else:
                    current_word += segmented_tokens[i + 1]
        detokenized.append(current_word)
        if (filter_vocab is None or current_word.lower() in filter_vocab) and (current_word not in special_tokens):
            current_word_end_idx = len(segmented_tokens) - 1
            final_hidden_states[:, len(extracted_words), :] = aggregate_repr(
                all_hidden_states,
                current_word_start_idx,
                current_word_end_idx,
                aggregation,
            )
            extracted_words.append(current_word)
    else:
        print("Unrecognized model name:", model_name)
        sys.exit()

    all_hidden_states = final_hidden_states[:, : len(extracted_words), :]
    print("De-subword : ", detokenized)
    print("Extracted  : ", extracted_words)

    final_hidden_states = np.zeros(
        (
            all_hidden_states.shape[0],
            len(sentence.split(" ")),
            all_hidden_states.shape[2],
        )
    )

    # Tokenization aggregation
    final_extracted_tokens = []
    current_word_start_idx = 0
    current_word_end_idx = 0
    for original_token_idx, original_token in enumerate(sentence.split(" ")):
        concatenated_token = "".join(
            extracted_words[current_word_start_idx : current_word_end_idx + 1]
        )
        print("Tokenization concatenation:", concatenated_token, "vs", original_token)
        while len(concatenated_token) != len(original_token):
            current_word_end_idx += 1
            concatenated_token = "".join(
                extracted_words[current_word_start_idx : current_word_end_idx + 1]
            )

        final_hidden_states[:, original_token_idx, :] = aggregate_repr(
            all_hidden_states,
            current_word_start_idx,
            current_word_end_idx,
            tokenization_aggregation,
        )
        current_word_start_idx = current_word_end_idx + 1
        current_word_end_idx = current_word_start_idx
        final_extracted_tokens.append(concatenated_token)
    print("Restored   : ", final_extracted_tokens)
    assert len(sentence.split(" ")) == len(final_extracted_tokens)

    return final_hidden_states, final_extracted_tokens


# from https://github.com/nelson-liu/contextual-repr-analysis
def make_hdf5_file(sentence_to_index, vectors, output_file_path):
    with h5py.File(output_file_path, "w") as fout:
        for key, embeddings in vectors.items():
            fout.create_dataset(
                str(key), embeddings.shape, dtype="float32", data=embeddings
            )
        sentence_index_dataset = fout.create_dataset(
            "sentence_to_index", (1,), dtype=h5py.special_dtype(vlen=str)
        )
        sentence_index_dataset[0] = json.dumps(sentence_to_index)


HDF5_SPECIAL_TOKENS = {
    ".": "__DOT__",
    "/": "__SLASH__"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Name of model")
    parser.add_argument(
        "input_corpus", help="Text file path with one sentence per line"
    )
    parser.add_argument(
        "output_file",
        help="Output file path where extracted representations will be stored",
    )
    parser.add_argument(
        "--filter_vocab",
        default=None,
        help="Set of words for which the extraction should take place",
    )
    parser.add_argument(
        "--model_path", help="Local path to load custom model from", default=None
    )
    parser.add_argument(
        "--aggregation",
        help="first, last or average aggregation for word representation in the case of subword segmentation",
        default="last",
    )
    parser.add_argument(
        "--tokenization-aggregation",
        help="first, last or average aggregation for word representation in the case of extra tokenization from tokenizer",
        default="average",
    )
    parser.add_argument(
        "--limit-max-occurrences",
        help="Limit the maximum number of occurrences saved for each word (Only supported in hdf5 format)",
        default=-1,
        type=int
    )
    parser.add_argument(
        "--output-type",
        choices=["hdf5", "json"],
        default="hdf5",
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

    print("Loading model")
    model, tokenizer, sep = get_model_and_tokenizer(
        args.model_name,
        device=device,
        random_weights=args.random_weights,
        model_path=args.model_path,
    )

    print("Reading input corpus")

    def corpus_generator(input_corpus_path):
        with open(input_corpus_path, "r") as fp:
            for line in fp:
                yield line.strip()
            return

    print("Reading filter vocabulary")
    filter_vocab = None
    if args.filter_vocab:
        filter_vocab = set()
        with open(args.filter_vocab, "r") as fp:
            for line in fp:
                filter_vocab.add(line.strip().lower())

    print("Preparing output file")
    if args.output_type == "hdf5":
        if not args.output_file.endswith(".hdf5"):
            print(
                "[WARNING] Output filename (%s) does not end with .hdf5, but output file type is hdf5."
                % (args.output_file)
            )
        output_file = h5py.File(args.output_file, "w")
        #output_file.create_group("tokens")
        sentence_to_index = {}
    elif args.output_type == "json":
        if not args.output_file.endswith(".json"):
            print(
                "[WARNING] Output filename (%s) does not end with .json, but output file type is json."
                % (args.output_file)
            )
        output_file = open(args.output_file, "w", encoding="utf-8")

    print("Extracting representations from model")
    count = 0
    for sentence_idx, sentence in enumerate(corpus_generator(args.input_corpus)):
        hidden_states, extracted_words = get_sentence_repr(
            sentence,
            model,
            tokenizer,
            sep,
            args.model_name,
            filter_vocab,
            device=device,
            include_embeddings=(not args.ignore_embeddings),
            aggregation=args.aggregation,
            tokenization_aggregation=args.tokenization_aggregation,
        )

        print("Hidden states: ", hidden_states.shape)
        print("Number Extracted words: ", len(extracted_words))
        #print ("One" , hidden_states[0], hidden_states[0].shape )
        #print ("Two", hidden_states[0][0].shape)
        #print("Number Extracted words: ", len(extracted_words), sentence)

        if args.output_type == "hdf5":

            sentence_to_index[sentence] = str(count)
            output_file.create_dataset(str(count),
                                hidden_states.shape,
                                dtype='float32',
                                data=hidden_states)
            count = count + 1
        elif args.output_type == "json":
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

    if args.output_type == "hdf5":
        sentence_index_dataset = output_file.create_dataset("sentence_to_index", (1,), dtype=h5py.special_dtype(vlen=str))
        sentence_index_dataset[0] = json.dumps(sentence_to_index)

    output_file.close()


if __name__ == "__main__":
    main()
