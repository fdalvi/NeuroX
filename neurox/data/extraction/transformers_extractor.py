"""Representations Extractor for ``transformers`` toolkit models.

Module that given a file with input sentences and a ``transformers``
model, extracts representations from all layers of the model. The script
supports aggregation over sub-words created due to the tokenization of
the provided model.

Can also be invoked as a script as follows:
    ``python -m neurox.data.extraction.transformers_extractor``
"""

import argparse
import sys

import numpy as np
import torch

from neurox.data.writer import ActivationsWriter

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def get_model_and_tokenizer(model_desc, device="cpu", random_weights=False):
    """
    Automatically get the appropriate ``transformers`` model and tokenizer based
    on the model description

    Parameters
    ----------
    model_desc : str
        Model description; can either be a model name like ``bert-base-uncased``,
        a comma separated list indicating <model>,<tokenizer> (since 1.0.8),
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
    model_desc = model_desc.split(",")
    if len(model_desc) == 1:
        model_name = model_desc[0]
        tokenizer_name = model_desc[0]
    else:
        model_name = model_desc[0]
        tokenizer_name = model_desc[1]
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if random_weights:
        print("Randomizing weights")
        model.init_weights()

    return model, tokenizer


def aggregate_repr(state, start, end, aggregation):
    """
    Function that aggregates activations/embeddings over a span of subword tokens.
    This function will usually be called once per word. For example, if we had the sentence::

        This is an example

    which is tokenized by BPE into::

        this is an ex @@am @@ple

    The function should be called 4 times::

        aggregate_repr(state, 0, 0, aggregation)
        aggregate_repr(state, 1, 1, aggregation)
        aggregate_repr(state, 2, 2, aggregation)
        aggregate_repr(state, 3, 5, aggregation)

    Returns a zero vector if end is less than start, i.e. the request is to
    aggregate over an empty slice.

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
    word_vector : numpy.ndarray
        Matrix of size [NUM_LAYERS x LAYER_DIM]
    """
    if end < start:
        sys.stderr.write(
            "WARNING: An empty slice of tokens was encountered. "
            + "This probably implies a special unicode character or text "
            + "encoding issue in your original data that was dropped by the "
            + "transformer model's tokenizer.\n"
        )
        return np.zeros((state.shape[0], state.shape[2]))
    if aggregation == "first":
        return state[:, start, :]
    elif aggregation == "last":
        return state[:, end, :]
    elif aggregation == "average":
        return np.average(state[:, start : end + 1, :], axis=1)


def extract_sentence_representations(
    sentence_pair,
    model,
    tokenizer,
    device="cpu",
    include_embeddings=True,
    aggregation="last",
    dtype="float32",
    include_special_tokens=False,
    seq2seq_component=None,
    tokenization_counts={},
    context_indicator="a",
):
    """
    Get representations for a single sentence

    The extractor runs a detokenization procedure to combine subwords
    automatically. For instance, a sentence "Hello, how are you?" may be
    tokenized by the model as "Hell @@o , how are you @@?". This extractor
    automatically detokenizes the subtokens back into the original token.


    Parameters
    ----------
    sentence : str
        Sentence for which the extraction needs to be done. The returned output
        will have representations for exactly the same number of elements as
        tokens in this sentence (counted by `sentence.split(' ')`).

    model : transformers model
        An instance of one of the transformers.modeling classes

    tokenizer : transformers tokenizer
        An instance of one of the transformers.tokenization classes

    device : str, optional
        Specifies the device (CPU/GPU) on which the extraction should be
        performed. Defaults to 'cpu'

    include_embeddings : bool, optional
        Whether the embedding layer should be included in the final output, or
        just regular layers. Defaults to True

    aggregation : {'first', 'last', 'average'}, optional
        Aggregation method for combining subword activations. Defaults to 'last'

    dtype : str, optional
        Data type in which the activations will be stored. Supports all numpy
        based tensor types. Common values are 'float32' and 'float16'. Defaults
        to 'float16'

    include_special_tokens : bool, optional
        Whether or not to special tokens in the extracted representations.
        Special tokens are tokens not present in the original sentence, but are
        added by the tokenizer, such as [CLS], [SEP] etc.

    tokenization_counts : dict, optional
        Tokenization counts to use across a dataset for efficiency

    context_indicator : str, optional
        A token that is guaranteed to be a unbroken after tokenization. Usually an
        element from the tokenizer's vocab. Defaults to 'a' which works for most
        models, but may break if 'a' is split into multiple sub-tokens (e.g. breaks
        for `google/mt5-base`).

    Returns
    -------
    final_hidden_states : numpy.ndarray
        Numpy Matrix of size [``NUM_LAYERs`` x ``NUM_TOKENS`` x ``NUM_NEURONS``].

    detokenizer : list
        List of detokenized words. This will have the same number of elements as
        tokens in the original sentence, plus special tokens if requested. Each element
        preserves tokenization artifacts (such as `##`, `@@` etc) to enable further
        automatic processing.
    """

    encoder_sentence, decoder_sentence = sentence_pair

    all_sentences = {}
    all_sentences["encoder"] = encoder_sentence
    all_sentences["decoder"] = decoder_sentence

    all_keys = []
    if seq2seq_component == "both":
        all_keys = ["encoder", "decoder"]
    elif seq2seq_component == "encoder":
        all_keys = ["encoder"]
    else:
        all_keys = ["decoder"]

    special_tokens = [
        x for x in tokenizer.all_special_tokens if x != tokenizer.unk_token
    ]
    special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)

    all_original_tokens = {}
    for key in all_keys:
        all_original_tokens[key] = all_sentences[key].split(" ")

    # Add letters and spaces around each word since some tokenizers are context sensitive
    all_tmp_tokens = {}
    for key in all_keys:
        all_tmp_tokens[key] = []
        if len(all_original_tokens[key]) > 0:
            all_tmp_tokens[key].append(
                f"{all_original_tokens[key][0]} {context_indicator}"
            )
        all_tmp_tokens[key] += [
            f"{context_indicator} {x} {context_indicator}"
            for x in all_original_tokens[key][1:-1]
        ]
        if len(all_original_tokens[key]) > 1:
            all_tmp_tokens[key].append(
                f"{context_indicator} {all_original_tokens[key][-1]}"
            )

        assert len(all_original_tokens[key]) == len(
            all_tmp_tokens[key]
        ), f"Original: {all_original_tokens[key]}, Temp: {all_tmp_tokens[key]}"

    all_ids = {}
    with torch.no_grad():
        # Get tokenization counts if not already available
        for key in all_keys:
            for token_idx, token in enumerate(all_tmp_tokens[key]):
                tok_ids = [
                    x for x in tokenizer.encode(token) if x not in special_tokens_ids
                ]
                # Ignore the added letter tokens
                if token_idx != 0 and token_idx != len(all_tmp_tokens[key]) - 1:
                    # Word appearing in the middle of the sentence
                    tok_ids = tok_ids[1:-1]
                elif token_idx == 0:
                    # Word appearing at the beginning
                    tok_ids = tok_ids[:-1]
                else:
                    # Word appearing at the end
                    tok_ids = tok_ids[1:]

                if token in tokenization_counts:
                    assert tokenization_counts[token] == len(
                        tok_ids
                    ), "Got different tokenization for already processed word"
                else:
                    tokenization_counts[token] = len(tok_ids)

        # Hugging Face format: tuple of torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
        # Tuple has 13 elements for base model: embedding outputs + hidden states at each layer
        hidden_states = {}
        if seq2seq_component == "encoder":
            all_ids["encoder"] = tokenizer.encode(encoder_sentence, truncation=True)
            encoder_input_ids = torch.tensor([all_ids["encoder"]]).to(device)

            all_ids["decoder"] = tokenizer.encode("", truncation=True)
            decoder_input_ids = torch.tensor([all_ids["decoder"]]).to(device)
            model_outputs = model(
                encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
            )
            hidden_states["encoder"] = model_outputs.encoder_hidden_states
        elif seq2seq_component == "decoder" or seq2seq_component == "both":
            all_ids["encoder"] = tokenizer.encode(encoder_sentence, truncation=True)
            encoder_input_ids = torch.tensor([all_ids["encoder"]]).to(device)

            all_ids["decoder"] = tokenizer.encode(decoder_sentence, truncation=True)
            decoder_input_ids = torch.tensor([all_ids["decoder"]]).to(device)
            model_outputs = model(
                encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
            )
            if seq2seq_component == "both":
                hidden_states["encoder"] = model_outputs.encoder_hidden_states
            hidden_states["decoder"] = model_outputs.decoder_hidden_states
        else:
            raise NotImplementedError

        if include_embeddings:
            for key in hidden_states:
                hidden_states[key] = [
                    hidden_states[0].cpu().numpy()
                    for hidden_states in hidden_states[key]
                ]
        else:
            for key in hidden_states:
                hidden_states[key] = [
                    hidden_states[0].cpu().numpy()
                    for hidden_states in hidden_states[key][1:]
                ]

        for key in hidden_states:
            hidden_states[key] = np.array(hidden_states[key], dtype=dtype)

    for key in all_keys:
        print('(%s) Sentence         : "%s"' % (key, all_sentences[key]))
        print(
            "(%s) Original    (%03d): %s"
            % (key, len(all_original_tokens[key]), all_original_tokens[key])
        )
        print(
            "(%s) Tokenized   (%03d): %s"
            % (
                key,
                len(tokenizer.convert_ids_to_tokens(all_ids[key])),
                tokenizer.convert_ids_to_tokens(all_ids[key]),
            )
        )

        assert key not in hidden_states or hidden_states[key].shape[1] == len(
            all_ids[key]
        )

    # Handle special tokens
    # filtered_ids will contain all ids if we are extracting with
    #  special tokens, and only normal word/subword ids if we are
    #  extracting without special tokens
    # all_hidden_states will also be filtered at this step to match
    #  the ids in filtered ids

    final_hidden_states = {}
    detokenized = {}
    for key in all_keys:
        ids = all_ids[key]
        tmp_tokens = all_tmp_tokens[key]
        original_tokens = all_original_tokens[key]
        all_hidden_states = hidden_states[key]
        filtered_ids = ids
        idx_special_tokens = [
            t_i for t_i, x in enumerate(ids) if x in special_tokens_ids
        ]
        special_token_ids = [ids[t_i] for t_i in idx_special_tokens]

        if not include_special_tokens:
            idx_without_special_tokens = [
                t_i for t_i, x in enumerate(ids) if x not in special_tokens_ids
            ]
            filtered_ids = [ids[t_i] for t_i in idx_without_special_tokens]
            all_hidden_states = all_hidden_states[:, idx_without_special_tokens, :]
            special_token_ids = []

        assert all_hidden_states.shape[1] == len(filtered_ids)
        print(
            "Filtered   (%03d): %s"
            % (
                len(tokenizer.convert_ids_to_tokens(filtered_ids)),
                tokenizer.convert_ids_to_tokens(filtered_ids),
            )
        )

        # Get actual tokens for filtered ids in order to do subword
        #  aggregation
        segmented_tokens = tokenizer.convert_ids_to_tokens(filtered_ids)

        # Perform subword aggregation/detokenization
        #  After aggregation, we should have |original_tokens| embeddings,
        #  one for each word. If special tokens are included, then we will
        #  have |original_tokens| + |special_tokens|
        counter = 0
        detokenized[key] = []
        final_hidden_states[key] = np.zeros(
            (
                all_hidden_states.shape[0],
                len(original_tokens) + len(special_token_ids),
                all_hidden_states.shape[2],
            ),
            dtype=dtype,
        )
        inputs_truncated = False

        # Keep track of what the previous token was. This is used to detect
        #  special tokens followed/preceeded by dropped tokens, which is an
        #  ambiguous situation for the detokenizer
        prev_token_type = "NONE"

        last_special_token_pointer = 0
        for token_idx, token in enumerate(tmp_tokens):
            # Handle special tokens
            if include_special_tokens and tokenization_counts[token] != 0:
                if last_special_token_pointer < len(idx_special_tokens):
                    while (
                        last_special_token_pointer < len(idx_special_tokens)
                        and counter == idx_special_tokens[last_special_token_pointer]
                    ):
                        assert prev_token_type != "DROPPED", (
                            "A token dropped by the tokenizer appeared next "
                            + "to a special token. Detokenizer cannot resolve "
                            + f"the ambiguity, please remove '{sentence}' from"
                            + "the dataset, or try a different tokenizer"
                        )
                        prev_token_type = "SPECIAL"
                        final_hidden_states[key][
                            :, len(detokenized[key]), :
                        ] = all_hidden_states[:, counter, :]
                        detokenized[key].append(
                            segmented_tokens[
                                idx_special_tokens[last_special_token_pointer]
                            ]
                        )
                        last_special_token_pointer += 1
                        counter += 1

            current_word_start_idx = counter
            current_word_end_idx = counter + tokenization_counts[token]

            # Check for truncated hidden states in the case where the
            # original word was actually tokenized
            if (
                tokenization_counts[token] != 0
                and current_word_start_idx >= all_hidden_states.shape[1]
            ) or current_word_end_idx > all_hidden_states.shape[1]:
                final_hidden_states[key] = final_hidden_states[key][
                    :,
                    : len(detokenized[key])
                    + len(special_token_ids)
                    - last_special_token_pointer,
                    :,
                ]
                inputs_truncated = True
                break

            if tokenization_counts[token] == 0:
                assert prev_token_type != "SPECIAL", (
                    "A token dropped by the tokenizer appeared next "
                    + "to a special token. Detokenizer cannot resolve "
                    + f"the ambiguity, please remove '{sentence}' from"
                    + "the dataset, or try a different tokenizer"
                )
                prev_token_type = "DROPPED"
            else:
                prev_token_type = "NORMAL"

            final_hidden_states[key][:, len(detokenized[key]), :] = aggregate_repr(
                all_hidden_states,
                current_word_start_idx,
                current_word_end_idx - 1,
                aggregation,
            )
            detokenized[key].append(
                "".join(segmented_tokens[current_word_start_idx:current_word_end_idx])
            )
            counter += tokenization_counts[token]

        if include_special_tokens:
            while counter < len(segmented_tokens):
                if last_special_token_pointer >= len(idx_special_tokens):
                    break

                if counter == idx_special_tokens[last_special_token_pointer]:
                    assert prev_token_type != "DROPPED", (
                        "A token dropped by the tokenizer appeared next "
                        + "to a special token. Detokenizer cannot resolve "
                        + f"the ambiguity, please remove '{sentence}' from"
                        + "the dataset, or try a different tokenizer"
                    )
                    prev_token_type = "SPECIAL"
                    final_hidden_states[key][
                        :, len(detokenized[key]), :
                    ] = all_hidden_states[:, counter, :]
                    detokenized[key].append(
                        segmented_tokens[idx_special_tokens[last_special_token_pointer]]
                    )
                    last_special_token_pointer += 1
                counter += 1

        print(
            "(%s) Detokenized (%03d): %s"
            % (key, len(detokenized[key]), detokenized[key])
        )
        print("(%s) Counter: %d" % (key, counter))

        if inputs_truncated:
            print("WARNING: Input truncated because of length, skipping check")
        else:
            assert counter == len(filtered_ids)
            assert len(detokenized[key]) == len(original_tokens) + len(
                special_token_ids
            )
    print("===================================================================")

    return final_hidden_states, detokenized


def extract_representations(
    model_desc,
    encoder_input_corpus,
    decoder_input_corpus,
    output_file,
    device="cpu",
    aggregation="last",
    output_type="json",
    random_weights=False,
    ignore_embeddings=False,
    decompose_layers=False,
    filter_layers=None,
    dtype="float32",
    include_special_tokens=False,
    seq2seq_component=None,
):
    """
    Extract representations for an entire corpus and save them to disk

    Parameters
    ----------
    model_desc : str
        Model description; can either be a model name like ``bert-base-uncased``,
        a comma separated list indicating <model>,<tokenizer> (since 1.0.8),
        or a path to a trained model

    input_corpus : str
        Path to the input corpus, where each sentence is on its separate line

    output_file : str
        Path to output file. Supports all filetypes supported by
        ``data.writer.ActivationsWriter``.

    device : str, optional
        Specifies the device (CPU/GPU) on which the extraction should be
        performed. Defaults to 'cpu'

    aggregation : {'first', 'last', 'average'}, optional
        Aggregation method for combining subword activations. Defaults to 'last'

    output_type : str, optional
        Explicit definition of output file type if it cannot be derived from the
        ``output_file`` path

    random_weights : bool, optional
        Whether the weights of the model should be randomized. Useful for analyses
        where one needs an untrained model. Defaults to False.

    ignore_embeddings : bool, optional
        Whether the embedding layer should be excluded in the final output, or
        kept with the regular layers. Defaults to False

    decompose_layers : bool, optional
        Whether each layer should have it's own output file, or all layers be saved
        in a single file. Defaults to False, i.e. single file

    filter_layers : str
        Comma separated list of layer indices to save. The format is the same as
        the one accepted by ``data.writer.ActivationsWriter``.

    dtype : str, optional
        Data type in which the activations will be stored. Supports all numpy
        based tensor types. Common values are 'float32' and 'float16'. Defaults
        to 'float16'

    include_special_tokens : bool, optional
        Whether or not to special tokens in the extracted representations.
        Special tokens are tokens not present in the original sentence, but are
        added by the tokenizer, such as [CLS], [SEP] etc.
    """
    print(f"Loading model: {model_desc}")
    model, tokenizer = get_model_and_tokenizer(
        model_desc, device=device, random_weights=random_weights
    )

    print("Reading input corpus")

    def corpus_generator(encoder_input_corpus_path, decoder_input_corpus_path):
        efp = open(encoder_input_corpus_path, "r")
        dfp = None
        if decoder_input_corpus_path is not None:
            dfp = open(decoder_input_corpus_path, "r")
        else:
            dfp = iter(lambda: None, 0)

        for encoder_line, decoder_line in zip(efp, dfp):
            yield (
                encoder_line.strip(),
                decoder_line.strip() if decoder_line is not None else None,
            )

        efp.close()
        if decoder_input_corpus_path:
            dfp.close()
        return

    print("Preparing output files")
    if seq2seq_component == "both" or seq2seq_component == "encoder":
        encoder_writer = ActivationsWriter.get_writer(
            f"encoder-{output_file}",
            filetype=output_type,
            decompose_layers=decompose_layers,
            filter_layers=filter_layers,
            dtype=dtype,
        )
    if seq2seq_component == "both" or seq2seq_component == "decoder":
        decoder_writer = ActivationsWriter.get_writer(
            f"decoder-{output_file}",
            filetype=output_type,
            decompose_layers=decompose_layers,
            filter_layers=filter_layers,
            dtype=dtype,
        )

    context_indicator = None
    special_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.all_special_tokens)

    for subtoken in tokenizer.vocab:
        pieces = [
            idx
            for idx in tokenizer(subtoken)["input_ids"]
            if idx not in special_token_ids
        ]
        if len(pieces) == 1:
            context_indicator = subtoken
            break

    assert context_indicator is not None

    print("Extracting representations from model")
    tokenization_counts = {}  # Cache for tokenizer rules
    for sentence_idx, sentence_pair in enumerate(
        corpus_generator(encoder_input_corpus, decoder_input_corpus)
    ):
        hidden_states, extracted_words = extract_sentence_representations(
            sentence_pair,
            model,
            tokenizer,
            device=device,
            include_embeddings=(not ignore_embeddings),
            aggregation=aggregation,
            dtype=dtype,
            include_special_tokens=include_special_tokens,
            tokenization_counts=tokenization_counts,
            seq2seq_component=seq2seq_component,
            context_indicator=context_indicator,
        )

        print(
            "Hidden states: ",
            hidden_states["encoder"].shape if "encoder" in hidden_states else None,
            hidden_states["decoder"].shape if "decoder" in hidden_states else None,
        )
        print(
            "# Extracted words: ",
            len(extracted_words["encoder"]) if "encoder" in hidden_states else None,
            len(extracted_words["decoder"]) if "decoder" in hidden_states else None,
        )

        if seq2seq_component == "both" or seq2seq_component == "encoder":
            encoder_writer.write_activations(
                sentence_idx, extracted_words["encoder"], hidden_states["encoder"]
            )

        if seq2seq_component == "both" or seq2seq_component == "decoder":
            decoder_writer.write_activations(
                sentence_idx, extracted_words["decoder"], hidden_states["decoder"]
            )

    if seq2seq_component == "both" or seq2seq_component == "encoder":
        encoder_writer.close()

    if seq2seq_component == "both" or seq2seq_component == "decoder":
        decoder_writer.close()


HDF5_SPECIAL_TOKENS = {".": "__DOT__", "/": "__SLASH__"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_desc", help="Name of model")
    parser.add_argument(
        "encoder_input_corpus",
        help="Text file path with one sentence per line, input to the encoder",
    )
    parser.add_argument(
        "decoder_input_corpus",
        nargs="?",
        help="Text file path with one sentence per line, input to the decoder",
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
        "--dtype",
        choices=["float16", "float32"],
        default="float32",
        help="Output dtype of the extracted representations",
    )
    parser.add_argument("--disable_cuda", action="store_true")
    parser.add_argument("--ignore_embeddings", action="store_true")
    parser.add_argument(
        "--random_weights",
        action="store_true",
        help="generate representations from randomly initialized model",
    )
    parser.add_argument(
        "--include_special_tokens",
        action="store_true",
        help="Include special tokens like [CLS] and [SEP] in the extracted representations",
    )
    parser.add_argument(
        "--seq2seq_component",
        choices=["both", "encoder", "decoder"],
        default="both",
        help="If the model is a seq2seq model, which component should the outputs be saved for",
    )

    ActivationsWriter.add_writer_options(parser)

    args = parser.parse_args()

    if args.seq2seq_component == "both" or args.seq2seq_component == "decoder":
        assert (
            args.decoder_input_corpus is not None
        ), "Decoder Input corpus must be provided if not extracting only from encoder component"

    assert args.aggregation in [
        "average",
        "first",
        "last",
    ], "Invalid aggregation option, please specify first, average or last."

    assert not (
        args.filter_layers is not None and args.ignore_embeddings is True
    ), "--filter_layers and --ignore_embeddings cannot be used at the same time"

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    extract_representations(
        args.model_desc,
        args.encoder_input_corpus,
        args.decoder_input_corpus,
        args.output_file,
        device=device,
        aggregation=args.aggregation,
        output_type=args.output_type,
        random_weights=args.random_weights,
        ignore_embeddings=args.ignore_embeddings,
        dtype=args.dtype,
        decompose_layers=args.decompose_layers,
        filter_layers=args.filter_layers,
        include_special_tokens=args.include_special_tokens,
        seq2seq_component=args.seq2seq_component,
    )


if __name__ == "__main__":
    main()
