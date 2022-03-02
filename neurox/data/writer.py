"""Representations Writers

Module with various writers for saving representations/activations. Currently,
two file types are supported:

1. ``hdf5``: This is a binary format, and results in smaller overall files.
   The structure of the file is as follows:

   * ``sentence_to_idx`` dataset: Contains a single json string at index 0 that
     maps sentences to indices
   * Indices ``0`` through ``N-1`` datasets: Each index corresponds to one
     sentence. The value of the dataset is a tensor with dimensions
     ``num_layers x sentence_length x embedding_size``, where ``embedding_size``
     may include multiple layers
2. ``json``: This is a human-readable format. There is some loss of precision,
   since each activation value is saved using 8 decimal places. Concretely, this
   results in a jsonl file, where each line is a json string corresponding to a
   single sentence. The structure of each line is as follows:

   * ``linex_idx``: Sentence index
   * ``features``: List of tokens (with their activations)

     * ``token``: The current token
     * ``layers``: List of layers

       * ``index``: Layer index (does not correspond to original model's layers)
       * ``values``: List of activation values for all neurons in the layer

The writers also support saving activations from specific layers only, using the
``filter_layers`` argument. Since activation files can be large, an additional
option for decomposing the representations into layer-wise files is also
provided.
"""

import argparse
import collections
import json

import h5py


class ActivationsWriter:
    """
    Class that encapsulates all available writers.

    This is the only class that should be used by the rest of the library.

    Attributes
    ----------
    filename : str
        Filename for storing the activations. May not be used exactly if
        ``decompose_layers`` is True.
    filetype : str
        An additional hint for the filetype. This argument is optional
        The file type will be detected automatically from the filename if
        none is supplied.
    decompose_layers : bool
        Set to true if each layer's activations should be saved in a
        separate file.
    filter_layers : str
        Comma separated list of layer indices to save.
    """

    def __init__(
        self, filename, filetype=None, decompose_layers=False, filter_layers=None
    ):
        self.filename = filename
        self.decompose_layers = decompose_layers
        self.filter_layers = filter_layers

    def open(self):
        """
        Method to open the underlying files. Will be called automatically
        by the class instance when necessary.
        """
        raise NotImplementedError("Use a specific writer or the `get_writer` method.")

    def write_activations(self, sentence_idx, extracted_words, activations):
        """Method to write a single sentence's activations to file"""
        raise NotImplementedError("Use a specific writer or the `get_writer` method.")

    def close(self):
        """Method to close the udnerlying files."""
        raise NotImplementedError("Use a specific writer or the `get_writer` method.")

    @staticmethod
    def get_writer(filename, filetype=None, decompose_layers=False, filter_layers=None):
        """Method to get the correct writer based on filename and filetype"""
        return ActivationsWriterManager(
            filename, filetype, decompose_layers, filter_layers
        )

    @staticmethod
    def add_writer_options(parser):
        """Method to return argparse arguments specific to activation writers"""
        parser.add_argument(
            "--output_type",
            choices=["autodetect", "hdf5", "json"],
            default="autodetect",
            help="Output format of the extracted representations. Default autodetects based on file extension.",
        )
        parser.add_argument(
            "--decompose_layers",
            action="store_true",
            help="Save activations from each layer in a separate file",
        )
        parser.add_argument(
            "--filter_layers",
            default=None,
            type=str,
            help="Comma separated list of layers to save activations for. The layers will be saved in the order specified in this argument.",
        )


class ActivationsWriterManager(ActivationsWriter):
    """
    Manager class that handles decomposition and filtering.

    Decomposition requires multiple writers (one per file) and filtering
    requires processing the activations to remove unneeded layer activations.
    This class sits on top of the actual activations writer to manage these
    operations.
    """

    def __init__(
        self, filename, filetype=None, decompose_layers=False, filter_layers=None
    ):
        super().__init__(
            filename,
            filetype=filetype,
            decompose_layers=decompose_layers,
            filter_layers=filter_layers,
        )

        if filename.endswith(".hdf5") or filetype == "hdf5":
            self.base_writer = HDF5ActivationsWriter
        elif filename.endswith(".json") or filetype == "json":
            self.base_writer = JSONActivationsWriter
        else:
            raise NotImplementedError("filetype not supported. Use `hdf5` or `json`.")

        self.filename = filename
        self.layers = None
        self.writers = None

    def open(self, num_layers):
        self.layers = list(range(num_layers))
        self.writers = []
        if self.filter_layers:
            self.layers = [int(l) for l in self.filter_layers.split(",")]
        if self.decompose_layers:
            for layer_idx in self.layers:
                local_filename = f"{self.filename[:-5]}-layer{layer_idx}.{self.filename[-4:]}"
                _writer = self.base_writer(local_filename)
                _writer.open()
                self.writers.append(_writer)
        else:
            _writer = self.base_writer(self.filename)
            _writer.open()
            self.writers.append(_writer)

    def write_activations(self, sentence_idx, extracted_words, activations):
        if self.writers is None:
            self.open(activations.shape[0])

        if self.decompose_layers:
            for writer_idx, layer_idx in enumerate(self.layers):
                self.writers[writer_idx].write_activations(
                    sentence_idx, extracted_words, activations[[layer_idx], :, :]
                )
        else:
            self.writers[0].write_activations(
                sentence_idx, extracted_words, activations[self.layers, :, :]
            )

    def close(self):
        for writer in self.writers:
            writer.close()


class HDF5ActivationsWriter(ActivationsWriter):
    def __init__(self, filename):
        super().__init__(filename, filetype="hdf5")
        if not self.filename.endswith(".hdf5"):
            raise ValueError(
                f"Output filename ({self.filename}) does not end with .hdf5, but output file type is hdf5."
            )
        self.activations_file = None

    def open(self):
        self.activations_file = h5py.File(self.filename, "w")
        self.sentence_to_index = {}

    def write_activations(self, sentence_idx, extracted_words, activations):
        if self.activations_file is None:
            self.open()

        self.activations_file.create_dataset(
            str(sentence_idx), activations.shape, dtype="float32", data=activations
        )

        # TODO: Replace with better implementation with list of indices
        sentence = " ".join(extracted_words)
        final_sentence = sentence
        counter = 1
        while final_sentence in self.sentence_to_index:
            counter += 1
            final_sentence = f"{sentence} (Occurrence {counter})"
        sentence = final_sentence
        self.sentence_to_index[sentence] = str(sentence_idx)

    def close(self):
        sentence_index_dataset = self.activations_file.create_dataset(
            "sentence_to_index", (1,), dtype=h5py.special_dtype(vlen=str)
        )
        sentence_index_dataset[0] = json.dumps(self.sentence_to_index)
        self.activations_file.close()


class JSONActivationsWriter(ActivationsWriter):
    def __init__(self, filename):
        super().__init__(filename, filetype="json")
        if not self.filename.endswith(".json"):
            raise ValueError(
                f"Output filename ({self.filename}) does not end with .json, but output file type is json."
            )

        self.activations_file = None

    def open(self):
        self.activations_file = open(self.filename, "w", encoding="utf-8")

    def write_activations(self, sentence_idx, extracted_words, activations):
        if self.activations_file is None:
            self.open()

        output_json = collections.OrderedDict()
        output_json["linex_index"] = sentence_idx
        all_out_features = []

        for word_idx, extracted_word in enumerate(extracted_words):
            all_layers = []
            for layer_idx in range(activations.shape[0]):
                layers = collections.OrderedDict()
                layers["index"] = layer_idx
                layers["values"] = [
                    round(x.item(), 8) for x in activations[layer_idx, word_idx, :]
                ]
                all_layers.append(layers)
            out_features = collections.OrderedDict()
            out_features["token"] = extracted_word
            out_features["layers"] = all_layers
            all_out_features.append(out_features)
        output_json["features"] = all_out_features
        self.activations_file.write(json.dumps(output_json) + "\n")

    def close(self):
        self.activations_file.close()
