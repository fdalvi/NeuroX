import collections
import json

import h5py


class ActivationsWriter:
    def __init__(self, filename, filetype=None, decompose_layers=False, filter_layers=None):
        self.filename = filename
        self.decompose_layers = decompose_layers
        self.filter_layers = filter_layers

    def open(self):
        raise NotImplementedError("Use a specific writer or the `get_writer` method.")

    def write_activations(self, sentence_idx, extracted_words, activations):
        raise NotImplementedError("Use a specific writer or the `get_writer` method.")

    def close(self):
        raise NotImplementedError("Use a specific writer or the `get_writer` method.")

    @staticmethod
    def get_writer(filename, filetype=None, decompose_layers=False, filter_layers=None):
        return DecomposableActivationsWriter(filename, filetype, decompose_layers, filter_layers)


class DecomposableActivationsWriter(ActivationsWriter):
    def __init__(self, filename, filetype=None, decompose_layers=False, filter_layers=None):
        super().__init__(filename, filetype=filetype, decompose_layers=decompose_layers, filter_layers=filter_layers)

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
                local_filename = f"{self.filename[:-5]}-layer{layer_idx}.hdf5"
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
                self.writers[writer_idx].write_activations(sentence_idx, extracted_words, activations[layer_idx, :, :])
        else:
            self.writers[0].write_activations(sentence_idx, extracted_words, activations[self.layers, :, :])

    def close(self):
        for writer in self.writers:
            writer.close()

class HDF5ActivationsWriter(ActivationsWriter):
    def __init__(self, filename):
        super().__init__(filename, filetype="hdf5")
        if not self.filename.endswith(".hdf5"):
            raise ValueError(f"Output filename ({self.filename}) does not end with .hdf5, but output file type is hdf5.")
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
            raise ValueError(f"Output filename ({self.filename}) does not end with .json, but output file type is json.")

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
