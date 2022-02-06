import collections
import json

import h5py


class ActivationsWriter:
    def __init__(self, filename, filetype=None):
        self.activations_file = None
        self.filename = filename

    def open_file(self):
        raise NotImplementedError("Use a specific writer or the `get_writer` method.")

    def write_activations(self, sentence_idx, extracted_words, activations):
        raise NotImplementedError("Use a specific writer or the `get_writer` method.")

    def close_file(self):
        raise NotImplementedError("Use a specific writer or the `get_writer` method.")

    @staticmethod
    def get_writer(filename, filetype=None):
        if filename.endswith(".hdf5") or filetype == "hdf5":
            return HDF5ActivationsWriter(filename)
        elif filename.endswith(".json") or filetype == "json":
            return JSONActivationsWriter(filename)
        else:
            raise NotImplementedError("filetype not supported. Use `hdf5` or `json`.")


class HDF5ActivationsWriter(ActivationsWriter):
    def __init__(self, filename):
        super().__init__(filename, filetype="hdf5")

    def open_file(self):
        if not self.filename.endswith(".hdf5"):
            print(
                "[WARNING] Output filename (%s) does not end with .hdf5, but output file type is hdf5."
                % (self.filename)
            )
        self.activations_file = h5py.File(self.filename, "w")
        self.sentence_to_index = {}

    def write_activations(self, sentence_idx, extracted_words, activations):
        self.activations_file.create_dataset(
            str(sentence_idx), activations.shape, dtype="float32", data=activations
        )

        # TODO: Replace with better implementation with list of indices
        final_sentence = " ".join(extracted_words)
        counter = 1
        while final_sentence in self.sentence_to_index:
            counter += 1
            final_sentence = f"{sentence} (Occurrence {counter})"
        sentence = final_sentence
        self.sentence_to_index[sentence] = str(sentence_idx)

    def close_file(self):
        sentence_index_dataset = self.activations_file.create_dataset(
            "sentence_to_index", (1,), dtype=h5py.special_dtype(vlen=str)
        )
        sentence_index_dataset[0] = json.dumps(self.sentence_to_index)
        self.activations_file.close()


class JSONActivationsWriter(ActivationsWriter):
    def __init__(self, filename):
        super().__init__(filename, filetype="json")

    def open_file(self):
        if not self.filename.endswith(".json"):
            print(
                "[WARNING] Output filename (%s) does not end with .json, but output file type is json."
                % (self.filename)
            )
        self.activations_file = open(self.filename, "w", encoding="utf-8")

    def write_activations(self, sentence_idx, extracted_words, activations):
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

    def close_file(self):
        self.activations_file.close()
