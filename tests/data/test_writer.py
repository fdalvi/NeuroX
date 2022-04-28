import json
import unittest

from tempfile import TemporaryDirectory
from unittest.mock import ANY, MagicMock, patch

import h5py
import torch

from neurox.data import loader

from neurox.data.writer import (
    ActivationsWriter,
    HDF5ActivationsWriter,
    JSONActivationsWriter,
)


class TestAutoDetect(unittest.TestCase):
    def setUp(self):
        self.tmpdir = TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_autodetect_file_type_hdf5(self):
        "Auto-detection test of hdf5 from filename"
        writer = ActivationsWriter.get_writer(f"{self.tmpdir.name}/somename.hdf5")
        self.assertEqual(writer.base_writer, HDF5ActivationsWriter)

    def test_autodetect_file_type_json(self):
        "Auto-detection test of json from filename"
        writer = ActivationsWriter.get_writer(f"{self.tmpdir.name}/somename.json")
        self.assertEqual(writer.base_writer, JSONActivationsWriter)


class TestNoUseBaseWriter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_nouse_base_writer_open(self):
        "Incorrect use of base writer (ActivationsWriter) class (open)"
        writer = ActivationsWriter(f"{self.tmpdir.name}/somename")
        self.assertRaises(NotImplementedError, writer.open)

    def test_nouse_base_writer_write_activations(self):
        "Incorrect use of base writer (ActivationsWriter) class (write_activations)"
        writer = ActivationsWriter(f"{self.tmpdir.name}/somename")
        self.assertRaises(
            NotImplementedError, writer.write_activations, None, None, None
        )

    def test_nouse_base_writer_close(self):
        "Incorrect use of base writer (ActivationsWriter) class (close)"
        writer = ActivationsWriter(f"{self.tmpdir.name}/somename")
        self.assertRaises(NotImplementedError, writer.close)


class TestUnsupportedFileType(unittest.TestCase):
    def setUp(self):
        self.tmpdir = TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_unsupported_file_type(self):
        "Unsupported writer file type"
        self.assertRaises(
            NotImplementedError,
            ActivationsWriter.get_writer,
            f"{self.tmpdir.name}/somename.txt",
        )


class TestInvalidFileType(unittest.TestCase):
    def setUp(self):
        self.tmpdir = TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_invalid_file_type_hdf5(self):
        "Invalid filetype argument for hdf5 writer"
        self.assertRaises(
            ValueError, HDF5ActivationsWriter, f"{self.tmpdir.name}/somename.txt"
        )

    def test_invalid_file_type_json(self):
        "Invalid filetype argument for json writer"
        self.assertRaises(
            ValueError, JSONActivationsWriter, f"{self.tmpdir.name}/somename.txt"
        )


class TestAutoOpen(unittest.TestCase):
    def setUp(self):
        self.tmpdir = TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_auto_open_hdf5(self):
        "File should auto open on first write_activations call if not already open"
        writer = HDF5ActivationsWriter(f"{self.tmpdir.name}/somename.hdf5")

        writer.open = MagicMock(side_effect=writer.open)
        writer.write_activations(0, ["word"], torch.rand((13, 1, 768)))
        writer.open.assert_called_once()
        writer.close()

    def test_auto_open_json(self):
        "File should auto open on first write_activations call if not already open"
        writer = JSONActivationsWriter(f"{self.tmpdir.name}/somename.json")

        writer.open = MagicMock(side_effect=writer.open)
        writer.write_activations(0, ["word"], torch.rand((13, 1, 768)))
        writer.open.assert_called_once()
        writer.close()


class TestHDF5DuplicateSentences(unittest.TestCase):
    def setUp(self):
        self.tmpdir = TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_duplicate_sentence_hdf5(self):
        "Duplicated sentences should we stored with occurence index"
        output_file = f"{self.tmpdir.name}/somename.hdf5"
        writer = HDF5ActivationsWriter(output_file)

        sentences = [
            "This is a sentence",
            "This is a another sentence",
            "This is a sentence",
            "This is a sentence",
        ]
        expected_sentences = [
            sentences[0],
            sentences[1],
            f"{sentences[0]} (Occurrence 2)",
            f"{sentences[0]} (Occurrence 3)",
        ]
        expected_activations = [
            torch.rand((13, len(sentence.split(" ")), 768)) for sentence in sentences
        ]

        for s_idx in range(len(sentences)):
            writer.write_activations(
                s_idx, sentences[s_idx].split(" "), expected_activations[s_idx]
            )

        writer.close()

        saved_activations = h5py.File(output_file, "r")

        # Check hdf5 structure
        self.assertEqual(len(saved_activations.keys()), len(sentences) + 1)
        self.assertTrue("sentence_to_index" in saved_activations)
        for idx in range(len(sentences)):
            self.assertTrue(str(idx) in saved_activations)

        # Check saved sentences
        self.assertEqual(len(saved_activations["sentence_to_index"]), 1)
        sentence_to_index = json.loads(saved_activations["sentence_to_index"][0])
        self.assertEqual(len(sentence_to_index), len(sentences))
        for sentence in sentence_to_index:
            self.assertEqual(
                sentence, expected_sentences[int(sentence_to_index[sentence])]
            )

        # Check saved activations
        for sentence in sentence_to_index:
            idx = sentence_to_index[sentence]
            self.assertTrue(
                torch.equal(
                    torch.FloatTensor(saved_activations[idx]),
                    expected_activations[int(idx)],
                )
            )


class TestWriterOptions(unittest.TestCase):
    @patch("argparse.ArgumentParser")
    def test_options_added(self, parser_mock):
        ActivationsWriter.add_writer_options(parser_mock)

        parser_mock.add_argument.assert_any_call(
            "--output_type", choices=ANY, help=ANY, default=ANY
        )
        parser_mock.add_argument.assert_any_call(
            "--decompose_layers", action=ANY, help=ANY
        )
        parser_mock.add_argument.assert_any_call(
            "--filter_layers", help=ANY, default=ANY, type=ANY
        )


class TestDecomposition(unittest.TestCase):
    def setUp(self):
        self.num_layers = 13
        self.tmpdir = TemporaryDirectory()

        self.sentences = [
            "This is a sentence",
            "This is a another sentence",
            "This is a sentence",
            "This is a sentence",
        ]
        self.expected_activations = [
            torch.rand((self.num_layers, len(sentence.split(" ")), 768))
            for sentence in self.sentences
        ]

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_decomposition_hdf5(self):
        "Test decomposition of all layers into separate files for hdf5"
        output_file = f"{self.tmpdir.name}/somename.hdf5"
        actual_output_files = [
            f"{self.tmpdir.name}/somename-layer{layer_idx}.hdf5"
            for layer_idx in range(self.num_layers)
        ]
        writer = ActivationsWriter.get_writer(output_file, decompose_layers=True)

        for s_idx in range(len(self.sentences)):
            writer.write_activations(
                s_idx,
                self.sentences[s_idx].split(" "),
                self.expected_activations[s_idx],
            )

        writer.close()

        for layer_idx, output_file in enumerate(actual_output_files):
            saved_activations, num_layers = loader.load_activations(output_file)
            # Decomposed files should only have 1 layer each
            self.assertEqual(1, num_layers)

            # Check saved activations
            for sentence_idx, sentence_activations in enumerate(saved_activations):
                curr_saved_activations = torch.FloatTensor(
                    saved_activations[sentence_idx]
                )
                curr_expected_activations = self.expected_activations[sentence_idx][
                    layer_idx, :, :
                ]
                self.assertTrue(
                    torch.equal(curr_saved_activations, curr_expected_activations)
                )

    def test_decomposition_json(self):
        "Test decomposition of all layers into separate files for json"
        output_file = f"{self.tmpdir.name}/somename.json"
        actual_output_files = [
            f"{self.tmpdir.name}/somename-layer{layer_idx}.json"
            for layer_idx in range(self.num_layers)
        ]
        writer = ActivationsWriter.get_writer(output_file, decompose_layers=True)

        for s_idx in range(len(self.sentences)):
            writer.write_activations(
                s_idx,
                self.sentences[s_idx].split(" "),
                self.expected_activations[s_idx],
            )

        writer.close()

        for layer_idx, output_file in enumerate(actual_output_files):
            saved_activations, num_layers = loader.load_activations(output_file)
            # Decomposed files should only have 1 layer each
            self.assertEqual(1, num_layers)

            # Check saved activations
            for sentence_idx, sentence_activations in enumerate(saved_activations):
                curr_saved_activations = torch.FloatTensor(
                    saved_activations[sentence_idx]
                )
                curr_expected_activations = self.expected_activations[sentence_idx][
                    layer_idx, :, :
                ]
                self.assertTrue(
                    torch.allclose(curr_saved_activations, curr_expected_activations)
                )


class TestFiltering(unittest.TestCase):
    def setUp(self):
        self.num_layers = 13
        self.tmpdir = TemporaryDirectory()

        self.sentences = [
            "This is a sentence",
            "This is a another sentence",
            "This is a sentence",
            "This is a sentence",
        ]
        self.expected_activations = [
            torch.rand((self.num_layers, len(sentence.split(" ")), 768))
            for sentence in self.sentences
        ]
        self.filter_layers = [5, 3, 2]

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_filter_layers_hdf5(self):
        "Test layer filtering for hdf5"
        output_file = f"{self.tmpdir.name}/somename.hdf5"
        writer = ActivationsWriter.get_writer(
            output_file, filter_layers=",".join(map(str, self.filter_layers))
        )

        for s_idx in range(len(self.sentences)):
            writer.write_activations(
                s_idx,
                self.sentences[s_idx].split(" "),
                self.expected_activations[s_idx],
            )

        writer.close()

        saved_activations, num_layers = loader.load_activations(output_file)
        self.assertEqual(len(self.filter_layers), num_layers)

        # Check saved activations
        for sentence_idx, sentence_activations in enumerate(saved_activations):
            curr_saved_activations = torch.FloatTensor(
                saved_activations[sentence_idx]
                .reshape(
                    (
                        self.expected_activations[sentence_idx].shape[1],
                        len(self.filter_layers),
                        -1,
                    )
                )
                .swapaxes(0, 1)
            )
            curr_expected_activations = self.expected_activations[sentence_idx][
                self.filter_layers, :, :
            ]
            self.assertTrue(
                torch.equal(curr_saved_activations, curr_expected_activations)
            )

    def test_filter_layers_json(self):
        "Test layer filtering for json"
        output_file = f"{self.tmpdir.name}/somename.json"
        writer = ActivationsWriter.get_writer(
            output_file, filter_layers=",".join(map(str, self.filter_layers))
        )

        for s_idx in range(len(self.sentences)):
            writer.write_activations(
                s_idx,
                self.sentences[s_idx].split(" "),
                self.expected_activations[s_idx],
            )

        writer.close()

        saved_activations, num_layers = loader.load_activations(output_file)
        self.assertEqual(len(self.filter_layers), num_layers)

        # Check saved activations
        for sentence_idx, sentence_activations in enumerate(saved_activations):
            curr_saved_activations = torch.FloatTensor(
                saved_activations[sentence_idx]
                .reshape(
                    (
                        self.expected_activations[sentence_idx].shape[1],
                        len(self.filter_layers),
                        -1,
                    )
                )
                .swapaxes(0, 1)
            )
            curr_expected_activations = self.expected_activations[sentence_idx][
                self.filter_layers, :, :
            ]
            self.assertTrue(
                torch.allclose(curr_saved_activations, curr_expected_activations)
            )


class TestDecompositionAndFiltering(unittest.TestCase):
    def setUp(self):
        self.num_layers = 13
        self.tmpdir = TemporaryDirectory()

        self.sentences = [
            "This is a sentence",
            "This is a another sentence",
            "This is a sentence",
            "This is a sentence",
        ]
        self.expected_activations = [
            torch.rand((self.num_layers, len(sentence.split(" ")), 768))
            for sentence in self.sentences
        ]
        self.filter_layers = [5, 3, 2]

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_decomposition_and_filter_layers_hdf5(self):
        "Test decomposition of specific layers into separate files for hdf5"
        output_file = f"{self.tmpdir.name}/somename.hdf5"
        actual_output_files = [
            f"{self.tmpdir.name}/somename-layer{layer_idx}.hdf5"
            for layer_idx in self.filter_layers
        ]
        writer = ActivationsWriter.get_writer(
            output_file,
            decompose_layers=True,
            filter_layers=",".join(map(str, self.filter_layers)),
        )

        for s_idx in range(len(self.sentences)):
            writer.write_activations(
                s_idx,
                self.sentences[s_idx].split(" "),
                self.expected_activations[s_idx],
            )

        writer.close()

        for layer_idx, output_file in enumerate(actual_output_files):
            saved_activations, num_layers = loader.load_activations(output_file)
            # Decomposed files should only have 1 layer each
            self.assertEqual(1, num_layers)

            # Check saved activations
            for sentence_idx, sentence_activations in enumerate(saved_activations):
                curr_saved_activations = torch.FloatTensor(
                    saved_activations[sentence_idx]
                )
                curr_expected_activations = self.expected_activations[sentence_idx][
                    self.filter_layers[layer_idx], :, :
                ]
                self.assertTrue(
                    torch.equal(curr_saved_activations, curr_expected_activations)
                )

    def test_decomposition_and_filter_layers_json(self):
        "Test decomposition of specific layers into separate files for json"
        output_file = f"{self.tmpdir.name}/somename.json"
        actual_output_files = [
            f"{self.tmpdir.name}/somename-layer{layer_idx}.json"
            for layer_idx in self.filter_layers
        ]
        writer = ActivationsWriter.get_writer(
            output_file,
            decompose_layers=True,
            filter_layers=",".join(map(str, self.filter_layers)),
        )

        for s_idx in range(len(self.sentences)):
            writer.write_activations(
                s_idx,
                self.sentences[s_idx].split(" "),
                self.expected_activations[s_idx],
            )

        writer.close()

        for layer_idx, output_file in enumerate(actual_output_files):
            saved_activations, num_layers = loader.load_activations(output_file)
            # Decomposed files should only have 1 layer each
            self.assertEqual(1, num_layers)

            # Check saved activations
            for sentence_idx, sentence_activations in enumerate(saved_activations):
                curr_saved_activations = torch.FloatTensor(
                    saved_activations[sentence_idx]
                )
                curr_expected_activations = self.expected_activations[sentence_idx][
                    self.filter_layers[layer_idx], :, :
                ]
                self.assertTrue(
                    torch.allclose(curr_saved_activations, curr_expected_activations)
                )
