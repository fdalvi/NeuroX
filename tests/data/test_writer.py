import json
import unittest

from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import h5py
import torch

from neurox.data.writer import ActivationsWriter, HDF5ActivationsWriter, JSONActivationsWriter

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
        self.assertRaises(NotImplementedError, writer.write_activations, None, None, None)

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
        self.assertRaises(NotImplementedError, ActivationsWriter.get_writer, f"{self.tmpdir.name}/somename.txt")

class TestInvalidFileType(unittest.TestCase):
    def setUp(self):
        self.tmpdir = TemporaryDirectory()
    def tearDown(self):
        self.tmpdir.cleanup()

    def test_invalid_file_type_hdf5(self):
        "Invalid filetype argument for hdf5 writer"
        self.assertRaises(ValueError, HDF5ActivationsWriter, f"{self.tmpdir.name}/somename.txt")

    def test_invalid_file_type_json(self):
        "Invalid filetype argument for json writer"
        self.assertRaises(ValueError, JSONActivationsWriter, f"{self.tmpdir.name}/somename.txt")

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
        "File should auto open on first write_activations call if not already open"
        output_file = f"{self.tmpdir.name}/somename.hdf5"
        writer = HDF5ActivationsWriter(output_file)

        sentences = ["This is a sentence", "This is a another sentence", "This is a sentence"]
        expected_sentences = [
            sentences[0],
            sentences[1],
            f"{sentences[0]} (Occurrence 2)"
        ]
        expected_activations = [
            torch.rand((13, len(sentence.split(" ")), 768)) for sentence in sentences
        ]

        for s_idx in range(len(sentences)):
            writer.write_activations(s_idx, sentences[s_idx].split(" "), expected_activations[s_idx])
        
        writer.close()

        saved_activations = h5py.File(output_file)

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
            self.assertEqual(sentence, expected_sentences[int(sentence_to_index[sentence])])

        # Check saved activations
        for sentence in sentence_to_index:
            idx = sentence_to_index[sentence]
            self.assertTrue(torch.equal(torch.FloatTensor(saved_activations[idx]), expected_activations[int(idx)]))