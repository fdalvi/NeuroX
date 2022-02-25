import unittest
from collections import Counter
from cProfile import label
from random import sample

import neurox.control_tasks.seq_labeling_ct as ct
import numpy as np

# def create_seq_labeling_control_task_datasets(train_tokens, sample_from='same', dev_source=None, test_source=None):


class TestCreateSeqLabelingControlTaskDatasets(unittest.TestCase):
    def setUp(self):
        self.three_label_data_skewed = {
            "source": [
                ["Fly", "me", "to", "the", "moon"],
                ["I", "am", "walking", "on", "the", "moon"],
                ["Now", "I", "am", "walking", "on", "a", "dream"],
            ],
            "target": [
                ["A", "A", "A", "B", "A"],
                ["A", "B", "A", "A", "B", "C"],
                ["B", "A", "B", "A", "A", "B", "A"],
            ],
        }
        self.two_labels_dataset_balanced = {
            "source": [
                ["I", "am", "walking", "on", "sunshine"],
                "Tambourines",
                "and",
                "elephants",
                "are",
                "playin",
                "in",
                "the",
                "band",
            ],
            "target": [
                ["B", "A", "A", "B", "A"],
                ["B", "A", "B", "A", "B", "A", "B", "A"],
            ],
        }
        self.small_dataset_four_labels = {
            "source": [["Is", "There", "life", "on", "Mars"]],
            "target": [["A", "B", "C", "D", "A"]],
        }

        ints_list_of_lists = [[str(i)] for i in range(1000)]
        modulo_seven_target = [["T"] if i % 7 == 0 else ["F"] for i in range(1000)]
        self.modulo_seven_dataset = {
            "source": ints_list_of_lists,
            "target": modulo_seven_target,
        }

    def test_same_words_have_same_ct_label_in_same_dataset(self):
        [ct_tokens] = ct.create_seq_labeling_control_task_datasets(
            self.three_label_data_skewed
        )
        the_moon = (
            ct_tokens["source"][0][3:] == ct_tokens["source"][1][4:]
            and ct_tokens["target"][0][3:] == ct_tokens["target"][1][4:]
        )
        eye_am_walking_on = (
            ct_tokens["source"][1][0:4] == ct_tokens["source"][2][1:5]
            and ct_tokens["target"][1][0:4] == ct_tokens["target"][2][1:5]
        )
        assert the_moon and eye_am_walking_on

    def test_same_words_have_same_ct_label_in_diff_datasets(self):
        [ct_tokens_train, ct_tokens_dev] = ct.create_seq_labeling_control_task_datasets(
            self.three_label_data_skewed,
            dev_source=self.two_labels_dataset_balanced["source"],
        )
        eye_am_walking_on = (
            ct_tokens_train["source"][1][0:4] == ct_tokens_dev["source"][0][0:4]
            and ct_tokens_train["target"][1][0:4] == ct_tokens_dev["target"][0][0:4]
        )
        assert eye_am_walking_on

    def test_num_ct_labels_exactly_matches_num_training_labels_two_labels(self):
        """The number of control task labels is smaller or equal to the number of labels in the task training data"""
        [ct_tokens] = ct.create_seq_labeling_control_task_datasets(
            self.two_labels_dataset_balanced, sample_from="uniform"
        )
        labels_flat = [l for sublist in ct_tokens["target"] for l in sublist]
        assert len(set(labels_flat)) == 2

    def test_num_ct_labels_roughly_matches_num_training_labels_three_labels(self):
        """The number of control task labels is smaller or equal to the number of labels in the task training data"""
        [ct_tokens] = ct.create_seq_labeling_control_task_datasets(
            self.three_label_data_skewed, sample_from="same"
        )
        labels_flat = [l for sublist in ct_tokens["target"] for l in sublist]
        assert len(set(labels_flat)) <= 3

    def test_num_ct_labels_roughly_matches_num_training_labels_four_labels(self):
        """The number of control task labels is smaller or equal to the number of labels in the task training data"""
        [ct_tokens] = ct.create_seq_labeling_control_task_datasets(
            self.small_dataset_four_labels, sample_from="same"
        )
        labels_flat = [l for sublist in ct_tokens["target"] for l in sublist]
        assert len(set(labels_flat)) <= 4

    def test_uniform_distribution_matches(self):
        """Regardless of the label distribution in the training data, the control task labels are in a more or less uniform distribution"""
        [ct_tokens] = ct.create_seq_labeling_control_task_datasets(
            self.modulo_seven_dataset, sample_from="uniform"
        )
        labels_flat = [l for sublist in ct_tokens["target"] for l in sublist]
        label_freqs_dict = Counter(labels_flat)
        assert max(label_freqs_dict.values()) < 600

    def test_same_distribution_matches(self):
        """The label distribution in the control task data roughly matches the label distribution in the training data"""
        [ct_tokens] = ct.create_seq_labeling_control_task_datasets(
            self.modulo_seven_dataset, sample_from="same"
        )
        labels_flat = [l for sublist in ct_tokens["target"] for l in sublist]
        label_freqs_dict = Counter(labels_flat)
        assert min(label_freqs_dict.values()) < 200
