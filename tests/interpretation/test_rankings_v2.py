import random
from ssl import RAND_add

import neurox.interpretation.gaussian_probe as guassian_probe
import neurox.interpretation.iou_probe as iou_probe
import neurox.interpretation.linear_probe as linear_probe

import neurox.interpretation.probeless as probeless
import numpy as np
import pytest
import torch


######################## Set the Main Seed ########################
main_seed = 76 
np.random.seed(main_seed)
random.seed(main_seed)
torch.manual_seed(main_seed)

######################## Generate The Other Seeds Based On The Main Seed ########################
HIGHEST_DIM = random.randint(0, 1000)
MAX_NUMBER_OF_CLASSES = 10
SEEDS = [random.randint(0, HIGHEST_DIM - 1) for _ in range(10)]

######################## Define  the fixtures  ########################
@pytest.fixture
def binary_test_cases():
    test_cases = []
    for seed in SEEDS: 
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        num_samples = random.randint(0, HIGHEST_DIM - 1)
        num_features = random.randint(0, HIGHEST_DIM - 1)
        X = np.random.rand(num_samples, num_features)
        y = np.random.randint(2, size=num_samples)
        test_cases.append((X,y))
    return test_cases


@pytest.fixture
def multiclass_test_cases():
    test_cases = []
    for seed in SEEDS: 
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        num_samples = random.randint(0, HIGHEST_DIM - 1)
        num_features = random.randint(0, HIGHEST_DIM - 1) 
        num_classes = random.randint(3, MAX_NUMBER_OF_CLASSES)
        X = np.random.rand(num_samples, num_features)
        y = np.random.randint(num_classes, size=num_samples)
        test_cases.append((X,y, num_classes))
    return test_cases
     

def test_binary_probeless_ranking(binary_test_cases, snapshot): 
    for idx, case in enumerate(binary_test_cases): 
        X, y = case
        probeless_ranking = probeless.get_neuron_ordering(X, y) 
        assert probeless_ranking == snapshot(name=f"binary_probeless_case_{idx}")

def test_binary_iou_ranking(binary_test_cases, snapshot): 
    for idx, case in enumerate(binary_test_cases): 
        X, y = case
        iou_ranking = list(iou_probe.get_neuron_ordering(X, y)) 
        assert iou_ranking == snapshot(name=f"binary_iou_case_{idx}")

def test_binary_guassian_ranking(binary_test_cases, snapshot): 
 
    X, y = random.choice(binary_test_cases) # test only on one case for guassian as it takes a lot of time to train a guassian probe
    g_probe= guassian_probe.train_probe(X, y)
    guassian_ranking = guassian_probe.get_neuron_ordering(g_probe, X.shape[-1])
    assert guassian_ranking == snapshot(name=f"binary_guassian_case")



def test_binary_logistic_regression_probe_lca(binary_test_cases, snapshot): 
    for idx, case in enumerate(binary_test_cases): 
        X, y = case
        l_probe = linear_probe.train_logistic_regression_probe(X, y, lambda_l2=0.1, lambda_l1=0.1)
        overall_ranking, per_class_ranking = linear_probe.get_top_neurons(l_probe, 1, {"class0": 0, "class1": 1})
        assert list(overall_ranking) == snapshot(name=f"binary_lca_overall_ranking_{idx}") 
        assert list(per_class_ranking["class0"]) == snapshot(name=f"binary_lca_class0_ranking_{idx}")
        assert list(per_class_ranking["class1"]) == snapshot(name=f"binary_lca_class1_ranking_{idx}")


def test_binary_logistic_regression_probe_no_reg(binary_test_cases, snapshot): 
    for idx, case in enumerate(binary_test_cases): 
        X, y = case
        l_probe = linear_probe.train_logistic_regression_probe(X, y, lambda_l2=0.0, lambda_l1=0.0)
        overall_ranking, per_class_ranking = linear_probe.get_top_neurons(l_probe, 1, {"class0": 0, "class1": 1})
        assert list(overall_ranking) == snapshot(name=f"binary_no_reg_overall_ranking_{idx}") 
        assert list(per_class_ranking["class0"]) == snapshot(name=f"binary_no_reg_class0_ranking_{idx}")
        assert list(per_class_ranking["class1"]) == snapshot(name=f"binary_no_reg_class1_ranking_{idx}")



def test_binary_logistic_regression_probe_lasso(binary_test_cases, snapshot): 
    for idx, case in enumerate(binary_test_cases): 
        X, y = case
        l_probe = linear_probe.train_logistic_regression_probe(X, y, lambda_l2=0.0, lambda_l1=0.1)
        overall_ranking, per_class_ranking = linear_probe.get_top_neurons(l_probe, 1, {"class0": 0, "class1": 1})
        assert list(overall_ranking) == snapshot(name=f"binary_lasso_overall_ranking_{idx}") 
        assert list(per_class_ranking["class0"]) == snapshot(name=f"binary_lasso_class0_ranking_{idx}")
        assert list(per_class_ranking["class1"]) == snapshot(name=f"binary_lasso_class1_ranking_{idx}")


def test_binary_logistic_regression_probe_ridge(binary_test_cases, snapshot): 
    for idx, case in enumerate(binary_test_cases): 
        X, y = case
        l_probe = linear_probe.train_logistic_regression_probe(X, y, lambda_l2=0.1, lambda_l1=0.0)
        overall_ranking, per_class_ranking = linear_probe.get_top_neurons(l_probe, 1, {"class0": 0, "class1": 1})
        assert list(overall_ranking) == snapshot(name=f"binary_ridge_overall_ranking_{idx}") 
        assert list(per_class_ranking["class0"]) == snapshot(name=f"binary_ridge_class0_ranking_{idx}")
        assert list(per_class_ranking["class1"]) == snapshot(name=f"binary_ridge_class1_ranking_{idx}")



def test_multiclass_probeless_ranking(multiclass_test_cases, snapshot): 
    for idx, case in enumerate(multiclass_test_cases): 
        X, y, num_classes = case 
        idx2label = {i:f"class{i}" for i in range(num_classes)}
        overall_ranking, top_neurons_per_class = probeless.get_neuron_ordering_for_all_tags(X, y, idx2label=idx2label)
        assert overall_ranking == snapshot(name=f"probless_multiclass_ranking_case_{idx}") 
        for v in list(idx2label.values()): 
            assert top_neurons_per_class[v] == snapshot(name=f"probeless_multiclass_ranking_case_{idx}_{v}") 



def test_multiclass_logistic_regression_probe_lca(multiclass_test_cases, snapshot):
    for idx, case  in enumerate(multiclass_test_cases):
         X, y, num_classes = case 
         class_to_idx = {f"class{i}":i for i in range(num_classes)}
         probe = linear_probe.train_logistic_regression_probe(X, y, lambda_l2=0.1, lambda_l1=0.1)
         overall_ranking, top_neurons_per_class = linear_probe.get_top_neurons(probe, 1, class_to_idx)
         assert list(overall_ranking) == snapshot(name=f"lca_probe_multiclass_overall_ranking_{idx}")
         for k in list(class_to_idx.keys()): 
             assert list(top_neurons_per_class[k]) == snapshot(name=f"lca_probe_multiclass_ranking_{idx}_{k}")



def test_multiclass_logistic_regression_probe_no_reg(multiclass_test_cases, snapshot):
    for idx, case  in enumerate(multiclass_test_cases):
         X, y, num_classes = case 
         class_to_idx = {f"class{i}":i for i in range(num_classes)}
         probe = linear_probe.train_logistic_regression_probe(X, y, lambda_l2=0.0, lambda_l1=0.0)
         overall_ranking, top_neurons_per_class = linear_probe.get_top_neurons(probe, 1, class_to_idx)
         assert list(overall_ranking) == snapshot(name=f"no_reg_probe_multiclass_overall_ranking_{idx}")
         for k in list(class_to_idx.keys()): 
            assert list(top_neurons_per_class[k]) == snapshot(name=f"no_reg_probe_multiclass_ranking_{idx}_{k}")



def test_multiclass_logistic_regression_probe_lasso(multiclass_test_cases, snapshot):
    for idx, case  in enumerate(multiclass_test_cases):
         X, y, num_classes = case 
         class_to_idx = {f"class{i}":i for i in range(num_classes)}
         probe = linear_probe.train_logistic_regression_probe(X, y, lambda_l2=0.0, lambda_l1=0.1)
         overall_ranking, top_neurons_per_class = linear_probe.get_top_neurons(probe, 1, class_to_idx)
         assert list(overall_ranking) == snapshot(name=f"lasso_probe_multiclass_overall_ranking_{idx}")
         for k in list(class_to_idx.keys()): 
            assert list(top_neurons_per_class[k]) == snapshot(name=f"lasso_probe_multiclass_ranking_{idx}_{k}")



def test_multiclass_logistic_regression_probe_ridge(multiclass_test_cases, snapshot):
    for idx, case  in enumerate(multiclass_test_cases):
         X, y, num_classes = case 
         class_to_idx = {f"class{i}":i for i in range(num_classes)}
         probe = linear_probe.train_logistic_regression_probe(X, y, lambda_l2=0.1, lambda_l1=0.0)
         overall_ranking, top_neurons_per_class = linear_probe.get_top_neurons(probe, 1, class_to_idx)
         assert list(overall_ranking) == snapshot(name=f"ridge_probe_multiclass_overall_ranking_{idx}")
         for k in list(class_to_idx.keys()): 
            assert list(top_neurons_per_class[k]) == snapshot(name=f"ridge_probe_multiclass_ranking_{idx}_{k}")