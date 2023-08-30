import random

import neurox.interpretation.gaussian_probe as guassian_probe
import neurox.interpretation.iou_probe as iou_probe
import neurox.interpretation.linear_probe as linear_probe

import neurox.interpretation.probeless as probeless
import numpy as np
import pytest
import torch


SEED = 99 # cannot set it as a fixture as we can only use fixtures from arguments of test functions 
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

@pytest.fixture
def first_binary_test_case(): # Is it better to keep them this way? Or is better to right one fixture with all my variables?
    X = np.random.rand(100, 768)
    y = np.random.randint(2, size=100)
    return X, y

@pytest.fixture 
def second_binary_test_case(): 
    X = np.random.rand(100, 100)
    y = np.random.randint(2, size=100)
    return X, y 

@pytest.fixture 
def third_binary_test_case(): 
    X = np.random.rand(150, 50)
    y = np.random.randint(2, size=150)
    return X,y

@pytest.fixture 
def guassian_first_test_case(): 
    X = np.random.rand(10, 15) 
    y = np.random.randint(2, size=10) 
    return X,y 
 
@pytest.fixture 
def guassian_second_test_case(): 
    X = np.random.rand(15, 10) 
    y = np.random.randint(2, size=15) 
    return X, y  

@pytest.fixture 
def guassian_third_test_case(): 
    X = np.random.rand(10, 10) 
    y= np.random.randint(2, size=10) 
    return X,y


@pytest.fixture
def first_multiclass_test_case(): 
    X =  np.random.rand(200, 768)
    y = np.random.randint(4, size=200)
    return X, y

@pytest.fixture
def second_multiclass_test_case(): 
    X =  np.random.rand(200, 200)
    y = np.random.randint(3, size=200)
    return X, y

@pytest.fixture
def third_multiclass_test_case(): 
    X =  np.random.rand(100, 50)
    y = np.random.randint(5, size=100)
    return X, y


def test_binary_probless_ranking(first_binary_test_case, second_binary_test_case, third_binary_test_case, snapshot):
    X_first = first_binary_test_case[0]
    y_first = first_binary_test_case[1]
    probeless_ordering_first = probeless.get_neuron_ordering(X_first, y_first)
    assert probeless_ordering_first == snapshot(name="first_case")
    X_second = second_binary_test_case[0]
    y_second = second_binary_test_case[1]
    probeless_ordering_second = probeless.get_neuron_ordering(X_second, y_second)
    assert probeless_ordering_second == snapshot(name="second_case")
    X_third = third_binary_test_case[0]
    y_third = third_binary_test_case[1] 
    probeless_ordering_third = probeless.get_neuron_ordering(X_third, y_third)
    assert probeless_ordering_third == snapshot(name="third_case")
    

def test_binary_iou_ranking(first_binary_test_case, second_binary_test_case, third_binary_test_case, snapshot):
    X_first = first_binary_test_case[0]
    y_first = first_binary_test_case[1]
    iou_ordering_first = list(iou_probe.get_neuron_ordering(X_first, y_first))
    assert iou_ordering_first == snapshot(name="first_case")
    X_second = second_binary_test_case[0]
    y_second = second_binary_test_case[1]
    iou_ordering_second = list(iou_probe.get_neuron_ordering(X_second, y_second))
    assert iou_ordering_second == snapshot(name="second_case")
    X_third = third_binary_test_case[0]
    y_third = third_binary_test_case[1] 
    iou_ordering_third = list(iou_probe.get_neuron_ordering(X_third, y_third))
    assert iou_ordering_third == snapshot(name="third_case")

def test_binary_guassian_ranking(guassian_first_test_case, guassian_second_test_case, guassian_third_test_case, snapshot): 
    X_first = guassian_first_test_case[0]
    y_first = guassian_first_test_case[1]
    first_probe = guassian_probe.train_probe(X_first, y_first)
    first_guassian_ordering =  guassian_probe.get_neuron_ordering(first_probe, X_first.shape[-1])
    assert first_guassian_ordering == snapshot(name="first_case")
    X_second = guassian_second_test_case[0]
    y_second = guassian_second_test_case[1]
    second_probe = guassian_probe.train_probe(X_second, y_second)
    second_guassian_ordering = guassian_probe.get_neuron_ordering(second_probe, X_second.shape[-1])
    assert second_guassian_ordering == snapshot(name="second_case")
    X_third = guassian_third_test_case[0]
    y_third = guassian_third_test_case[1] 
    third_probe = guassian_probe.train_probe(X_third, y_third)
    third_guassian_ordering = guassian_probe.get_neuron_ordering(third_probe, X_third.shape[-1])
    assert third_guassian_ordering == snapshot(name="third_case")

  
def test_binary_logistic_regression_probe_lca(first_binary_test_case, second_binary_test_case, third_binary_test_case, snapshot):
    X_first = first_binary_test_case[0]
    y_first = first_binary_test_case[1]
    first_probe = linear_probe.train_logistic_regression_probe(X_first, y_first, lambda_l2=0.1, lambda_l1=0.1)
    first_ranking , _ = linear_probe.get_top_neurons(first_probe, 1, {"class0": 0, "class1": 1}) 
    first_ranking = list(first_ranking)
    assert first_ranking == snapshot(name="first_case")
    X_second = second_binary_test_case[0]
    y_second = second_binary_test_case[1]
    second_probe = linear_probe.train_logistic_regression_probe(X_second, y_second, lambda_l2=0.1, lambda_l1=0.1)
    second_ranking , _ = linear_probe.get_top_neurons(second_probe, 1, {"class0": 0, "class1": 1}) 
    second_ranking = list(second_ranking)
    assert second_ranking == snapshot(name="second_case")
    X_third = third_binary_test_case[0]
    y_third = third_binary_test_case[1]
    third_probe = linear_probe.train_logistic_regression_probe(X_third, y_third, lambda_l2=0.1, lambda_l1=0.1)
    third_ranking , _ = linear_probe.get_top_neurons(third_probe, 1, {"class0": 0, "class1": 1}) 
    third_ranking = list(third_ranking)
    assert third_ranking == snapshot(name="third_case")


def test_binary_logistic_regression_probe_no_reg(first_binary_test_case, second_binary_test_case, third_binary_test_case, snapshot):
    X_first = first_binary_test_case[0]
    y_first = first_binary_test_case[1]
    first_probe = linear_probe.train_logistic_regression_probe(X_first, y_first, lambda_l2=0.0, lambda_l1=0.0)
    first_ranking , _ = linear_probe.get_top_neurons(first_probe, 1, {"class0": 0, "class1": 1}) 
    first_ranking = list(first_ranking)
    assert first_ranking == snapshot(name="first_case")
    X_second = second_binary_test_case[0]
    y_second = second_binary_test_case[1]
    second_probe = linear_probe.train_logistic_regression_probe(X_second, y_second, lambda_l2=0.0, lambda_l1=0.0)
    second_ranking , _ = linear_probe.get_top_neurons(second_probe, 1, {"class0": 0, "class1": 1}) 
    second_ranking = list(second_ranking)
    assert second_ranking == snapshot(name="second_case")
    X_third = third_binary_test_case[0]
    y_third = third_binary_test_case[1]
    third_probe = linear_probe.train_logistic_regression_probe(X_third, y_third, lambda_l2=0.0, lambda_l1=0.0)
    third_ranking , _ = linear_probe.get_top_neurons(third_probe, 1, {"class0": 0, "class1": 1}) 
    third_ranking = list(third_ranking)
    assert third_ranking == snapshot(name="third_case")


def test_binary_logistic_regression_probe_lasso(first_binary_test_case, second_binary_test_case, third_binary_test_case, snapshot):
    X_first = first_binary_test_case[0]
    y_first = first_binary_test_case[1]
    first_probe = linear_probe.train_logistic_regression_probe(X_first, y_first, lambda_l2=0.0, lambda_l1=0.1)
    first_ranking , _ = linear_probe.get_top_neurons(first_probe, 1, {"class0": 0, "class1": 1}) 
    first_ranking = list(first_ranking)
    assert first_ranking == snapshot(name="first_case")
    X_second = second_binary_test_case[0]
    y_second = second_binary_test_case[1]
    second_probe = linear_probe.train_logistic_regression_probe(X_second, y_second, lambda_l2=0.0, lambda_l1=0.1)
    second_ranking , _ = linear_probe.get_top_neurons(second_probe, 1, {"class0": 0, "class1": 1}) 
    second_ranking = list(second_ranking)
    assert second_ranking == snapshot(name="second_case")
    X_third = third_binary_test_case[0]
    y_third = third_binary_test_case[1]
    third_probe = linear_probe.train_logistic_regression_probe(X_third, y_third, lambda_l2=0.0, lambda_l1=0.1)
    third_ranking , _ = linear_probe.get_top_neurons(third_probe, 1, {"class0": 0, "class1": 1}) 
    third_ranking = list(third_ranking)
    assert third_ranking == snapshot(name="third_case")




def test_binary_logistic_regression_probe_ridge(first_binary_test_case, second_binary_test_case, third_binary_test_case, snapshot):
    X_first = first_binary_test_case[0]
    y_first = first_binary_test_case[1]
    first_probe = linear_probe.train_logistic_regression_probe(X_first, y_first, lambda_l2=0.1, lambda_l1=0.0)
    first_ranking , _ = linear_probe.get_top_neurons(first_probe, 1, {"class0": 0, "class1": 1}) 
    first_ranking = list(first_ranking)
    assert first_ranking == snapshot(name="first_case")
    X_second = second_binary_test_case[0]
    y_second = second_binary_test_case[1]
    second_probe = linear_probe.train_logistic_regression_probe(X_second, y_second, lambda_l2=0.0, lambda_l1=0.1)
    second_ranking , _ = linear_probe.get_top_neurons(second_probe, 1, {"class0": 0, "class1": 1}) 
    second_ranking = list(second_ranking)
    assert second_ranking == snapshot(name="second_case")
    X_third = third_binary_test_case[0]
    y_third = third_binary_test_case[1]
    third_probe = linear_probe.train_logistic_regression_probe(X_third, y_third, lambda_l2=0.0, lambda_l1=0.1)
    third_ranking , _ = linear_probe.get_top_neurons(third_probe, 1, {"class0": 0, "class1": 1}) 
    third_ranking = list(third_ranking)
    assert third_ranking == snapshot(name="third_case")



def test_probless_multiclass_ranking(first_multiclass_test_case, second_multiclass_test_case, third_multiclass_test_case, snapshot):

    X_first = first_multiclass_test_case[0]
    y_first = first_multiclass_test_case[1]
    top_neurons_first, top_neurons_per_class_first = probeless.get_neuron_ordering_for_all_tags(X_first, y_first, idx2label={0: "class0", 1: "class1", 2:"class2", 3:"class3"})
    assert top_neurons_first == snapshot(name="overall_ranking_first") 
    assert top_neurons_per_class_first["class0"] == snapshot(name="class0_ranking_first") 
    assert top_neurons_per_class_first["class1"] == snapshot(name="class1_ranking_first") 
    assert top_neurons_per_class_first["class2"] == snapshot(name="class2_ranking_first") 
    assert top_neurons_per_class_first["class3"] == snapshot(name="class3_ranking_first") 

    # test the second multiclass case
    X_second = second_multiclass_test_case[0]
    y_second = second_multiclass_test_case[1]
    top_neurons_second, top_neurons_per_class_second = probeless.get_neuron_ordering_for_all_tags(X_second, y_second, idx2label={0: "class0", 1: "class1", 2:"class2"})
    assert top_neurons_second == snapshot(name="overall_ranking_second") 
    assert top_neurons_per_class_second["class0"] == snapshot(name="class0_ranking_second") 
    assert top_neurons_per_class_second["class1"] == snapshot(name="class1_ranking_second") 
    assert top_neurons_per_class_second["class2"] == snapshot(name="class2_ranking_second") 

    # test the third multiclass case
    X_third = third_multiclass_test_case[0]
    y_third = third_multiclass_test_case[1]
    top_neurons_third , top_neurons_per_class_third= probeless.get_neuron_ordering_for_all_tags(X_third, y_third, idx2label={0: "class0", 1: "class1", 2:"class2", 3:"class3", 4:"class4"})
    assert top_neurons_third == snapshot(name="overall_ranking_third") 
    assert top_neurons_per_class_third["class0"] == snapshot(name="class0_ranking_third") 
    assert top_neurons_per_class_third["class1"] == snapshot(name="class1_ranking_third") 
    assert top_neurons_per_class_third["class2"] == snapshot(name="class2_ranking_third") 
    assert top_neurons_per_class_third["class3"] == snapshot(name="class3_ranking_third") 
    assert top_neurons_per_class_third["class4"] == snapshot(name="class4_ranking_third") 

    



def test_multiclass_logistic_regression_probe_lca(first_multiclass_test_case, second_multiclass_test_case, third_multiclass_test_case, snapshot):
    # Test the first multiclass case
    X_first = first_multiclass_test_case[0]
    y_first = first_multiclass_test_case[1]
    first_probe = linear_probe.train_logistic_regression_probe(X_first, y_first, lambda_l2=0.1, lambda_l1=0.1)
    _, top_neurons_first = linear_probe.get_top_neurons(first_probe, 1, {"class0": 0, "class1": 1, "class2": 2, "class3":3}) 
    assert list(top_neurons_first["class0"]) == snapshot(name="class0_ranking_first") 
    assert list(top_neurons_first["class1"]) == snapshot(name="class1_ranking_first") 
    assert list(top_neurons_first["class2"]) == snapshot(name="class2_ranking_first") 
    assert list(top_neurons_first["class3"]) == snapshot(name="class3_ranking_first") 
    # Test the second multiclass case 
    X_second = second_multiclass_test_case[0]
    y_second = second_multiclass_test_case[1]
    second_probe = linear_probe.train_logistic_regression_probe(X_second, y_second, lambda_l2=0.1, lambda_l1=0.1)
    _, top_neurons_second = linear_probe.get_top_neurons(second_probe, 1, {"class0": 0, "class1": 1, "class2": 2}) 
    assert list(top_neurons_second["class0"]) == snapshot(name="class0_ranking_second") 
    assert list(top_neurons_second["class1"]) == snapshot(name="class1_ranking_second") 
    assert list(top_neurons_second["class2"]) == snapshot(name="class2_ranking_second") 
    # Test third multiclass case 
    X_third = third_multiclass_test_case[0]
    y_third = third_multiclass_test_case[1]
    third_probe = linear_probe.train_logistic_regression_probe(X_third, y_third, lambda_l2=0.1, lambda_l1=0.1)
    _, top_neurons_third = linear_probe.get_top_neurons(third_probe, 1,  {"class0": 0, "class1": 1, "class2": 2, "class3":3, "class4":4})
    assert list(top_neurons_third["class0"]) == snapshot(name="class0_ranking_third") 
    assert list(top_neurons_third["class1"]) == snapshot(name="class1_ranking_third") 
    assert list(top_neurons_third["class2"]) == snapshot(name="class2_ranking_third") 
    assert list(top_neurons_third["class3"]) == snapshot(name="class3_ranking_third") 
    assert list(top_neurons_third["class4"]) == snapshot(name="class4_ranking_third") 






def test_multiclass_logistic_regression_probe_lca(first_multiclass_test_case, second_multiclass_test_case, third_multiclass_test_case, snapshot):
    # Test the first multiclass case
    X_first = first_multiclass_test_case[0]
    y_first = first_multiclass_test_case[1]
    first_probe = linear_probe.train_logistic_regression_probe(X_first, y_first, lambda_l2=0.1, lambda_l1=0.1)
    _, top_neurons_first = linear_probe.get_top_neurons(first_probe, 1, {"class0": 0, "class1": 1, "class2": 2, "class3":3}) 
    assert list(top_neurons_first["class0"]) == snapshot(name="class0_ranking_first") 
    assert list(top_neurons_first["class1"]) == snapshot(name="class1_ranking_first") 
    assert list(top_neurons_first["class2"]) == snapshot(name="class2_ranking_first") 
    assert list(top_neurons_first["class3"]) == snapshot(name="class3_ranking_first") 
    # Test the second multiclass case 
    X_second = second_multiclass_test_case[0]
    y_second = second_multiclass_test_case[1]
    second_probe = linear_probe.train_logistic_regression_probe(X_second, y_second, lambda_l2=0.1, lambda_l1=0.1)
    _, top_neurons_second = linear_probe.get_top_neurons(second_probe, 1, {"class0": 0, "class1": 1, "class2": 2}) 
    assert list(top_neurons_second["class0"]) == snapshot(name="class0_ranking_second") 
    assert list(top_neurons_second["class1"]) == snapshot(name="class1_ranking_second") 
    assert list(top_neurons_second["class2"]) == snapshot(name="class2_ranking_second") 
    # Test third multiclass case 
    X_third = third_multiclass_test_case[0]
    y_third = third_multiclass_test_case[1]
    third_probe = linear_probe.train_logistic_regression_probe(X_third, y_third, lambda_l2=0.1, lambda_l1=0.1)
    _, top_neurons_third = linear_probe.get_top_neurons(third_probe, 1,  {"class0": 0, "class1": 1, "class2": 2, "class3":3, "class4":4})
    assert list(top_neurons_third["class0"]) == snapshot(name="class0_ranking_third") 
    assert list(top_neurons_third["class1"]) == snapshot(name="class1_ranking_third") 
    assert list(top_neurons_third["class2"]) == snapshot(name="class2_ranking_third") 
    assert list(top_neurons_third["class3"]) == snapshot(name="class3_ranking_third") 
    assert list(top_neurons_third["class4"]) == snapshot(name="class4_ranking_third")



def test_multiclass_logistic_regression_probe_no_reg(first_multiclass_test_case, second_multiclass_test_case, third_multiclass_test_case, snapshot):
    # Test the first multiclass case
    X_first = first_multiclass_test_case[0]
    y_first = first_multiclass_test_case[1]
    first_probe = linear_probe.train_logistic_regression_probe(X_first, y_first, lambda_l2=0.0, lambda_l1=0.0)
    _, top_neurons_first = linear_probe.get_top_neurons(first_probe, 1, {"class0": 0, "class1": 1, "class2": 2, "class3":3}) 
    assert list(top_neurons_first["class0"]) == snapshot(name="class0_ranking_first") 
    assert list(top_neurons_first["class1"]) == snapshot(name="class1_ranking_first") 
    assert list(top_neurons_first["class2"]) == snapshot(name="class2_ranking_first") 
    assert list(top_neurons_first["class3"]) == snapshot(name="class3_ranking_first") 
    # Test the second multiclass case 
    X_second = second_multiclass_test_case[0]
    y_second = second_multiclass_test_case[1]
    second_probe = linear_probe.train_logistic_regression_probe(X_second, y_second, lambda_l2=0.0, lambda_l1=0.0)
    _, top_neurons_second = linear_probe.get_top_neurons(second_probe, 1, {"class0": 0, "class1": 1, "class2": 2}) 
    assert list(top_neurons_second["class0"]) == snapshot(name="class0_ranking_second") 
    assert list(top_neurons_second["class1"]) == snapshot(name="class1_ranking_second") 
    assert list(top_neurons_second["class2"]) == snapshot(name="class2_ranking_second") 
    # Test third multiclass case 
    X_third = third_multiclass_test_case[0]
    y_third = third_multiclass_test_case[1]
    third_probe = linear_probe.train_logistic_regression_probe(X_third, y_third, lambda_l2=0.0, lambda_l1=0.0)
    _, top_neurons_third = linear_probe.get_top_neurons(third_probe, 1,  {"class0": 0, "class1": 1, "class2": 2, "class3":3, "class4":4})
    assert list(top_neurons_third["class0"]) == snapshot(name="class0_ranking_third") 
    assert list(top_neurons_third["class1"]) == snapshot(name="class1_ranking_third") 
    assert list(top_neurons_third["class2"]) == snapshot(name="class2_ranking_third") 
    assert list(top_neurons_third["class3"]) == snapshot(name="class3_ranking_third") 
    assert list(top_neurons_third["class4"]) == snapshot(name="class4_ranking_third")


def test_multiclass_logistic_regression_probe_lasso(first_multiclass_test_case, second_multiclass_test_case, third_multiclass_test_case, snapshot):
    # Test the first multiclass case
    X_first = first_multiclass_test_case[0]
    y_first = first_multiclass_test_case[1]
    first_probe = linear_probe.train_logistic_regression_probe(X_first, y_first, lambda_l2=0.0, lambda_l1=0.1)
    _, top_neurons_first = linear_probe.get_top_neurons(first_probe, 1, {"class0": 0, "class1": 1, "class2": 2, "class3":3}) 
    assert list(top_neurons_first["class0"]) == snapshot(name="class0_ranking_first") 
    assert list(top_neurons_first["class1"]) == snapshot(name="class1_ranking_first") 
    assert list(top_neurons_first["class2"]) == snapshot(name="class2_ranking_first") 
    assert list(top_neurons_first["class3"]) == snapshot(name="class3_ranking_first") 
    # Test the second multiclass case 
    X_second = second_multiclass_test_case[0]
    y_second = second_multiclass_test_case[1]
    second_probe = linear_probe.train_logistic_regression_probe(X_second, y_second, lambda_l2=0.0, lambda_l1=0.1)
    _, top_neurons_second = linear_probe.get_top_neurons(second_probe, 1, {"class0": 0, "class1": 1, "class2": 2}) 
    assert list(top_neurons_second["class0"]) == snapshot(name="class0_ranking_second") 
    assert list(top_neurons_second["class1"]) == snapshot(name="class1_ranking_second") 
    assert list(top_neurons_second["class2"]) == snapshot(name="class2_ranking_second") 
    # Test third multiclass case 
    X_third = third_multiclass_test_case[0]
    y_third = third_multiclass_test_case[1]
    third_probe = linear_probe.train_logistic_regression_probe(X_third, y_third, lambda_l2=0.0, lambda_l1=0.1)
    _, top_neurons_third = linear_probe.get_top_neurons(third_probe, 1,  {"class0": 0, "class1": 1, "class2": 2, "class3":3, "class4":4})
    assert list(top_neurons_third["class0"]) == snapshot(name="class0_ranking_third") 
    assert list(top_neurons_third["class1"]) == snapshot(name="class1_ranking_third") 
    assert list(top_neurons_third["class2"]) == snapshot(name="class2_ranking_third") 
    assert list(top_neurons_third["class3"]) == snapshot(name="class3_ranking_third") 
    assert list(top_neurons_third["class4"]) == snapshot(name="class4_ranking_third")


def test_multiclass_logistic_regression_probe_ridge(first_multiclass_test_case, second_multiclass_test_case, third_multiclass_test_case, snapshot):
    # Test the first multiclass case
    X_first = first_multiclass_test_case[0]
    y_first = first_multiclass_test_case[1]
    first_probe = linear_probe.train_logistic_regression_probe(X_first, y_first, lambda_l2=0.1, lambda_l1=0.0)
    _, top_neurons_first = linear_probe.get_top_neurons(first_probe, 1, {"class0": 0, "class1": 1, "class2": 2, "class3":3}) 
    assert list(top_neurons_first["class0"]) == snapshot(name="class0_ranking_first") 
    assert list(top_neurons_first["class1"]) == snapshot(name="class1_ranking_first") 
    assert list(top_neurons_first["class2"]) == snapshot(name="class2_ranking_first") 
    assert list(top_neurons_first["class3"]) == snapshot(name="class3_ranking_first") 
    # Test the second multiclass case 
    X_second = second_multiclass_test_case[0]
    y_second = second_multiclass_test_case[1]
    second_probe = linear_probe.train_logistic_regression_probe(X_second, y_second, lambda_l2=0.1, lambda_l1=0.0)
    _, top_neurons_second = linear_probe.get_top_neurons(second_probe, 1, {"class0": 0, "class1": 1, "class2": 2}) 
    assert list(top_neurons_second["class0"]) == snapshot(name="class0_ranking_second") 
    assert list(top_neurons_second["class1"]) == snapshot(name="class1_ranking_second") 
    assert list(top_neurons_second["class2"]) == snapshot(name="class2_ranking_second") 
    # Test third multiclass case 
    X_third = third_multiclass_test_case[0]
    y_third = third_multiclass_test_case[1]
    third_probe = linear_probe.train_logistic_regression_probe(X_third, y_third, lambda_l2=0.1, lambda_l1=0.0)
    _, top_neurons_third = linear_probe.get_top_neurons(third_probe, 1,  {"class0": 0, "class1": 1, "class2": 2, "class3":3, "class4":4})
    assert list(top_neurons_third["class0"]) == snapshot(name="class0_ranking_third") 
    assert list(top_neurons_third["class1"]) == snapshot(name="class1_ranking_third") 
    assert list(top_neurons_third["class2"]) == snapshot(name="class2_ranking_third") 
    assert list(top_neurons_third["class3"]) == snapshot(name="class3_ranking_third") 
    assert list(top_neurons_third["class4"]) == snapshot(name="class4_ranking_third")

