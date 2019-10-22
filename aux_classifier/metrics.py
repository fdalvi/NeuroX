import numpy as np

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef as mcc, f1_score


def numpyfy(x):
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def accuracy(preds, labels):
    preds = numpyfy(preds)
    labels = numpyfy(labels)
    return (preds == labels).mean()


def f1(preds, labels):
    preds = numpyfy(preds)
    labels = numpyfy(labels)
    return f1_score(y_true=labels, y_pred=preds)


def accuracy_and_f1(preds, labels):
    preds = numpyfy(preds)
    labels = numpyfy(labels)
    acc = accuracy(preds, labels)
    f1_s = f1_score(y_true=labels, y_pred=preds)
    return (acc + f1_s) / 2


def pearson(preds, labels):
    preds = numpyfy(preds)
    labels = numpyfy(labels)
    return pearsonr(preds, labels)[0]


def spearman(preds, labels):
    preds = numpyfy(preds)
    labels = numpyfy(labels)
    return spearmanr(preds, labels)[0]


def pearson_and_spearman(preds, labels):
    preds = numpyfy(preds)
    labels = numpyfy(labels)
    return ((pearson(preds, labels) + spearman(preds, labels)) / 2,)


def matthews_corrcoef(preds, labels):
    preds = numpyfy(preds)
    labels = numpyfy(labels)
    return mcc(preds, labels)


def compute_score(preds, labels, metric):
    if metric == 'accuracy':
        return accuracy(preds, labels)
    elif metric == 'f1':
        return f1(preds, labels)
    elif metric == 'accuracy_and_f1':
        return accuracy_and_f1(preds, labels)
    elif metric == 'pearson':
        return pearson(preds, labels)
    elif metric == 'spearman':
        return spearman(preds, labels)
    elif metric == 'pearson_and_spearman':
        return pearson_and_spearman(preds, labels)
    elif metric == 'matthews_corrcoef':
        return matthews_corrcoef(preds, labels)