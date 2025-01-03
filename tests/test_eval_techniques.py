import numpy as np
from src.eval_techniques import (
    error_rate,
    accuracy,
    true_positive_rate,
    false_positive_rate,
    precision,
    f1_score,
    roc_curve,
    auc_roc_score,
)


def test_error_rate():
    predictions = np.array([True, False, True, False])
    labels = np.array([True, True, False, False])
    assert np.isclose(error_rate(predictions, labels), 0.5)


def test_accuracy():
    predictions = np.array([True, False, True, False])
    labels = np.array([True, False, True, False])
    assert np.isclose(accuracy(predictions, labels), 1.0)


def test_true_positive_rate():
    predictions = np.array([True, False, True, False])
    labels = np.array([True, True, True, False])
    assert np.isclose(true_positive_rate(predictions, labels), 0.6667, atol=1e-4)


def test_false_positive_rate():
    predictions = np.array([True, False, True, False])
    labels = np.array([False, False, True, True])
    assert np.isclose(false_positive_rate(predictions, labels), 0.5, atol=1e-4)


def test_precision():
    predictions = np.array([True, False, True, False])
    labels = np.array([True, True, True, False])
    assert np.isclose(precision(predictions, labels), 1.0, atol=1e-4)


def test_f1_score():
    predictions = np.array([True, False, True, False])
    labels = np.array([True, True, True, False])
    assert np.isclose(f1_score(predictions, labels), 0.8, atol=1e-4)


def test_roc_curve():
    predictions = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = np.array([1, 1, 0, 0, 0])
    tpr, fpr = roc_curve(predictions, labels)
    assert len(tpr) == len(fpr)
    assert np.all(tpr >= 0) and np.all(tpr <= 1)
    assert np.all(fpr >= 0) and np.all(fpr <= 1)


def test_auc_roc_score():
    predictions = np.array([0.9, 0.8, 0.7, 0.6, 0.4])
    labels = np.array([1, 1, 1, 1, 0])
    auc = auc_roc_score(predictions, labels)
    assert np.isclose(auc, 0.625, atol=1e-4)
