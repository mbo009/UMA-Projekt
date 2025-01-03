import numpy as np
import matplotlib.pyplot as plt


def error_rate(predictions, labels):
    return np.mean(predictions != labels)


def accuracy(predictions, labels):
    return np.mean(predictions == labels)


def true_positive_rate(predictions, labels):
    TP = np.sum((predictions == True) & (labels == True))
    FN = np.sum((predictions == False) & (labels == True))
    return TP / (TP + FN)


def false_positive_rate(predictions, labels):
    FP = np.sum((predictions == True) & (labels == False))
    TN = np.sum((predictions == False) & (labels == False))
    return FP / (FP + TN)


def precision(predictions, labels):
    TP = np.sum((predictions == True) & (labels == True))
    FP = np.sum((predictions == True) & (labels == False))
    return TP / (TP + FP)


def f1_score(predictions, labels):
    prec = precision(predictions, labels)
    sens = true_positive_rate(predictions, labels)
    return 2 * prec * sens / (prec + sens)


def roc_curve(predictions, labels):
    thresholds = np.sort(predictions)
    tpr, fpr = [], []

    for threshold in thresholds:
        y_pred = (predictions >= threshold).astype(int)
        tpr.append(true_positive_rate(y_pred, labels))
        fpr.append(false_positive_rate(y_pred, labels))

    tpr = np.array(tpr)
    fpr = np.array(fpr)

    sorted_indices = np.argsort(fpr)
    return tpr[sorted_indices], fpr[sorted_indices]


def auc_roc_score(predictions, labels):
    tpr, fpr = roc_curve(predictions, labels)
    auc = np.trapz(tpr, fpr)
    return auc
