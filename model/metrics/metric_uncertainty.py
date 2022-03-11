"""
Evaluation Metrics for Uncertainty
==================================
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn.functional as F
from torchvision import transforms
import os



def get_entropy(prob, n_classes=None):
    if n_classes is None:
        max_entropy = 1  # no normalization made
    else:
        max_entropy = -np.log(1 / n_classes)

    entropy = torch.sum(-prob * torch.log(prob), 1) * 1 / max_entropy
    entropy[entropy != entropy] = 0  # nan to zero
    return entropy



def softmax_uncertainty(outputs, labels, use_softmax=False):
    """
    outputs : torch.Tensor, [n_samples, n_classes]
        Logit or softmax outputs of your model. (n_classes)
    torch.Tensor, [n_samples]
        Ground Truth Labels between (0 - (n_classes-1))
    """
    labels = labels.numpy()
    if use_softmax:
        softmaxes = F.softmax(outputs, 1)
    else:
        softmaxes = outputs
    confidences, predictions = softmaxes.max(1)

    confidences = confidences.numpy()
    predictions = predictions.numpy()
    accuracies = np.equal(predictions, labels).astype(float)

    return confidences, accuracies


def ece(confidences, accuracies, n_bins=20):
    """
    ECE

    Arguments:
        confidences {torch.Tensor} -- confidence for each prediction
        accuracies {torch.Tensor} -- corrects: vector of TRUE and FALSE indicating whether the prediction was correct for each prediction

    Keyword Arguments:
        n_bins {int} -- How many bins should be used for the plot (default: {20})
    """
    accuracies = accuracies.numpy().astype(float)
    confidences = confidences.numpy()

    # Calibration Curve Calculation
    bins = np.linspace(0, 1, n_bins + 1)
    bins[-1] = 1.0001
    width = bins[1] - bins[0]

    bin_indices = [
        np.greater_equal(confidences, bin_lower) * np.less(confidences, bin_upper)
        for bin_lower, bin_upper in zip(bins[:-1], bins[1:])
    ]
    bin_corrects = np.array([np.mean(accuracies[bin_index]) for bin_index in bin_indices])  # Height of bars
    bin_scores = np.array([np.mean(confidences[bin_index]) for bin_index in bin_indices])  # confidence range

    # ECE Calculation

    B = np.sum(np.array(bin_indices), 1)
    n = np.sum(B)
    weights = B / n
    d_acc_conf = np.abs(bin_corrects - bin_scores)
    d_acc_conf = np.nan_to_num(d_acc_conf)
    ece = np.sum(d_acc_conf * weights)

    return ece*100, np.nan_to_num(bin_corrects), np.nan_to_num(bins)



def calculate_auroc(preds_in, preds_out, use_softmax=False, n_classes=10, confidence_measure="max_softmax"):
    """
    Calculate AUROC with confidences (max).
    
    Parameters
    ----------
    preds_in : torch.Tensor, [n_samples, n_classes]
        Predictions of the in-distribution (with or without softmax ), [n_samples, n_classes]
    preds_out : torch.Tensor, [n_samples, n_classes]
        Predictions of the out-distribution (with or without softmax )
    use_softmax : bool, optional
        Test , by default False
    
    Returns 
    -------
    float
        AUROC for confidences (max) in range 0-1
    """
    with torch.no_grad():
        if use_softmax:
            preds_in_soft = F.softmax(preds_in, 1)
            preds_out_soft = F.softmax(preds_out, 1)
        else:
            preds_in_soft = preds_in
            preds_out_soft = preds_out

        if confidence_measure == "max_softmax":
            confidences_in, prediction_in = preds_in_soft.max(1)
            confidences_out, prediction_out = preds_out_soft.max(1)
        else:
            confidences_in = 1 - get_entropy(preds_in_soft, n_classes)
            confidences_out = 1 - get_entropy(preds_out_soft, n_classes)

        confidences_in = confidences_in.numpy()
        confidences_out = confidences_out.numpy()
        labels_in = np.ones(len(confidences_in))
        labels_out = np.zeros(len(confidences_out))

        confs = np.concatenate((confidences_in, confidences_out))
        labels = np.concatenate((labels_in, labels_out))
        auroc = roc_auc_score(y_true=labels, y_score=confs)

    return auroc

