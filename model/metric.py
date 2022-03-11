import torch
from sklearn.metrics import f1_score as sci_f1_score
from sklearn.metrics import precision_score as sci_precision
from sklearn.metrics import recall_score as sci_recall
import torch.nn.functional as F


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def f1_score(output, target):
    with torch.no_grad():
        output = F.sigmoid(output).round().cpu().numpy()
        target = target.cpu().numpy()

        return sci_f1_score(y_true=target, y_pred=output, average="weighted")


def precision(output, target):
    with torch.no_grad():
        output = F.sigmoid(output).round().cpu().numpy()
        target = target.cpu().numpy()
        return sci_precision(y_true=target, y_pred=output, average="weighted")


def recall(output, target):
    with torch.no_grad():
        output = F.sigmoid(output).round().cpu().numpy()
        target = target.cpu().numpy()
        return sci_recall(y_true=target, y_pred=output, average="weighted")


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
