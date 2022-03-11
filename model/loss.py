import torch.nn.functional as F
from torch import nn
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy(output, target):
    return F.cross_entropy(output, target)

def mixup(output, target, valid=False):
    if valid:
        return F.cross_entropy(output, target)
    else:
        targets1, targets2, lam = target
        criterion = nn.CrossEntropyLoss(reduction="mean")
        return lam * criterion(output, targets1) + (1 - lam) * criterion(output, targets2)


