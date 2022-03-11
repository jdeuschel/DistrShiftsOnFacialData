# The code here is based on the code at
# https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np


def optimal_temp_scale(probs, labels, lr=0.05, max_iter=100, logits_given=False, criterion=None):
    """tempterature scaling which is applied on the logits at inference.
    
    Parameters
    ----------
    probs : torch.tensor
        model outputs; usually probit values where softmax already applied
    labels : torch.tensor
        targets
    lr : float, optional
        learning rate, by default 0.01
    max_iter : int, optional
        maximal number of iterations taken for optimization, by default 50
    logits_given : bool, optional
        wheather logits are given instead of probits. 
        False: probits given, True: logits (without that softmax has been applied) , by default False
    criterion : [type], optional
        Temporary! Optional loss function (can be deleted), by default None
    
    Returns
    -------
    torch
        optimal temperature wrt the loss
    """
    labels = labels.long()
    temp_grid = np.logspace(-1, 1, 1000)
    logits = torch.log(probs + 1e-12)

    nll_criterion = nn.CrossEntropyLoss()

    before_temperature_nll = nll_criterion(logits, labels)

    best_nll = before_temperature_nll
    best_temp = 1.0

    for T in temp_grid:
        loss = nll_criterion(logits / T, labels)
        if loss < best_nll:
            best_temp = T
            best_nll = loss

    return best_temp  # .item()


