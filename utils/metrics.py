from typing import Callable, Union
from sklearn.metrics import confusion_matrix
import torch
from ignite.metrics import EpochMetric
import numpy as np



def balanced_accuracy_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):
    cm = confusion_matrix(y_targets.cpu().numpy(), y_preds.cpu().numpy())

    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    balanced_accuracy = (specificity + sensitivity) / 2

    return balanced_accuracy

class BalancedAccuracy(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu")) -> None:
        super().__init__(balanced_accuracy_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn, device=device)




def auroc_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):
    from sklearn.metrics import roc_auc_score

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return roc_auc_score(y_true, y_pred)

class AUROC(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu")) -> None:
        super().__init__(auroc_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn, device=device)


def auprc_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):
    from sklearn.metrics import average_precision_score

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()

    return average_precision_score(y_true, y_pred)

class AUPRC(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu")) -> None:
        super().__init__(auprc_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn, device=device)




def f1_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):
    from sklearn.metrics import f1_score

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return f1_score(y_true, y_pred, average='binary')

class F1(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu")) -> None:
        super().__init__(f1_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn, device=device)



def precision_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):
    from sklearn.metrics import precision_score

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return precision_score(y_true, y_pred, average='binary')

class Precision(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu")) -> None:
        super().__init__(precision_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn, device=device)



def recall_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):
    from sklearn.metrics import recall_score

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return recall_score(y_true, y_pred, average='binary')

class Recall(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu")) -> None:
        super().__init__(recall_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn, device=device)



def specificity_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):
    cm = confusion_matrix(y_targets.cpu().numpy(), y_preds.cpu().numpy())

    TN = cm[0, 0]
    FP = cm[0, 1]
    specificity = TN / (TN + FP)

    return specificity

class Specificity(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu")) -> None:
        super().__init__(specificity_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn, device=device)



def npv_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):
    cm = confusion_matrix(y_targets.cpu().numpy(), y_preds.cpu().numpy())

    TN = cm[0, 0]
    FN = cm[1, 0]

    npv = TN / (TN + FN)
    return npv

class NPV(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu")) -> None:
        super().__init__(npv_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn, device=device)

class BrierScore(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu")) -> None:
        super().__init__(brier_score_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn, device=device)

def brier_score_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):\
    # y_preds: (N,)
    from sklearn.metrics import brier_score_loss

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return brier_score_loss(y_true, y_pred)

class ECE(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu")) -> None:
        super().__init__(ece_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn, device=device)

def ece_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):
    # y_preds: (N,)
    from torchmetrics.functional import calibration_error

    y_true = y_targets.cpu()
    y_pred = y_preds.cpu()
    return calibration_error(y_pred, y_true, task='binary', norm = "l1")

class NLL(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu")) -> None:
        super().__init__(nll_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn, device=device)

def nll_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):
    # y_preds: (N, C)
    from torch.nn.functional import nll_loss

    y_true = y_targets.cpu()
    y_pred = y_preds.cpu()
    return nll_loss(torch.log(y_pred), y_true)