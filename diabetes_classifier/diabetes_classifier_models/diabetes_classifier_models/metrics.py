from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Classifier(Protocol):
    def predict_proba(X) -> np.ndarray:
        """"""

    def predict(X) -> np.ndarray:
        """"""


def score_classifier(
    clf: Classifier,
    X: pd.DataFrame,
    y: pd.DataFrame,
    beta: float,
) -> dict[str, any]:
    """Scores a classifier on some datasets, then returns the metrics created in the form of a dictionary

    Args:
        clf (XGBClassifier): classifier to get metrics on
        X (pd.DataFrame): Eval or test set to get metrics on
        y (pd.DataFrame): Correct answers for the eval or test set
        beta (float): beta value to use for the f_beta score

    Returns:
        dict[str, any]: dictionary of metrics
    """
    pred_probas = clf.predict_proba(X)[:, 1]
    preds = clf.predict(X)

    metrics = {}

    metrics["auc"] = roc_auc_score(y, pred_probas)
    metrics["recall"] = recall_score(y, preds)
    metrics["f_score"] = fbeta_score(y, preds, beta=beta)
    metrics["precision"] = precision_score(y, preds)
    metrics["accuracy_score"] = accuracy_score(y, preds)

    return metrics
