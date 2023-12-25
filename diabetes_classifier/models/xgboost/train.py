import argparse
import warnings
from argparse import Namespace
from collections import defaultdict
from functools import partial

import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def score_xgboost_classifier(
    clf: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    beta: float,
) -> dict[str, any]:
    sns.set_style("darkgrid")
    pred_probas = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    metrics = {}

    metrics["auc"] = roc_auc_score(y_test, pred_probas)
    metrics["recall"] = recall_score(y_test, preds)
    metrics["f_score"] = fbeta_score(y_test, preds, beta=beta)
    metrics["precision"] = precision_score(y_test, preds)
    metrics["accuracy_score"] = accuracy_score(y_test, preds)

    return metrics


def hyperopt_optimize_function(
    space: dict[str, any],
    X_train_list_cv: list[pd.DataFrame],
    y_train_list_cv: list[pd.Series],
    X_eval_list_cv: list[pd.DataFrame],
    y_eval_list_cv: list[pd.Series],
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, any]:
    with mlflow.start_run(nested=True):
        eval_metrics_cv: dict[str, any] = defaultdict(lambda: [])
        for X_train_cv, y_train_cv, X_eval_cv, y_eval_cv in zip(
            X_train_list_cv, y_train_list_cv, X_eval_list_cv, y_eval_list_cv
        ):
            clf = XGBClassifier(**space)
            clf.fit(
                X_train_cv,
                y_train_cv,
                eval_set=[(X_eval_cv, y_eval_cv)],
                early_stopping_rounds=20,
                eval_metric="auc",
                verbose=True,
            )
            beta = 1.2
            eval_metrics = score_xgboost_classifier(
                clf=clf,
                X_test=X_eval_cv,
                y_test=y_eval_cv,
                beta=beta,
            )

            for metric_name, metric_val in eval_metrics.items():
                eval_metrics_cv[metric_name].append(metric_val)
        clf = XGBClassifier(**space)
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            eval_metric="auc",
            verbose=True,
        )
        test_metrics = score_xgboost_classifier(
            clf,
            X_test=X_test,
            y_test=y_test,
            beta=beta,
        )

        for metric_name, metric_val in test_metrics.items():
            mlflow.log_metric(key=f"{metric_name}_test", value=metric_val)

        avg_eval_metrics = {key: np.mean(vals) for key, vals in eval_metrics_cv.items()}

        for metric_name, metric_val in avg_eval_metrics.items():
            mlflow.log_metric(key=f"{metric_name}_eval", value=metric_val)

        # mlflow.xgboost.log_model(clf, "model")
        for param_name, param_value in space.items():
            mlflow.log_param(key=param_name, value=param_value)
        mlflow.xgboost.log_model(clf, "model")
        return {
            "loss": -np.mean(avg_eval_metrics["auc"]),
            "status": STATUS_OK,
            "model": clf,
        }


def create_cross_validation(
    df_train: pd.DataFrame, features: list[str], target_column: str, n_folds: int = 5
) -> tuple[
    list[pd.DataFrame],
    list[pd.Series],
    list[pd.DataFrame],
    list[pd.Series],
    pd.DataFrame,
    pd.Series,
]:
    smt = SMOTE(random_state=42, k_neighbors=3)
    X_train_total = df_train[features]
    y_train_total = df_train[target_column]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    split = kf.split(X_train_total, y_train_total)

    X_train_list: list[pd.DataFrame] = []
    y_train_list: list[pd.Series] = []

    X_eval_list: list[pd.DataFrame] = []
    y_eval_list: list[pd.Series] = []

    print("creating cross validation")
    for _, (train_index, test_index) in enumerate(split):
        X_train, y_train = (
            X_train_total.iloc[train_index],
            y_train_total.iloc[train_index],
        )
        X_eval, y_eval = X_train_total.iloc[test_index], y_train_total.iloc[test_index]

        X_train_smt, y_train_smt = smt.fit_resample(X_train, y_train)

        X_train_list.append(X_train_smt)
        y_train_list.append(y_train_smt)

        X_eval_list.append(X_eval)
        y_eval_list.append(y_eval)
    print("done creating cross validation")
    return (
        X_train_list,
        y_train_list,
        X_eval_list,
        y_eval_list,
        X_train_smt,
        y_train_smt,
    )


def train_xgboost_classifier(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: list[str],
    target_column: str,
    n_folds: int = 5,
) -> XGBClassifier:
    space = {
        "eta": hp.uniform("eta", 0, 1.0),
        "max_depth": scope.int(hp.quniform("max_depth", 2, 20, 1)),
        "subsample": hp.uniform("subsample", 0, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0, 1),
        "n_estimators": scope.int(hp.quniform("n_estimators", 20, 5000, 25)),
        "gamma": hp.uniform("gamma", 0, 0.3),
        "min_child_weight": hp.uniform("min_child_weight", 0, 1),
        "nthread": 4,
    }
    (
        X_train_list_cv,
        y_train_list_cv,
        X_eval_list_cv,
        y_eval_list_cv,
        X_train,
        y_train,
    ) = create_cross_validation(
        df_train,
        features,
        target_column,
        n_folds,
    )

    X_test = df_test[features]
    y_test = df_test[target_column]

    func_to_optimize = partial(
        hyperopt_optimize_function,
        X_train_list_cv=X_train_list_cv,
        y_train_list_cv=y_train_list_cv,
        X_eval_list_cv=X_eval_list_cv,
        y_eval_list_cv=y_eval_list_cv,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    trials = Trials()
    rstate = np.random.default_rng(42)
    _ = fmin(
        fn=func_to_optimize,
        space=space,
        algo=tpe.suggest,
        max_evals=10,
        trials=trials,
        rstate=rstate,
    )

    sorted_trials = sorted(trials, key=lambda x: x["result"]["loss"])
    sorted_trials[0]["result"]["model"]
    return


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-folds", type=int)
    parser.add_argument("--train-data")
    parser.add_argument("--test-data")
    parser.add_argument("--target-column")

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    train_data_source = args.train_data
    test_data_source = args.test_data
    target_column = args.target_column
    n_folds = args.num_folds

    df_train = pd.read_pickle(train_data_source)
    df_test = pd.read_pickle(test_data_source)

    features = [col for col in df_train.columns if col != target_column]
    mlflow.set_experiment("diabetes_prediction")
    with mlflow.start_run():
        _ = train_xgboost_classifier(
            df_train=df_train,
            df_test=df_test,
            features=features,
            target_column=target_column,
            n_folds=n_folds,
        )


if __name__ == "__main__":
    main()
