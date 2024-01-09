import argparse
import warnings
from argparse import Namespace
from collections import defaultdict
from functools import partial

import mlflow
import numpy as np
import pandas as pd
from data_prep import create_cross_validation, upsample_training_cv_data
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
from metrics import score_classifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


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
            verbose=False,
        )
        beta = 2
        eval_metrics = score_classifier(
            clf=clf,
            X=X_eval_cv,
            y=y_eval_cv,
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
        verbose=False,
    )
    test_metrics = score_classifier(
        clf,
        X=X_test,
        y=y_test,
        beta=beta,
    )

    avg_eval_metrics = {key: np.mean(vals) for key, vals in eval_metrics_cv.items()}

    return {
        "loss": -np.mean(avg_eval_metrics["f_score"]),
        "status": STATUS_OK,
        "model": clf,
        "params": space,
        "avg eval metrics": avg_eval_metrics,
        "test metrics": test_metrics,
    }


def train_xgboost_classifier(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: list[str],
    target_column: str,
    n_folds: int = 5,
    num_hyperopt_evals: int = 5,
    num_hyperopt_trials_to_log: int = 5,
):
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

    # Create list of cross validation training sets
    (
        X_train_cv_list,
        y_train_cv_list,
        X_eval_cv_list,
        y_eval_cv_list,
    ) = create_cross_validation(
        df_train,
        features,
        target_column,
        n_folds,
    )

    X_train = df_train[features]
    y_train = df_train[target_column]

    # Upsample the training (but not evaluation) cross validation datasets
    X_train_cv_list, y_train_cv_list, X_train, y_train = upsample_training_cv_data(
        X_train_cv_list, y_train_cv_list, X_train, y_train
    )

    # Create the total, non upsampled training/test sets
    X_test = df_test[features]
    y_test = df_test[target_column]

    # Find optimal hyperparameters
    func_to_optimize = partial(
        hyperopt_optimize_function,
        X_train_list_cv=X_train_cv_list,
        y_train_list_cv=y_train_cv_list,
        X_eval_list_cv=X_eval_cv_list,
        y_eval_list_cv=y_eval_cv_list,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    # Find good hyperparameters, log the hyperopt trials
    trials = Trials()
    rstate = np.random.default_rng(42)
    _ = fmin(
        fn=func_to_optimize,
        space=space,
        algo=tpe.suggest,
        max_evals=num_hyperopt_evals,
        trials=trials,
        rstate=rstate,
    )
    mlflow_log_best_trials(trials, num_hyperopt_trials_to_log)


def mlflow_log_best_trials(trials: Trials, num_trials_to_log: int = 5):
    """From the hyperopt trials, logs the num_trials_to_log best models, ranked
    according to their losses

    Args:
        trials (Trials): hyperopt trials
        num_trials_to_log (int, optional): number of trials to log. Defaults to 5.
    """
    sorted_trials = sorted(trials, key=lambda x: x["result"]["loss"])
    for i in range(num_trials_to_log):
        with mlflow.start_run(nested=True):
            params = sorted_trials[i]["result"]["params"]
            avg_eval_metrics = sorted_trials[i]["result"]["avg eval metrics"]
            test_metrics = sorted_trials[i]["result"]["test metrics"]
            clf = sorted_trials[i]["result"]["model"]

            for metric_name, metric_val in avg_eval_metrics.items():
                mlflow.log_metric(key=f"{metric_name}_eval", value=metric_val)
            for metric_name, metric_val in test_metrics.items():
                mlflow.log_metric(key=f"{metric_name}_test", value=metric_val)

            # mlflow.xgboost.log_model(clf, "model")
            for param_name, param_value in params.items():
                mlflow.log_param(key=param_name, value=param_value)
            mlflow.xgboost.log_model(clf, "model")


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-folds", type=int)
    parser.add_argument("--train-data")
    parser.add_argument("--test-data")
    parser.add_argument("--target-column")
    parser.add_argument("--num-hyperopt-evals", type=int)
    parser.add_argument("--num-hyperopt-trials-to-log", type=int)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    train_data_source = args.train_data
    test_data_source = args.test_data
    target_column = args.target_column
    n_folds = args.num_folds
    num_hyperopt_evals = args.num_hyperopt_evals
    num_hyperopt_trials_to_log = args.num_hyperopt_trials_to_log

    if num_hyperopt_trials_to_log > num_hyperopt_evals:
        raise ValueError("Cannot log more hyperopt trials than evals")

    df_train = pd.read_pickle(train_data_source)
    df_test = pd.read_pickle(test_data_source)

    features = [col for col in df_train.columns if col != target_column]
    mlflow.set_experiment("diabetes_prediction")
    with mlflow.start_run():
        train_xgboost_classifier(
            df_train=df_train,
            df_test=df_test,
            features=features,
            target_column=target_column,
            n_folds=n_folds,
            num_hyperopt_evals=num_hyperopt_evals,
            num_hyperopt_trials_to_log=num_hyperopt_trials_to_log,
        )


if __name__ == "__main__":
    main()
