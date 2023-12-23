import argparse
import warnings
from argparse import Namespace
from functools import partial

import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
from imblearn.over_sampling import SMOTE
from sklearn.metrics import fbeta_score, recall_score, roc_auc_score
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

    return metrics


def hyperopt_optimize_function(
    space: dict[str, any],
    X_train_list: list[pd.DataFrame],
    y_train_list: list[pd.Series],
    X_eval_list: list[pd.DataFrame],
    y_eval_list: list[pd.Series],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, any]:
    with mlflow.start_run(nested=True):
        losses = []

        for X_train, y_train, X_eval, y_eval in zip(
            X_train_list, y_train_list, X_eval_list, y_eval_list
        ):
            clf = XGBClassifier(**space)
            print("starting")
            clf.fit(X_train, y_train)
            print("ending")
            beta = 1.2
            eval_metrics = score_xgboost_classifier(
                clf=clf,
                X_test=X_eval,
                y_test=y_eval,
                beta=beta,
            )

            for key, val in eval_metrics.items():
                mlflow.log_metric(key=key, value=val)
            losses.append(eval_metrics["f_score"])
            test_metrics = score_xgboost_classifier(
                clf,
                X_test=X_test,
                y_test=y_test,
                beta=beta,
            )
            for key, val in test_metrics.items():
                mlflow.log_metric(key=f"{key}_test", value=val)
            mlflow.xgboost.log_model(clf, "model")
            for param_name, param_value in space.items():
                mlflow.log_param(key=param_name, value=param_value)

        return {"loss": -np.mean(losses), "status": STATUS_OK, "model": clf}


def create_cross_validation(
    df_train: pd.DataFrame, features: list[str], target_column: str, n_folds: int = 5
) -> tuple[list[pd.DataFrame], list[pd.Series], list[pd.DataFrame], list[pd.Series]]:
    print(df_train.info())

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    split = kf.split(df_train)

    X_train_list: list[pd.DataFrame] = []
    y_train_list: list[pd.Series] = []

    X_eval_list: list[pd.DataFrame] = []
    y_eval_list: list[pd.Series] = []

    print("creating cross validation with smote")
    for i, (train_index, test_index) in enumerate(split):
        print(f"On cross validation set {i+1} of {n_folds}")
        df_eval = df_train.iloc[test_index]
        df_train_ = df_train.iloc[train_index]

        X_train = df_train_[features]
        y_train = df_train_[target_column]

        X_eval = df_eval[features]
        y_eval = df_eval[target_column]

        smt = SMOTE()
        X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
        X_train_list.append(X_train_sm)
        y_train_list.append(y_train_sm)

        X_eval_list.append(X_eval)
        y_eval_list.append(y_eval)
    print("done creating cross validation with smote")
    return X_train_list, y_train_list, X_eval_list, y_eval_list


def train_xgboost_classifier(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: list[str],
    target_column: str,
    n_folds: int = 5,
) -> XGBClassifier:
    space = {
        "eta": hp.uniform("eta", 0.05, 0.8),
        "max_depth": scope.int(hp.quniform("max_depth", 2, 200, 1)),
        "subsample": hp.uniform("subsample", 0, 1),
        "n_estimators": scope.int(hp.quniform("n_estimators", 20, 10000, 25)),
        "gamma": hp.uniform("gamma", 0, 1),
        "min_child_weight": hp.uniform("min_child_weight", 0, 100000),
        "nthread": 4,
    }
    X_train_list, y_train_list, X_eval_list, y_eval_list = create_cross_validation(
        df_train,
        features,
        target_column,
        n_folds,
    )

    X_test = df_test[features]
    y_test = df_test[target_column]

    func_to_optimize = partial(
        hyperopt_optimize_function,
        X_train_list=X_train_list,
        y_train_list=y_train_list,
        X_eval_list=X_eval_list,
        y_eval_list=y_eval_list,
        X_test=X_test,
        y_test=y_test,
    )
    trials = Trials()
    rstate = np.random.default_rng(42)
    best = fmin(
        fn=func_to_optimize,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        rstate=rstate,
    )
    print(best)
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
