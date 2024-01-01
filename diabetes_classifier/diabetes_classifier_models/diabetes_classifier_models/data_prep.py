import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold


def upsample_training_cv_data(
    X_train_cv_list: list[pd.DataFrame],
    y_train_cv_list: list[pd.Series],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[list[pd.DataFrame], list[pd.Series]]:
    """Function for upsampling the cross validation training data to account for the inbalance in the dataset.
    Only the training datasets, not evaluation datasets, should be upsampled.

    Args:
        X_train_cv_list (list[pd.DataFrame]): List of cross validation training sets X_trains
        y_train_cv_list (list[pd.Series]): List of cross validation y_trains
        X_train (pd.DataFrame): The total training dataset for upsampling
        y_train (pd.DataFrame): The correct answers to be upsampled

    Returns:
        tuple[list[pd.DataFrame], list[pd.Series]]: Upsampled list of X_trains, y_trains
    """
    smt = SMOTE(random_state=42, k_neighbors=3)
    X_train_cv_list_upsample: list[pd.DataFrame] = []
    y_train_cv_list_upsample: list[pd.Series] = []
    for X_train_cv, y_train_cv in zip(X_train_cv_list, y_train_cv_list):
        X_train_cv_upsample, y_train_cv_upsample = smt.fit_resample(
            X_train_cv, y_train_cv
        )
        X_train_cv_list_upsample.append(X_train_cv_upsample)
        y_train_cv_list_upsample.append(y_train_cv_upsample)
    X_train_upsample, y_train_upsample = smt.fit_resample(X_train, y_train)
    return (
        X_train_cv_list_upsample,
        y_train_cv_list_upsample,
        X_train_upsample,
        y_train_upsample,
    )


def create_cross_validation(
    df_train: pd.DataFrame, features: list[str], target_column: str, n_folds: int = 5
) -> tuple[list[pd.DataFrame], list[pd.Series], list[pd.DataFrame], list[pd.Series],]:
    """Creates a list of cross validation training and evaluation sets using sklearn's kfold to train and evaluate models

    Args:
        df_train (pd.DataFrame): training dataset
        features (list[str]): features to use in the training data set
        target_column (str): the binary column name identifying if diabetes is present
        n_folds (int, optional): number of folds to use in kfold validation. Defaults to 5.

    Returns:
        tuple[ list[pd.DataFrame], list[pd.Series], list[pd.DataFrame], list[pd.Series]]: list of cross validation sets, in order: X_train, y_train, X_eval, y_eval
    """
    X_train_total = df_train[features]
    y_train_total = df_train[target_column]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    split = kf.split(X_train_total, y_train_total)

    X_train_cv_list: list[pd.DataFrame] = []
    y_train_cv_list: list[pd.Series] = []

    X_eval_cv_list: list[pd.DataFrame] = []
    y_eval_cv_list: list[pd.Series] = []

    print("creating cross validation")
    for _, (train_index, test_index) in enumerate(split):
        X_train, y_train = (
            X_train_total.iloc[train_index],
            y_train_total.iloc[train_index],
        )
        X_eval, y_eval = X_train_total.iloc[test_index], y_train_total.iloc[test_index]
        X_train_cv_list.append(X_train)
        y_train_cv_list.append(y_train)

        X_eval_cv_list.append(X_eval)
        y_eval_cv_list.append(y_eval)
    print("done creating cross validation")
    return (
        X_train_cv_list,
        y_train_cv_list,
        X_eval_cv_list,
        y_eval_cv_list,
    )
