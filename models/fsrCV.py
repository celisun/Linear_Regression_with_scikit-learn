from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
import pandas as pd

def FSRCV(X_train, y_train, forward=True, cv=10):
    """
    Sequential Feature Selector from mlxtend package, used to
    implement a brute-force forward/backward selection.

    https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/.
    They also have a GridSearchCV function.

    Add/remove the feature that reports the best/worst MSE score
    in the model and run cross validation (default 10-fold).

    Args:
        X_train: numpy
        Y_train: numpy
        forward: FSR, iteratively adding features
        cv: default=10
    Return:
        sfs1: model
        cv_scores: tuple (mean, std error)
        X_train_sfs: the new subsets based on the selected features
    """
    # FSR
    estimator = LinearRegression()
    sfs1 = SFS(estimator,
               k_features=X_train.shape[1],
               forward=forward,
               floating=False,
               verbose=2,
               scoring='neg_mean_absolute_error',
               cv=cv)
    sfs1 = sfs1.fit(X_train, y_train)
    X_train_sfs = sfs1.tranform(X_train)  # selected best

    # get cv scores
    fsr_results = sfs1.get_metric_dict()
    fsr = pd.DataFrame.from_dict(fsr_results).T

    score = fsr.avg_score
    std = fsr.std_err
    cv_scores = [x for x in zip(score, std)]

    return sfs1, cv_scores, X_train_sfs
