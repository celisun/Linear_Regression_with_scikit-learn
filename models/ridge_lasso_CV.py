from sklearn.linear_model import RidgeCV, LassoCV
import numpy as np


def ridgeCV(X_train, y_train, lambds, cv=10):
    """Train Ridge on a sequence of lambdas (such as range(), or np.logspace())
    with cross validation and report every cv scores.
    Return:
        model_r
        scores_r: cv scores with mean and std error, shape (len(lambds), )
    """
    # Ridge
    model_r = RidgeCV(lambds, store_cv_values=True).fit(X_train, y_train)  # cv_values_ shape=(1168, 100)

    scores_mean_r = model_r.cv_values_.mean(0)  # shape (100,)
    scores_std_r = np.std(model_r.cv_values_, 0) / X_train.shape[0] ** 0.5  # one standard error  1/k sqrt(sum(R-CV)2)
    scores_r = [s for s in zip(scores_mean_r, scores_std_r)]

    return model_r, scores_r


def lassoCV(X_train, y_train, lambds, cv=10):
    """Train LASSO on a sequence of lambdas with cross validation
    and report every cv scores
    Return:
        model_l
        scores_r: cv scores with mean and std error, shape (len(lambds), )
    """
    # Lasso
    model_l = LassoCV(alphas=lambds, cv=cv).fit(X_train, y_train)  # mse_path_ shape (n_lambda, cv) = (100,10)
    # max_iter = 100000, normalize

    scores_mean_l = model_l.mse_path_.mean(1)          # shape (n_alpha,) = (100, )
    scores_std_l = np.std(model_l.mse_path_, 1) / cv ** 0.5
    scores_l = [s for s in zip(scores_mean_l, scores_std_l)]

    return model_l, scores_l