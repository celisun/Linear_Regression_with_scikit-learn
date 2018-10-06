from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import *


def knnCV(X_train, y_train, scale, cv=10):
    """
    try a scale of k on KNN with cross validation and report the cv_scores for each k.

    Args:
        X_train: numpy
        Y_train: numpy
        scale: function like np.logspace or range()
        cv: default 10
    Returns:
        cv_scores: shape (len(scale), )
    """
    cv_scores = []
    for k in scale:
        # initialize model
        model = KNeighborsClassifier(n_neighbors=k)
        # run CV on model
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')  # CV with MSE
        # add cv score
        cv_scores.append((scores.mean(), scores.std() / cv ** 0.5))

    return cv_scores
