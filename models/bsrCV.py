import statsmodels.formula.api as smf
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, RegressorMixin


class statsmodel(BaseEstimator, RegressorMixin):
    """a simple wrapper for statsmodel estimator so that
    can be used as sklearn function such as cross validation.
    Reference:
    https://sukhbinder.wordpress.com/2018/08/07/cross-validation-score-with-statsmodels/
    """
    def __init__(self, sm_class, formula):
        self.sm_class = sm_class
        self.formula = formula
        self.model = None
        self.result = None

    def fit(self, data):
        self.model = self.sm_class(self.formula, data)
        self.result = self.model.fit()

    def predict(self, X):
        return self.result.predict(X)


def BSRCV(X, name_space, cv=10, threshold_out=None, verbose=True):
    """Backward Stepwise Regression with cross validation (default 10-fold)
       the feature with the largest p value are dropped (and above threshold_out
       if defined)
    Args:
        threshold_out: safe to exclude a feature if its p-value > threshold_out
                       , otherwise stop bsr here. usually 0.05
        X: dataframe including training data x and y
        cv: default 10 fold cv
        name_space: list of all feature name
        verbose: whether print
    Return:
        bsr: statsmodel OLS model
        cv_scores: shape (len(name_space)-2, )
        n_f_: number of features selected, shape (len(name_space)-2, )
        features: r formulas for all models in BSR, shape (len(name_space)-2, )
    """
    included = [f for f in name_space]  # list of all features to start with

    r_formula = 'y~' + '+'.join([inc for inc in included])
    cv_scores_ = []
    n_f_, features_ = [], []
    # BSR
    while len(included) > 1:
        bsr = smf.ols(r_formula, data=X).fit()  # run with included features and extract p-values for each

        pvalues = bsr.pvalues.iloc[1:]  # use all coefs except intercept
        worst_pval = pvalues.max()      # null if pvalues is empty

        # if worst is smaller than threshold, means all features
        # are significant and not need to drop more break
        if threshold_out and worst_pval < threshold_out:
            break
        # remove worst feature and run CV on model
        worst_feature = pvalues.idxmax()
        included.remove(worst_feature)  # remove feature
        r_formula = 'y~' + '+'.join([inc for inc in included])  # update r formula

        # run cv with updated feature list
        clf = statsmodel(smf.ols, r_formula)
        score = cross_val_score(clf, X, X['y'], cv=10, scoring='neg_mean_squared_error')
        cv_scores_.append((score.mean(), score.std() / cv ** 0.5))
        n_f_.append(len(included))
        features_.append(r_formula)

        if verbose:
            print('Drop {:30} with p-value {:.4},  {} / {}, cv = {:.4}'.format(worst_feature, worst_pval, len(included),
                                                                               len(name_space), score.mean()))

    return bsr, cv_scores_, n_f_, features_

