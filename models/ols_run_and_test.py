import statsmodels.formula.api as smf


def ols_run_and_test(X_train, X_test, r_formula):
    """
    run statsomdel OLS mode and test with test data
    Args:
        r_formula:
        X_train: dataframe
        X_test:  dataframe
    Return:
        ols: model
        y_pred: prediction from test set
    """
    # fit
    ols = smf.ols(r_formula, data=X_train).fit()
    # Test
    y_pred = ols.predict(X_test)

    return ols, y_pred