from sklearn.metrics import mean_squared_error, r2_score


def test_report(y_predict, y_test):
    """report MSE and r2 score,
       given y_predict and y_test"""

    mse = mean_squared_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)

    print("Test Accurracy\nmse: {}".format(str(mse)))
    print("r2: {}%\n".format(str(r2 * 100)[:7]))

    return mse, r2
