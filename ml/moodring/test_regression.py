from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def test_regression(y_true, y_pred):
    MAE = mean_absolute_error(y_true, y_pred)
    MSE = mean_squared_error(y_true, y_pred)
    # R2 = r2_score(y_true, y_pred)

    return MAE, MSE