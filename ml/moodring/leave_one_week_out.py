import pandas as pd
import numpy as np
import os
from normal_stand import normal_stand
from test_regression import test_regression
import xgboost as xgb
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from depression_degree import depression_degree
from performance import performance

data_path = 'C:\\Users\\huzec\\Desktop\\Doryab\\moodring\\agg_features'
output_path = 'C:\\Users\\huzec\\Desktop\\Doryab\\moodring\\results\\latest'
out_feature_path = 'C:\\Users\\huzec\\Desktop\\Doryab\\moodring\\results\\latest\\features_importances'
##path needs to be changed

data_features = pd.read_csv('data_features.csv') #input data_features from where you output

lasso = linear_model.Lasso(alpha = 0.1)
lr = LinearRegression()
rfr = RandomForestRegressor(random_state=0)
xgb = xgb.XGBRegressor()

model = lr #lasso, rfr, xgb

users = data_features.groupby('device_label')
users_names = users.size().index
model_outputs = []
for user in users_names:
    df = data_features[data_features['device_label' ]== user]
    device_label = user
    weeks = list(df['week'])
    # print(weeks)
    # MAE_lst = []
    # MSE_lst = []
    actual_lst = []
    predict_lst = []
    for week in weeks:
        df_test = df.loc[df['week'] == week]
        df_train = df.loc[df['week'] != week]
        device_id = list(df_test['device_id'])[0]
        train_y = np.array(df_train['score'])
        df_train = df_train.drop(['score', 'device_id', 'week', 'device_label', 'label'], axis = 1)
        train_x = normal_stand(df_train)
        test_y = np.array(df_test['score'])
        # print(int(test_y))
        test_y_degree = depression_degree(test_y)
        # print(test_y_degree)
        df_test = df_test.drop(['score', 'device_id', 'week', 'device_label', 'label'], axis = 1)
        test_x = normal_stand(df_test)
        model.fit(train_x, train_y)
        model_pred = model.predict(test_x)
        model_pred_degree = depression_degree(model_pred)
        model_MAE, model_MSE = test_regression(test_y, model_pred)
        # MAE_lst.append(model_MAE)
        # MSE_lst.append(model_MSE)
        actual_lst.append(int(test_y))
        predict_lst.append(float(model_pred))
        diff = model_pred_degree - test_y_degree
        perform = performance(model_pred_degree, test_y_degree)
        model_outputs.append \
            ([device_label, device_id, week, model_pred, model_pred_degree, test_y, test_y_degree, model_MAE, model_MSE, diff, perform])
    # print('============')
    # print(actual_lst)
    # print(predict_lst)
    # print('============')
    ##PLOT ACTUAL & PRED
    # df_plot = pd.DataFrame({'Actual': actual_lst, 'Predicted': predict_lst})
    # print(df_plot)
    # df_plot.plot(kind='bar',figsize=(16,10))
    # plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    # plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    # plt.title(str(user))
    # plt.show()
    ##PLOT MSE & MAE
    # plt.plot(weeks,MAE_lst,'s-',color = 'r',label='MAE')
    # plt.plot(weeks,MSE_lst,'o-',color = 'g',label='MSE')
    # plt.title(str(user))
    # plt.xlabel('week')
    # plt.ylabel('error')
    # plt.legend(loc = 'best')
    # plt.show()


os.chdir(out_path)
model_results = pd.DataFrame(model_outputs, columns = ['device_label', 'device_id', 'week', 'predicted_score', 'predicted_depression_degree', 'true_score', 'true_depression_degree', 'MAE', 'MSE', 'difference', 'performance'])
model_results.to_csv('model_results_leaveoneweekout.csv', index = False) ##model needs to be changed