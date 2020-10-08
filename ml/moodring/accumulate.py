import pandas as pd
import numpy as np
import os
from normal_stand import normal_stand
from test_regression import test_regression
import matplotlib as plt
import xgboost as xgb
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from depression_degree import depression_degree
from performance import performance
from feat_agg import feat_agg
from feat_sel import feat_sel
from output_dict import output_dict

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


model_dict_feat = {}
users = data_features.groupby('device_label')
df_n = data_features.groupby('device_label').size().reset_index(name='counts')
users_names = users.size().index
model_outputs = []
for user in users_names:
    n = list(df_n[df_n['device_label']==user]['counts'])[0]
    df = data_features[data_features['device_label']==user]
    device_label = user
    weeks = []
    actual_lst = []
    predict_lst = []
    for i in range(2, n):
        df_test = df[i:i+1]
        pred_week_lst = list(df_test['week'])
        df_train = df.head(i)
        train_week_lst = list(df_train['week'])
        device_id = list(df_test['device_id'])[0]
        train_y = np.array(df_train['score'])
        df_train = df_train.drop(['score', 'device_id', 'week', 'device_label', 'label'], axis = 1)
        train_x = normal_stand(df_train)
        test_y = np.array(df_test['score'])
        #print(test_y)
        test_y_degree = depression_degree(test_y)
        df_test = df_test.drop(['score', 'device_id', 'week', 'device_label', 'label'], axis = 1)
        test_x = normal_stand(df_test)
        feature_list = list(df_train.columns)
        model.fit(train_x, train_y)
        model_pred = xgb.predict(test_x)
        model_pred_degree = depression_degree(model_pred)
        model_MAE, model_MSE = test_regression(test_y, model_pred)
        perform = performance(model_pred_degree, test_y_degree)
        actual_lst.append(int(test_y))
        predict_lst.append(float(model_pred))
        weeks.append(i)
        diff = model_pred_degree - test_y_degree
        model_feat_output, model_feat_imp = feat_sel(model, feature_list)
        model_dict_feat = feat_agg(model_feat_output, model_feat_imp, model_dict_feat)
        model_outputs.append([device_label, device_id, train_week_lst, pred_week_lst, model_pred, model_pred_degree, test_y, test_y_degree, model_MAE, model_MSE, diff, perform])
        i = i + 1
    plt.plot(weeks, predict_lst, 's-', color = 'r', label="predict value")
    plt.plot(weeks, actual_lst, 'o-', color = 'g', label="true value")
    my_x_ticks = np.arange(2, n, 1)
    plt.xticks(my_x_ticks)
    plt.xlabel("week")
    plt.ylabel("depression score")
    plt.legend(loc = "best")
    plt.title(user)
    plt.show()


os.chdir(output_path)
model_results = pd.DataFrame(model_outputs, columns = ['device_label', 'device_id', 'train_week', 'predict_week', 'predict_score', 'predict_depression_degree', 'true_score', 'true_depression_degree', 'MAE', 'MSE', 'difference', 'performance'])
model_results.to_csv('model_results_accumulate.csv', index = False)
##model needs to be changed

os.chdir(out_feature_path)
df_model_dict_feat = output_dict(model_dict_feat)
df_model_dict_feat.to_csv('model_features_importances.csv', index = False)
##model needs to be changed
