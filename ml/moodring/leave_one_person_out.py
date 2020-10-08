import pandas as pd
import numpy as np
import os
from normal_stand import normal_stand
from test_regression import test_regression
import xgboost as xgb
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from feat_agg import feat_agg
from feat_sel import feat_sel
from output_dict import output_dict

data_path = 'C:\\Users\\huzec\\Desktop\\Doryab\\moodring\\agg_features' ##data_path needs to be changed
output_path = 'C:\\Users\\huzec\\Desktop\\Doryab\\moodring\\results\\latest'
out_feature_path = 'C:\\Users\\huzec\\Desktop\\Doryab\\moodring\\results\\latest\\features_importances'

data_features = pd.read_csv('data_features.csv') #input data_features from where you output

lasso = linear_model.Lasso(alpha = 0.1)
lr = LinearRegression()
rfr = RandomForestRegressor(random_state=0)
xgb = xgb.XGBRegressor()

model = lr #lasso, rfr, xgb

model_outputs = []
model_dict_feat = {}
data_features.dropna(axis = 0, how = 'any', inplace = True)
#data split
group = data_features.groupby('device_label')
users = group.size().index
for user in users:
    device_label = user
    df_train = data_features[data_features['device_label']!=user]
    df_test = data_features[data_features['device_label']==user]
    train_y = np.array(df_train['score'])
    df_train = df_train.drop(['score', 'device_id', 'week', 'device_label', 'label'], axis = 1)
    train_x = normal_stand(df_train)
    test_y = np.array(df_test['score'])
    df_test = df_test.drop(['score', 'device_id', 'week', 'device_label', 'label'], axis = 1)
    test_x = normal_stand(df_test)
    feature_list = list(df_train.columns)
    model.fit(train_x, train_y)
    model_pred = model.predict(test_x)
    model_pred_train = model.predict(train_x)
    model_MAE, model_MSE = test_regression(train_y, model_pred_train)
    model_outputs.append([device_label, model_pred, test_y, model_MAE, model_MSE])
    model_feat_output, model_feat_imp = feat_sel(model, feature_list)
    model_dict_feat = feat_agg(model_feat_output, model_feat_imp, model_dict_feat)
    model_outputs.append([device_label, model_pred, test_y, model_MAE, model_MSE])

os.chdir(output_path)
model_results = pd.DataFrame(model_outputs, columns = ['device_label', 'predicted_score', 'true_score', 'MAE', 'MSE'])
model_results.to_csv('model_results_general.csv', index = False)  ##model needs to be changed

os.chdir(out_feature_path)
df_model_dict_feat = output_dict(model_dict_feat)
df_model_dict_feat.to_csv('model_features_importances_across_patients.csv', index = False) ##model needs to be changed