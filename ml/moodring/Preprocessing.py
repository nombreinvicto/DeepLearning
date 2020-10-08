import pandas as pd
import numpy as np
import os

data_path = 'C:\\Users\\huzec\\Desktop\\Doryab\\moodring\\agg_features'
output_path = 'C:\\Users\\huzec\\Desktop\\Doryab\\moodring\\results\\latest'

os.chdir(data_path)
data = pd.read_csv('agg_features.csv')

##Remove unlabeled rows
label_null_index = data[data['depression'].isnull()].index
data = data.drop(index = label_null_index)

##Two threshold to handle missing features
nans_per_col = data.isnull().sum(axis=0)
selected_nans_per_col = list(nans_per_col[nans_per_col <= 0.8 * len(data)].index.values)
data = data[selected_nans_per_col]
nans_per_row = data.isnull().sum(axis=1)
selected_nans_per_row_index = list(nans_per_row[nans_per_row <= 0.8 * len(data.columns)].index.values)
data = data[data.index.isin(selected_nans_per_row_index)]

##Get label
label_lst = []
score_lst = list(data['score'])
for score in score_lst:
    if score <= 14:
        label_lst.append(0)
    elif score > 14:
        label_lst.append(1)
#print(label_lst)
device_id_lst = list(data['device_id'])
#len(device_id_lst)
week_lst = list(data['group'])
#week_lst
device_label_lst = list(data['device_label'])
#device_label_lst

##Remove non-numerical columns
data_no_object = data.select_dtypes(exclude = ['object'])
##Get feature list
data_features = data_no_object.drop(['depression', 'score'], axis = 1)
##Get feature list
feature_lst = list(data_features.columns)
#len(feature_lst)

##Filling in NaN via forward interpolation
data_features = data_features.interpolate(method = 'linear', limit_direction = 'forward', axis = 0)
##Filling in NaN via backward interpolation
data_features = data_features.interpolate(method = 'linear', limit_direction = 'backward', axis = 0)

##Correlation (regression)
data_features.insert(0, 'score', score_lst)
data_features_corr_regression = data_features.corr()
#print(data_features_corr_regression['score'].index[np.where(np.isnan(data_features_corr_regression['score']))[0]]) #[u'home_stay_time_percent_100m', u'home_stay_time_percent_10m', u'percentage_silence_total', u'percentage_silence_total_with_unknown']

data_features = data_features.drop(['home_stay_time_percent_100m','home_stay_time_percent_10m', 'percentage_silence_total', 'percentage_silence_total_with_unknown'], axis = 1)
data_features_corr_regression = data_features.corr()
data_features = data_features.drop(['score'], axis = 1)
#print(data_features_corr_regression['score'])

data_features.insert(0, 'week', week_lst)
data_features.insert(0, 'device_label', device_label_lst)
data_features.insert(0, 'device_id', device_id_lst)
data_features.insert(0, 'score', score_lst)
data_features.insert(0, 'label', label_lst)
##Output the data_features

