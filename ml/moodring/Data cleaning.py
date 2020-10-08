import pandas as pd
import numpy as np
import os

data_path = 'C:\\Users\\huzec\\Desktop\\Doryab\\moodring\\agg_features'
output_path = 'C:\\Users\\huzec\\Desktop\\Doryab\\moodring\\results\\latest'
out_feature_path = 'C:\\Users\\huzec\\Desktop\\Doryab\\moodring\\results\\latest\\features_importances'

os.chdir(data_path)
df = pd.read_csv('plugin_ios_activity_recognition_cleaned.csv') ##sensor data files
df = df.drop(columns = ['name', 'index'])
device_label_lst = ['25BY', '44TA', '52NK', '203RN', '63WN', '87WR', '2210NK', '194SR',
                    '512SO', '105PE', '169KS', '49ZR', '219WT', '231TW', '19HL', '33mo ',
                    '249my', '169k6', '31CN'] ##unique device label list
df = df.loc[df['device_label'].isin(device_label_lst)]
df['timestamp'] = df['timestamp'].apply(pd.to_numeric)
#for device_id has null
del df['device_id']
df['device_id'] = df['device_id_1']
df.isnull().any()

os.chdir(out_path)
df.to_csv('activity.csv', index = False)##Output cleaned data