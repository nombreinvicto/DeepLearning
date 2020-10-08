import pandas as pd
import numpy as np
import os

data_path = 'C:\\Users\\huzec\\Desktop\\Doryab\\moodring\\extracted_features'
out_path = 'C:\\Users\\huzec\\Desktop\\Doryab\\moodring\\agg_features'

os.chdir(data_path)
df_device = pd.read_csv('aware_device.csv') #read all devices

ls_sensors = ['act', 'audio', 'call', 'loc', 'screen', 'wifi'] #the list of sensors that can be used

ls_device_ids = list(set(list(df_device['device_id'])))
df_col = pd.DataFrame()
for ls_device_id in ls_device_ids:
    df_row = pd.DataFrame()
    for ls_sensor in ls_sensors:
        file_path = 'f_' + ls_sensor + ls_device_id + '.csv'
        try:
            df = pd.read_csv(file_path)
            print('succeed')
            df = df[(df['group'] == 'week') & pd.isna(df['weekday']) & (df['epoch'] == 'allday')]
            df = df.drop(['device_id', 'epoch', 'weekday', 'group'], axis=1)
            df_row = pd.concat([df_row, df], axis=1)
        except:
            print('fail')

    df_row.insert(0, 'device_id', ls_device_id)
    df_col = pd.concat([df_col, df_row], axis=0)

df_final = df_col.drop(['device_id'], axis = 1)
df_final.insert(0, 'device_id', df_col['device_id'])

os.chdir(out_path)
df_final.to_csv('agg_features.csv', index = False)