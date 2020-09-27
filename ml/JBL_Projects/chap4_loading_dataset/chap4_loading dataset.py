import pandas as pd
import os

#%%

path = os.getcwd()
data = pd.read_csv(f"{path}/pima-indians-diabetes.data.csv",
                   )
data.head()
#%%