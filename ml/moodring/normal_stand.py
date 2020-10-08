from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

##Normalization & Standardization
def normal_stand(df):
    df_stand = StandardScaler().fit_transform(np.array(df))
    df_normal = Normalizer().fit_transform(df_stand)

    return df_normal