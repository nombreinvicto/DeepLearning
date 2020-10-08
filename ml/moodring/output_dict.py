import numpy as np
import pandas as pd

def output_dict(dict_feat):
    output_result = []
    feat_names = dict_feat.keys()
    for feat_name in feat_names:
        output_result_row = [feat_name]
        freq = len(dict_feat[feat_name])
        avg = np.mean(dict_feat[feat_name])
        output_result_row.append(freq)
        output_result_row.append(avg)
        output_result.append(output_result_row)

    df_output_result = pd.DataFrame(output_result, columns=['features', 'frequence', 'avg_importance'])

    return df_output_result