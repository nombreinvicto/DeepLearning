import numpy as np
import pandas as pd
import json
import os

cwd = os.getcwd()
json_file = f"{cwd}//p_at_k.json"


# %%

def calculate_mAP(file_path):
    ap = []
    with open(file_path, mode="r") as apfile:
        lines = apfile.readlines()
    for p in lines:
        ap.append(float(p.split("\n")[0]))
    return len(lines), np.mean(ap)


# %%

def calculate_avg_pk(json_file_path, only_odds=True):
    out_dict = {}
    df = pd.DataFrame(columns=['type', 'p@1', 'p@2', 'p@3', 'p@4', 'p@5',
                               'p@6', 'p@7', 'p@8', 'p@9', 'p@10'])

    with open(json_file_path, mode="r") as pks_file:
        content_dict = json.loads(pks_file.read())

    for k, v in content_dict.items():
        out_dict[k] = list(map(lambda x: round(x, 2), np.mean(v, axis=0)))

    for i, (k, v) in enumerate(out_dict.items()):
        df.loc[i] = [k] + list(v)

    if only_odds:
        selected_cols = []
        for i in range(1, len(df.columns)-2, 2):
            selected_cols.append(df.columns[i])
        return df[['type'] + selected_cols + ['p@10']].copy()

    return df.copy()


# %%

df = calculate_avg_pk(json_file, only_odds=False)
df.head(10)
print(df.mean().mean())
