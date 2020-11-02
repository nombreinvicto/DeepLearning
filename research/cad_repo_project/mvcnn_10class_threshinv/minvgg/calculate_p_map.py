import numpy as np


# %%

def calculate_mAP(file_path):
    ap = []
    with open(file_path, mode="r") as apfile:
        lines = apfile.readlines()
    for p in lines:
        ap.append(float(p.split("\n")[0]))
    return len(lines), np.mean(ap)
# %%
