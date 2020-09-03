from loader_util.utils import rank5_accuracy
from sklearn.linear_model import LogisticRegression
import pickle
import h5py

# %%
# construct the argument parser
data_dir = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\animals\extracted_features"
args_dict = {
    "db": f"{data_dir}//features.hdf5",
    "model": f"{data_dir}//animals.pickle"
}

# %%
# load the pretrained model
model = pickle.loads(
    open(args_dict["model"], mode="rb").read())  # type: LogisticRegression

# %%
# open the dataset
db = h5py.File(args_dict["db"], mode="r")
i = int(db["labels"].shape[0] * 0.75)
# %%

# make predictions on the testing set and get rank1 and rank5 accuracies
print(f"[INFO] predicting.....")
testx = db["extracted_features"][i:]
print(f"Shape of Testx: {testx.shape}")
preds = model.predict_proba(testx)
print(f"Shape of prediction probs: {preds.shape}")
# %%
# get the ranked accuracies
print(f"Shape of labels: {db['labels'][i:].shape}")
rank1, rank5 = rank5_accuracy(preds, db["labels"][i:])
# %%
# display the accuracies
print(f"Rank1 acc: {rank1 * 100:0.2f} and Rank5 acc: {rank5 * 100:0.2f}")
db.close()
