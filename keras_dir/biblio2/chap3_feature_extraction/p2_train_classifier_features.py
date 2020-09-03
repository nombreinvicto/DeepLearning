# import the necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
import h5py

# %%

# construct the argument parser
data_dir = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\caltech-101\extracted_features"
args_dict = {
    'db': f"{data_dir}//features.hdf5",
    "model": f"{data_dir}//caltech.pickle",
    "jobs": -1,
}

# %%

db = h5py.File(args_dict['db'], mode="r")
i = int(db["labels"].shape[0] * 0.75)
# %%

# define the set of params for ML
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0]}
model = GridSearchCV(LogisticRegression(multi_class="auto"),
                     param_grid=params,
                     cv=3,
                     n_jobs=args_dict["jobs"])
trainx = db["extracted_features"][:i]
trainy = db["labels"][:i]
print(trainy.shape)
model.fit(trainx, trainy)
print(f"[INFO] best hyperparams: {model.best_params_}")
# %%

# evaluate the model
preds = model.predict(db["extracted_features"][i:])
label_names = db["label_names"]
label_names = [names.decode() for names in label_names]
print(classification_report(db["labels"][i:], preds,
                            target_names=label_names))

# %%
f = open(args_dict["model"], "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()
db.close()
