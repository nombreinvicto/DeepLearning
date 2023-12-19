# import the necessary packages
import sys
sys.path.append(r"/content/drive/MyDrive")
from loader_util.preprocessing import ImageToArrayPreprocessor
from loader_util.preprocessing import SimplePreprocessor
from loader_util.preprocessing import MeanPreprocessor
from loader_util.preprocessing import CropPreprocessor
from loader_util.io import HDF5DatasetGenerator
from tensorflow.keras.models import Model
from loader_util.utils import rankn_accuracy
from tensorflow.keras.models import load_model
import numpy as np
import progressbar
import json
# %% ##################################################################
# script constants
saved_model_path = r"/content/drive/MyDrive/Colab Notebooks/ImageDataset/book2/kaggle_dogs_vs_cats/hdf5/output/saved_model.h5"
mean_json_path = r"/content/drive/MyDrive/Colab Notebooks/pyimagesearch/bibilio2/chapter10_alexnet_dogs_cats/mean_dog_vs_cat.json"
test_hdf5_path = r"/content/drive/MyDrive/Colab Notebooks/ImageDataset/book2/kaggle_dogs_vs_cats/hdf5/test.hdf5"
# %% ##################################################################
# load the rgb means
means = json.loads(open(mean_json_path).read())

# init the preprocessors
sp = SimplePreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
cp = CropPreprocessor(227, 227)
iap = ImageToArrayPreprocessor()

# load the pretrained network
print(f"[INFO] modeling model......")
model = load_model(saved_model_path, compile=False)  # type: Model
# %% ##################################################################
# init the test gen
print(f"[INFO] predicting on test data no crops......")
test_gen = HDF5DatasetGenerator(test_hdf5_path,
                                batch_size=64,
                                preprocessors=[sp, mp, iap])
predictions = model.predict_generator(test_gen.generator(),
                                      steps=test_gen.num_images // 64,
                                      max_queue_size=10)

# compute the rank and rank acc
rank1, _ = rankn_accuracy(preds=predictions,
                          labels=test_gen.db["labels"])
print(f"[INFO] rank1 acc no crop: {rank1 * 100}......")
test_gen.close()
# %% ##################################################################
# now to repeat everything with cropp accuracy
print(f"[INFO] now evaluating model with crop accuracy......")
test_gen = HDF5DatasetGenerator(test_hdf5_path,
                                batch_size=64,
                                preprocessors=[mp])
predictions = []

# init the progressbar
# initialize the progress bar
widgets = ["Evaluating: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=test_gen.num_images // 64,
                               widgets=widgets).start()

# loop over a single pass of the test data, extract a batch each loop
for i, (images, labels) in enumerate(test_gen.generator(passes=1)):
    # loop over images in the batch
    for image in images:
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(c) for c in crops], dtype="float32")
        crops_preds = model.predict(crops)
        predictions.append(crops_preds.mean(axis=0))

    # update the pbar when a batch is complete
    pbar.update(i)

# once all images processed
pbar.finish()
print(f"[INFO] predicting on test data with crops......")
rank1, _ = rankn_accuracy(np.array(predictions), test_gen.db["labels"])
print(f"[INFO] rank1 accuracy: {rank1 * 100}......")
test_gen.close()
# %% ##################################################################
