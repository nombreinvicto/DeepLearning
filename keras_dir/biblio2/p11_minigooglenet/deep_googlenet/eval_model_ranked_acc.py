import tiny_imagenet_config as config
from loader_util.preprocessing import ImageToArrayPreprocessor, \
    SimplePreProcessor, MeanPreprocessor
from loader_util.utils import rank5_accuracy
from loader_util.io import HDF5DatasetGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import json

# %%
# load the RGB means
means = json.loads(open(config.dataset_mean).read())

# init the preprocessors
sp = SimplePreProcessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# init the testing dataset generator
testgen = HDF5DatasetGenerator(dbPath=config.test_hdf5,
                               batchSize=64,
                               preprocessors=[sp, mp, iap],
                               classes=config.num_classes)
# %%
print(f"[INFO] loading model......")
model = load_model(config.model_path)  # type: Model
# %%
print(f"[INFO] predicting on test data......")
preds = model.predict_generator(testgen.generator(),
                                steps=testgen.numImages // 64,
                                max_queue_size=10)
# compute accuracies
rank1, rank5 = rank5_accuracy(preds, testgen.db["labels"])
print(f"[INFO] rank1: {rank1} and rank5: {rank5}......")
testgen.close()
#%%