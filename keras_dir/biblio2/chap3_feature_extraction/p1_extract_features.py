# %%
import progressbar, random, os
import numpy as np
import seaborn as sns

sns.set()
# %%
# import the necessary keras packages
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from loader_util.io import HDF5DatasetWriter

##
from tensorflow.keras.preprocessing.image import ImageDataGenerator, \
    img_to_array, load_img
from tensorflow.keras.applications import VGG16, imagenet_utils
from tensorflow.keras.models import Model
from imutils import paths

# %%

# create the argeparse dict

data_dir = r"C:\Users\mhasa\Google Drive\Tutorial " \
           r"Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\caltech-101\images"

output_dir = r"C:\Users\mhasa\Google Drive\Tutorial " \
             r"Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets" \
             r"\caltech-101\extracted_features\extracted_features_caltech.hdf5"

args = {
    "dataset": data_dir,
    "output": output_dir,
    "batch_size": 32,
    "buffer_size": 1000
}

# %%

print(f'[INFO] loading images.....')
image_paths = list(paths.list_images(args["dataset"]))
random.shuffle(image_paths)

# extract the class labels from image paths
lables = [p.split(os.path.sep)[-2] for p in image_paths]

# encode the labels
le = LabelEncoder()
encoded_labels = le.fit_transform(lables)

# %%
print(f'[INFO] loading the pretrained network.....')
model = VGG16(weights="imagenet", include_top=False)  # type: Model

# init the HdF5 dataset writer
dataset = HDF5DatasetWriter(dims=(len(image_paths), 512 * 7 * 7),
                            outputPath=args["output"],
                            dataKey="extracted_features",
                            bufSize=args["buffer_size"])
dataset.storeClassLabels(le.classes_)
# %%

widgets = ["Extracting Features : ", " ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(image_paths),
                               widgets=widgets).start()

# loop over the images in batches
for i in range(0, len(image_paths), args["batch_size"]):
    batch_paths = image_paths[i:i + args["batch_size"]]
    batch_labels = encoded_labels[i:i + args["batch_size"]]
    batch_images = []

    # loop over the images and labels in the current batch
    for j, image_path in enumerate(batch_paths):
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        batch_images.append(image)

    batch_images = np.vstack(batch_images)
    extracted_features = model.predict(batch_images,
                                       batch_size=args["batch_size"])
    extracted_features = extracted_features. \
        reshape((extracted_features.shape[0], -1))
    dataset.add(extracted_features, batch_labels)
    pbar.update(i)

dataset.close()
pbar.finish()
# %%
