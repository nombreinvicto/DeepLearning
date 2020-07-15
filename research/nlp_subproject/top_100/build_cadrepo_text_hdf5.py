from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from loader_util.io import HDF5DatasetWriterWithTokens
from loader_util.preprocessing import AspectAwarePreprocessor, \
    MeanSubtractionPreProcessor
from imutils import paths
import numpy as np
import progressbar
from cv2 import cv2
import os

# %%

# imagePath = r"C:\Users\mhasa\Google Drive\Tutorial
# Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\Unique3DClusters"
imagePath = r"C:\Users\mhasa\Desktop\NLP_100_choice"
dbPath = r"C:\Users\mhasa\Desktop"

# grab paths to training images and then extract train class labels and encode
trainPaths = list(paths.list_images(imagePath))

trainLabels = [p.split(os.path.sep)[-2] for p in trainPaths]
print("Unique Classes: ", len(np.unique(trainLabels)))
# %%
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)
# %%
class_labels = np.array(le.classes_)

#%% TokenProcessing
# setting global variables
dbBase = r"refined_dataset"
all_sentences = []
all_labels = []

#%%

# reading the words from file and creating a dataset
types = ['Bearings', 'Bolts', 'Collets', 'Springs', 'Sprockets']
class_num = len(types)

for label, type in enumerate(types):

    with open(f"{dbBase}//{type}_100.txt", mode='r') as partFile:
        content = partFile.readlines()
        content = [c.replace("\n", "") for c in content]

        all_sentences.extend(content)
        all_labels.extend([label] * len(content))

all_sentences = np.array(all_sentences)
all_labels = np.array(all_labels)

# %%

# perform stratified sampling from train set to construct validation set
split = train_test_split(trainPaths,
                         all_sentences,
                         trainLabels,
                         test_size=0.167,
                         stratify=trainLabels,
                         random_state=42)
trainpaths, testpaths, trainsent, testsent, trainlabels, testlabels = split


# %%

## Global Variables


vocab_size = 425
max_length = 4
trunc_type = "post"
oov_tok = "<OOV>"

IMAGE_READ_MODE = [cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE]
TARGET_SIZE = 28
CHANNLE_DIM = 1
IMAGE_READ_INDEX = 1

#%% Tokenization


# tokenize the train set
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(trainsent)
train_sequences = np.array(tokenizer.texts_to_sequences(trainsent))
padded_train_sequences = pad_sequences(train_sequences, maxlen=max_length,
                                       truncating=trunc_type)
# tokenise and pad the test set
test_sequences = np.array(tokenizer.texts_to_sequences(testsent))
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_length)

# check out the word index
print(len(tokenizer.word_index.keys()))
print(tokenizer.word_index)

#%%


# construct list pair
datasets = [
    ('train', trainpaths, trainlabels, padded_train_sequences,
     f"{dbPath}//train_gan_threshinv_5classNLP1002_28px1px_pristine"
     f".hdf5"),
    ('val', testpaths, testlabels, padded_test_sequences,
     f"{dbPath}//validate_gan_threshinv_5classNLP100_28px1px_pristine"
     f".hdf5")
]


#%%
# initialise the preprocessors
aap = AspectAwarePreprocessor(TARGET_SIZE, TARGET_SIZE)
mp = MeanSubtractionPreProcessor()

# create dataset loop over the dataset tuples
for dataType, paths, labels, token_sequences, output in datasets:
    # create HDF5 writer
    print(f"[INFO] building {output}.....")
    writer = HDF5DatasetWriterWithTokens(dims=(len(paths),
                                     TARGET_SIZE,
                                     TARGET_SIZE,
                                     CHANNLE_DIM),
                               outputPath=output,
                               tokenSize=max_length)

    # initialise the progressbar
    widgets = [f"Building Dataset: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    # now loop over the image paths
    for (i, (path, label, token)) in enumerate(zip(paths, labels,
                                               token_sequences)):
        # load the image and preprocess it
        image = cv2.imread(path, IMAGE_READ_MODE[IMAGE_READ_INDEX])
        image = aap.preprocess(image)
        # image = mp.preprocess(image)
        image = image.astype('float32')
        image = np.expand_dims(image, axis=-1)

        #image = image / 255.0 # dont use for gan sets


        # add the image and label to the HDF5 dataset
        writer.add([image], [label], [token])
        pbar.update(i)

    # store class labels before exiting
    writer.storeClassLabels(class_labels)

    # close the writer
    pbar.finish()
    writer.close()
# %%
