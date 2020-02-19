# import the necessary packages
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from loader_util.nn.conv import LeNet
from imutils import paths
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import imutils
import cv2
import os

sns.set()
# %%

# construct the args dict
path = r'C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\DeepLearning\DeepLearning-DL4CV\Edition3\SB_Code\SB_Code\datasets\SMILEsmileD\SMILEs'
args = {
    'dataset': path,
    'model': 'lenet_smiles.hdf5'
}

# initialise the list of data and labels
data = []
labels = []

# %%
# loop over the input images
for imagePath in sorted(list(paths.list_images(args['dataset']))):
    # load the image and then preprocess it
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("size1: ", image.size)
    image = imutils.resize(image, width=28)
    print("size2: ", image.size)
    image = img_to_array(image)
    print("size3: ", image.size)
    data.append(image)
    print("=" * 40)

    # extract class labels from iameg path and update label list
    label = imagePath.split(os.path.sep)[-3]
    label = 'smiling' if label == 'positives' else 'not_smiling'
    labels.append(label)

# %%
# scale the raw pixel intensities to [0,1]
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder()
labels = to_categorical(le.fit_transform(labels))

# account for skew in labelled data
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# %%
# now split the data
trainx, testx, trainy, testy = train_test_split(data, labels, test_size=0.2,
                                                random_state=42,
                                                stratify=labels)
# initialise the model
model = LeNet.build(28, 28, 1, classes=2)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
    'accuracy'])
H = model.fit(trainx, trainy, validation_data=(testx, testy),
              class_weight=classWeight,
              batch_size=64, epochs=15, verbose=1)

# evaluate the network
preds = model.predict(testx, batch_size=64)
print(classification_report(testy.argmax(axis=1), preds.argmax(axis=1),
                            target_names=le.classes_))

# save model to disk
model.save(args['model'])
#%%
# plot graph
epochs = range(1, 16)
loss = H.history['loss']
accuracy = H.history['accuracy']
val_loss = H.history['val_loss']
val_accuracy = H.history['val_accuracy']
plot_df = pd.DataFrame(
    data=np.c_[epochs, loss, accuracy, val_loss, val_accuracy],
    columns=['epochs', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])

sns.set(font_scale=1)
f, ax = plt.subplots(1, 1, figsize=(15, 8))
sns.lineplot(data=plot_df, x='epochs', y='loss', ax=ax, label='train loss',
             linewidth=3)
sns.lineplot(data=plot_df, x='epochs', y='accuracy', ax=ax,
             label='train accuracy', linewidth=3)
sns.lineplot(data=plot_df, x='epochs', y='val_loss', ax=ax, label='val loss',
             linewidth=3)
sns.lineplot(data=plot_df, x='epochs', y='val_accuracy', ax=ax,
             label='val_accuracy', linewidth=3)
ax.set_ylabel('Loss or Accuracy')
ax.set_xlabel('Epochs')
plt.setp(ax.get_legend().get_texts(), fontsize='18');  # for legend text
#%%
