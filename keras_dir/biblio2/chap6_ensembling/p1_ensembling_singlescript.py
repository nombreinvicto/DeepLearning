# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.axes._axes as axes

sns.set()
# %%
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from loader_util.preprocessing import ImageToArrayPreprocessor, \
    AspectAwarePreprocessor
from loader_util.datasets import SimpleDatasetLoader
from loader_util.nn.conv import FCHeadNet, MinVGGNet
##
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from imutils import paths

# %%
# construct the argument parser

data_dir = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\biblio2chap6"
args_dict = {
    "output": f"{data_dir}//outputs",
    "models": f"{data_dir}//models",
    "num_models": 5
}
# %%

(trainx, trainy), (testx, testy) = cifar10.load_data()
trainx = trainx.astype("float32") / 255.0
testx = testx.astype("float32") / 255.0

# binarise the labels
le = LabelBinarizer()
trainy = le.fit_transform(trainy)
testy = le.transform(testy)

class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
               "horse", "ship", "truck"]
# %%
# construct the image generator
aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         fill_mode='nearest')
# %%
epoch_num = 40
batch_size = 64
# loop over the no of models to be trained
for model_no in range(args_dict["num_models"]):
    print(f"[INFO] Initialising model {model_no + 1}"
          f"/{args_dict['num_models']}")
    opt = SGD(lr=0.01, decay=0.01 / epoch_num, momentum=0.9, nesterov=True)
    model = MinVGGNet.build(32, 32, 3, classes=len(class_names))
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])
    # trian the network
    print(f"[INFO] Starting training of model: {model_no + 1}")
    H = model.fit_generator(aug.flow(trainx, trainy, batch_size=batch_size),
                            validation_data=(testx, testy),
                            epochs=epoch_num,
                            steps_per_epoch=len(trainx) // batch_size,
                            verbose=1)
    print(f"[INFO] Training complete......")

    # save the model
    model.save(filepath=f'{args_dict["models"]}//model{model_no}.pt')

    # evaluate network
    preds = model.predict(testx, batch_size=batch_size)
    report = classification_report(testy.argmax(axis=1),
                                   preds.argmax(axis=1),
                                   target_names=class_names)

    # save report
    f = open(f'{args_dict["output"]}//report{model_no}.txt', mode="w")
    f.write(report)
    f.close()


    # plot loss accuracy and then save
    # plot the performance
    epochs = range(1, epoch_num)
    loss = H.history['loss']
    accuracy = H.history['acc']
    val_loss = H.history['val_loss']
    val_accuracy = H.history['val_acc']
    plot_df = pd.DataFrame(
        data=np.c_[epochs, loss, accuracy, val_loss, val_accuracy],
        columns=['epochs', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])

    # do the actual plots
    sns.set(font_scale=1)
    f, ax = plt.subplots(1, 1, figsize=(15, 8))
    sns.lineplot(data=plot_df, x='epochs', y='loss', ax=ax, label='train loss',
                 linewidth=3)
    sns.lineplot(data=plot_df, x='epochs', y='accuracy', ax=ax,
                 label='train accuracy', linewidth=3)
    sns.lineplot(data=plot_df, x='epochs', y='val_loss', ax=ax,
                 label='val loss', linewidth=3)
    sns.lineplot(data=plot_df, x='epochs', y='val_accuracy', ax=ax,
                 label='val_accuracy', linewidth=3)
    ax.set_ylabel('Loss or Accuracy')
    ax.set_xlabel('Epochs')
    plt.setp(ax.get_legend().get_texts(), fontsize='18');  # for legend text
    plt.savefig(f'{args_dict["output"]}//model{model_no}_plot.png')
    plt.close()
# %%
