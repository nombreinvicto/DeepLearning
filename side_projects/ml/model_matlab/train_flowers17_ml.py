import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.axes._axes as axes

sns.set()
# %%
# import scoring and utility functions from sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, ShuffleSplit
from sklearn.metrics import f1_score, make_scorer, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, LabelBinarizer, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn import manifold
#
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# import the necessary keras packages
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, ShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import  LogisticRegression
from loader_util.preprocessing import ImageToArrayPreprocessor, \
    AspectAwarePreprocessor
from loader_util.datasets import SimpleDatasetLoader
from loader_util.nn.conv import FCHeadNet, MinVGGNet
##
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from imutils import paths

# %%

data_dir = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\flowers17\images"

args = {
    "dataset": data_dir
}

image_paths = list(paths.list_images(data_dir))
class_names = [pt.split(os.path.sep)[-2] for pt in image_paths]
unique_class_names = np.unique(class_names)
# %%

aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
data, labels = sdl.load(image_paths)
data = data.astype('float') / 255.0 # type: np.ndarray
#%%

# convert the 64x64x64 images to vectors
data = data.reshape((data.shape[0], -1))

# %%

trainx, testx, trainy, testy = train_test_split(data, labels,
                                                test_size=0.25,
                                                random_state=42,
                                                stratify=labels)

# encode labels
lb = LabelEncoder()
trainy = lb.fit_transform(trainy)
testy = lb.transform(testy)
# %%


# conf matrix utility function
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap='viridis')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#%%

def grid_search(clf, parameters, scorer, train_data, test_data, cv=5):

    #Perform grid search on the classifier using 'scorer' as the scoring method.
    grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

    # Fit the grid search object to the training data and find the optimal parameters.
    grid_fit = grid_obj.fit(train_data[0], train_data[1])

    # Get the estimator.
    best_clf = grid_fit.best_estimator_

    # Fit the new model.
    best_clf.fit(train_data[0], train_data[1])

    # Make predictions using the new model.
    best_train_predictions = best_clf.predict(train_data[0])
    best_test_predictions = best_clf.predict(test_data[0])

    # Calculate the acc score of the new model.
    print('The training acc Score is', accuracy_score(best_train_predictions, train_data[1]))
    print('The testing acc Score is', accuracy_score(best_test_predictions, test_data[1]))
    print('The testing acc Score STD', np.mean(grid_obj.cv_results_['std_test_score']))


    # Let's also explore what parameters ended up being used in the new model.
    return grid_obj, best_clf



#%%

def logistic_cv(train_data, test_data, class_names, cv=10):

    # stringify class names
    class_names = list(map(str, class_names))

    # init model and params
    parameters = {'C':[1.0, 10, 100, 1000]}
    scorer = make_scorer(accuracy_score)
    logreg = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10_000)

    # grisearch
    grid_obj, best_clf = grid_search(logreg, parameters, scorer, train_data, test_data, cv)

    # confusion matrix
    preds = best_clf.predict(test_data[0])
    print(classification_report(test_data[1], preds,target_names=class_names))
    cm = confusion_matrix(test_data[1], preds)
    print_confusion_matrix(cm, class_names=class_names)

    return best_clf


#%%
best_clf = logistic_cv((trainx, trainy),
                       (testx, testy),
                       unique_class_names,
                       10)

#%%
model_dir = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\PyCharm Projects\MMTDataAnalysis\p3_matlab_python_bridge"

#%%
import pickle

filename='LR_model.pikl'
#pickle.dump(best_clf, open(f"{model_dir}//{filename}", 'wb'))
#%%

loaded_model = pickle.load(open(f"{model_dir}//{filename}", "rb")) # type: LogisticRegression
#%%

sample_image = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\flowers17\images\buttercup\image_1124.jpg"

#%%
from cv2 import cv2
image = cv2.imread(sample_image)
image = cv2.resize(image, dsize=(64, 64))
image = image.astype('float') / 255.0
image = np.ndarray.flatten(image)
image = np.expand_dims(image, axis=0)
print(image.shape)
#%%

pred = loaded_model.predict(image)
pred
#%%


