from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse

# add path to current directory to sys modules
import os, sys
current_dir = os.getcwd()
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from loader_util.preprocessing import SimplePreProcessor
from loader_util.datasets import SimpleDatasetLoader
from imutils import paths

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",
                required=True, help="path to the input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=5,
                help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of jobs for k-NN distance")
args = vars(ap.parse_args())

# grab the list of images that we will be describing
print(f"[INFO] loading images.......")
imapePaths = list(paths.list_images(args["dataset"]))

# init the image preprocessor, load dataset from disk and reshape
sp = SimplePreProcessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
data, labels = sdl.load(imapePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the images
print(f"[INFO] features matrix: {data.nbytes / (1024.0 * 1000.0)}")

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

trainx, testx, trainy, testy = train_test_split(data, labels,
                                                test_size=0.25,
                                                random_state=42)

# train and evaluate a k-NN classifier on raw pixel intensities
print(f"[INFO] evaluating k-NN classifier......")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
                             n_jobs=args["jobs"])
model.fit(trainx, trainy)
print(classification_report(testy, model.predict(testx),
                            target_names=le.classes_))
