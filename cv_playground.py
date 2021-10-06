# %%
# import the necessary packages
import inspect
#%%
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from loader_util.preprocessing import ImageToArrayPreprocessor, \
    AspectAwarePreprocessor, SimplePreProcessor, MeanPreprocessor
from loader_util.io import HDF5DatasetGenerator
from loader_util.datasets import SimpleDatasetLoader
from loader_util.nn.conv import FCHeadNet
from loader_util.callbacks import EpochCheckpoint, TrainingMonitor
##
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense, \
    BatchNormalization, MaxPooling2D, AveragePooling2D, Dropout
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model

# %%

# ground truth
true_weights = tf.constant(list(range(5)), dtype=tf.float32)
true_weights = tf.reshape(true_weights, shape=(true_weights.shape[0], -1))
# %%

# some random trianing data
x = tf.constant(tf.random.uniform((32, 5)), dtype=tf.float32)
y = tf.constant(x @ true_weights, dtype=tf.float32)
# %%

# model parameters
weights = tf.Variable(tf.random.uniform((5, 1)), dtype=tf.float32)
# %%
for it in range(1001):
    with tf.GradientTape() as tape:
        y_hat = tf.linalg.matmul(x, weights)
        loss = tf.reduce_mean(tf.square(y - y_hat))

    if not it % 100:
        print(f"[INFO] mean squared loss at iteration: {it} is {loss}......")

    gradients = tape.gradient(loss, weights)
    weights.assign_add(-0.05 * gradients)


# %%


def f(a, b, power=2, d=3):
    return tf.pow(a, power) + d * b


converted_f = tf.autograph.to_graph(f)
print(inspect.getsource(converted_f))
#%%
