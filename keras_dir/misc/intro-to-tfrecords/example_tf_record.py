# USAGE
# python example_tf_record.py

# import the necessary packages
from pyimagesearch import config
from pyimagesearch import utils
from tensorflow.keras.utils import get_file
import tensorflow as tf
import os

# a single instance of structured data will consist of an image and its
# corresponding class name
# Downloads a file from a URL if it not already in the cache -> returns path to downloaded file
imagePath = get_file(
    config.IMAGE_FNAME,  # the name to save
    config.IMAGE_URL,  # the url to download from
)
image = utils.load_image(pathToImage=imagePath)  # returns the 16x16 float32 image tensor
class_name = config.IMAGE_CLASS

# check to see if the output folder exists, if not, build the output
# folder
if not os.path.exists(config.OUTPUT_PATH):
    os.makedirs(config.OUTPUT_PATH)

# save the resized image
utils.save_image(image=image, saveImagePath=config.RESIZED_IMAGE_PATH)

# build the image and the class name feature
# LIST -> FEATURE -> FEATURES -> EXAMPLE
imageFeature = tf.train.Feature(
    bytes_list=tf.train.BytesList(value=[
        # notice how we serialize the image
        tf.io.serialize_tensor(image).numpy(),
    ])
)
classNameFeature = tf.train.Feature(
    bytes_list=tf.train.BytesList(value=[
        class_name.encode(),
    ])
)

# wrap the image feature and class feature with a features dictionary and then
# wrap the features into an example
features = tf.train.Features(feature={
    "image": imageFeature,
    "class_name": classNameFeature,
})
example = tf.train.Example(features=features)

# serialize the entire example
serializedExample = example.SerializeToString()

# write the serialized example into a TFRecord
with tf.io.TFRecordWriter(config.TFRECORD_EXAMPLE_FNAME) as recordWriter:
    recordWriter.write(serializedExample)

# build the feature schema and the TFRecord dataset
# This schematic will be used to parse each example
featureSchema = {
    "image": tf.io.FixedLenFeature([], dtype=tf.string),
    "class_name": tf.io.FixedLenFeature([], dtype=tf.string),
}
dataset = tf.data.TFRecordDataset(config.TFRECORD_EXAMPLE_FNAME)

# iterate over the dataset
for element in dataset:
    # get the serialized example and parse it with the feature schema
    element = tf.io.parse_example(element, featureSchema)

    # grab the serialized class name and the image
    className = element["class_name"].numpy().decode()
    image = tf.io.parse_tensor(
        element["image"].numpy(),
        out_type=tf.dtypes.float32
    )

    # save the de-serialized image along with the class name
    utils.save_image(
        image=image,
        saveImagePath=config.DESERIALIZED_IMAGE_PATH,
        title=className
    )
