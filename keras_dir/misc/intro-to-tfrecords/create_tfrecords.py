# USAGE
# python create_tfrecords.py

# import the necessary packages
from pyimagesearch import advance_config as config
import tensorflow_datasets as tfds
import tensorflow as tf
import os

# define AUTOTUNE object
AUTO = tf.data.AUTOTUNE


def pre_process(element):
    # grab the low and high resolution images
    lrImage = element["lr"]
    hrImage = element["hr"]

    # convert the low and high resolution images from tensor to
    # serialized TensorProto proto
    lrByte = tf.io.serialize_tensor(lrImage)
    hrByte = tf.io.serialize_tensor(hrImage)

    # return the low and high resolution proto objects
    return (lrByte, hrByte)


def create_dataset(dataDir, split, shardSize):
    # load the dataset, save it to disk, and preprocess it
    ds = tfds.load(config.DATASET, split=split, data_dir=dataDir)
    ds = (ds
          .map(pre_process, num_parallel_calls=AUTO)
          .batch(shardSize)
          )

    # return the dataset
    return ds


def create_serialized_example(lrByte, hrByte):
    # create low and high resolution image byte list
    lrBytesList = tf.train.BytesList(value=[lrByte])
    hrBytesList = tf.train.BytesList(value=[hrByte])

    # build low and high resolution image feature from the byte list
    lrFeature = tf.train.Feature(bytes_list=lrBytesList)
    hrFeature = tf.train.Feature(bytes_list=hrBytesList)

    # build a low and high resolution image feature map
    featureMap = {
        "lr": lrFeature,
        "hr": hrFeature,
    }

    # build a collection of features, followed by building example
    # from features, and serializing the example
    features = tf.train.Features(feature=featureMap)
    example = tf.train.Example(features=features)
    serializedExample = example.SerializeToString()

    # return the serialized example
    return serializedExample


def prepare_tfrecords(dataset, outputDir, name, printEvery=50):
    # check whether output directory exists
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # loop over the dataset and create TFRecords
    # images -> (lrByte, hrByte)
    for (index, images) in enumerate(dataset):
        # get the shard size and build the filename
        shardSize = images[0].numpy().shape[0]
        tfrecName = f"{index:02d}-{shardSize}.tfrec"
        filename = outputDir + f"/{name}-" + tfrecName

        # write to the tfrecords
        with tf.io.TFRecordWriter(filename) as outFile:
            # write shard size serialized examples to each TFRecord
            for i in range(shardSize):
                serializedExample = create_serialized_example(
                    images[0].numpy()[i], images[1].numpy()[i])
                outFile.write(serializedExample)

            # print the progress to the user
            if index % printEvery == 0:
                print("[INFO] wrote file {} containing {} records..."
                      .format(filename, shardSize))


# create training and validation dataset of the div2k images
print("[INFO] creating div2k training and testing dataset...")
trainDs = create_dataset(dataDir=config.DIV2K_PATH, split="train",
                         shardSize=config.SHARD_SIZE)
testDs = create_dataset(dataDir=config.DIV2K_PATH, split="validation",
                        shardSize=config.SHARD_SIZE)

# create training and testing TFRecords and write them to disk
print("[INFO] preparing and writing div2k TFRecords to disk...")
prepare_tfrecords(dataset=trainDs, name="train",
                  outputDir=config.GPU_DIV2K_TFR_TRAIN_PATH)
prepare_tfrecords(dataset=testDs, name="test",
                  outputDir=config.GPU_DIV2K_TFR_TEST_PATH)
