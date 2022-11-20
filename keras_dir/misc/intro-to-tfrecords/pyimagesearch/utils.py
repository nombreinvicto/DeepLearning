# import the necessary packages
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def load_image(pathToImage):
    # read the image from the path and decode the image
    image = tf.io.read_file(pathToImage)  # reads image from disk
    image = tf.image.decode_image(image, channels=3)  # converts tflow tensor of dtpye uint8

    # convert the image data type and resize it
    image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    image = tf.image.resize(image, (16, 16))

    # return the processed image
    return image


def save_image(image, saveImagePath, title=None):
    # show the image
    plt.imshow(image)

    # check if title is provided, if so, add the title to the plot
    if title:
        plt.title(title)

    # turn off the axis and save the plot to disk
    plt.axis("off")
    plt.savefig(saveImagePath)
