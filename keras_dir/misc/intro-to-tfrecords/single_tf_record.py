# USAGE
# python single_tf_record.py

# import the necessary packages
from pyimagesearch import config
import tensorflow as tf

# build a byte-string that will be our binary record
record = "12345"
binaryRecord = record.encode()  # encode the string into a bytestring. binary records = byte strings

# print the original data and the encoded data
print(f"Original data: {record}")  # 12345
print(f"Encoded data: {binaryRecord}")  # b'12345'

# use the with context to initialize the record writer
with tf.io.TFRecordWriter(config.TFRECORD_SINGLE_FNAME) as recordWriter:
    # write the binary record into the TFRecord
    recordWriter.write(binaryRecord)

# open the TFRecord file
with open(config.TFRECORD_SINGLE_FNAME, "rb") as filePointer:
    # print the binary record from the TFRecord
    # this is tthe serialised bbinary rrecord
    # b'\x05\x00\x00\x00\x00\x00\x00\x00\xea\xb2\x04>12345z\x1c\xed\xe8'
    print(f"Data from the TFRecord: {filePointer.read()}")

# build a dataset from the TFRecord and iterate over it to read
# the data in the decoded format
dataset = tf.data.TFRecordDataset(config.TFRECORD_SINGLE_FNAME)
for element in dataset:
    # fetch the string from the binary record and then decode it
    element = element.numpy().decode()

    # print the decoded data
    print(f"Decoded data: {element}")
