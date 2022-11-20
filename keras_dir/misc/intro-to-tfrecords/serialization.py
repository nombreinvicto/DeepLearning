# USAGE
# python serialization.py

# import the necessary packages
import tensorflow as tf

# build the original data
originalData = tf.constant(
    value=[1, 2, 3, 4],
    dtype=tf.dtypes.uint8
)

# serialize the data into binary records
serializedData = tf.io.serialize_tensor(originalData)

# read the serialized data into the original format
parsedData = tf.io.parse_tensor(
    serializedData,
    out_type=tf.dtypes.uint8
)

# print the original, encoded, and the decoded data
print(f"Original Data: {originalData}\n")  # Original Data: [1 2 3 4]
print(f"Encoded Data: {serializedData}\n")  # Encoded Data: b'\x08\x04\x12\x04\x12\x02\x08\x04"\x04\x01\x02\x03\x04'
print(f"Decoded Data: {parsedData}\n")  # Decoded Data: [1 2 3 4]
