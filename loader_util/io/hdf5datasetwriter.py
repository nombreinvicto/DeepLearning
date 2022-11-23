# import the required packages
import h5py
import os


# %% ##################################################################
class HDF5DatasetWriter:
    def __init__(self,
                 dims,
                 output_path,
                 data_key="data",
                 buf_size=1000):
        # check to see if the output path exists
        if os.path.exists(output_path):
            raise ValueError(f"the supplied path {output_path}"
                             f"already exists and cannot be"
                             f"overwritten. Delete the file"
                             f"and then continue")
        # open the HDF5 "database" for writing and create
        # two datasets one for image features and the other
        # for the labels
        self.db = h5py.File(output_path, mode="w")
        self.data = self.db.create_dataset(data_key,
                                           shape=dims,
                                           dtype="float")
        self.labels = self.db.create_dataset("labels",
                                             shape=(dims[0],),
                                             dtype="int")
        self.label_names = None

        # store the buffer size
        self.buf_size = buf_size
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        # add rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # check to see if buffer needs to be flushed to disk
        if len(self.buffer["data"]) > self.buf_size:
            self.flush()

    def flush(self):
        # write buffers to disk
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def store_class_labels(self, class_labels):
        # create a dataset to store actual class label names
        self.label_names = self.db.create_dataset("label_names",
                                                  shape=len(class_labels),
                                                  dtype=h5py.string_dtype())
        self.label_names[:] = class_labels

    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()
        self.db.close()
