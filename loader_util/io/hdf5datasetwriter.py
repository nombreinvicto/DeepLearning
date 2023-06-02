# import the necessary packages
import h5py
import os


class HDF5DatasetWriter:
    def __init__(self,
                 dims,  # tensor dimension of the data to be stored
                 outpath,
                 datakey="features",  # name of the dataset
                 bufsize=1000):
        # check to see if the outpath exists
        if os.path.exists(outpath):
            os.remove(outpath)

        # create the "database" which encapsulates the datasets
        self.db = h5py.File(outpath, mode="w")
        self.features = self.db.create_dataset(name=datakey,
                                               shape=dims,
                                               dtype="float")

        # labels saved as integers i.e ordinal nos e.g
        # output of LabelEncoder
        self.labels = self.db.create_dataset(name="labels",
                                             shape=(dims[0],),
                                             dtype="int")
        self.bufsize = bufsize
        self.buffer = {"features": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        # add the rows(features) and labels to buffer
        self.buffer["features"].extend(rows)
        self.buffer["labels"].extend(labels)

        if len(self.buffer["features"]) >= self.bufsize:
            self._flush()

    def _flush(self):
        # write the buffer to disk then reset the buffer
        i = self.idx + len(self.buffer["features"])
        self.features[self.idx:i] = self.buffer["features"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"features": [], "labels": []}

    def store_string_feature_labels(self, class_labels):
        dt = h5py.string_dtype(encoding='utf-8')
        label_db = self.db.create_dataset("label_names",
                                          shape=len(class_labels),
                                          dtype=dt)
        label_db[:] = class_labels

    def close(self):
        if len(self.buffer["features"]) > 0:
            self._flush()
        self.db.close()
