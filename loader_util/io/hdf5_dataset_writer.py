# import the necessary packages
import h5py
import os


class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufSize=1000):
        # check to see if outpath exists, if so raise exception
        if os.path.exists(outputPath):
            raise ValueError("The supplied outputPath already exists and "
                             "cannot be overwritten. Manually delete the "
                             "file before continuing.", outputPath)

        # open the HDF5 database for writing and create 2 datasets: one to
        # sstore the images/features and the other to store the class labels
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, dtype='float')
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype='int')

        # store the buffer size then initialise the buffer itself along with
        # the index into the datasets
        self.bufSize = bufSize
        self.buffer = {'data': [], 'labels': []}
        self.idx = 0

    # method to add data to BUFFER
    def add(self, rows, labels):
        # add the rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # check to see if buffer needs to be flashed to disk
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()
        # method writes the buffers to disk then resets buffer

    def flush(self):
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer['data']
        self.labels[self.idx:i] = self.buffer['labels']
        self.idx = i
        self.buffer = {'data': [], 'labels': []}

    def storeClassLabels(self, classLabels):
        # create a dataset to store the actual class label names
        dt = h5py.string_dtype(encoding='ascii')

        labelSet = self.db.create_dataset("label_names",
                                          shape=(len(classLabels),),
                                          dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        # check to see if there are any entries in the buffer that need to
        # be flushed to disk
        if len(self.buffer['data']) > 0:
            self.flush()

        # close the dataset
        self.db.close()
