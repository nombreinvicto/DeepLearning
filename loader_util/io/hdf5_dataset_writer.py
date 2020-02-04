# import the necessary packages
import h5py
import os


class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufSize=1000):
        # check to see if outpath exists, if so raise exception
        if os.path.exists(outputPath):
            raise ValueError("The supplied outputPath already exists and "
                             "cannot be overwritten. Manually delete the "
                             "file before continuing.")

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
