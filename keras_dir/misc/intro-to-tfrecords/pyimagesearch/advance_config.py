# import the necessary packages
import os

# name of the TFDS dataset we will be using
DATASET = "div2k/bicubic_x4"

# define the shard size and batch size
SHARD_SIZE = 256
TRAIN_BATCH_SIZE = 64
INFER_BATCH_SIZE = 8

# dataset specs
HR_SHAPE = [96, 96, 3]
LR_SHAPE = [24, 24, 3]
SCALING_FACTOR = 4

# define the path to the dataset
BASE_DATA_PATH = "dataset"
DIV2K_PATH = os.path.join(BASE_DATA_PATH, "div2k")

# define the path to the tfrecords for GPU training
GPU_BASE_TFR_PATH = "tfrecord"
GPU_DIV2K_TFR_TRAIN_PATH = os.path.join(GPU_BASE_TFR_PATH, "train")
GPU_DIV2K_TFR_TEST_PATH = os.path.join(GPU_BASE_TFR_PATH, "test")

# define the path to the tfrecords for TPU training
TPU_BASE_TFR_PATH = "gs://<PATH_TO_GCS_BUCKET>/tfrecord"
TPU_DIV2K_TFR_TRAIN_PATH = os.path.join(TPU_BASE_TFR_PATH, "train")
TPU_DIV2K_TFR_TEST_PATH = os.path.join(TPU_BASE_TFR_PATH, "test")