from enum import Enum
import os

class Hyperparameters(Enum):
    IMG_WIDTH, IMG_HEIGHT = 224, 224
    BATCH_SIZE = 32

class DatasetOriginal(Enum):
    TRAINING_ORIGINAL_DATA_DIR = 'datasets/divided_datasets/divide_dataset/train' 
    VALIDATION_ORIGINAL_DATA_DIR = 'datasets/divided_datasets/divide_dataset/val/'
    TEST_ORIGINAL_DATA_DIR = 'datasets/divided_datasets/divide_dataset/test/'

    ORIGINAL_TRAIN_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(TRAINING_ORIGINAL_DATA_DIR)])
    ORIGINAL_VALIDATION_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(VALIDATION_ORIGINAL_DATA_DIR)])
    ORIGINAL_TEST_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(TEST_ORIGINAL_DATA_DIR)])

    NUM_ORIGINAL_CLASSES = len(next(os.walk(TEST_ORIGINAL_DATA_DIR))[1])

class DatasetOriginalAug(Enum):
    TRAINING_ORIGINAL_DATA_DIR = 'datasets/divided_datasets/divide_dataset_aug/train' 
    VALIDATION_ORIGINAL_DATA_DIR = 'datasets/divided_datasets/divide_dataset_aug/val/'
    TEST_ORIGINAL_DATA_DIR = 'datasets/divided_datasets/divide_dataset_aug/test/'

    ORIGINAL_TRAIN_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(TRAINING_ORIGINAL_DATA_DIR)])
    ORIGINAL_VALIDATION_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(VALIDATION_ORIGINAL_DATA_DIR)])
    ORIGINAL_TEST_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(TEST_ORIGINAL_DATA_DIR)])

    NUM_ORIGINAL_CLASSES = len(next(os.walk(TEST_ORIGINAL_DATA_DIR))[1])

class DatasetAFT(Enum):
    TRAINING_AFT_DATA_DIR = 'datasets/divided_datasets/divide_dataset_aft/train' 
    VALIDATION_AFT_DATA_DIR = 'datasets/divided_datasets/divide_dataset_aft/val/'
    TEST_AFT_DATA_DIR = 'datasets/divided_datasets/divide_dataset_aft/test/'

    AFT_TRAIN_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(TRAINING_AFT_DATA_DIR)])
    AFT_VALIDATION_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(VALIDATION_AFT_DATA_DIR)])
    AFT_TEST_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(TEST_AFT_DATA_DIR)])

    NUM_AFT_CLASSES = len(next(os.walk(TEST_AFT_DATA_DIR))[1])

class DatasetFITA(Enum):
    TRAINING_FITA_DATA_DIR = 'datasets/divided_datasets/divide_dataset_fita/train' 
    VALIDATION_FITA_DATA_DIR = 'datasets/divided_datasets/divide_dataset_fita/val/'
    TEST_FITA_DATA_DIR = 'datasets/divided_datasets/divide_dataset_fita/test/'

    FITA_TRAIN_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(TRAINING_FITA_DATA_DIR)])
    FITA_VALIDATION_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(VALIDATION_FITA_DATA_DIR)])
    FITA_TEST_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(TEST_FITA_DATA_DIR)])

    NUM_FITA_CLASSES = len(next(os.walk(TEST_FITA_DATA_DIR))[1])

class DatasetFITAAug(Enum):
    TRAINING_FITA_DATA_DIR = 'datasets/divided_datasets/divide_dataset_fita_aug/train' 
    VALIDATION_FITA_DATA_DIR = 'datasets/divided_datasets/divide_dataset_fita_aug/val/'
    TEST_FITA_DATA_DIR = 'datasets/divided_datasets/divide_dataset_fita_aug/test/'

    FITA_TRAIN_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(TRAINING_FITA_DATA_DIR)])
    FITA_VALIDATION_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(VALIDATION_FITA_DATA_DIR)])
    FITA_TEST_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(TEST_FITA_DATA_DIR)])

    NUM_FITA_CLASSES = len(next(os.walk(TEST_FITA_DATA_DIR))[1])

class DatasetOther(Enum):
    TRAINING_OTHER_DATA_DIR = 'datasets/divided_datasets/divide_dataset_other/train' 
    VALIDATION_OTHER_DATA_DIR = 'datasets/divided_datasets/divide_dataset_other/val/'
    TEST_OTHER_DATA_DIR = 'datasets/divided_datasets/divide_dataset_other/test/'

    OTHER_TRAIN_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(TRAINING_OTHER_DATA_DIR)])
    OTHER_VALIDATION_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(VALIDATION_OTHER_DATA_DIR)])
    OTHER_TEST_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(TEST_OTHER_DATA_DIR)])

    NUM_OTHER_CLASSES = len(next(os.walk(TEST_OTHER_DATA_DIR))[1])

class DatasetOtherAug(Enum):
    TRAINING_OTHER_DATA_DIR = 'datasets/divided_datasets/divided_dataset_other_aug/train' 
    VALIDATION_OTHER_DATA_DIR = 'datasets/divided_datasets/divided_dataset_other_aug/val/'
    TEST_OTHER_DATA_DIR = 'datasets/divided_datasets/divided_dataset_other_aug/test'

    OTHER_TRAIN_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(TRAINING_OTHER_DATA_DIR)])
    OTHER_VALIDATION_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(VALIDATION_OTHER_DATA_DIR)])
    OTHER_TEST_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(TEST_OTHER_DATA_DIR)])

    NUM_OTHER_CLASSES = len(next(os.walk(TEST_OTHER_DATA_DIR))[1])


