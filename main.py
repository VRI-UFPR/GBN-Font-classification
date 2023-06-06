import os
from architectures import VGG19, renet50, handbuild_vgg19
from enums.dataset_enum import DatasetFITA, DatasetFITAAug, DatasetOther, Hyperparameters, DatasetAFT, DatasetOriginal, DatasetOtherAug, DatasetOriginalAug
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():

    vgg = handbuild_vgg19.vgg19_architecture(batch_size=Hyperparameters.BATCH_SIZE.value, 
    img_height=Hyperparameters.IMG_HEIGHT.value,
    img_width=Hyperparameters.IMG_WIDTH.value,
    num_classes=DatasetOriginalAug.NUM_ORIGINAL_CLASSES.value,
    test_data_dir=DatasetOriginalAug.TEST_ORIGINAL_DATA_DIR.value,
    test_sample=DatasetOriginalAug.ORIGINAL_TEST_SAMPLES.value,
    train_sample=DatasetOriginalAug.ORIGINAL_TRAIN_SAMPLES.value,
    training_data_dir=DatasetOriginalAug.TRAINING_ORIGINAL_DATA_DIR.value,
    validation_data_dir=DatasetOriginalAug.VALIDATION_ORIGINAL_DATA_DIR.value,
    validation_sample=DatasetOriginalAug.ORIGINAL_VALIDATION_SAMPLES.value)
    
    # vgg = renet50.resnet_architecture(batch_size=Hyperparameters.BATCH_SIZE.value, 
    # img_height=Hyperparameters.IMG_HEIGHT.value,
    # img_width=Hyperparameters.IMG_WIDTH.value,
    # num_classes=DatasetOtherAug.NUM_OTHER_CLASSES.value,
    # test_data_dir=DatasetOtherAug.TEST_OTHER_DATA_DIR.value,
    # test_sample=DatasetOtherAug.OTHER_TEST_SAMPLES.value,
    # train_sample=DatasetOtherAug.OTHER_RAIN_SAMPLES.value,
    # training_data_dir=DatasetOtherAug.TRAINING_OTHER_DATA_DIR.value,
    # validation_data_dir=DatasetOtherAug.VALIDATION_OTHER_DATA_DIR.value,
    # validation_sample=DatasetOtherAug.OTHER_VALIDATION_SAMPLES.value)

    # vgg = handbuild_vgg19.vgg19_architecture(batch_size=Hyperparameters.BATCH_SIZE.value, 
    # img_height=Hyperparameters.IMG_HEIGHT.value,
    # img_width=Hyperparameters.IMG_WIDTH.value,
    # num_classes=DatasetFITAAug.NUM_FITA_CLASSES.value,
    # test_data_dir=DatasetFITAAug.TEST_FITA_DATA_DIR.value,
    # test_sample=DatasetFITAAug.FITA_TEST_SAMPLES.value,
    # train_sample=DatasetFITAAug.FITA_TRAIN_SAMPLES.value,
    # training_data_dir=DatasetFITAAug.TRAINING_FITA_DATA_DIR.value,
    # validation_data_dir=DatasetFITAAug.VALIDATION_FITA_DATA_DIR.value,
    # validation_sample=DatasetFITAAug.FITA_VALIDATION_SAMPLES.value)

    # vgg = renet50.resnet_architecture(batch_size=Hyperparameters.BATCH_SIZE.value, 
    # img_height=Hyperparameters.IMG_HEIGHT.value,
    # img_width=Hyperparameters.IMG_WIDTH.value,
    # num_classes=DatasetOtherAug.NUM_OTHER_CLASSES.value,
    # test_data_dir=DatasetOtherAug.TEST_OTHER_DATA_DIR.value,
    # test_sample=DatasetOtherAug.OTHER_TEST_SAMPLES.value,
    # train_sample=DatasetOtherAug.OTHER_TRAIN_SAMPLES.value,
    # training_data_dir=DatasetOtherAug.TRAINING_OTHER_DATA_DIR.value,
    # validation_data_dir=DatasetOtherAug.VALIDATION_OTHER_DATA_DIR.value,
    # validation_sample=DatasetOtherAug.OTHER_VALIDATION_SAMPLES.value)

    model_id = sum([len(filenames) for dirpath, dirnames, filenames in os.walk('./models/vgg')])

    print(model_id)

    model_name = "model_id_" + str(model_id + 1) 
    # model_name = "model_id_30"
    print("Modelo: " + model_name)
    

    # vgg.generate_images(model_name)
    vgg.train_network(model_name)

main()