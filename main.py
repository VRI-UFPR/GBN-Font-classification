import os
from architectures.VGG19 import vgg19_architecture
from enums.dataset_enum import DatasetOther, Hyperparameters, DatasetAFT

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    vgg = vgg19_architecture(batch_size=Hyperparameters.BATCH_SIZE.value, 
    img_height=Hyperparameters.IMG_HEIGHT.value,
    img_width=Hyperparameters.IMG_WIDTH.value,
    num_classes=DatasetAFT.NUM_AFT_CLASSES.value,
    test_data_dir=DatasetAFT.TEST_AFT_DATA_DIR.value,
    test_sample=DatasetAFT.AFT_TEST_SAMPLES.value,
    train_sample=DatasetAFT.AFT_TRAIN_SAMPLES.value,
    training_data_dir=DatasetAFT.TRAINING_AFT_DATA_DIR.value,
    validation_data_dir=DatasetAFT.VALIDATION_AFT_DATA_DIR.value,
    validation_sample=DatasetAFT.AFT_VALIDATION_SAMPLES.value)

    model_id = sum([len(filenames) for dirpath, dirnames, filenames in os.walk('./models/vgg')])

    print(model_id)

    model_name = "model_id_" + str(model_id + 1) 
    print("Modelo: " + model_name)

    vgg.generate_images(model_name)

main()