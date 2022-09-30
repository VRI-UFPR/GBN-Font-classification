from gc import callbacks
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input, GlobalAveragePooling2D, Dense, Dropout, Flatten
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import backend as K
from collections import Counter
from PIL import ImageFile
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import os.path
import math
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TRAINING_DATA_DIR = 'divide_dataset_aft/train/' 
VALIDATION_DATA_DIR = 'divide_dataset_aft/val/'
TEST_DATA_DIR = 'divide_dataset_aft/test/'

TRAIN_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(TRAINING_DATA_DIR)])
VALIDATION_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(VALIDATION_DATA_DIR)])
TEST_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(TEST_DATA_DIR)])

imgs = [
"divide_dataset/test/Script/Der-Pionier-1887-08-06_P03_r1_line0008_word0000.png",
"divide_dataset/test/Script/Der-Pionier-1889-08-21_P02_r1_line0012_word0000.png",
"divide_dataset/test/Script/Der-Pionier-1889-08-21_P02_r1_line0012_word0001.png",
"divide_dataset/test/Script/Kolonie-Zeitung-1888-09-25_P01_r1_line0005_word0001.png"]

# fonts = ['Antiqua', 'Fraktur', 'Italic', 'Kanzlei', 'Script', 'Textura']

NUM_CLASSES = len(next(os.walk(TRAINING_DATA_DIR))[1])


IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32

def f1(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

ImageFile.LOAD_TRUNCATED_IMAGES = True

train_datagen = ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   preprocessing_function=preprocess_input,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                #    zoom_range=1.0,
                                   #channel_shift_range = 1.0,
                                   horizontal_flip = True,
                                   vertical_flip = True)


val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = train_datagen.flow_from_directory(TRAINING_DATA_DIR,
                                                    target_size=(IMG_WIDTH,
                                                                 IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True, # Training pictures are shuffled to introduce more randomness during the training process
                                                    seed=42,
                                                    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(VALIDATION_DATA_DIR,
                                                       target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=False,
                                                       class_mode='categorical')

labels = (train_generator.class_indices)


# Save the dictionnary to file for future use
pickle.dump(labels, open('./labels.pickle', 'wb'))

print(labels)

# callback_01 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0)
# callback_02 = tf.keras.callbacks.TensorBoard(log_dir='logs',
#                                  histogram_freq=0, 
#                                  write_graph=True, 
#                                  write_images=True,    
#                                  update_freq='epoch', 
#                                  profile_batch=2, 
#                                  embeddings_freq=0,    
#                                  embeddings_metadata=None)

# #Import the pretrained model without the top classification layer
# base_model = ResNet50(include_top=False,
#                       input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# # for layer in base_model.layers[:-8]:
# #     layer.trainable = False

# custom_model = GlobalAveragePooling2D()(base_model.output)
# custom_model = Dense(512, activation='relu')(custom_model)
# custom_model = Dropout(0.5)(custom_model)

# # Final layer : the number of neurons is equal to the number of classes we want to predict
# # Since we have more than two classes, we choose 'softmax' as the activation function.
# custom_model = Dense(NUM_CLASSES, activation = 'softmax')(custom_model)

# model = Model(inputs=base_model.input, outputs=custom_model)

# model.compile(loss='kullback_leibler_divergence',
#               optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#               metrics=['AUC'])

# model.fit(train_generator,
#                     steps_per_epoch = math.ceil(float(TRAIN_SAMPLES) / BATCH_SIZE),
#                     validation_data = validation_generator,
#                     validation_steps = math.ceil(float(VALIDATION_SAMPLES) / BATCH_SIZE),
#                     callbacks = [callback_01, callback_02],
#                     epochs=256)



# # Save the model to Keras HDF5 format
# model.save('./model-finetuned_id_aft_aug_7.h5', save_format='h5')
           
# # Save the model to TensorFlow format (for deployment to Google AI Platform)
# model.save('TFSavedModel_id_aft_aug', save_format='tf')

model = load_model('./model-finetuned_id_aft_aug_7.h5')

# for img in imgs:
#         img = image.load_img(img, target_size=(IMG_HEIGHT, IMG_WIDTH))
#         img_array = image.img_to_array(img)
#         img_batch = np.expand_dims(img_array, axis=0)
#         img_preprocessed = preprocess_input(img_batch)
#         print(img)

#         prediction = model.predict(img_preprocessed)

#         print(prediction)


score = model.evaluate_generator(validation_generator, math.ceil(float(VALIDATION_SAMPLES) / BATCH_SIZE))
print("Validation accuracy : {:.2f} %".format(score[1]*100))

test_generator = val_datagen.flow_from_directory(TEST_DATA_DIR,
                                                 target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=False,
                                                 class_mode='categorical')

score = model.evaluate_generator(test_generator, math.ceil(float(TEST_SAMPLES) / BATCH_SIZE))
print("Test accuracy : {:.2f} %".format(score[1]*100))

predictions = model.predict_generator(test_generator, math.ceil(float(TEST_SAMPLES) / BATCH_SIZE))

# Get the list of top predictions (index of class with highest probability) for all images
top_index = np.argmax(predictions, axis=1)

# Compute the confusion matrix: the horizontal axis shows the predicted classes, the vertical axis shows the actual class
matrix = confusion_matrix(test_generator.classes, top_index)

print('Classification Report')
print(classification_report(test_generator.classes, top_index, target_names=labels))
