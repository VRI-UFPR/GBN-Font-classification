from dataclasses import dataclass
import tensorflow as tf
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout,AveragePooling2D, Convolution2D, ZeroPadding2D
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import math
import pickle
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

path = 'datasets/base_datasets/dataset_aft_sift'


def TANNSB_KF_v1():
    train_examples = []
    train_labels = []
    test_examples = []
    test_labels = []    


    with np.load('dataset.npz', allow_pickle=True) as data:
        train_examples = data['x']
        train_labels = data['y']
        
    
    
    train_labels = train_labels[:, np.newaxis]
    train_labels = train_labels[:, :, np.newaxis]
    
    # train_examples = np.column_stack(train_examples)
    # train_labels = np.repeat(train_labels, 19)
    
    # print(train_labels.shape)
    
    # for i in range(len(train_examples)):
    #     train_examples[i] = train_examples[i].astype(np.int32)
    
    train_data, val_data, train_labels, val_labels = train_test_split(train_examples, train_labels, test_size=0.2, random_state=42)
    
    print(train_labels.shape)
    
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    # test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
    
    # print(train_dataset)

    # train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE)
    # test_dataset = test_dataset.batch(BATCH_SIZE)
    
    # print(train_dataset)
    
    # create the input layer
    inputs = tf.keras.layers.Input(shape=(20, 128))

    # define the number of filters and kernel size for the convolutional layer
    num_filters = 64
    kernel_size = 3

    # add a convolutional layer to extract features from the input data
    conv_layer = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu')(inputs)

    # add a max pooling layer to reduce the spatial dimensions of the output of the convolutional layer
    pooling_layer = tf.keras.layers.MaxPooling1D(pool_size=2)(conv_layer)

    # flatten the output of the pooling layer to create a feature vector
    flatten_layer = tf.keras.layers.Flatten()(pooling_layer)

    # add a dense layer to learn the non-linear mapping from the feature vector to the output
    dense_layer = tf.keras.layers.Dense(units=128, activation='relu')(flatten_layer)

    # add an output layer to produce the final output
    output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')(dense_layer)

    # create the model with the input and output layers
    model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)

    # compile the model with a loss function and optimizer
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

    # print the summary of the model
    model.summary()
    
    history = model.fit(
        train_data,
        train_labels,
        epochs=10,
        validation_data=(val_data, val_labels),
        shuffle=True
    )


    

TANNSB_KF_v1()
    