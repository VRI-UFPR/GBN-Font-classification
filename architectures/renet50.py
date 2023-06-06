from gc import callbacks
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout,GlobalAveragePooling2D
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


@dataclass
class resnet_architecture():
    training_data_dir: str
    validation_data_dir: str
    test_data_dir: str
    num_classes:int
    img_width: int
    img_height: int
    batch_size: int
    train_sample: int
    validation_sample: int
    test_sample: int

    def train_network(self, model_name):
    # Inicia as imagens que serao usadas pela rede
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        train_datagen = ImageDataGenerator(
                                        preprocessing_function=preprocess_input)


        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


        train_generator = train_datagen.flow_from_directory(self.training_data_dir,
                                                            target_size=(self.img_width,
                                                                        self.img_height),
                                                            batch_size=self.batch_size,
                                                            shuffle=True, # Training pictures are shuffled to introduce more randomness during the training process
                                                            seed=42,
                                                            class_mode='categorical')

        validation_generator = val_datagen.flow_from_directory(self.validation_data_dir,
                                                            target_size=(self.img_width, self.img_height),
                                                            batch_size=self.batch_size,
                                                            shuffle=False,
                                                            class_mode='categorical')                                                  

        labels = (train_generator.class_indices)

        pickle.dump(labels, open('./logs/labels.pickle', 'wb'))

        ## Monta o modelo que sera usado para o treinamento
        # Monta callback 
        callback_01 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0)

        #Import the pretrained model without the top classification layer
        base_model = ResNet50(include_top=False,
                            input_shape=(self.img_width, self.img_height, 3))

        for layer in base_model.layers[:-8]:
            layer.trainable = False

        custom_model = GlobalAveragePooling2D()(base_model.output)
        vgg_x = Dense(1024, activation = 'relu', kernel_initializer='he_uniform')(custom_model)
        vgg_x = Dropout(0.5)(vgg_x)

        # Final layer : the number of neurons is equal to the number of classes we want to predict
        # Since we have more than two classes, we choose 'softmax' as the activation function.
        custom_model = Dense(self.num_classes, activation = 'softmax')(custom_model)

        model = Model(inputs=base_model.input, outputs=custom_model)

        model.compile(loss='kullback_leibler_divergence',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    metrics=[
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),  
      tf.keras.metrics.AUC(name='auc')])

            # treina o modelols 
        model.fit_generator(train_generator,
                        steps_per_epoch = math.ceil(float(self.train_sample) / self.batch_size),
                        validation_data = validation_generator,
                        validation_steps = math.ceil(float(self.validation_sample) / self.batch_size),
                        callbacks = [callback_01],
                        epochs=256)



        # Save the model to Keras HDF5 format
        model.save(f'./models/resnet50/{model_name}.h5', save_format='h5')

        model = load_model(f'./models/resnet50/{model_name}.h5')

        test_generator = val_datagen.flow_from_directory(self.test_data_dir,
                                                target_size=(self.img_width, self.img_height),
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                class_mode='categorical')   

        # avalia o modelo
        score = model.evaluate_generator(test_generator, math.ceil(float(self.test_sample) / self.batch_size))
        print("Test accuracy : {:.2f} %".format(score[1]*100))

        predictions = model.predict_generator(test_generator, math.ceil(float(self.test_sample) / self.batch_size))

        top_index = np.argmax(predictions, axis=1)

        # Calcula a matriz de confusao
        matrix = confusion_matrix(test_generator.classes, top_index)

        print('Classification Report')
        print(classification_report(test_generator.classes, top_index, target_names=labels))
