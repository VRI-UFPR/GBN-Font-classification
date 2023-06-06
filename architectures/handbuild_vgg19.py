from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout,AveragePooling2D, Convolution2D, ZeroPadding2D
from sklearn.metrics import classification_report, confusion_matrix
from PIL import ImageFile
import numpy as np
import math
import pickle
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@dataclass
class vgg19_architecture():
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

        train_datagen = ImageDataGenerator( preprocessing_function=preprocess_input)


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
        
        # Monta arquitetura da rede
        inputs = keras.Input(shape=(self.img_height, self.img_width, 3))
        x = layers.ZeroPadding2D((3, 3),
                        name='block1_zeroPd')(inputs)
        x = layers.Conv2D(64, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block1_conv1')(x)
        x = layers.BatchNormalization(axis=3, name='block1_batchN1')(x)
        x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
        x = layers.BatchNormalization(axis=3, name='block1_batchN2')(x)

        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = layers.Conv2D(128, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block2_conv1')(x)
        x = layers.BatchNormalization(axis=3, name='block2_batchN1')(x)
        x = layers.Conv2D(128, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block2_conv2')(x)
        x = layers.BatchNormalization(axis=3, name='block2_batchN2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = layers.Conv2D(256, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block3_conv1')(x)
        x = layers.BatchNormalization(axis=3, name='block3_batchN1')(x)
        x = layers.Conv2D(256, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block3_conv2')(x)
        x = layers.BatchNormalization(axis=3, name='block3_batchN2')(x)
        x = layers.Conv2D(256, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block3_conv3')(x)
        x = layers.BatchNormalization(axis=3, name='block3_batchN3')(x)
        x = layers.Conv2D(256, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block3_conv4')(x)
        x = layers.BatchNormalization(axis=3, name='block3_batchN4')(x)

        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block4_conv1')(x)
        x = layers.BatchNormalization(axis=3, name='block4_batchN1')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block4_conv2')(x)
        x = layers.BatchNormalization(axis=3, name='block4_batchN2')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block4_conv3')(x)
        x = layers.BatchNormalization(axis=3, name='block4_batchN3')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block4_conv4')(x)
        x = layers.BatchNormalization(axis=3, name='block4_batchN4')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block5_conv1')(x)
        x = layers.BatchNormalization(axis=3, name='block5_batchN1')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block5_conv2')(x)
        x = layers.BatchNormalization(axis=3, name='block5_batchN2')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block5_conv3')(x)
        x = layers.BatchNormalization(axis=3, name='block5_batchN3')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block5_conv4')(x)
        x = layers.BatchNormalization(axis=3, name='block5_batchN4')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        
        # Block 6
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block6_conv1')(x)
        x = layers.BatchNormalization(axis=3, name='block6_batchN1')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block6_conv2')(x)
        x = layers.BatchNormalization(axis=3, name='block6_batchN2')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block6_conv3')(x)
        x = layers.BatchNormalization(axis=3, name='block6_batchN3')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block6_conv4')(x)
        x = layers.BatchNormalization(axis=3, name='block6_batchN4')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(x)

        # Block 7
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block7_conv1')(x)
        x = layers.BatchNormalization(axis=3, name='block7_batchN1')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block7_conv2')(x)
        x = layers.BatchNormalization(axis=3, name='block7_batchN2')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block7_conv3')(x)
        x = layers.BatchNormalization(axis=3, name='block7_batchN3')(x)
        x = layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block7_conv4')(x)
        x = layers.BatchNormalization(axis=3, name='block7_batchN4')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block7_pool')(x)
        
        model = Model(inputs=inputs, outputs=x)

        # Modifica a ultima camada da rede (para adequacao aos dados que estao sendo utilizados)
        vgg_x = Flatten()(x)
        vgg_x = BatchNormalization()(vgg_x)
        vgg_x = Dense(512, activation = 'relu', kernel_initializer='he_uniform')(vgg_x)
        vgg_x = Dropout(0.3)(vgg_x)
        vgg_x = BatchNormalization()(vgg_x)
        vgg_x = Dense(512, activation = 'relu', kernel_initializer='he_uniform')(vgg_x)
        vgg_x = Dropout(0.3)(vgg_x)
        vgg_x = BatchNormalization()(vgg_x)
        vgg_x = Dense(512, activation = 'relu', kernel_initializer='he_uniform')(vgg_x)
        vgg_x = Dropout(0.3)(vgg_x)
        vgg_x = Dense(self.num_classes, activation = 'softmax')(vgg_x)
        
        model = Model(inputs=model.input, outputs=vgg_x)
        

        # loss and metrics
        model.compile(loss = tf.keras.losses.KLDivergence(), optimizer= 'adam', metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),  
        tf.keras.metrics.AUC(name='auc')])

        # treina o modelo
        model.fit(train_generator,
                            steps_per_epoch = math.ceil(float(self.train_sample) / self.batch_size),
                            validation_data = validation_generator,
                            validation_steps = math.ceil(float(self.validation_sample) / self.batch_size),
                            callbacks = [callback_01],
                            epochs=256)


        # Salva o modelo treinado
        model.save(f'./models/vgg/{model_name}.h5', save_format='h5')
        model = load_model(f'./models/vgg/{model_name}.h5')

        # model = load_model(f'./models/vgg/model_id_21.h5')

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
        print("Matriz:")
        print(matrix)

        print('Classification Report')
        print(classification_report(test_generator.classes, top_index, target_names=labels))
