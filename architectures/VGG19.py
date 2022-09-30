from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from sklearn.metrics import classification_report, confusion_matrix
from PIL import ImageFile
import numpy as np
import math
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

   def generate_images(self, model_name):
      # Inicia as imagens que serao usadas pela rede
      ImageFile.LOAD_TRUNCATED_IMAGES = True

      train_datagen = ImageDataGenerator(featurewise_center=True,
                                       featurewise_std_normalization=True,
                                       preprocessing_function=preprocess_input,
                                       rotation_range=30,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       horizontal_flip = True,
                                       vertical_flip = True)


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
      base_model = VGG19(include_top=False,
                        pooling='avg',
                        weights='imagenet',
                           input_shape=(self.img_width, self.img_height, 3))

      # desabilita o treinamento para hidden layer (ja foram treinados com a imagenet)
      for layers in base_model.layers:
                  layers.trainable=False

      last_output = base_model.layers[-1].output

      # Modifica a ultima camada da rede (para adequacao aos dados que estao sendo utilizados)
      vgg_x = Flatten()(last_output)
      vgg_x = Dense(128, activation = 'relu')(vgg_x)
      vgg_x = Dense(self.num_classes, activation = 'softmax')(vgg_x)
      model = Model(base_model.input, vgg_x)
      model.compile(loss = 'kullback_leibler_divergence', optimizer= 'adam', metrics=['AUC'])

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
