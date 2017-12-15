from __future__ import division, absolute_import

from os.path import isfile
from Constants import *

from tensorflow import keras as k
import tensorflow as tf
from PIL import Image
import h5py

class EmotionRecognition:
  model = 0

  def __init__(self):
    pass

  def build_network(self):

    print('Building the CNN model')
    #Model
    self.model = k.models.Sequential()

    self.model.add(k.layers.Conv2D(32, 3, 3, input_shape=(48, 48, 1)))
    self.model.add(k.layers.Activation('relu'))
    self.model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    self.model.add(k.layers.Flatten())
    self.model.add(k.layers.Dense(128, kernel_initializer='lecun_uniform'))
    self.model.add(k.layers.Dropout(0.4))
    self.model.add(k.layers.Activation('relu'))
    self.model.add(k.layers.Dense(7))
    self.model.add(k.layers.Activation('softmax'))

    print('Compile the model')
    self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  def full_training(self):
    batch_size = 50

    #Build the CNN network
    self.build_network()

    #Generate a test dataset form the images
    train_set, validation_set = self.generate_test_data(batch_size)

    #Start the training
    print('Training....')
    #self.model.fit(train_set, epochs=nb_epoch, batch_size=batch_size, validation_split=0.1, shuffle=True, verbose=1)
    self.model.fit_generator(train_set,
                             steps_per_epoch=int(7307 / batch_size),
                             epochs=EPOCHS,
                             verbose=1,
                             validation_steps=1025,
                             validation_data=validation_set)

    print('Training finished')

    # Show the result of the model
    #loss_and_metrics = self.model.evaluate(train_set, batch_size=batch_size, verbose=1)

    #print('Loss: ', loss_and_metrics[0])
    #print(' Acc: ', loss_and_metrics[1])

  def generate_test_data(self, batch_size):

    # TODO train and testset spit
    #train_gen, test_gen = sk.train_test_split(train_gen, test_size=0.2)
    #Get a list of all images and split into train an test set
    #Image_List = glob.glob(os.path.join('data/train', '*.png'))
    #train_samples, validation_samples = train_test_split(Image_List, test_size=0.1)

    # Load the training data
    # this is the augmentation configuration we will use for training
    train_data_gen = k.preprocessing.image.ImageDataGenerator(
      rescale=1. / 255.
    )

    test_data_gen = k.preprocessing.image.ImageDataGenerator(
      rescale=1. / 255.
    )

    train_gen = train_data_gen.flow_from_directory(
      "data/train",
      target_size=(SIZE_FACE, SIZE_FACE),
      batch_size=batch_size,
      class_mode="categorical",
      color_mode="grayscale"
    )

    test_gen = test_data_gen.flow_from_directory(
      "data/test",
      target_size=(SIZE_FACE, SIZE_FACE),
      batch_size=batch_size,
      class_mode="categorical",
      color_mode="grayscale"
    )
    return (train_gen, test_gen)

  def predict(self, image):
    if image is None:
      return None
    image = image.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    return self.model.predict(image)

  def save_model(self, filename):
    filepath = MODEL_DIRECTORY + filename + '.h5'
    k.models.save_model(self.model, filepath)
    print('[+] Model trained and saved at ' + MODEL_DIRECTORY + filename)

  def load_model(self, filename):
    print('Model loaded from ' + MODEL_DIRECTORY + filename + '.h5')

    if isfile(MODEL_DIRECTORY + filename + '.h5'):
      self.model = k.models.load_model(MODEL_DIRECTORY + filename + '.h5')
    else:
      print('Model could not be loaded')
