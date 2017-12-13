from __future__ import division, absolute_import

from os.path import isfile
from Constants import *

from tensorflow import keras as k
import tensorflow as tf

class EmotionRecognition:

  def __init__(self):
    pass

  def build_network(self, filename):

    print('Building the CNN model')
    #Model
    self.model = k.models.Sequential()
    self.model.add(k.layers.Conv2D(32, 3, 3, border_mode='same', activation='relu', input_shape=(1, 48, 48)))
    self.model.add(k.layers.Conv2D(32, 3, 3, border_mode='same', activation='relu'))
    self.model.add(k.layers.Conv2D(32, 3, 3, border_mode='same', activation='relu'))
    self.model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))

    self.model.add(k.layers.Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    self.model.add(k.layers.Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    self.model.add(k.layers.Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    self.model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))

    self.model.add(k.layers.Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    self.model.add(k.layers.Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    self.model.add(k.layers.Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    self.model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))

    self.model.add(k.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
    self.model.add(k.layers.Dense(64, activation='relu'))
    self.model.add(k.layers.Dense(64, activation='relu'))
    self.model.add(k.layers.Dense(2, activation='softmax'))

    print('Compile the model')
    self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Load the model only if a existing model is used
    self.load_model(filename)

  def full_training(self):
    batch_size = 50

    self.build_network()

    train_set = self.generate_test_data(batch_size)

    print('Training....')
    #self.model.fit(train_set, epochs=nb_epoch, batch_size=batch_size, validation_split=0.1, shuffle=True, verbose=1)
    self.model.fit_generator(train_set,
                             epochs=100,
                             batch_size=batch_size,
                             validation_split=0.1,
                             shuffle=True,
                             verbose=1,
                             show_metric=True,
                             snapshot_step=200,
                             snapshot_epoch=True,
                             )
    print('Training finished')

    # Show the result of the model
    loss_and_metrics = self.model.evaluate(train_set, batch_size=batch_size, verbose=1)

    print('Loss: ', loss_and_metrics[0])
    print(' Acc: ', loss_and_metrics[1])

    # model logging:
    #notes = 'medium set 100'
    #self.save_model(self.model.to_json(), './Models/')
    #self.save_config(self.model.get_config(), './Models/')
    #self.save_result(loss_and_metrics, notes, './Models/')

  def generate_test_data(self, batch_size):
    test_set = 0
    validation_set = 0

    import sklearn.model_selection as sk

    # TODO train and testset spit
    # train_samples, validation_samples = sk.train_test_split(Image_List, test_size=0.2)

    # Load the training data

    # this is the augmentation configuration we will use for training
    train_datagen = k.preprocessing.image.ImageDataGenerator(
      rescale=1. / 255,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True)


    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = k.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # subfolers of 'Images', and indefinitely generate
    # batches of augmented image data
    data = train_datagen.flow_from_directory(
      'Images',  # this is the target directory
      target_size=(150, 150),  # all images will be resized to 150x150
      batch_size=batch_size,
      class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    return (test_set, validation_set)

  def predict(self, image):
    if image is None:
      return None
    image = image.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    return self.model.predict(image)

  def save_model(self, filename):
    filepath = MODEL_DIRECTORY + filename
    self.model.save(filepath)
    print('[+] Model trained and saved at ' + MODEL_DIRECTORY + filename)

  def load_model(self, filename):
    filepath = MODEL_DIRECTORY + filename
    if isfile(filepath + ".meta"):
      self.model.load(filepath)
      print('Model loaded from ' + MODEL_DIRECTORY + filename)
    else:
      print('Model could not be loaded')
