import tensorflow.contrib.keras as k
import time
from PIL import Image
from tensorflow.contrib.keras import layers, models, optimizers
import numpy as np


class SelfCnn:

    target_size = None
    input_shape = None
    save_dir = None

    def __init__(self, save_dir="runs/"):
        self.target_size = (64, 64)
        self.input_shape = (64, 64, 1)
        self.save_dir = save_dir + str(round(time.time() * 1000))

    def train(self):

        train_data_gen = k.preprocessing.image.ImageDataGenerator(
            rescale=1./255.,
            shear_range=0.5,
            zoom_range=0.01,
            horizontal_flip=False
        )

        test_data_gen = k.preprocessing.image.ImageDataGenerator(
            rescale=1./255.
        )

        train_gen = train_data_gen.flow_from_directory(
            "data/train",
            target_size=self.target_size,
            batch_size=32,
            class_mode="categorical",
            color_mode="grayscale"
        )

        validation_gen = test_data_gen.flow_from_directory(
            "data/test",
            target_size=self.target_size,
            batch_size=8,
            class_mode="categorical",
            color_mode="grayscale"
        )

        eval_gen = test_data_gen.flow_from_directory(
            "data/eval",
            target_size=self.target_size,
            batch_size=1,
            class_mode="categorical",
            color_mode="grayscale"
        )

        model = models.Sequential()

        model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=self.input_shape))
        model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=self.input_shape))
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(128, (4, 4), activation='relu'))
        #model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        model.add(layers.Dense(3072, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(3, activation='softmax'))

        adam = optimizers.Adam()
        model.compile(loss='categorical_crossentropy', optimizer=adam)

        tbCallBack = k.callbacks.TensorBoard(
            log_dir=self.save_dir,
            histogram_freq=0,
            write_grads=1,
            write_graph=True,
            write_images=True
        )

        model.fit_generator(
            train_gen,
            steps_per_epoch=2,
            epochs=1,
            validation_data=validation_gen,
            validation_steps=10,
            callbacks=[tbCallBack]
        )

        score = model.evaluate_generator(eval_gen, steps=10)

        print(model.metrics_names)

    def predict(self, model, path):
        img = Image.open(path).convert('LA')
        img_ = np.asarray(img.resize(self.target_size))
        img_np = np.array([img_]).T
        return model.predict_on_batch(img_np)

    def save(self, model, path):
        model.save(path)


#model.save(save_dir + "/model-self-cnn.hdf5")

#plt.figure()
#plt.imshow(img)
#plt.show()
#print(predictions_gen)
#print(predictions_files)
