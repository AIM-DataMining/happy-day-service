import tensorflow.contrib.keras as k
import time
from PIL import Image
from tensorflow.contrib.keras import layers, models, optimizers
import numpy as np


class SelfCNN:

    target_size = None
    input_shape = None
    save_dir = None
    model = None

    def __init__(self, save_dir="runs/"):
        self.target_size = (48, 48)
        self.input_shape = (48, 48, 1)
        self.save_dir = save_dir + str(round(time.time() * 1000))

    def train(self, train_steps=20, epochs=50, data_path="data", validation_steps=20, batch_size=20):

        train_data_gen = k.preprocessing.image.ImageDataGenerator(
            rescale=1./255.,
            shear_range=0.05,
            zoom_range=0.01,
            rotation_range=0.02,
            width_shift_range=0.01,
            height_shift_range=0.01,
            vertical_flip=False,
            horizontal_flip=False,
        )

        test_data_gen = k.preprocessing.image.ImageDataGenerator(
            rescale=1./255.
        )

        train_gen = train_data_gen.flow_from_directory(
            data_path + "/train",
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode="categorical",
            color_mode="grayscale"
        )

        validation_gen = test_data_gen.flow_from_directory(
            data_path + "/test",
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode="categorical",
            color_mode="grayscale"
        )

        self.model = models.Sequential()

        self.model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=self.input_shape))
        self.model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(layers.Dropout(0.25))

        self.model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        self.model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(layers.Dropout(0.25))

        self.model.add(layers.Conv2D(128, (4, 4), activation='relu'))
        # self.model.add(layers.Conv2D(128, (2, 4), activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.25))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(3072, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(3, activation='softmax'))

        adam = optimizers.Adamax()
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adam,
                           metrics=['acc'])

        tbCallBack = k.callbacks.TensorBoard(
            log_dir=self.save_dir,
            histogram_freq=0,
            write_grads=1,
            write_graph=True,
            write_images=True
        )

        self.model.fit_generator(
            train_gen,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=validation_gen,
            validation_steps=validation_steps,
            callbacks=[tbCallBack]
        )

        print(self.model.metrics_names)

    def predict(self, path):
        img = Image.open(path).convert('L')
        img_ = np.asarray(img.resize(self.target_size))
        img_np = np.array([np.array([img_]).T])
        print(np.shape(img_np))
        pred = self.model.predict_on_batch(img_np).tolist()
        print(pred)
        return {"model": "self_cnn",
                "sad": pred[0][0],
                "smile": pred[0][1],
                "neutral": pred[0][2]}

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = k.models.load_model(path)


if __name__ == "__main__":
    cnn = SelfCNN()

    #cnn.load("runs/1515086732388/model-self-cnn.hdf5")
    cnn.train(train_steps=400, epochs=50, data_path="/tmp/happy-day", validation_steps=10, batch_size=3)
    cnn.save(cnn.save_dir + "/model-self-cnn.hdf5")
    pred = cnn.predict("test/img/test/sad-woman2.jpg")

    print(pred)
