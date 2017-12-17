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
        self.target_size = (64, 64)
        self.input_shape = (64, 64, 1)
        self.save_dir = save_dir + str(round(time.time() * 1000))

    def train(self, steps=20, epochs=50):

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

        self.model = models.Sequential()

        self.model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=self.input_shape))
        self.model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=self.input_shape))
        self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(layers.Dropout(0.25))

        self.model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        self.model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(layers.Dropout(0.25))

        self.model.add(layers.Conv2D(128, (4, 4), activation='relu'))
        #model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.25))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(3072, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(3, activation='softmax'))

        adam = optimizers.Adam()
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['mae', 'acc'])

        tbCallBack = k.callbacks.TensorBoard(
            log_dir=self.save_dir,
            histogram_freq=0,
            write_grads=1,
            write_graph=True,
            write_images=True
        )

        self.model.fit_generator(
            train_gen,
            steps_per_epoch=steps,
            epochs=epochs,
            validation_data=validation_gen,
            validation_steps=10,
            callbacks=[tbCallBack]
        )

        score = self.model.evaluate_generator(eval_gen, steps=10)

        print(self.model.metrics_names)

    def predict(self, path):
        img = Image.open(path).convert('LA')
        img_ = np.asarray(img.resize(self.target_size))
        img_np = np.array([img_]).T
        pred = self.model.predict_on_batch(img_np)
        print(pred)
        return {"self_cnn": pred.tolist()}

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = k.models.load_model(path)


if __name__ == "__main__":
    cnn = SelfCNN()
    cnn.load("runs/1513535046095/model-self-cnn.hdf5")
    cnn.train(steps=20, epochs=20)
    cnn.save(cnn.save_dir + "/model-self-cnn.hdf5")
    pred = cnn.predict("data/eval/sad/1512140526.jpg")

    print(pred)
