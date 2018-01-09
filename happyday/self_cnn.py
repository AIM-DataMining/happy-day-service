import tensorflow.contrib.keras as k
import time
from PIL import Image
from tensorflow.contrib.keras import layers, models, optimizers, callbacks
import numpy as np


class SelfCNN:

    target_size = None
    input_shape = None
    save_dir = None
    model = None
    model_path = None

    def __init__(self, save_dir="runs/"):
        self.target_size = (48, 48)
        self.input_shape = (48, 48, 1)
        self.save_dir = save_dir + str(round(time.time() * 1000))
        self.model_path = self.save_dir + "/self-cnn.{epoch:02d}-{val_loss:.2f}.hdf5"

    def train(self, train_steps=20, epochs=50, data_path="data", validation_steps=20, batch_size=20):

        train_data_gen = k.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255.,
            shear_range=0.1,
            zoom_range=0.3,
            rotation_range=0.5,
            width_shift_range=0.2,
            height_shift_range=0.2,
            vertical_flip=False,
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

        tb_callback = callbacks.TensorBoard(
            log_dir=self.save_dir,
            histogram_freq=0,
            write_graph=True,
            write_images=False,
        )

        save_callback = callbacks.ModelCheckpoint(
            self.model_path,
            period=10,
            save_best_only=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.001
        )

        self.model.fit_generator(
            train_gen,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=validation_gen,
            validation_steps=validation_steps,
            callbacks=[tb_callback, save_callback, reduce_lr],
            workers=4,
            use_multiprocessing=True
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

    def evaluate(self):
        eval_data_gen = k.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255.
        )
        eval_gen = eval_data_gen.flow_from_directory(
            "data/eval",
            target_size=self.target_size,
            batch_size=1,
            class_mode="categorical",
            color_mode="grayscale"
        )

        metrics = self.model.evaluate_generator(eval_gen, steps=10)
        return {"model": "self-cnn",
                "metrics": metrics,
                "names": self.model.metrics_names
                }


if __name__ == "__main__":
    cnn = SelfCNN()

    cnn.load("models/self-cnn.209-0.25.hdf5")
    cnn.train(train_steps=200,
              epochs=500,
              data_path="/home/oli/tmp/happy-day",
              validation_steps=50,
              batch_size=6)
    _pred = cnn.predict("/home/oli/schrolmcloud/Studium/DataMining/happy-day/smile/IMG_6457.JPG")
    print(_pred)
    print(cnn.evaluate())
