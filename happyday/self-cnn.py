import tensorflow as tf
import tensorflow.contrib.keras as k
from tensorflow.contrib.keras import layers, models, optimizers

from matplotlib import pyplot as plt

target_size = (48, 48)
input_shape = (48, 48, 3)
train_data_gen = k.preprocessing.image.ImageDataGenerator(
    rescale=1./255.
)

test_data_gen = k.preprocessing.image.ImageDataGenerator(
    rescale=1./255.
)

train_gen = train_data_gen.flow_from_directory(
    "data/train",
    target_size=target_size,
    batch_size=32,
    class_mode="categorical"
)

test_gen = test_data_gen.flow_from_directory(
    "data/test",
    target_size=target_size,
    batch_size=32,
    class_mode="categorical"
)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

adam = optimizers.Adam()
model.compile(loss='categorical_crossentropy', optimizer=adam)

tbCallBack = k.callbacks.TensorBoard(
    log_dir='./Graph/'+str(round(time.time() * 1000)),
    histogram_freq=0,
    write_graph=True,
    write_images=True
)

model.fit_generator(
    train_gen,
    steps_per_epoch=200,
    epochs=20,
    validation_data=test_gen,
    validation_steps=100
    validation_steps=10,
    callbacks=[tbCallBack]
)
score = model.evaluate_generator(train_gen, steps=32)