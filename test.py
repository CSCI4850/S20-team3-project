#!/usr/bin/python3

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(4, kernel_size=(2, 2), activation='relu', input_shape=[100,100,1]))
model.add(tf.keras.layers.Conv2D(16, (2, 2), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(3, activation='relu'))
model.add(tf.keras.layers.Dense(6, activation='softmax'))
model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()
