#!/usr/bin/env python3

import sys
import keras

# ten enemy types and one projectile type so 11 feature detectors

class GalagaAgent:
        def __init__(self, state_size, action_size, image_width, image_height, num_channels):
            self.state_size = state_size
            self.action_size = action_size
            self.image_width = image_width
            self.image_height = image_height
            self.num_channels = num_channels
            self.model = make_model(state_size, action_size)

        def make_model(state_size, action_size):
            model = keras.Sequential()
            model.add(keras.layers.Conv2D(11, kernel_size=(15, 15), activation='relu', input_shape=state_size))
            model.add(keras.layers.Conv2D(22, (15, 15), activation='relu'))
            model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
            model.add(keras.layers.Dropout(.20))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(22, activation='relu'))
            model.add(keras.layers.Dense(action_size, activation='softmax'))
            model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Nadam(learning_rate = 0.01),
                            metrics=['accuracy'])
            model.summary()

        
