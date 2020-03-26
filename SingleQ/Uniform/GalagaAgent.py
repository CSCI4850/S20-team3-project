#!/usr/bin/env python3

import sys
import keras
import numpy as np

sys.path.append('../../') # Get top-level
from HyperParameters import *

# ten enemy types and one projectile type so 11 feature detectors

class GalagaAgent:
        def __init__(self, action_size, image_width, image_height, num_channels):
            self.action_size = action_size
            self.image_width = image_width
            self.image_height = image_height
            self.num_channels = num_channels
            #variables used to set input shape for Conv2D layer
            self.model = make_model(self.image_height, self.image_width, self.num_channels, self.action_size)  
        
        def set_weights(self,weights):
            self.model.set_weights(weights)

        def get_weights(self):
            self.model.get_weights()
        
        def fit(self, start_states, target_state):
            self.model.fit(start_states, target_state,
                           batch_size = 'BATCHES',
                           epochs = 'EPOCHS',
                           verbose = 1)
        
        def get_action(self,):
            
            
        def __make_model(rows, cols, channels, action_size):
            model = keras.Sequential()
            model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', strides=4, input_shape = [rows, cols,  channels]))
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(keras.layers.Conv2D(32, kernal_size=(3, 3), activation='relu', strides=2))
            model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(keras.layers.Dropout(.60))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(512, activation='relu'))
            model.add(keras.layers.dropout(.30)
            model.add(keras.layers.Dense(action_size, activation='softmax'))

            model.compile(loss=keras.losses.categorical_crossentropy, 
                          optimizer=keras.optimizers.Nadam('LEARNING_RATE'),
                          metrics=['accuracy'])
            
            model.summary()

        
