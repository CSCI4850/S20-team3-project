#!/usr/bin/env python3

import sys
import keras
import numpy as np

sys.path.append('../../') # Get top-level
from HyperParameters import *
from utils import map_actions


class GalagaAgent:
        def __init__(self, action_size, image_width, image_height, num_channels):
            #variables used to set input shape for model
            self.action_size = action_size
            self.image_width = image_width
            self.image_height = image_height
            self.num_channels = num_channels
            #creates model and stores it
            self.model = self.build_model()  
        
        #wrapper function for set_weights
        def set_weights(self,weights):
            return self.model.set_weights(weights)

        #wrapper function for get_weights
        def get_weights(self):
            return self.model.get_weights()
        
        #wrapper function for fit function
        def fit(self, start_states, target_state):
            return self.model.fit(start_states, target_state,
                           batch_size = 'BATCHES',
                           epochs = 'EPOCHS',
                           verbose = 0)
            
        #takes a state and returns an appropriate action
        #uses utils.py map_action to map result to proper input 
        def get_action(self, current_state):
            return map_actions(np.argmax(self.model.predict(current_state)))

        def build_model(self):
            model = keras.Sequential()
            model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', strides=4, input_shape = [self.image_height,
                                                                                                               self.image_width,
                                                                                                               self.num_channels]))
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(keras.layers.Conv2D(32, kernal_size=(3, 3), activation='relu', strides=2))
            model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(keras.layers.Dropout(.60))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(512, activation='relu'))
            model.add(keras.layers.dropout(.30)
            model.add(keras.layers.Dense(self.action_size, activation='softmax'))

            model.compile(loss=keras.losses.categorical_crossentropy, 
                          optimizer=keras.optimizers.Nadam('LEARNING_RATE'),
                          metrics=['accuracy'])
            
            model.summary()

        
