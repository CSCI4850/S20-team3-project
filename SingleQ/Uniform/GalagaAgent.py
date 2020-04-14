#!/usr/bin/env python3

import sys
import os
import tensorflow as tf
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
            self.model.set_weights(weights)

        #wrapper function for get_weights
        def get_weights(self):
            return self.model.get_weights()

        #wrapper function for save_weights
        def save_weights(self, w_file):
            self.model.save_weights(w_file)

        #wrapper function for load_weights
        def load_weights(self, w_file):
            if os.path.isfile(w_file):
	            self.model.load_weights(w_file)
	            
        #wrapper function for fit function
        def fit(self, states, targets, batch_size):
            states = states.reshape(states.shape[0], params['IMG_WIDTH'], params['IMG_HEIGHT'],
                                                  1 if params['GRAYSCALE'] else 3)

            return self.model.fit(states,targets,
                           batch_size = batch_size,
                           epochs = 1,
                           verbose = 0)

        def predict(self, states):
            n_states = states.shape[0] if (states.ndim is 4 and not params['GRAYSCALE']) or (states.ndim is 3 and params['GRAYSCALE']) else 1 # if single state passed, expand to (1, w, h, c), else (n, w, h, c)
            states = states.reshape(n_states, params['IMG_WIDTH'], params['IMG_HEIGHT'],
                                                  1 if params['GRAYSCALE'] else 3)
            return self.model.predict(states)

        #takes a state and returns an appropriate action
        #uses utils.py map_action to map result to proper input 
        def get_action(self, current_state):
            current_state = current_state.reshape(1, params['IMG_WIDTH'], params['IMG_HEIGHT'],
                                                  1 if params['GRAYSCALE'] else 3)
            Q = self.model.predict(current_state)
            return map_actions(np.argmax(Q)), Q

        def build_model(self):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(64, kernel_size=(7, 7), activation='relu', input_shape = [self.image_width,
                                                                                                               self.image_height,
                                                                                                               self.num_channels]))
            model.add(tf.keras.layers.MaxPooling2D(2,2))
            model.add(tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu'))
            model.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D(2,2))
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(1024, activation='relu'))
            model.add(tf.keras.layers.Dense(512, activation='relu'))
            model.add(tf.keras.layers.Dense(self.action_size, activation='softmax'))

            model.compile(loss=tf.keras.losses.categorical_crossentropy,
                          optimizer=tf.keras.optimizers.Nadam(params['LEARNING_RATE']),
                          metrics=['accuracy'])
            
            model.summary()
            return model

        def get_summary(self):
            summary = []
            self.model.summary(print_fn=lambda x: summary.append(x))
            summary_str = "\n".join(summary)
            return summary_str
