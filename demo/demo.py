#!/usr/bin/env python3.7

# Basal program for the Single Q-Learning Uniform Replay implementation

import sys
import os
from gym import core, spaces
import retro
import numpy as np
from collections import deque

os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('../') # Get top-level
from HyperParameters import *
from utils import preprocess, map_actions, log_create, log_params, log_output
from GalagaAgent import GalagaAgent

def main():

    env = retro.make(game=params['ENVIRONMENT'],
                     use_restricted_actions=retro.Actions.DISCRETE)

    action_space = env.action_space.n if params['USE_FULL_ACTION_SPACE'] else params['SMALL_ACTION_SPACE']
    env.action_space = spaces.Discrete(action_space)

    use_time_cutoff = params['USE_TIME_CUTOFF']

    img_width = params['IMG_WIDTH']
    img_height = params['IMG_HEIGHT']
    channels = 1 if params['GRAYSCALE'] else 3

    model = GalagaAgent(action_space, img_width, img_height, channels)
    
    weight_files = ["m_weights_SQU.h5", "m_weights_SQP.h5", "m_weights_DQU.h5", "m_weights_DQP.h5"]
    labels = ["Single-Q Uniform", "Single-Q Prioritized", "Double-Q Uniform", "Double-Q Prioritized"]
    
    target_update_every = params['TARGET_UPDATE_EVERY']

    replay_memory_size = params['REPLAY_MEMORY_SIZE']
    score_window = deque(maxlen=replay_memory_size)
    
    epochs = 1
    epoch_length = 10000
    frame_count = 0
    for i in range(4):
        model.load_weights(weight_files[i])
        print(labels[i])
        
        for epoch in range(epochs):
            state = env.reset()
            done = False
            last_score = 0
            time = 0
            reward_window = deque(maxlen=epoch_length)

            while not done:

                state = preprocess(state, img_width, img_height, channels)

                action = model.get_action(state)

                next_state, reward, done, info = env.step(action[0])

                state = next_state

                reward_window.append(reward)

                if time > epoch_length:
                    break

                time += 1
                env.render()

            score_window.append(info['score'])
            mean_score = np.mean(score_window)
            
            print("\rEpisode: %d/%d, Mean Score: %d, Mean Reward: %f" % (epoch+1, epochs, mean_score, np.mean(reward_window)))
    

if __name__ == "__main__":
    np.random.seed(params['NUMPY_SEED'])
    main()
