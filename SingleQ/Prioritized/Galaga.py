#!/usr/bin/env python3

# Basal program for the Single Q-Learning Prioritized Replay implementation

import sys
from gym import core, spaces
import retro
import numpy as np

sys.path.append('../../') # Get top-level
from HyperParameters import *
from utils import preprocess


def main():

    env = retro.make(game=params['ENVIRONMENT'],
                     use_restricted_actions=retro.Actions.DISCRETE)

    action_space = env.action_space.n if params['USE_FULL_ACTION_SPACE'] else params['SMALL_ACTION_SPACE']
    env.action_space = spaces.Discrete(action_space)
    epsilon = params['EPSILON']
    epsilon_gamma = params['EPSILON_GAMMA']
    epsilon_min = params['EPSILON_MIN']
    epochs = params['EPOCHS']
    epoch_length = params['EPOCH_MAX_LENGTH']
    use_time_cutoff = params['USE_TIME_CUTOFF']

    img_width = params['IMG_WIDTH']
    img_height = params['IMG_HEIGHT']
    channels = 1 if params['GRAYSCALE'] else 3
    input_space = (env.observation_space.shape[0])

    for epoch in range(epochs):
        state = env.reset();
        done = False

        while not done:
            state = preprocess(state, channels, img_width, img_height)
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)

            # reward, memory replay, etc

            state = next_state

            if use_time_cutoff and time > epoch_length:
                break

            if "--play" in sys.argv:
                env.render()

        epsilon = epsilon * epsilon_gamma if epsilon > epsilon_min else epsilon_min

if __name__ == "__main__":
    np.random.seed(params['NUMPY_SEED'])
    main()
