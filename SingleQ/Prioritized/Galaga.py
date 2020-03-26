#!/usr/bin/env python3

# Basal program for the Single Q-Learning Prioritized Replay implementation

import sys
from gym import core, spaces
import retro
import numpy as np

sys.path.append('../../') # Get top-level
from HyperParameters import *
from utils import preprocess
from ReplayMemory import ReplayMemory

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

    replay_iterations = params['REPLAY_ITERATIONS']
    replay_sample_size = params['REPLAY_SAMPLE_SIZE']
    replay_memory_size = params['REPLAY_MEMORY_SIZE']
    replay_alpha = params['REPLAY_ALPHA']

    q_learning_gamma = params['Q_LEARNING_GAMMA']

    memory = ReplayMemory(replay_memory_size, img_width, img_height, action_space)

    for epoch in range(epochs):
        state = env.reset();
        done = False

        while not done:
            state = preprocess(state, channels, img_width, img_height)
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)

            # Reward

            # Memory Replay

            pp_next = preprocess(next_state)

            td_error = memory.td_error(model, target_model,
                                       state, pp_next,
                                       reward,
                                       q_learning_gamma)

            memory.remember(state,
                            action,
                            reward,
                            pp_next,
                            done,
                            td_error)

            state = next_state

            if use_time_cutoff and time > epoch_length:
                break

            if "--play" in sys.argv:
                env.render()

        epsilon = epsilon * epsilon_gamma if epsilon > epsilon_min else epsilon_min
        memory.replay(model, target_model, replay_sample_size, q_learning_gamma, replay_alpha)

if __name__ == "__main__":
    np.random.seed(params['NUMPY_SEED'])
    main()
