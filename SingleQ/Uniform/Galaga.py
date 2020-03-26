#!/usr/bin/env python3.7

# Basal program for the Single Q-Learning Uniform Replay implementation

import sys
from gym import core, spaces
import retro
import numpy as np

sys.path.append('../../') # Get top-level
from HyperParameters import *
from utils import preprocess
<<<<<<< HEAD
from GalagaAgent import GalagaAgent
from ReplayMemory import ReplayMemory
=======

>>>>>>> PrioritizedReplay

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

    q_learning_gamma = params['Q_LEARNING_GAMMA']

    model = GalagaAgent(action_space, img_width, img_height, channels)
    target = GalagaAgent(action_space, img_width, img_height, channels)


    memory = ReplayMemory(replay_memory_size, img_width, img_height, channels)

    for epoch in range(epochs):
        state = env.reset();
        done = False
       
        while not done:
            state = preprocess(state, channels, img_width, img_height)
            action = model.get_action(state) if np.random.random() > epsilon else map_action(np.random.randint(0, action_space+1))
            next_state, reward, done, info = env.step(action)

            # reward, memory replay, etc
            pp_next = preprocess(next_state)
            memory.remember(state, action, reward, pp_next, done)

            state = next_state

            if use_time_cutoff and time > epoch_length:
                break

            if "--play" in sys.argv:
                env.render()

        memory.replay(model, target, replay_iterations, replay_sample_size, q_learning_gamma)
        epsilon = epsilon * epsilon_gamma if epsilon > epsilon_min else epsilon_min

if __name__ == "__main__":
    np.random.seed(params['NUMPY_SEED'])
    main()
