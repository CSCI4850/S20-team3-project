#!/usr/bin/env python3.7

from gym import core, spaces
import retro
#import tensorflow as tf

def main():

    env = retro.make(game='GalagaDemonsOfDeath-Nes', use_restricted_actions=retro.Actions.DISCRETE)
    obs = env.reset()
    env.action_space = spaces.Discrete(5)
   
    input_space = env.observation_space.shape[0]

    while True:
        # Chooses a random action from the space
        # obs: observation of the screen after the action
        # rew: reward gained from action
        # done: if done state reached
        # info: debugging info, using this disqualifies official grading
        action = env.action_space.sample()+1
        obs, rew, done, info = env.step(action)
        env.render()
        if done:
            # Restarts the environment
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
