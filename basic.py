#!/usr/bin/env python3.7

from gym import core, spaces
import retro
#import tensorflow as tf

def main():
    


    env = retro.make(game='GalagaDemonsOfDeath-Nes')
    obs = env.reset()
   
    input_space = env.observation_space.shape[0]
    i = 1
    while True:
        # Chooses a random action from the space
        # obs: observation of the screen after the action
        # rew: reward gained from action
        # done: if done state reached
        # info: debugging info, using this disqualifies official grading
        action = env.action_space.sample()
        print(action)
        obs, rew, done, info = env.step([0,0,0,0,0,0,0,1,1])
        env.render()
        a = 0
        if i == 1:
            for x in obs:
                for y in x:
                    if y.any() != 0:
                        a += 1
            print(a)
        i = 2
        if done:
            # Restarts the environment
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
