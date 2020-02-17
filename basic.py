#!/usr/bin/env python3

import retro

def main():
    env = retro.make(game='GalagaDemonsOfDeath-Nes')
    obs = env.reset()
    while True:
        # Chooses a random action from the space
        # obs: observation of the screen after the action
        # rew: reward gained from action
        # done: if done state reached
        # info: debugging info, using this disqualifies official grading
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            # Restarts the environment
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
