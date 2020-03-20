#!/usr/bin/env python3

# Prioritized Memory Replay for DQN Agents
# Source: "Prioritized Memory Replay", Schaul, et al.
# Uses the temporal difference error to approximate how useful an experience is for learning
#

import heapq
import numpy as np

class ReplayMemory:
    def __init__(self, memory_size, frame_width, frame_height, channels, alpha=0.1):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.size = 0
        self.maxsize = memory_size
        self.current_index = 0
        self.current_state = np.zeros([memory_size, frame_width, frame_height, channels])
        self.action = [0]*memory_size
        self.reward = np.zeros([memory_size])
        self.next_state = np.zeros([memory_size, frame_width, frame_height, channels])
        self.done = [False]*memory_size
        self.td_errors = np.zeros([memory_size])

    def remember(self, current_state, action, reward, next_state, done, error):
        self.current_state[self.current_index, :, :] = current_state
        self.action[self.current_index] = action
        self.reward[self.current_index] = reward
        self.next_state[self.current_index, :, :] = next_state
        self.done[self.current_index] = done
        self.td_errors[self.current_index] = error
        self.current_index = (self.current_index+1) % self.maxsize
        self.size = max(self.current_index, self.size)

    def replay(model, target, num_samples, sample_size, gamma, alpha=1):
        for i in range(num_samples):
            delta = 0

            probabilities = get_sample_probabilities()

            current_sample = np.random.choice(self.size, sample_size, replace=False, p=probabilities)

            #TODO: Apply sampled states to model

            model.model.set_weights(model.model.get_weights() + delta)

        target.model.set_weights(model.model.get_weights())

    def get_sample_probabilities(self, alpha):
        ranks = np.argsort(np.absolute(self.td_errors)) # Ranked by |delta|, where delta = td_error
        probabilities = [1 / i for i in range(len(ranks))] # 1/rank(i) (uniform for all sets of size j)
        probabilities = [pow(probabilities[i], alpha) / pow(np.sum(probabilities), alpha) for i in range(len(probabilities))] # pi / sum(p)

        # Matches the probabilities to the order of state memories
        sample_probs = [0]*len(probabilities)
        for i in range(len(ranks)):
            sample_probs[ranks[i]] = probabilities[i]

        return sample_probs

    def td_error(model, target, state, next_state, reward, gamma):
        delta = r + gamma*target.getAction(next_state) - model.getAction()
        return delta

class SumTree:
    def __init__(self):
        return None
