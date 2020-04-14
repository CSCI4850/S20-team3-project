#!/usr/bin/env python3

# Prioritized Memory Replay for DQN Agents
# Source: "Prioritized Memory Replay", Schaul, et al.
# Uses the temporal difference error to approximate how useful an experience is for learning
#

import heapq
import numpy as np

class ReplayMemory:
    def __init__(self, memory_size, image_width, image_height, channels, alpha=0.1):
        self.image_width = image_width
        self.image_height = image_height
        self.size = 0
        self.maxsize = memory_size
        self.current_index = 0
        self.current_state = np.zeros([memory_size, image_width, image_height, channels]) if channels > 1 else np.zeros([memory_size, image_width, image_height])
        self.action = [0]*memory_size
        self.reward = np.zeros([memory_size])
        self.next_state = np.zeros([memory_size, image_width, image_height, channels]) if channels > 1 else np.zeros([memory_size, image_width, image_height])
        self.done = [False]*memory_size
        self.td_errors = np.zeros([memory_size])
        self.channels = channels

    def remember(self, current_state, action, reward, next_state, done, error):
        self.current_state[self.current_index, :, :] = current_state
        self.action[self.current_index] = action
        self.reward[self.current_index] = reward
        self.next_state[self.current_index, :, :] = next_state
        self.done[self.current_index] = done
        self.td_errors[self.current_index] = error
        self.current_index = (self.current_index+1) % self.maxsize
        self.size = max(self.current_index, self.size)

    def replay(self, model, target, num_samples, sample_size, gamma, alpha=1):
        for i in range(num_samples):
            delta = 0

            probabilities = self.get_sample_probabilities(alpha)

            print(self.size, sample_size)
            print(probabilities) #TODO: Make sure probs are right shape

            current_sample = np.random.choice(self.size, sample_size, replace=False, p=np.array(probabilities))

            if self.channels > 1:
                current_state = self.current_state[current_sample, :, :, :]
            else:
                current_state = self.current_state[current_sample, :, :]
            action = [self.action[j] for j in current_sample]
            reward = self.reward[current_sample]
            if self.channels > 1:
                next_state = self.next_state[current_sample, :, :, :]
            else:
                next_state = self.next_state[current_sample, :, :]
            done = [self.done[j] for j in current_sample]

            model_targets = model.predict(current_state)

            targets = reward + gamma * np.amax(target_model.predict(next_state))
            targets[done] = reward[done]

            model_targets[range(sample_size), action] = targets

            model.fit(current_state, model_targets, epochs=1, verbose=0, batch_size=sample_size)

            model.set_weights(model.get_weights() + delta)

        target.set_weights(model.get_weights())

    def get_sample_probabilities(self, alpha):
        td_errors = self.td_errors[0:self.current_index] if self.size < self.maxsize else self.td_errors
        ranks = np.argsort(np.absolute(td_errors)) # Ranked by |delta|, where delta = td_error
        probabilities = [1 / (i+1) for i in range(len(ranks))] # 1/rank(i) (uniform for all sets of size j)
        probabilities = [pow(probabilities[i], alpha) / pow(np.sum(probabilities), alpha) for i in range(len(probabilities))] # pi / sum(p)

        # Matches the probabilities to the order of state memories
        sample_probs = [0]*len(probabilities)
        for i in range(len(ranks)):
            sample_probs[ranks[i]] = probabilities[i]

        return sample_probs

    def td_error(self, model, target, state, next_state, reward, gamma):
        delta = reward + gamma*target.get_action(next_state) - model.get_action(state)
        return delta

    def td_error(self, model_Q, target_Q, reward, gamma):
        delta = reward + gamma*np.max(target_Q) - np.max(model_Q)
        return delta
