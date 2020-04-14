#!/usr/bin/env python3

# Prioritized Memory Replay for DQN Agents
# Source: "Prioritized Memory Replay", Schaul, et al.
# Uses the temporal difference error to approximate how useful an experience is for learning

import heapq
import numpy as np
from collections import deque

class ReplayMemory:
    def __init__(self, memory_size, epsilon):
        self.maxsize = memory_size
        self.experiences = deque(maxlen=memory_size)
        self.priorities = deque(maxlen=memory_size)
        self.epsilon = epsilon

    # Adds an experience to the buffer
    def remember(self, experience):
        self.experiences.append(experience) # state, next, action, reward, done
        self.priorities.append(max(self.priorities, default=1))

    def replay(self, model, target, num_samples, sample_size, gamma, alpha, beta):
        for i in range(num_samples):

            current_sample, importance, indices = self.sample(sample_size, alpha)

            (current_state, next_state, action, reward, done) = current_sample

            current_state = np.array(current_state)
            next_state = np.array(next_state)
            action = np.array(action)
            reward = np.array(reward)
            done = np.array(done)

            model_targets = model.predict(current_state)
            is_weights = importance[indices]

            innerQ = model.predict(next_state)
            #print("model.predict : ", model.predict(next_state)[0])
            #print("np.argmax : ", np.argmax(model.predict(next_state)))
            #print("innerQ[0] : ", innerQ[0])
            targetQ = target_model.predict(next_state)[[np.argmax(row) for row in innerQ]]
            
            targets = is_weights * (reward + gamma * (targetQ - np.amax(model_targets))) + np.amax(model_targets) # TD-Error
            errors = targets
            targets[done] = reward[done]

            model_targets[range(sample_size), action] = targets

            model.fit(current_state, model_targets, batch_size=sample_size)

            errors = abs(errors) + self.epsilon
            self.set_priorities(indices, errors)

    # Gathers a sample to fit the model on
    def sample(self, batch_size, alpha=1.0, beta=1.0):
        sample_size = min(len(self.experiences), batch_size)
        probs = self.get_sample_probabilities(alpha)
        indices = np.random.choice(range(len(self.experiences)), size=sample_size, p=probs)
        importance = self.get_importance(probs, beta)# Importance sampling weights
        samples = np.array(self.experiences)[indices]
        return map(list, zip(*samples)), importance, indices

    # Determine the importance sampling weights
    def get_importance(self, probabilities, beta):
        importance = pow(1/len(self.experiences) * 1/probabilities, beta) # (( 1 / N ) * 1 / Prob(i))^beta
        importance_normalized = importance / max(importance) # Normalize
        return importance_normalized

    # Determine rank-based probabilities for all states
    def get_sample_probabilities(self, alpha=1.0):
        scaled_priors = pow(np.array(self.priorities), alpha)
        probs = scaled_priors / np.sum(scaled_priors) # Pi^alpha / sum(Pk^alpha)
        return probs

    def set_priorities(self, indices, errors):
        for i,e in zip(indices, errors):
            self.priorities[i] = e
