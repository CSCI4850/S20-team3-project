#!/usr/bin/env python3

import heapq

class ReplayMemory:
    def __init__(self, memory_size, frame_width, frame_height, alpha=0.1):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.size = 0
        self.maxsize = memory_size
        self.current_index = 0
        self.current_state = np.zeros([memory_size, frame_width, frame_height])
        self.action = [0]*memory_size
        self.reward = np.zeros([memory_size])
        self.next_state = np.zeros([memory_size, frame_width, frame_height])
        self.done = [False]*memory_size

    def remember(self, current_state, action, reward, next_state, done):
        return None
   
    def replay():
        return None

    def sample():
        return None

    def td_error(model, target, state, next_state, reward, gamma):
        delta = r + gamma*target.getAction(next_state) - model.getAction()
        return delta

class SumTree:
    def __init__(self):
        return None
