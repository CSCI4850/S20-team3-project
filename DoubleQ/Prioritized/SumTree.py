#!/usr/bin/env python3

# SumTree implementation for PRE DQN inspired and assisted by the following implementations:
# MorvanZhou: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
# OpenAI Baseline: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/segment_tree.py#L93
# PythonLessons: https://github.com/pythonlessons/Reinforcement_Learning/blob/master/05_CartPole-reinforcement-learning_PER_D3QN/PER.py

import numpy as np

# SumTree works as a binary tree where the root node's value is the sum of all values below it.
# Therefore, all nodes are the sum of the nodes below them.

class SumTree(self):

    data_index = 0

    # Creates a sum tree with max_size nodes, but using numpy arrays
    def __init__(self, max_size):
        self.max_size = max_size

        self.tree = np.zeros(2 * max_size - 1) # Binary tree
        self.data = np.zeros(max_size, dtype=object) # relational array of data for tree nodes

    # Adds a state index to the tree
    def add(self, priority, data):
        current_index = self.data_index + self.max_size - 1
        self.data[self.data_index] = data
        self.update(tree_index, priority)
        self.data_index = self.data_index + 1 if self.data_index < self.max_size else 0

    # Changes the ranking of a state index and propogates the tree
    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    #
    def get_leaf(self, v):
        parent_index = 0

        # The index of any node relative to its parent is (2*parent + 1)
        # The relation between left and right nodes is a difference of 1
        while True:
            left_index = 2 * parent_index + 1
            right_index = left_index + 1

            if left_index >= len(self.tree): # Reached the bottom
                leaf_index = parent_index
                break
            else: # Keep searching down for a higher priority
                if v <= self.tree[left_index]:
                    parent_index = left_index
                else:
                    v -= self.tree[left_index]
                    parent_index = right_index

        data_index = leaf_index - self.max_size + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    # Returns the root node, which represents the sum of all priority values
    def get_total_priority(self):
        return self.tree[0]
