#!/usr/bin/env python3
import retro
import numpy as np
from gym import core, spaces
import sys
import keras

def main():
    class ReplayMemory:
        def __init__(self, memory_size, state_size, action_size):
            self.state_size = state_size
            self.action_size = action_size
            self.size = 0
            self.maxsize = memory_size
            self.current_index = 0
            self.current_state = np.zeros([memory_size, env.observation_space.shape[0], env.observation_space.shape[1],
                                           env.observation_space.shape[2]])
            self.action = [0] * memory_size
            self.reward = np.zeros([memory_size])
            self.next_state = np.zeros([memory_size, env.observation_space.shape[0], env.observation_space.shape[1],
                                        env.observation_space.shape[2]])
            self.done = [False] * memory_size  # Boolean (terminal transition?)

        def remember(self, current_state, action, reward, next_state, done):
            # Stores a single memory item
            self.current_state[self.current_index, :, :] = current_state
            self.action[self.current_index] = action
            self.reward[self.current_index] = reward
            self.next_state[self.current_index, :, :] = next_state
            self.done[self.current_index] = done
            self.current_index = (self.current_index + 1) % self.maxsize
            self.size = max(self.current_index, self.size)

        def replay(self, model, target_model, num_samples, sample_size, gamma):
            # Run replay!

            # Can't train if we don't yet have enough samples to begin with...
            if self.size < sample_size:
                return

            for i in range(num_samples):
                # Select sample_size memory indices from the whole set
                current_sample = np.random.choice(self.size, sample_size, replace=False)

                # Slice memory into training sample
                current_state = self.current_state[current_sample, :, :]
                action = [self.action[j] for j in current_sample]
                reward = self.reward[current_sample]
                next_state = self.next_state[current_sample, :, :]
                done = [self.done[j] for j in current_sample]

                # Obtain model's current Q-values
                model_targets = model.predict(current_state)

                # Create targets from argmax(Q(s+1,a+1))
                # Use the target model!
                targets = reward + gamma * np.amax(target_model.predict(next_state))
                # Absorb the reward on terminal state-action transitions
                targets[done] = reward[done]
                # Update just the relevant parts of the model_target vector...
                model_targets[range(sample_size), action] = targets

                # Update the weights accordingly
                model.fit(current_state, model_targets,
                          epochs=1, verbose=0, batch_size=sample_size)

            # Once we have finished training, update the target model
            target_model.set_weights(model.get_weights())

    def make_model(state_size, action_size):
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(4, kernel_size=(2, 2), activation='relu', input_shape=state_size))
        model.add(keras.layers.Conv2D(8, (2, 2), activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(3, activation='relu'))
        model.add(keras.layers.Dense(action_size, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Nadam(learning_rate = 0.005),
                      metrics=['accuracy'])
        model.summary()
        return model

    env = retro.make(game='GalagaDemonsOfDeath-Nes', use_restricted_actions=retro.Actions.DISCRETE)
    observation = env.reset()

    x_train = env.observation_space.shape
    y_train = env.observation_space.shape

    env.action_space = spaces.Discrete(16)

    from collections import deque

    gamma = 0.95
    epsilon = 0.7
    epsilon_decay = 0.95
    epsilon_min = 0.01
    episodes = 25

    replay_iterations = 100
    replay_sample_size = 256
	
    times_window = deque(maxlen=100)
    mean_times = deque(maxlen=episodes)

    model = make_model(env.observation_space.shape,env.action_space.n)
    target_model = make_model(env.observation_space.shape,env.action_space.n)
    memory = ReplayMemory(250,env.observation_space.shape,env.action_space.n)
    
    print("Epsilon :", epsilon)
    for episode in range(episodes):
        current_state = env.reset()
        for time in range (5000):
            Q = model.predict(np.expand_dims(current_state,axis=0))
            action = env.action_space.sample() if np.random.random() < epsilon else np.argmax(Q)
            next_state, reward, done, info = env.step(action)
            if done:
                reward = -10.0
            memory.remember(current_state,action,reward,next_state,done)
            current_state = next_state
            if (done):
                break
            if "--play" in sys.argv:
                env.render()
            print('\rEpisdoe: %d/%d epslion: %f'%(episode+1,episodes,epsilon),end='')
        memory.replay(model,target_model,replay_iterations,replay_sample_size,gamma)
        epsilon = epsilon * epsilon_decay if epsilon > epsilon_min else epsilon_min 
    print()
    
    # Play
    print("play")
    observation = env.reset()
    while True:
        Q = model.predict(np.expand_dims(observation,axis=0))
        action = np.argmax(Q)
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            break
main()
