#!/usr/bin/env python3

params = {
    # --- Input Image Parameters ---
    'IMG_WIDTH': 100,
    'IMG_HEIGHT': 100,
    'GRAYSCALE': True,
    # --- Epsilon Parameters ---
    'EPSILON': 0.99,
    'EPSILON_GAMMA': 0.99,
    'EPSILON_MIN': 0.01,
    # --- Training Parameters ---
    'EPOCHS': 100,
    'EPOCH_MAX_LENGTH': 10000,
    'LEARNING_RATE': 0.002,
    'USE_TIME_CUTOFF': False,
    'Q_LEARNING_GAMMA': 0.1,
    # --- Memory Replay Parameters ---
    'REPLAY_ITERATIONS': 64,
    'REPLAY_SAMPLE_SIZE': 128,
    'REPLAY_MEMORY_SIZE': 1000,
    'REPLAY_GAMMA': 0.1,
    'REPLAY_ALPHA': 0.1,
    # --- Constant Parameters ---
    'ENVIRONMENT': 'GalagaDemonsOfDeath-Nes',
    'USE_FULL_ACTION_SPACE': True,
    'SMALL_ACTION_SPACE': 15,
    'NUMPY_SEED': 1239,
    # --- Huber Loss Parameters ---
    'HUBER_DELTA':1.0,
}
