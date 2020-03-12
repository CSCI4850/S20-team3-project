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
    'USE_TIME_CUTOFF': False,
    # --- Memory Replay Parameters ---
    'REPLAY_ITERATIONS': 64,
    'REPLAY_SAMPLE_SIZE': 500,
    # --- Constant Parameters ---
    'ENVIRONMENT': 'GalagaDemonsOfDeath-Nes',
    'USE_FULL_ACTION_SPACE': True,
    'SMALL_ACTION_SPACE': 5,
    'NUMPY_SEED': 1239,

}
