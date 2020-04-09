#!/usr/bin/env python3

params = {
    # --- Input Image Parameters ---
    'IMG_WIDTH': 100,
    'IMG_HEIGHT': 100,
    'GRAYSCALE': True,
    # --- Epsilon Parameters ---
    'EPSILON': 0.99,
    'EPSILON_GAMMA': 0.995,
    'EPSILON_MIN': 0.01,
    # --- Training Parameters ---
    'EPOCHS': 500,
    'EPOCH_MAX_LENGTH': 20000,
    'LEARNING_RATE': 0.00025,
    'USE_TIME_CUTOFF': True,
    'Q_LEARNING_GAMMA': 0.99,
    'FRAMES_SINCE_SCORE_LIMIT': 500,
    # --- Memory Replay Parameters ---
    'REPLAY_ITERATIONS': 4,
    'REPLAY_SAMPLE_SIZE': 8,
    'REPLAY_MEMORY_SIZE': 100000,
    'REPLAY_ALPHA': 0.6,
    'REPLAY_BETA': 0.4,
    'REPLAY_EPSILON': 0.01,
    'TARGET_UPDATE_EVERY': 5,
    # --- Constant Parameters ---
    'ENVIRONMENT': 'GalagaDemonsOfDeath-Nes',
    'USE_FULL_ACTION_SPACE': False,
    'SMALL_ACTION_SPACE': 6,
    'NUMPY_SEED': 1239,
    # --- Huber Loss Parameters ---
    'HUBER_DELTA':1.0,
}
