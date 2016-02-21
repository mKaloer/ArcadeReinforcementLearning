import numpy as np
from collections import deque
import random

class Agent():

    _FRAME_HISTORY_LEN = 10
    _REPLAY_MEMORY_LEN = 1000000
    _INITIAL_EPSILON = 1
    _MIN_EPSILON = 0.1
    _EPSILON_ANNEALING_STEPS = 1000000

    def __init__(self, action_set, frame_size=(128,128), rand_seed=0):
        self.action_set = action_set
        self.frame_size = frame_size
        self.frames = deque(maxlen=Agent._FRAME_HISTORY_LEN)
        self.replay_mem = deque(maxlen=Agent._REPLAY_MEMORY_LEN)
        self.epsilon = Agent._INITIAL_EPSILON
        random.seed(rand_seed)

    def set_state(self, frame):
        self.frames.append(frame)

    def get_action(self):
        if random.random() <= self.epsilon:
            # Random action
            return self._random_action()
        else:
            return self._optimal_action()

        return 0

    def reward(self, reward):
        self._update_state()
        pass

    def _update_state(self):
        # Update epsilon (epsilon-greedy policy)
        self.epsilon -= (Agent._INITIAL_EPSILON - Agent._MIN_EPSILON) / Agent._EPSILON_ANNEALING_STEPS

    def _random_action(self):
        return random.choice(self.action_set)

    def optimal_action(self):
        frame = self.frames[-1]
        # TODO: Perform optimal action
        return self._random_action()
