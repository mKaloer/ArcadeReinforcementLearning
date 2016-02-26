import numpy as np
import scipy.misc
from collections import deque
import random
from conv_net import ConvolutionalNetwork

class Agent():

    _REPLAY_MEMORY_LEN = 100000
    _INITIAL_EPSILON = 1
    _MIN_EPSILON = 0.1
    _EPSILON_ANNEALING_STEPS = 1000000
    _MINIBATCH_SIZE = 32
    _DISCOUNT_FACTOR = 0.9
    _NUM_SKIP_FRAMES = 4

    def __init__(self, action_set, frame_size=(128,128), rand_seed=0, restore=False):
        self.action_set = action_set
        self.frame_size = frame_size
        self.frames = deque(maxlen=Agent._NUM_SKIP_FRAMES)
        self.replay_mem = deque(maxlen=Agent._REPLAY_MEMORY_LEN)
        self.epsilon = Agent._INITIAL_EPSILON
        self._prev_state = None
        self._current_state = {}
        self._model = ConvolutionalNetwork()
        self._model.start_session(restore=restore)
        self._current_skip = 0
        self._train_skips = 0
        random.seed(rand_seed)

    def stop_session(self):
        self._model.stop_session()

    def set_frame(self, frame):
        img = Agent.rgb2gray(frame)
        small_img = scipy.misc.imresize(img, (84,84), mode='L')
        self.frames.append(small_img)

    def get_action(self, force_random=False):
        # Return prev action if not new state
        if self._current_skip > 0:
            return self._current_state['action']
        elif force_random or random.random() <= self.epsilon or len(self.frames) < 4:
            # Random action
            action = self._random_action()
        else:
            action = self._optimal_action()

        self._current_state['action'] = action
        return action

    def reward(self, reward):
        if self._current_skip > 0:
            self._current_state['reward'] = max(self._current_state['reward'], reward)
            if self._current_skip == Agent._NUM_SKIP_FRAMES - 1:
                self._learn_from_state()
                self._current_skip = 0 # Reset skip
            else:
                self._current_skip += 1
        else:
            self._current_state['reward'] = reward
            self._current_skip += 1

    def _learn_from_state(self):
        prev_state = self._prev_state

        self._current_state['prev_state'] = prev_state
        state_repr = self._represent_state(self._current_state)
        # Store transition in replay memory
        self.replay_mem.append({
            'prev': self._represent_state(self._prev_state),
            'action': self._current_state['action'],
            'reward': self._current_state['reward'],
            'current': state_repr
        })

        if self._train_skips > 100:
            self._train_skips = 0
            sample = random.sample(self.replay_mem, min(len(self.replay_mem), Agent._MINIBATCH_SIZE))
            frames = []
            q_vals = []
            for transition in sample:
                frames.append(transition['current'][0])
                if transition['current'][1]:
                    # Is terminal
                    max_q = 0
                else:
                    max_q = max(self._model.predict(transition['current'][0]))
                q_vals.append(transition['reward'] + Agent._DISCOUNT_FACTOR * max_q)

            # Train
            self._model.train_batch(np.vstack(frames), q_vals)

        self._train_skips += 1
        # Update epsilon (epsilon-greedy policy)
        self.epsilon -= (Agent._INITIAL_EPSILON - Agent._MIN_EPSILON) / Agent._EPSILON_ANNEALING_STEPS
        # Store previous state and create new current state
        self._prev_state = self._current_state
        self._current_state = {}

    @staticmethod
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


    def _represent_state(self, state):
        # TODO: return (frame_features, is_terminal)
        if state is None:
            return (np.zeros((84, 84, Agent._NUM_SKIP_FRAMES)), 0)
        stacked_frames = np.array(list(self.frames))
        return (np.array([stacked_frames.flatten()]), 0)

    def _random_action(self):
        return random.choice(self.action_set)

    def _optimal_action(self):
        # Use frame from prev_state as we have reset state in _learn_from_state
        vals = self._model.predict(self._represent_state(self._prev_state)[0])
        action = np.argmax(vals)
        if not action in self.action_set:
            action = self._random_action()
        return action
