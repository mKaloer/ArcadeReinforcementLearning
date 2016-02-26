from ale_python_interface import ALEInterface
import random
import sys
from game import Game
from rl_agent import Agent

ENDURO_ROM = "roms/Enduro.A26"
AMIDAR_ROM  = "roms/Amidar.A26"
BERZERK_ROM  = "roms/Berzerk.A26"
USE_SDL = False
RESTORE_MODEL = True

game = Game(BERZERK_ROM, USE_SDL)

# Get the list of actions
# legal_actions = game.get_legal_action_set()
legal_actions = game.get_minimal_action_set()
print(legal_actions)
agent = Agent(legal_actions, restore=RESTORE_MODEL)
agent.set_frame(game.get_frame())
num_frames = 0
num_games = 0
try:
    while True:
        first = True
        num_games += 1
        while not game.game_over():
            if first:
                # Force random action (we have too few frames to decide on something)
                agent.set_frame(game.get_frame())
                first = False
                continue

            a = agent.get_action()
            # Apply an action and get the resulting reward
            reward = game.act(a)
            if game.game_over():
                reward = -1
            agent.set_frame(game.get_frame())
            agent.reward(reward)
            if reward != 0:
                print("Reward: " + str(reward))
            num_frames += 1
            if num_frames % 100000 == 0:
                print("Frame: %d\tGame: %d" % (num_frames, num_games))
        game.reset_game()
except KeyboardInterrupt:
    print("Stopping")
finally:
    print("Saving")
    agent.stop_session()
