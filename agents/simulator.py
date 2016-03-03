import os
import argparse
from ale_python_interface import ALEInterface
import random
import sys
from game import Game
from rl_agent import Agent
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Arcade Reinforcement Learning')
    parser.add_argument('romfile', type=argparse.FileType('r'),
                        help='The path to the rom file to use')
    parser.add_argument('-g, --gui', default=False, action='store_true', dest='gui',
                        help='If set, a GUI frame of the game will be shown')
    parser.add_argument('-c, --continue', default=False, action='store_true', dest='cont',
                        help='If set, the agent will continue on a previously saved state')

    return parser.parse_args()

def run():
    args = parse_args()
    game = Game(args.romfile.name, args.gui)
    save_dir = "model_data/%s" % (os.path.splitext(os.path.basename(args.romfile.name))[0],)
    save_file = "%s/agent.ckpt" % (save_dir,)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("Model will be saved in %s" % (save_file,))
    # Get the list of actions
    legal_actions = game.get_minimal_action_set()
    print(legal_actions)
    agent = Agent(legal_actions, restore=args.cont, save_path=save_file)
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
                agent.set_frame(game.get_frame())
                if game.game_over():
                    reward = -1
                if reward > 0:
                    reward = 1
                elif reward < 0:
                    reward = -1
                agent.reward(reward, game.game_over())
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


if __name__ == '__main__':
    run()
