import sys
from ale_python_interface import ALEInterface

class Game():
    """
    Wrapper around the ALEInterface class.
    """

    def __init__(self, rom_file, sdl=False):
        self.ale = ALEInterface()
        # Setup SDL
        if sdl:
            if sys.platform == 'darwin':
                import pygame
                pygame.init()
                self.ale.setBool(b'sound', False) # Sound doesn't work on OSX
            elif sys.platform.startswith('linux'):
                self.ale.setBool(b'sound', True)
            self.ale.setBool(b'display_screen', True)

        # Load rom
        self.ale.loadROM(str.encode(rom_file))

    def get_action_set(self):
        return self.ale.getLegalActionSet()

    def get_minimal_action_set(self):
        return self.ale.getMinimalActionSet()

    def game_over(self):
        return self.ale.game_over()

    def act(self, action):
        return self.ale.act(action)

    def reset_game(self):
        self.ale.reset_game()

    def get_frame(self):
        return self.ale.getScreenRGB()
