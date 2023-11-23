import numpy as np
import numpy as np
import cv2
from ale_py import ALEInterface, SDL_SUPPORT

class Game:
    def __init__(self, environment_pathlib, game_name, frame_skip: int = 1, mode:int = 0, difficulty:int = 0):
        # Set up ale environment
        self.name = game_name
        self.ale = ALEInterface()
        self.ale.loadROM(environment_pathlib)

        self.ale.setInt("random_seed", 123)
        self.ale.setInt("frame_skip", frame_skip)

        # Check if we can display the screen
        if SDL_SUPPORT:
            self.ale.setBool("sound", True)
            self.ale.setBool("display_screen", True)
        # Set mode and difficulty level
        avail_modes = self.ale.getAvailableModes()
        avail_diff = self.ale.getAvailableDifficulties()

        self.ale.setDifficulty(avail_diff[difficulty])
        self.ale.setMode(avail_modes[mode])

        self.legal_actions = self.ale.getLegalActionSet()
        self.minimal_actions = self.ale.getMinimalActionSet()

        # tensor for a stack of 4 frames
        self.obs_4 = np.zeros((4, 84, 84))
        # buffer to keep maximum of last 2 frames
        self.obs_2_max = np.zeros((2, 84, 84))
        # Keep track of the episode rewards
        self.rewards = []
        # number of lives left
        self.lives = self.ale.lives()
    def step(self, action):
        reward = 0.0
        done = None

        # run for 4 steps
        for i in range(4):
            r = self.ale.act(action)
            done = self.ale.game_over()
            self.lives = self.ale.lives()
            obs = self.ale.getScreenRGB()
            if i >= 2:
                self.obs_2_max[i % 2] = self._process_obs(obs)

            reward += r
            # reset if a life is lost
            if self.lives == 0:
                done = True
                break

        # maintain rewards for each step
        self.rewards.append(reward)

        if done:
            # if finished, set episode information if episode is over, and reset
            episode_info = {"reward": sum(self.rewards), "length": len(self.rewards)}
            self.reset()
        else:
            episode_info = None

            # get the max of last two frames
            obs = self.obs_2_max.max(axis=0)

            # push it to the stack of 4 frames
            self.obs_4 = np.roll(self.obs_4, shift=-1, axis=0)
            self.obs_4[-1] = obs.copy()

        return self.obs_4, reward, done, episode_info
    def reset(self):
        """
        ### Reset environment
        Clean up episode info and 4 frame stack
        """
        self.ale.reset_game()
        # reset caches
        obs = self._process_obs(self.ale.getScreenRGB())
        for i in range(4):
            self.obs_4[i] = obs.copy()
        self.rewards = []

        self.lives = self.ale.lives()

        return self.obs_4
    @staticmethod
    def _process_obs(obs:np.ndarray):
        """
        #### Process game frames
        Convert game frames to gray and rescale to 84x84
        """
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs