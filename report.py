import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import ale_py
from actor_critic import ActorCritic
from game import Game
from ultis import *
from ale_py.roms import MsPacman, Phoenix, Alien, Assault, KungFuMaster, Defender
import random
# Use CUDA
use_cuda = not torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


# Create the environment
game_list = []

game_list.append(Game(Alien, "Alien"))
game_list.append(Game(Assault, "Assault"))
game_list.append(Game(Defender, "Defender"))
game_list.append(Game(KungFuMaster, "KungFuMaster"))
game_list.append(Game(MsPacman, "MsPacman"))
game_list.append(Game(Phoenix, "Phoenix"))


for game in game_list:
    observation = game.reset()
    cv2.imwrite("report/{}.png".format(game.name), game.ale.getScreenRGB())