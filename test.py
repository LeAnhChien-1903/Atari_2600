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

policy = ActorCritic().to(device)
policy.load_state_dict(torch.load(os.path.join("Parameters", "parameters.pt"), map_location= device))
print("Load model!")

colors = ['red', 'green', 'blue', 'orange', 'brown', 'purple']
# fig, axs = plt.subplots(3, 2)

# counter = 0
# fig.suptitle('Phần thưởng tích lũy trong quá trình đào tạo')
# for i in range(3):
#     for j in range(2):
#         rewards = np.loadtxt(os.path.join(game_list[counter].name, "cumulative_reward.txt"))
#         axs[i, j].plot(rewards, color = colors[counter])
#         axs[i, j].grid(True)
#         # axs[i, j].set_title(game_list[counter].name)
#         axs[i, j].text(0, max(rewards) * 0.95, game_list[counter].name, fontsize= 15, color='k')
#         counter+= 1
# for ax in axs.flat:
#     ax.set(xlabel='Số lần đào tạo', ylabel='Phần thưởng')

# plt.show()    
epsilon = 0.1
print("Testing policy")
num_of_test = 50
total_reward = np.zeros((6, num_of_test))

# for i in range(num_of_test):
    # counter = 0
for game in game_list:
    observation = game.reset()
    image = game.ale.getScreenRGB()
    out = cv2.VideoWriter("results/{}.avi".format(game.name), cv2.VideoWriter_fourcc(*'DIVX'), 24, (image.shape[1], image.shape[0]))
    observation = game.reset()
    out.write(game.ale.getScreenRGB())
    tested_reward = 0.0
    done = False
    test_counter = 0

    while done == False and test_counter < 2000:
        # Get action
        obs = convertToTensor(observation, device = device)
        random_value = random.random()
        # if random_value < epsilon:
        #     action, _, _, _ = policy.get_action(obs)
        # else:
        action = policy.exploit_policy(obs)
        observation, reward, done, info = game.step(game.legal_actions[action])
        out.write(game.ale.getScreenRGB())
        # cv2.imshow("Game", game.ale.getScreenRGB())
        # if cv2.waitKey(10) & ord('q') == 0xFF:
        #     break
        tested_reward += reward
        test_counter += 1
    out.release()
    print("Total tested reward of {}: {} with {} frames".format(game.name, tested_reward, test_counter))
        # total_reward[counter, i] = tested_reward
        # cv2.destroyAllWindows()
        # counter +=1 
    # print(i)

# fig, axs = plt.subplots(3, 2)
# counter = 0
# fig.suptitle('Phần thưởng tích lũy chạy thử  với epsilon = 0.2')
# for i in range(3):
#     for j in range(2):
#         axs[i, j].plot(total_reward[counter], colors[counter])
#         axs[i, j].grid(True)
#         # axs[i, j].set_title(game_list[counter].name)
#         axs[i, j].text(0, max(total_reward[counter]) * 0.95, game_list[counter].name, fontsize= 10, color='k')
#         counter+= 1
# for ax in axs.flat:
#     ax.set(xlabel='Số lần kiểm thử', ylabel='Phần thưởng')
    
# plt.show()
