import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import ale_py
from actor_critic import ActorCritic
from game import Game
from ultis import *
from ale_py.roms import MsPacman, Phoenix, Alien, Assault, KungFuMaster, MarioBros

# Use CUDA
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
# Hyper parameters
size_of_obs = (84, 84)
num_of_update = 10000
num_of_epochs = 20
batch_size = 500
gamma = 0.99
lambda_ = 0.95
clip_coef = 0.2
value_coef = 0.5
entropy_coef = 0.01

# Create the environment
game_list = []

game_list.append(Game(Alien, "Alien"))
game_list.append(Game(Assault, "Assault"))
game_list.append(Game(KungFuMaster, "KungFuMaster"))
game_list.append(Game(MarioBros, "MarioBros"))
game_list.append(Game(MsPacman, "MsPacman"))
game_list.append(Game(Phoenix, "Phoenix"))

# Initialize ActorCritic Model
policy = ActorCritic().to(device)
optimizer = torch.optim.Adam(policy.parameters(),  lr = 3e-4)

for game in game_list:
    if not os.path.exists(game.name):
        os.mkdir(game.name)

    if not os.path.exists(os.path.join(game.name, "cumulative_reward.txt")):
        np.savetxt(os.path.join(game.name, "cumulative_reward.txt"), np.zeros(1),fmt='%.1f')

if not os.path.exists("Parameters"):
    os.mkdir("Parameters")

if not os.path.exists(os.path.join("Parameters", "parameters.pt")):
    torch.save(policy.state_dict(), os.path.join("Parameters", "parameters.pt"))
    print("Save initialized model!")
else:
    policy.load_state_dict(torch.load(os.path.join("Parameters", "parameters.pt"), map_location= device))
    print("Load model!")

for game in game_list:
    observation = game.reset()
    for iteration in range(num_of_update):
        with torch.no_grad():
            obs_batch = torch.zeros((batch_size, 4, size_of_obs[0], size_of_obs[1])).to(device)
            action_batch = torch.zeros(batch_size).to(device)
            log_prob_batch = torch.zeros(batch_size).to(device)
            entropy_batch = torch.zeros(batch_size).to(device)
            reward_batch = torch.zeros(batch_size).to(device)
            value_batch = torch.zeros(batch_size + 1).to(device)
            done_batch = torch.zeros(batch_size).to(device)
            # Collect data
            for t in range(batch_size):
                # Get action
                obs = convertToTensor(observation, device = device)
                action, log_prob, entropy, value = policy.get_action(obs)
                observation, reward, done, info = game.step(game.legal_actions[action])

                # Collect action
                obs_batch[t] = obs
                action_batch[t] = action
                log_prob_batch[t] = log_prob
                entropy_batch[t] = entropy
                value_batch[t] = value

                reward_batch[t] = reward

                if t == batch_size - 1:
                    obs = convertToTensor(observation, device= device)
                    value_batch[batch_size] =  policy.get_value(obs)

        total_reward = reward_batch.sum().to('cpu').item()

        previous_rewards = np.loadtxt(os.path.join(game.name, "cumulative_reward.txt"))
        np.savetxt( os.path.join(game.name, "cumulative_reward.txt"),
                    np.append(previous_rewards, total_reward), fmt='%.1f')
        print("Cumulative reward at {} iteration of {}: {}".format(iteration, game.name, total_reward))

        # Calculate the Advantage Function by using GAE
        advantage_batch = calculateGAE(reward_batch, value_batch, done_batch, device, gamma, lambda_)

        value_batch = value_batch[0:-1]
        return_batch = advantage_batch + value_batch
        # Training
        for epoch in range(num_of_epochs):
            new_log_prob_batch, new_entropy_batch, new_value_batch = policy.evaluate(obs_batch.detach(), action_batch.detach())
            new_value_batch = new_value_batch.reshape((-1))

            ratio = (new_log_prob_batch - log_prob_batch).exp()
            loss1 = ratio * advantage_batch
            loss2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantage_batch

            actor_loss  = - torch.min(loss1, loss2).mean()
            critic_loss = (return_batch - new_value_batch).pow(2).mean()

            loss = (0.5 * critic_loss + actor_loss - 0.001 * entropy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        torch.save(policy.state_dict(), os.path.join("Parameters", "parameters.pt"))

        # Test policy
        if iteration % 100 == 0 and iteration != 0:
            print("Testing policy")
            observation = game.reset()
            tested_reward = 0.0
            done = False
            test_counter = 0

            while done == False and test_counter < 5000:
                # Get action
                obs = convertToTensor(observation, device = device)
                action, log_prob, entropy, value = policy.get_action(obs)
                observation, reward, done, info = game.step(game.legal_actions[action])
                cv2.imshow("Game", game.ale.getScreenRGB())
                if cv2.waitKey(10) & ord('q') == 0xFF:
                    break
                tested_reward += reward
                test_counter += 1
            print("Total tested reward: ", tested_reward)
            cv2.destroyAllWindows()
        # Clear
        del obs_batch
        del action_batch
        del log_prob_batch
        del entropy_batch
        del reward_batch
        del value_batch
        del done_batch
        del new_log_prob_batch
        del new_entropy_batch
        del new_value_batch
        del advantage_batch
        del return_batch