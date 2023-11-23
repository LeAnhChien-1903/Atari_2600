import torch
import numpy as np

def calculateGAE(reward_batch: torch.Tensor, value_batch: torch.Tensor, done_batch: torch.Tensor,
                device, gamma: float = 0.99, lambda_: float= 0.95):
    batch_size = reward_batch.shape[0]
    advantage_batch = torch.zeros(batch_size).to(device)

    last_advantage = torch.zeros(1).to(device)
    last_value = value_batch[-1]

    for t in reversed(range(batch_size)):
        mask = 1.0 - done_batch[t]
        last_value = last_value * mask
        last_advantage = last_advantage * mask

        delta = reward_batch[t] + gamma * last_value - value_batch[t]

        last_advantage = delta + gamma * lambda_ * last_advantage

        advantage_batch[t] = last_advantage

        last_value = value_batch[t]


    return advantage_batch

def convertToTensor(data: np.ndarray, device):
    return torch.tensor(data, dtype= torch.float32, device= device)/255