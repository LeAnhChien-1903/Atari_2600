import torch
import torch.nn as nn
from torch.distributions import Categorical


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class ActorCritic(nn.Module):
    def __init__(self, num_input_channels = 4, num_of_actions = 18):
        super(ActorCritic, self).__init__()
        self.num_of_actions_ = num_of_actions
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels= num_input_channels, out_channels= 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= 4, stride= 2),
            nn.ReLU(),

            nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 3, stride= 1),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(in_features= 7*7*64, out_features= 512),
            nn.ReLU(),

            nn.Linear(in_features= 512, out_features= num_of_actions),

            nn.Softmax(dim = 1)
        )
        self.critic = nn.Sequential(
            nn.Conv2d(in_channels= num_input_channels, out_channels= 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= 4, stride= 2),
            nn.ReLU(),

            nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 3, stride= 1),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(in_features= 7*7*64, out_features= 512),
            nn.ReLU(),

            nn.Linear(in_features= 512, out_features= 1),
        )
        self.apply(init_weights)

    def get_action(self, state: torch.Tensor):
        if len(state.shape) == 3:
            state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
        actor_probs: torch.Tensor = self.actor(state)
        state_value = self.critic(state)
        distribution = Categorical(probs = actor_probs)
        action = distribution.sample()

        action_log_prob: torch.Tensor = distribution.log_prob(action)
        return action, action_log_prob, distribution.entropy(), state_value


    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        if len(state.shape) == 3:
            state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
            action = action.reshape(1, action.shape[0])

        actor_probs: torch.Tensor = self.actor(state)
        state_value = self.critic(state)
        distribution = Categorical(actor_probs)
        action_log_prob: torch.Tensor = distribution.log_prob(action)

        return action_log_prob, distribution.entropy(),  state_value

    def exploit_policy(self, state: torch.Tensor):
        if len(state.shape) == 3:
            state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
        actor_probs: torch.Tensor = self.actor(state)

        return actor_probs.argmax(dim=1)
    def get_value(self, state: torch.Tensor):
        if len(state.shape) == 3:
            state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
        return self.critic(state)