import numpy as np
import torch
from torch import nn
from gymnasium import spaces

from drama.restrictions import DiscreteVectorRestriction, DiscreteSetRestriction
from drama.restrictors import Restrictor, RestrictorActionSpace
from drama.utils import flatdim, DiscreteVectorActionSpace, DiscreteSetActionSpace

from examples.utils import ReplayBuffer
from examples.traffic_new.utils import to_edges


class QNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, flatdim(action_space)),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class TrafficRestrictor(Restrictor):
    def __init__(
        self, edges, routes, valid_edge_restrictions, total_timesteps, seed=42
    ) -> None:
        self.rng = np.random.default_rng(seed)

        observation_space = spaces.Box(0, np.inf, (len(edges),))
        action_space = spaces.Discrete(len(valid_edge_restrictions))
        # action_space = DiscreteVectorActionSpace(spaces.Discrete(len(edges)))
        self.network_action_space = spaces.Discrete(len(valid_edge_restrictions))

        super().__init__(observation_space, action_space)

        self.edges = edges
        self.routes = routes
        self.valid_edge_restrictions = valid_edge_restrictions
        self.start_e, self.end_e = 1.0, 0.01
        self.exploration_fraction = 0.5
        self.total_timesteps = total_timesteps
        self.learning_rate = 2.5e-4
        self.batch_size = 128
        self.gamma = 0.99
        self.target_network_frequency = 500
        self.tau = 1
        self.training_start = 10_000
        self.training_frequency = 10
        self.cuda = True

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cuda else "cpu"
        )

        self.q_network = QNetwork(observation_space, self.network_action_space).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.learning_rate
        )
        self.target_network = QNetwork(observation_space, self.network_action_space).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.replay_buffer = ReplayBuffer(
            state_dim=flatdim(observation_space), action_dim=1
        )

        self.global_step = 0

    def act(self, observation: spaces.Space) -> RestrictorActionSpace:
        epsilon = linear_schedule(
            self.start_e,
            self.end_e,
            self.exploration_fraction * self.total_timesteps,
            self.global_step,
        )
        if self.rng.random() < epsilon:
            network_action = self.network_action_space.sample()
        else:
            q_values = self.q_network(torch.Tensor(observation).to(self.device))
            network_action = torch.argmax(q_values).cpu().numpy()

        return network_action

        # return DiscreteVectorRestriction(
        #     base_space=spaces.Discrete(len(self.edges)),
        #     allowed_actions=self.valid_edge_restrictions[network_action],
        # )

    def learn(self, obs, action, next_obs, reward, done):
        self.record(obs, action, next_obs, reward, done)

        if self.global_step >= self.training_start:
            if (self.global_step - self.training_start) % self.training_frequency == 0:
                self.update_q_network()

            if (
                self.global_step - self.training_start
            ) % self.target_network_frequency == 0:
                self.update_target_network()

        self.global_step += 1

    def record(self, obs, action, next_obs, reward, done):
        self.replay_buffer.add(obs, action, next_obs, reward, done)

    def update_q_network(self):
        states, actions, next_states, rewards, not_dones = self.replay_buffer.sample(
            self.batch_size
        )

        actions = actions.type(torch.int64)
        rewards = rewards.flatten()
        not_dones = not_dones.flatten()

        # print(f'{states=}, {actions=}, {next_states=}, {rewards=}, {not_dones=}')

        # Get values from target network
        with torch.no_grad():
            target_max, _ = self.target_network(next_states).max(dim=1)

            # print(f'{target_max=}')

            td_target = rewards.flatten() + self.gamma * target_max * not_dones

        # Get values from q network
        # print(f'{self.q_network(states)=}')
        old_val = self.q_network(states).gather(1, actions).squeeze()

        # Compute loss
        loss = nn.functional.mse_loss(td_target, old_val)

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        for target_network_param, q_network_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_network_param.data.copy_(
                self.tau * q_network_param.data
                + (1.0 - self.tau) * target_network_param.data
            )

    def postprocess_restriction(self, restriction):
        edge_restriction = DiscreteVectorRestriction(base_space=spaces.Discrete(len(self.edges)), allowed_actions=self.valid_edge_restrictions[restriction])
        allowed_routes = [
            all(edge_restriction.contains(edge) for edge in to_edges(route, self.edges))
            for route in self.routes
        ]

        return DiscreteVectorRestriction(
            base_space=spaces.Discrete(len(self.routes)), allowed_actions=allowed_routes
        )
