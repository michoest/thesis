from typing import List
from collections import Counter

import numpy as np

from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from examples.traffic_new.utils import latency


class TrafficEnvironment(AECEnv):
    metadata = {"render_modes": ["human"], "name": "traffic_v1"}

    def __init__(
        self,
        graph,
        agents,
        possible_start_and_target_nodes,
        routes,
        number_of_steps,
        seed: int = 42,
    ):
        super().__init__()

        self.possible_agents = agents
        self.possible_start_and_target_nodes = possible_start_and_target_nodes
        self.routes = routes
        self.edges = {edge: i for i, edge in enumerate(graph.edges)}
        self.edge_latencies = {
            i: data["latency"] for i, [_, _, data] in enumerate(graph.edges(data=True))
        }

        self._number_of_nodes = graph.number_of_nodes()
        self._number_of_edges = graph.number_of_edges()
        self._number_of_routes = len(routes)

        self._observation_space = spaces.Dict(
            {
                "position": spaces.Discrete(self._number_of_nodes),
                "target": spaces.Discrete(self._number_of_nodes),
                "travel_times": spaces.Box(0, np.inf, shape=(self._number_of_edges,)),
            }
        )
        self._action_space = spaces.Discrete(self._number_of_routes)

        self.number_of_steps = number_of_steps
        self.render_mode = "human"
        self._actions = None

        self.rng = np.random.default_rng(seed)

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def observe(self, agent):
        position, target = self._properties[agent]

        agents_per_edge = Counter(
            edge
            for _agent, route in self._routes_taken.items()
            if _agent != agent
            for edge in self._to_edges(route)
        )
        travel_times = {
            edge: latency(
                self.edge_latencies[edge], agents_per_edge[edge] / self.max_num_agents
            )
            for edge in range(self._number_of_edges)
        }

        return {
            "position": position,
            "target": target,
            "travel_times": np.array(list(travel_times.values())),
        }

    def close(self):
        pass

    def reset(self, seed=42, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._properties = {
            agent: self.rng.choice(self.possible_start_and_target_nodes)
            for agent in self.agents
        }
        self._actions = {agent: None for agent in self.agents}
        self._routes_taken = {agent: tuple() for agent in self.agents}
        self._travel_times = {
            edge: latency(self.edge_latencies[edge], 0)
            for edge in range(self._number_of_edges)
        }
        self._state = {agent: None for agent in self.agents}
        self.current_step = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        self._actions[self.agent_selection] = action
        self._routes_taken[self.agent_selection] = self.routes[action]

        agents_per_edge = Counter(
            edge
            for route in self._routes_taken.values()
            for edge in self._to_edges(route)
        )

        self._travel_times = {
            edge: latency(
                self.edge_latencies[edge], agents_per_edge[edge] / self.max_num_agents
            )
            for edge in range(self._number_of_edges)
        }

        self.rewards = {
            agent: -sum(
                self._travel_times[edge]
                for edge in self._to_edges(self._routes_taken[agent])
            )
            for agent in self.agents
        }
        self._cumulative_rewards = self.rewards.copy()

        self.truncations = {
            agent: self.current_step >= self.number_of_steps for agent in self.agents
        }

        self.agent_selection = self._agent_selector.next()
        self.current_step += 1

    def state(self):
        return np.array(list(self._travel_times.values()))

    def _to_edges(self, route: List[int]) -> List[int]:
        return [self.edges[(v, w)] for v, w in zip(route[:-1], route[1:])]
