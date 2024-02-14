from typing import List, Dict, Tuple

import numpy as np
from gymnasium.spaces import Discrete


class TrafficAgent:
    def __init__(
        self, routes: List[List[int]], edges: Dict[Tuple[int, int], int], seed: int = 42
    ) -> None:
        # A route is a list of nodes
        self.routes = routes
        self.edges = edges

        self.action_space = Discrete(len(self.routes))

        self.rng = np.random.default_rng(seed)

    def act(self, observation: Dict) -> int:
        # print(f'{observation=}')
        # Destructure observation, either with or without restriction
        if "restriction" in observation:
            observation, restriction = (
                observation["observation"],
                observation["restriction"],
            )
        else:
            restriction = None

        position, target, edge_travel_times = (
            observation["position"],
            observation["target"],
            observation["travel_times"],
        )

        # Calculate travel times for all routes, with np.inf for infeasible routes
        travel_times = np.array(
            [
                sum(edge_travel_times[edge] for edge in self._to_edges(route))
                if (restriction is None or restriction.contains(i))
                and (route[0] == position and route[-1] == target)
                else np.inf
                for i, route in enumerate(self.routes)
            ]
        )

        # print(f"{travel_times=}")

        # Since only valid restrictions can be chosen by the governance, there is
        # always a path
        assert travel_times.min() < np.inf

        # Select routes with minimal travel time
        optimal_routes = np.argmin(travel_times, keepdims=True)
        # print(f"{optimal_routes=}")

        # Choose a route randomly from optimal routes
        return self.rng.choice(optimal_routes)

    def _to_edges(self, route: List[int]) -> List[int]:
        return [self.edges[(v, w)] for v, w in zip(route[:-1], route[1:])]
