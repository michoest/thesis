import numpy as np
import sklearn.preprocessing
import functools
import itertools
import random

from src.utils import state_to_number, delta


class Governance:
    def __init__(self, gamma, _lambda):
        self.gamma = lambda state: gamma(self, state)
        self._lambda = lambda old_state, chosen_actions, new_state: _lambda(
            self, old_state, chosen_actions, new_state
        )

    def reset(self):
        pass

    def restrict_actions(self, state, fundamental_actions):
        return self.gamma(state, fundamental_actions)

    def learn(self, old_state, chosen_actions, new_state):
        return self._lambda(old_state, chosen_actions, new_state)


class EUMASGovernance(Governance):
    def __init__(
        self, number_of_agents, number_of_variables, cost_table, cost_threshold
    ):
        # Internal state of governance: Number of occurrences of action per agent per
        # state
        self.params = (
            number_of_agents,
            number_of_variables,
            cost_table,
            cost_threshold,
        )
        self.state = np.zeros(
            (number_of_agents, 2**number_of_variables, number_of_variables + 1)
        )

        self.cost_table = cost_table
        self.cost_threshold = cost_threshold

        self.cost = lambda state: self.cost_table[state_to_number(state)]

        def gamma(state, fundamental_actions):
            # Extract current state from state table
            state_table = np.copy(self.state[:, state_to_number(state)])

            # Normalize rows
            state_table = sklearn.preprocessing.normalize(
                state_table, axis=1, norm="l1"
            )

            # Assume uniform distribution if no observations are available
            m = np.all(state_table == 0, axis=1)
            state_table[m] = np.ones(number_of_variables + 1) / (
                number_of_variables + 1
            )

            # Make n-dimensional matrix out of state table
            state_matrix = functools.reduce(np.multiply.outer, state_table)

            # Build n-dimensional cost matrix
            cost_table = list(
                map(
                    lambda joint_action: self.cost_table[
                        state_to_number(delta(state, list(joint_action)))
                    ],
                    itertools.product(*fundamental_actions),
                )
            )
            cost_matrix = np.array(cost_table).reshape(
                tuple([number_of_variables + 1] * number_of_agents)
            )

            # Compute n-dimensional expected cost matrix
            expected_cost_matrix = np.multiply(state_matrix, cost_matrix)

            # Initialize action sets
            allowed_actions = fundamental_actions.copy()

            # Check if cost of neutral action is above threshold; if so, increase alpha
            # accordingly
            alpha = (
                self.cost_threshold
                if self.cost_threshold
                >= expected_cost_matrix[tuple([0] * number_of_agents)]
                else 1.5 * expected_cost_matrix[tuple([0] * number_of_agents)]
            )

            # Reduce expected cost matrix until cost threshold is crossed
            while np.sum(np.abs(expected_cost_matrix)) > alpha:
                # Find all actions with maximum cost reduction
                action_cost_reduction = []
                for agent in range(number_of_agents):
                    swapped_cost_matrix = np.copy(expected_cost_matrix).swapaxes(
                        0, agent
                    )
                    for action_index in range(len(allowed_actions[agent])):
                        action_cost_reduction.append(
                            [
                                agent,
                                action_index,
                                np.sum(np.abs(swapped_cost_matrix[action_index])),
                            ]
                        )

                action_cost_reduction_matrix = np.array(action_cost_reduction)

                max_cost = np.amax(action_cost_reduction_matrix[:, 2])

                max_action_cost_reduction_matrix = action_cost_reduction_matrix[
                    action_cost_reduction_matrix[:, 2] == max_cost, 0:2
                ]

                optimal_actions = [
                    (int(action[0]), int(action[1]))
                    for action in max_action_cost_reduction_matrix
                ]

                if not optimal_actions:
                    raise RuntimeError("Could not find an action to remove!")
                else:
                    # Choose randomly between optimal actions
                    removed_agent, removed_action = random.choice(optimal_actions)

                    # Remove action from cost matrix and action set
                    expected_cost_matrix = np.delete(
                        expected_cost_matrix, removed_action, removed_agent
                    )
                    allowed_actions[removed_agent] = np.delete(
                        allowed_actions[removed_agent], removed_action
                    )

            return allowed_actions

        self.gamma = gamma

        def _lambda(old_state, chosen_actions, new_state):
            for agent, action in enumerate(chosen_actions):
                self.state[agent][state_to_number(old_state)][action + 1] += 1

        self._lambda = _lambda

    def reset(self):
        number_of_agents, number_of_variables, _, _ = self.params
        self.state = np.zeros(
            (number_of_agents, 2**number_of_variables, number_of_variables + 1)
        )


class PassiveGovernance(Governance):
    def __init__(self, number_of_agents, number_of_variables, cost_table):
        self.gamma = lambda state, fundamental_actions: fundamental_actions
        self._lambda = lambda old_state, chosen_actions, new_state: None
        self.cost = lambda state: cost_table[state_to_number(state)]

    def reset(self):
        pass
