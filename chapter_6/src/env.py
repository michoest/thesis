import numpy as np

from gymnasium.spaces import Discrete, Box, Dict, MultiBinary
from ray.rllib.env import MultiAgentEnv


class FMAS_Environment(MultiAgentEnv):
    """
    Governance and agent reward is only 1 if all actions coincide.
    """

    def __init__(self, env_config: dict = {}):
        assert "NUMBER_OF_ACTIONS" in env_config
        assert "NUMBER_OF_AGENTS" in env_config
        assert "NUMBER_OF_STEPS_PER_EPISODE" in env_config
        assert "ALPHA" in env_config

        self.NUMBER_OF_ACTIONS = env_config["NUMBER_OF_ACTIONS"]
        self.NUMBER_OF_STEPS_PER_EPISODE = env_config["NUMBER_OF_STEPS_PER_EPISODE"]
        self.ALPHA = env_config["ALPHA"]
        self.NUMBER_OF_AGENTS = env_config["NUMBER_OF_AGENTS"]
        self.agents = [str(i) for i in range(self.NUMBER_OF_AGENTS)]

        self.agent_observations = {
            agent: np.array(
                np.mod([index - 1, index, index + 1], self.NUMBER_OF_AGENTS)
            )
            for index, agent in enumerate(self.agents)
        }

        # Define obs and action spaces for agents
        self.observation_space = Dict(
            {
                "obs": Box(0, self.NUMBER_OF_ACTIONS - 1, shape=(3,)),
                "allowed_actions": MultiBinary(self.NUMBER_OF_ACTIONS),
            }
        )
        self.action_space = Discrete(self.NUMBER_OF_ACTIONS)

        self.state = None
        self.current_step = None
        self.current_agent_index = None

        self.allowed_actions = None

        self.is_reward_step = None

    def step(self, actions):
        self.is_reward_step = False
        if "gov" in actions:
            # Only the governance has acted
            assert len(actions) == 1

            # Governance action is a set of allowed actions; save it for later
            self.allowed_actions[self.agents[self.current_agent_index]] = actions[
                "gov"
            ].astype(bool)
            self.current_agent_index += 1
        else:
            # All agents have acted
            assert len(actions) == len(self.agents)

            # Execute transition
            self.state = np.array([actions[agent] for agent in self.agents])

            state_reward = 1 if np.all(self.state == self.state[0]) else 0
            degree_of_restriction = np.mean(
                [
                    1 - (np.sum(allowed_actions) / self.NUMBER_OF_ACTIONS)
                    for allowed_actions in self.allowed_actions.values()
                ]
            )
            restriction_reward = -self.ALPHA * degree_of_restriction
            gov_reward = state_reward + restriction_reward

            self.current_step += 1
            self.current_agent_index = 0
            self.allowed_actions = {}  # Could be removed for speed-up

            # Governance needs to decide on allowed actions for first agent on the list
            # (if episode is not over)
            self.is_reward_step = True
            return (
                {
                    "gov": {
                        "state": self.state,
                        "obs": self.get_observation(
                            self.agents[self.current_agent_index]
                        ),
                    }
                },
                {"gov": gov_reward},
                {"__all__": self.current_step >= self.NUMBER_OF_STEPS_PER_EPISODE},
                {"gov": (state_reward, restriction_reward, degree_of_restriction)},
            )

        # Governance needs to decide on allowed actions for next agent on the list
        if self.current_agent_index < len(self.agents):
            return (
                {
                    "gov": {
                        "state": self.state,
                        "obs": self.get_observation(
                            self.agents[self.current_agent_index]
                        ),
                    }
                },
                {"gov": 0},
                {"__all__": False},
                {"gov": {}},
            )

        # Agents need to act
        else:
            agent_reward = 1 if np.all(self.state == self.state[0]) else 0

            return (
                {
                    agent: {
                        "obs": self.get_observation(agent),
                        "allowed_actions": self.allowed_actions[agent],
                    }
                    for agent in self.agents
                },
                {agent: agent_reward for agent in self.agents},
                {"__all__": False},
                {},
            )

    def reset(self):
        self.current_step = 0
        self.current_agent_index = 0

        self.state = np.random.randint(0, self.NUMBER_OF_ACTIONS, (len(self.agents),))

        self.allowed_actions = {}

        return {
            "gov": {
                "state": self.state,
                "obs": self.get_observation(self.agents[self.current_agent_index]),
            }
        }

    def get_observation(self, agent: str):
        return self.state[self.agent_observations[agent]]


class GMAS_Environment(MultiAgentEnv):
    """
    Governance reward is only 1 if all actions coincide.
    Agents get their rewards if their observations coincide.
    """

    def __init__(self, env_config: dict = {}):
        assert "NUMBER_OF_ACTIONS" in env_config
        assert "NUMBER_OF_AGENTS" in env_config
        assert "NUMBER_OF_STEPS_PER_EPISODE" in env_config
        assert "ALPHA" in env_config

        self.NUMBER_OF_ACTIONS = env_config["NUMBER_OF_ACTIONS"]
        self.NUMBER_OF_STEPS_PER_EPISODE = env_config["NUMBER_OF_STEPS_PER_EPISODE"]
        self.ALPHA = env_config["ALPHA"]
        self.NUMBER_OF_AGENTS = env_config["NUMBER_OF_AGENTS"]
        self.agents = [str(i) for i in range(self.NUMBER_OF_AGENTS)]

        self.agent_observations = {
            agent: np.array(
                np.mod([index - 1, index, index + 1], self.NUMBER_OF_AGENTS)
            )
            for index, agent in enumerate(self.agents)
        }

        # Define obs and action spaces for agents
        self.observation_space = Dict(
            {
                "obs": Box(0, self.NUMBER_OF_ACTIONS - 1, shape=(3,)),
                "allowed_actions": MultiBinary(self.NUMBER_OF_ACTIONS),
            }
        )
        self.action_space = Discrete(self.NUMBER_OF_ACTIONS)

        self.state = None
        self.current_step = None
        self.current_agent_index = None

        self.allowed_actions = None

        self.is_reward_step = None

    def step(self, actions):
        self.is_reward_step = False
        if "gov" in actions:
            # Only the governance has acted
            assert len(actions) == 1

            # Governance action is a set of allowed actions; save it for later
            self.allowed_actions[self.agents[self.current_agent_index]] = actions[
                "gov"
            ].astype(bool)
            self.current_agent_index += 1
        else:
            # All agents have acted
            assert len(actions) == len(self.agents)

            # Execute transition
            self.state = np.array([actions[agent] for agent in self.agents])

            state_reward = 1 if np.all(self.state == self.state[0]) else 0
            degree_of_restriction = np.mean(
                [
                    1 - (np.sum(allowed_actions) / self.NUMBER_OF_ACTIONS)
                    for allowed_actions in self.allowed_actions.values()
                ]
            )
            restriction_reward = -self.ALPHA * degree_of_restriction
            gov_reward = state_reward + restriction_reward

            self.current_step += 1
            self.current_agent_index = 0
            self.allowed_actions = {}  # Could be removed for speed-up

            # Governance needs to decide on allowed actions for first agent on the list
            # (if episode is not over)
            self.is_reward_step = True
            return (
                {
                    "gov": {
                        "state": self.state,
                        "obs": self.get_observation(
                            self.agents[self.current_agent_index]
                        ),
                    }
                },
                {"gov": gov_reward},
                {"__all__": self.current_step >= self.NUMBER_OF_STEPS_PER_EPISODE},
                {"gov": (state_reward, restriction_reward, degree_of_restriction)},
            )

        # Governance needs to decide on allowed actions for next agent on the list
        if self.current_agent_index < len(self.agents):
            return (
                {
                    "gov": {
                        "state": self.state,
                        "obs": self.get_observation(
                            self.agents[self.current_agent_index]
                        ),
                    }
                },
                {"gov": 0},
                {"__all__": False},
                {"gov": {}},
            )

        # Agents need to act
        else:
            agent_observations = {
                agent: self.state[self.agent_observations[agent]]
                for agent in self.agents
            }
            agent_rewards = {
                agent: 1 if np.all(obs == obs[0]) else 0
                for agent, obs in agent_observations.items()
            }

            return (
                {
                    agent: {
                        "obs": self.get_observation(agent),
                        "allowed_actions": self.allowed_actions[agent],
                    }
                    for agent in self.agents
                },
                {agent: agent_rewards[agent] for agent in self.agents},
                {"__all__": False},
                {},
            )

    def reset(self):
        self.current_step = 0
        self.current_agent_index = 0

        self.state = np.random.randint(0, self.NUMBER_OF_ACTIONS, (len(self.agents),))

        self.allowed_actions = {}

        return {
            "gov": {
                "state": self.state,
                "obs": self.get_observation(self.agents[self.current_agent_index]),
            }
        }

    def get_observation(self, agent: str):
        return self.state[self.agent_observations[agent]]
