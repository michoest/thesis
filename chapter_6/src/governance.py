import numpy as np

from gymnasium.spaces import MultiDiscrete
from ray.rllib.policy.policy import Policy


class PassiveGovernancePolicy(Policy):
    """
    Always allows all actions
    """

    def __init__(self, observation_space, action_space, config={}):
        super().__init__(observation_space, action_space, config)

        assert isinstance(
            action_space, MultiDiscrete
        ), f"action_space is not MultiDiscrete ({type(action_space)})"
        self.NUMBER_OF_ACTIONS = len(action_space.nvec)

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs,
    ):

        return (
            [np.ones(self.NUMBER_OF_ACTIONS).astype(bool) for _ in obs_batch],
            state_batches,
            {},
        )

    def get_weights(self):
        return {}

    def set_weights(self, weights) -> None:
        pass
