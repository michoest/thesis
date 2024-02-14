import numpy as np

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class CustomMetricsLogger(DefaultCallbacks):
    def __init__(self):
        super().__init__()

        # Unknown until on_episode_start
        self.NUMBER_OF_AGENTS = self.NUMBER_OF_ACTIONS = (
            self.NUMBER_OF_STEPS_PER_EPISODE
        ) = self.ALPHA = None

    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        if self.NUMBER_OF_AGENTS is None:
            env = base_env.get_unwrapped()[0]

            self.NUMBER_OF_AGENTS = env.NUMBER_OF_AGENTS
            self.NUMBER_OF_ACTIONS = env.NUMBER_OF_ACTIONS
            self.NUMBER_OF_STEPS_PER_EPISODE = env.NUMBER_OF_STEPS_PER_EPISODE
            self.ALPHA = env.ALPHA

        episode.user_data["gov_info"] = []

    def on_episode_step(
        self, *, worker, base_env, policies=None, episode, env_index, **kwargs
    ):
        env = base_env.get_unwrapped()[0]

        if env.is_reward_step:
            episode.user_data["gov_info"].append(np.array(episode.last_info_for("gov")))

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        gov_info = np.vstack(episode.user_data["gov_info"])
        (
            episode_state_reward,
            episode_restriction_reward,
            episode_degree_of_restriction,
        ) = np.mean(gov_info, axis=0)

        episode.custom_metrics["episode_degree_of_restriction/gov"] = (
            episode_degree_of_restriction
        )
        episode.custom_metrics["episode_state_reward/gov"] = episode_state_reward
        episode.custom_metrics["episode_restriction_reward/gov"] = (
            episode_restriction_reward
        )
        episode.custom_metrics["episode_reward/gov"] = (
            episode_state_reward + episode_restriction_reward
        )

    def on_postprocess_trajectory(
        self,
        *,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs
    ):
        if agent_id == "gov":
            rewards = postprocessed_batch["rewards"]
            assert len(rewards.shape) == 1

            # Distribute rewards over all gov_actions in one environment step, not just
            # the last one
            for i in range(len(rewards)):
                if rewards[i] != 0:
                    rewards[max(0, i - self.NUMBER_OF_AGENTS + 1):(i + 1)] = (
                        rewards[i] / self.NUMBER_OF_AGENTS
                    )
