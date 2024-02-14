import tensorflow as tf
from gymnasium.spaces import Discrete, Box

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork


class ParametricAgentModel(TFModelV2):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, *args, **kwargs
    ):
        super(ParametricAgentModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, *args, **kwargs
        )

        assert isinstance(
            action_space, Discrete
        ), f"action_space is a {type(action_space)}, but should be Discrete!"

        true_obs_shape = (3,)
        action_embed_size = action_space.n

        self.action_embed_model = FullyConnectedNetwork(
            Box(0, 1, shape=true_obs_shape),
            action_space,
            action_embed_size,
            model_config,
            name + "_action_embedding",
        )

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["allowed_actions"]
        action_embedding, _ = self.action_embed_model({"obs": input_dict["obs"]["obs"]})
        intent_vector = tf.expand_dims(action_embedding, 1)
        action_logits = tf.reduce_sum(intent_vector, axis=1)
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()
