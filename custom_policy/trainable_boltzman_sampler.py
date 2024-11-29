from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from tianshou.utils.net.common import MLP, Net


class TrainableBoltzmanSampler(nn.Module):
    """
    Boltzman Sampler with a trainable temperature parameter.
    """

    def __init__(
        self,
        feature_net: nn.Module | Net,
        action_shape: Sequence[int] | int,
        hidden_sizes: Sequence[int] = (),
        device: str | int | torch.device = "cpu",
    ):
        super().__init__()
        self.device = device
        self.feature_net = feature_net
        self.action_shape = action_shape
        input_dim = getattr(feature_net, "output_dim")
        self.temp_net = MLP(
            input_dim,
            1, # temperature is a scalar 
            hidden_sizes,
            device=self.device,
        )

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Mapping: s_B -> action_values_BA, hidden_state_BH | None.

        :param obs:
        :param state: q_values from the dqn 
        :param info: unused

        Returns a tensor representing the values of each action, i.e, of shape
        `(n_actions, )`, and
        a hidden state (which may be None). If `self.softmax_output` is True, they are the
        probabilities for taking each action. Otherwise, they will be action values.
        The hidden state is only
        not None if a recurrent net is used as part of the learning algorithm.
        """
        assert(state.shape[-1] == self.action_shape)
        features, hidden_BH = self.feature_net(obs)
        temperature = self.temp_net(features)

        # Here state are the q_values from seperate DQN
        scaled_q_values = state / temperature 
        action_dist = torch.softmax(scaled_q_values, dim=-1)

        output_BA = action_dist
        return output_BA, hidden_BH
