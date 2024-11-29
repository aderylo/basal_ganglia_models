from dataclasses import dataclass
import logging
import time
from tianshou.utils.net.common import Net
import torch
import torch.nn.functional as F
from tianshou.data import Batch
from tianshou.policy import BasePolicy, TrainingStats
from typing import Any, Dict, Optional, Tuple, cast
from collections.abc import Callable
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as, to_torch


from typing import Any, Generic, Literal, TypeVar, cast
from tianshou.data.batch import BatchProtocol

import gymnasium as gym
import numpy as np
import torch

from tianshou.data.types import (
    BatchWithReturnsProtocol,
    DistBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)


from hybrid_buffer import HybridReplayBuffer

from tianshou.utils.torch_utils import policy_within_training_step, torch_train_mode



from tianshou.policy import BasePolicy
from tianshou.policy.base import TLearningRateScheduler, TrainingStats
from tianshou.utils import RunningMeanStd
from tianshou.utils.net.continuous import ActorProb
from tianshou.utils.net.discrete import Actor

# Dimension Naming Convention
# B - Batch Size
# A - Action
# D - Dist input (usually 2, loc and scale)
# H - Dimension of hidden, can be None
TDistFnContinuous = Callable[
    [tuple[torch.Tensor, torch.Tensor]],
    torch.distributions.Distribution,
]
TDistFnDiscrete = Callable[[torch.Tensor], torch.distributions.Categorical]
TDistFnDiscrOrCont = TDistFnContinuous | TDistFnDiscrete

@dataclass
class DQNAttr:
    model: torch.nn.Module | Net
    optim: torch.optim.Optimizer
    action_space: gym.spaces.Discrete
    discount_factor: float = 0.99
    estimation_step: int = 1
    target_update_freq: int = 0
    reward_normalization: bool = False
    is_double: bool = True
    clip_loss_grad: bool = False
    observation_space: gym.Space | None = None
    lr_scheduler: TLearningRateScheduler | None = None

@dataclass
class PGAttr:
    actor: torch.nn.Module | ActorProb | Actor
    optim: torch.optim.Optimizer
    dist_fn: TDistFnDiscrOrCont
    action_space: gym.Space
    discount_factor: float = 0.99
    # TODO: rename to return_normalization?
    reward_normalization: bool = False
    deterministic_eval: bool = False
    observation_space: gym.Space | None = None
    # TODO: why change the default from the base?
    action_scaling: bool = True
    action_bound_method: Literal["clip", "tanh"] | None = "clip"
    lr_scheduler: TLearningRateScheduler | None = None

class MBDPolicy(BasePolicy):
    def __init__(
        self,
        *,
        action_space: gym.spaces.Discrete,
        observation_space: gym.Space | None = None,
        dqn_atrr: DQNAttr,
        pg_atrr: PGAttr,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=False,
            action_bound_method=None,
            lr_scheduler=None,
        )

    def _error_if_not_within_training_step(self):
        if not self.is_within_training_step:
            raise RuntimeError(
                f"update() was called outside of a training step as signalled by {self.is_within_training_step=} "
                f"If you want to update the policy without a Trainer, you will have to manage the above-mentioned "
                f"flag yourself. You can to this e.g., by using the contextmanager {policy_within_training_step.__name__}.",
            )

    def hybrid_update(
        self,
        sample_size: int | None,
        buffer: HybridReplayBuffer | None,
        **kwargs: Any,
    ) -> TrainingStats:
        """Update the policy network and replay buffer.

        It includes 3 function steps: process_fn, learn, and post_process_fn. In
        addition, this function will change the value of ``self.updating``: it will be
        False before this function and will be True when executing :meth:`update`.
        Please refer to :ref:`policy_state` for more detailed explanation. The return
        value of learn is augmented with the training time within update, while smoothed
        loss values are computed in the trainer.

        :param sample_size: 0 means it will extract all the data from the buffer,
            otherwise it will sample a batch with given sample_size. None also
            means it will extract all the data from the buffer, but it will be shuffled
            first. TODO: remove the option for 0?
        :param buffer: the corresponding replay buffer.

        :return: A dataclass object containing the data needed to be logged (e.g., loss) from
            ``policy.learn()``.
        """
        self._error_if_not_within_training_step()

        if buffer is None:
            return TrainingStats()  # type: ignore[return-value]

        start_time = time.time()
        offpolicy_batch, offpolicy_indices = buffer.sample(sample_size, on_policy_only=False)
        onpolicy_batch, onpolicy_indices = buffer.sample(sample_size, on_policy_only=True)

        self.updating = True
        offpolicy_batch = self.offpolicy_process_fn(offpolicy_batch, buffer, offpolicy_indices)
        onpolicy_batch = self.onpolicy_process_fn(onpolicy_batch, buffer, onpolicy_indices)

        with torch_train_mode(self):
            training_stat = self.learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indices)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.updating = False
        training_stat.train_time = time.time() - start_time
        return training_stat


    def offpolicy_process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: HybridReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithReturnsProtocol:
        """Compute the n-step return for Q-learning targets.

        More details can be found at
        :meth:`~tianshou.policy.BasePolicy.compute_nstep_return`.
        """
        return self.compute_nstep_return(
            batch=batch,
            buffer=buffer,
            indices=indices,
            target_q_fn=self._target_q,
            gamma=self.gamma,
            n_step=self.n_step,
            rew_norm=self.rew_norm,
        )

    def onpolicy_process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: HybridReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithReturnsProtocol:
        r"""Compute the discounted returns (Monte Carlo estimates) for each transition.

        They are added to the batch under the field `returns`.
        Note: this function will modify the input batch!

        .. math::
            G_t = \sum_{i=t}^T \gamma^{i-t}r_i

        where :math:`T` is the terminal time step, :math:`\gamma` is the
        discount factor, :math:`\gamma \in [0, 1]`.

        :param batch: a data batch which contains several episodes of data in
            sequential order. Mind that the end of each finished episode of batch
            should be marked by done flag, unfinished (or collecting) episodes will be
            recognized by buffer.unfinished_index().
        :param buffer: the corresponding replay buffer.
        :param numpy.ndarray indices: tell batch's location in buffer, batch is equal
            to buffer[indices].
        """
        v_s_ = np.full(indices.shape, self.ret_rms.mean)
        # gae_lambda = 1.0 means we use Monte Carlo estimate
        unnormalized_returns, _ = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_=v_s_,
            gamma=self.gamma,
            gae_lambda=1.0,
        )
        # TODO: overridden in A2C, where mean is not subtracted. Subtracting mean
        #  can be very detrimental! It also has no theoretical grounding.
        #  This should be addressed soon!
        if self.rew_norm:
            batch.returns = (unnormalized_returns - self.ret_rms.mean) / np.sqrt(
                self.ret_rms.var + self._eps,
            )
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns

        batch: BatchWithReturnsProtocol
        return batch

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any
    ) -> DistBatchProtocol:
        # Forward pass through the Q-Network and Temperature Network
        q_values_BA = self.q_network(batch.obs)
        temperature_B = self.temperature_network(batch.obs)

        # Compute the Boltzmann policy probabilities
        action_probs_BA = F.softmax(q_values_BA / temperature_B.detach(), dim=-1)

        # Create the Categorical distribution
        dist = torch.distributions.Categorical(action_probs_BA)

        # Sample actions from the Boltzmann policy
        actions_B = dist.sample()

        result = Batch(logits=q_values_BA, act=actions_B, state=temperature_B, dist=dist)
        return cast(DistBatchProtocol, result)


    def learn(
        self, 
        offpolicy_batch: RolloutBatchProtocol,
        onpolicy_batch: BatchWithReturnsProtocol,
        *args: Any, 
        **kwargs: Any
    ) -> Dict[str, float]:
        self.optim_dqn.zero_grad()

        # Update the Q-Network using standard DQN loss
        weight = offpolicy_batch.pop("weight", 1.0)
        q = self(offpolicy_batch).logits
        q = q[torch.arange(len(q)), offpolicy_batch.act]
        returns = to_torch_as(offpolicy_batch.returns.flatten(), q)
        td_error = returns - q
        dqn_loss = (td_error.pow(2) * weight).mean()
        dqn_loss.backward()
        self.optim_dqn.step()

        # Update the Temperature Network using policy gradient loss
        self.optim_pg.zero_grad()
        result = self(onpolicy_batch)
        dist = result.dist
        act = to_torch_as(onpolicy_batch.act, result.act)
        ret = to_torch(onpolicy_batch.returns, torch.float, result.act.device)
        log_prob = dist.log_prob(act).reshape(len(ret), -1).transpose(0, 1)
        pg_loss = -(log_prob * ret).mean()
        pg_loss.backward()
        self.optim_pg.step()

        self._iter += 1
        return {'dqn_loss': dqn_loss.item(), 'pg_loss': pg_loss.item()}

