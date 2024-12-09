from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast

import gymnasium as gym
import numpy as np
from tianshou.utils.net.common import Net
from torch import nn
import torch


from tianshou.policy import BasePolicy
from tianshou.policy.base import TrainingStats
from tianshou.policy.modelfree.dqn import DQNTrainingStats
from tianshou.policy.modelfree.pg import PGTrainingStats

from tianshou.data import ReplayBuffer, SequenceSummaryStats, Batch, to_torch, to_torch_as
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import RolloutBatchProtocol, DistBatchProtocol, ObsBatchProtocol

from tianshou.utils import RunningMeanStd

class BatchWithTwoReturnsProtocol(RolloutBatchProtocol, Protocol):
    """With added returns, usually computed with GAE."""
    nstep_returns: torch.Tensor | np.ndarray
    episodic_returns: torch.Tensor | np.ndarray


@dataclass(kw_only=True)
class MBDTrainingStats(TrainingStats):
    pg_loss: SequenceSummaryStats
    dqn_loss: float

TMBDTrainingStats = TypeVar("TMBDTrainingStats", bound=PGTrainingStats)

@dataclass
class DQNParams:
    discount_factor: float = 0.99
    estimation_step: int = 1
    target_update_freq: int = 0
    reward_normalization: bool = False
    is_double: bool = True
    clip_loss_grad: bool = False

class PGParams:
    discount_factor: float = 0.99

class MBDPolicy(BasePolicy[TMBDTrainingStats], Generic[TMBDTrainingStats]):
    def __init__(
        self,
        *,
        dqn_net: nn.Module | Net,
        dqn_optim: torch.optim.Optimizer,
        sampler_net: nn.Module | Net,
        sampler_optim: torch.optim.Optimizer,
        sampler_update_freq: int = 1,
        action_space: gym.spaces.Discrete,
        observation_space: gym.Space | None = None,
        dqn_params: DQNParams,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=False,
            action_bound_method=None,
            lr_scheduler=None,
        )

        self.dqn_net = dqn_net
        self.dqn_optim = dqn_optim
        self.sampler_net = sampler_net
        self.sampler_optim = sampler_optim
        self.sampler_update_freq = sampler_update_freq
        self.dqn_params = dqn_params

        self._target = dqn_params.target_update_freq > 0
        self._iter = 0
        if self._target:
            self.dqn_model_old = deepcopy(self.dqn_net)
            self.dqn_model_old.eval()

        self.ret_rms = RunningMeanStd()
        self.gamma = 0.99

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=[None] * len(indices),
        )  # obs_next: s_{t+n}
        result = self(obs_next_batch)
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self(obs_next_batch, model="model_old").logits
        else:
            target_q = result.logits
        if self.dqn_params.is_double:
            return target_q[np.arange(len(result.act)), result.act]
        # Nature DQN, over estimate
        return target_q.max(dim=1)[0]


    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithTwoReturnsProtocol:
        """Compute the n-step return for Q-learning targets.

        More details can be found at
        :meth:`~tianshou.policy.BasePolicy.compute_nstep_return`.
        """
        processed_batch = self.compute_nstep_return(
            batch=batch,
            buffer=buffer,
            indices=indices,
            target_q_fn=self._target_q,
            gamma=self.dqn_params.discount_factor,
            n_step=self.dqn_params.estimation_step,
            rew_norm=self.dqn_params.reward_normalization,
        )
        processed_batch.nstep_returns = processed_batch.returns

        v_s_ = np.full(indices.shape, self.ret_rms.mean)
        unnormalized_epi_returns, _ = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_=v_s_,
            gamma=self.gamma,
            gae_lambda=1.0,
        )
        processed_batch.episodic_returns = unnormalized_epi_returns
        return cast(BatchWithTwoReturnsProtocol, processed_batch)


    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> DistBatchProtocol:

        obs = batch.obs
        obs_next = obs.obs if hasattr(obs, "obs") else obs

        q_values_BA, hidden_BH = self.dqn_net(obs_next, state=state, info=batch.info)
        action_probs_BA, _ = self.sampler_net(obs, state=q_values_BA, info=batch.info)


        dist = torch.distributions.Categorical(action_probs_BA)
        act_B = dist.sample()

        result = Batch(logits=q_values_BA, act=act_B, state=hidden_BH, dist=dist)
        return result


    def _dqn_learn(
            self,
            batch: BatchWithTwoReturnsProtocol, 
            *args: Any,
            **kwargs: Any
    ) -> DQNTrainingStats:
        if self._target and self._iter % self.dqn_params.target_update_freq == 0:
            self.sync_weight()

        self.dqn_optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        q = self(batch).logits
        q = q[np.arange(len(q)), batch.act]
        returns = to_torch_as(batch.nstep_returns.flatten(), q)
        td_error = returns - q

        if self.dqn_params.clip_loss_grad:
            y = q.reshape(-1, 1)
            t = returns.reshape(-1, 1)
            loss = torch.nn.functional.huber_loss(y, t, reduction="mean")
        else:
            loss = (td_error.pow(2) * weight).mean()

        batch.weight = td_error  # prio-buffer
        loss.backward()
        self.dqn_optim.step()

        return DQNTrainingStats(loss=loss.item())  # type: ignore[return-value]


    def _pg_learn(  # type: ignore
        self,
        batch: BatchWithTwoReturnsProtocol,
        batch_size: int | None,
        repeat: int,
        backward: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> PGTrainingStats:
        losses = []
        split_batch_size = batch_size or -1

        with torch.no_grad() if not backward else torch.enable_grad():
            for _ in range(repeat):
                for minibatch in batch.split(split_batch_size, merge_last=True):
                    self.sampler_optim.zero_grad()
                    result = self(minibatch)
                    dist = result.dist
                    act = to_torch_as(minibatch.act, result.act)
                    ret = to_torch(minibatch.episodic_returns, torch.float, result.act.device)
                    log_prob = dist.log_prob(act).reshape(len(ret), -1).transpose(0, 1)
                    loss = -(log_prob * ret).mean()
                    if backward:
                        loss.backward()
                        self.sampler_optim.step()
                    losses.append(loss.item())

        loss_summary_stat = SequenceSummaryStats.from_sequence(losses)

        return PGTrainingStats(loss=loss_summary_stat)  # type: ignore[return-value]


    def learn( # type: ignore
        self,
        batch: BatchWithTwoReturnsProtocol,
        batch_size: int | None,
        repeat: int,
        *args: Any,
        **kwargs: Any
    ) -> MBDTrainingStats:
        """
        Perform a learning step on the given batch for both q-network and the temperatrure network.

        :param batch: 
        :param batch_size: necessary for policy gradient learning step.
        :param repeat: necessary for policy gradient learning step. 
        """
        sampler_compute_grad = self._iter % self.sampler_update_freq == 0
        pg_stats = self._pg_learn(batch, batch_size, repeat, backward=sampler_compute_grad)
        dqn_stats = self._dqn_learn(batch)
        self._iter += 1

        return MBDTrainingStats(
            pg_loss=pg_stats.loss,
            dqn_loss=dqn_stats.loss
        )




