import numpy as np

from tianshou.data import ReplayBuffer
from tianshou.data.types import RolloutBatchProtocol


class HybridReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        """Initialize the HybridReplayBuffer."""
        super().__init__(*args, **kwargs)
        # Initialize is_offpolicy field to False for all transitions
        self._is_onpolicy = np.zeros(self.maxsize, dtype=bool)

    def add(
        self,
        batch: RolloutBatchProtocol,
        buffer_ids: np.ndarray | list[int] | None = None,
        is_onpolicy: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Add a batch of data into the replay buffer.

        :param batch: the input data batch.
        :param buffer_ids: to make consistent with other buffer's add function; if it
            is not None, we assume the input batch's first dimension is always 1.
        :param is_offpolicy: Whether the transitions are off-policy. Default is False.
        """
        indices, ep_rew, ep_len, ep_idx = super().add(batch, buffer_ids)
        # Set the is_offpolicy flag for the new transitions
        self._is_onpolicy[indices] = is_onpolicy
        return indices, ep_rew, ep_len, ep_idx

    def mark_all_offpolicy(self) -> None:
        """Mark all transitions in the buffer as off-policy."""
        self._is_onpolicy[:self._size] = False

    def sample(self, 
               batch_size: int | None,
               on_policy_only: bool = False,
        ) -> tuple[RolloutBatchProtocol, np.ndarray]:
        """Get a random sample from buffer with size = batch_size.

        Return all the data in the buffer if batch_size is 0.

        :return: Sample data and its corresponding index inside the buffer.
        """
        indices = self.sample_indices(batch_size, on_policy_only)
        return self[indices], indices


    def sample_indices(self, 
                       batch_size: int | None,
                       on_policy_only: bool = False
        ) -> np.ndarray:
        """Get a random sample of index with size = batch_size.

        Return all available indices in the buffer if batch_size is 0; return an empty
        numpy array if batch_size < 0 or no available index can be sampled.

        :param batch_size: the number of indices to be sampled. If None, it will be set
            to the length of the buffer (i.e. return all available indices in a
            random order).
        :param on_policy_only: if True, only sample indices that are marked as on-policy.
        """
        if batch_size is None:
            batch_size = len(self)

        mask = self._is_onpolicy[:self._size]
        available_indices = np.where(mask)[0] if on_policy_only else np.arange(self._size)

        if len(available_indices) == 0 or batch_size < 0:
            return np.array([], int)

        # Sampling logic without stacking
        if self.stack_num == 1 or not self._sample_avail:
            if batch_size > 0:
                return np.random.choice(available_indices, batch_size, replace=False)
            if batch_size == 0:
                return available_indices
            return np.array([], int)

        #TODO: Review the logic with stacking
        # Sampling logic with stacking
        all_indices = prev_indices = np.concatenate([
            np.arange(self._index, self._size),
            np.arange(self._index),
        ])
        for _ in range(self.stack_num - 2):
            prev_indices = self.prev(prev_indices)
        all_indices = all_indices[prev_indices != self.prev(prev_indices)]

        # Filter stacked indices for off-policy if needed
        if on_policy_only:
            all_indices = np.intersect1d(all_indices, available_indices, assume_unique=True)

        if batch_size > 0:
            return np.random.choice(all_indices, batch_size, replace=False)
        return all_indices

