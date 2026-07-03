from typing import Dict
import torch
import numpy as np
import copy
from termcolor import cprint
from algos.imitate.common.pytorch_util import dict_apply
from algos.imitate.common.replay_buffer import ReplayBuffer
from algos.imitate.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from algos.imitate.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from dataset.base_dataset import BaseDataset
from transforms3d.quaternions import quat2mat


class TrajectoryDataset(BaseDataset):
    def __init__(self, zarr_path, horizon=1, pad_before=0, pad_after=0, seed=42, val_ratio=0.0, max_train_episodes=None, task_name=None, noisy_points=False, noisy_states=False, point_cs="world"):
        super().__init__()
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=['state', 'action', 'point_cloud'])
        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=horizon, pad_before=pad_before, pad_after=pad_after, episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.noisy_points = noisy_points
        self.noisy_states = noisy_states
        self.point_cs = point_cs

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=self.horizon, pad_before=self.pad_before, pad_after=self.pad_after, episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {'action': self.replay_buffer['action'],}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:, :22].astype(np.float32)
        point_cloud = sample['point_cloud'][:,].astype(np.float32)

        if self.point_cs == "target":
            point_cloud = np.concatenate([point_cloud, point_cloud - sample['state'][:, 22:25][:, None]], -1)
        elif self.point_cs == "hand":
            pcs_list = []
            pcs_list.append(point_cloud)
            pcs_list.append(point_cloud - sample['state'][:, 22:25][:, None])
            trans_mat = sample['state'][:, 25:].reshape(point_cloud.shape[0], -1, 4, 4)
            for idx in range(trans_mat.shape[1]):
                trans_pcs = point_cloud - trans_mat[:, idx, :3, 3][:, None]
                trans_pcs = (trans_mat[:, idx, :3, :3].transpose(0, 2, 1) @ trans_pcs.transpose(0, 2, 1)).transpose(0, 2, 1)
                pcs_list.append(trans_pcs)
            point_cloud = np.concatenate(pcs_list, -1)

        data = {'obs': {'point_cloud': point_cloud, 'agent_pos': agent_pos,}, 'action': sample['action'].astype(np.float32)}
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        if self.noisy_points:
            torch_data['obs']['point_cloud'] *= (1 + torch.randn_like(torch_data['obs']['point_cloud']) * 0.03)
        if self.noisy_states:
            torch_data['obs']['agent_pos'] += torch.randn_like(torch_data['obs']['agent_pos']) * 0.03
        
        return torch_data
