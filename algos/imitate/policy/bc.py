from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from termcolor import cprint
import copy
import time
# import pytorch3d.ops as torch3d_ops

from algos.imitate.model.common.normalizer import LinearNormalizer
from algos.imitate.policy.base_policy import BasePolicy
from algos.imitate.common.pytorch_util import dict_apply
from algos.imitate.common.model_util import print_params
from algos.imitate.model.vision.pointnet_extractor import DP3EncoderV2, create_mlp

class BC(BasePolicy):
    def __init__(self, shape_meta: dict, encoder_output_dim=256, crop_shape=None, use_pc_color=False, pointnet_type="pointnet", pointcloud_encoder_cfg=None, **kwargs):
        super().__init__()
        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        obs_encoder = DP3EncoderV2(observation_space=obs_dict, img_crop_shape=crop_shape, out_channel=encoder_output_dim, pointcloud_encoder_cfg=pointcloud_encoder_cfg, use_pc_color=use_pc_color, pointnet_type=pointnet_type)
        obs_feature_dim = obs_encoder.output_shape()

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[BCPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[BCPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")

        model = nn.Sequential(*create_mlp(input_dim=obs_feature_dim, output_dim=action_dim, net_arch=[256, 256], activation_fn=nn.Tanh))

        self.obs_encoder = obs_encoder
        self.model = model
        
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.kwargs = kwargs

        print_params(self)
        
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # if not self.use_pc_color:
        #     nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:1,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        naction_pred = self.model(nobs_features)

        # unnormalize prediction
        action_pred = self.normalizer['action'].unnormalize(naction_pred).unsqueeze(1)

        result = {'action': action_pred, 'action_pred': action_pred}
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        target = nactions.squeeze(1)

        # if not self.use_pc_color:
        #     nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]

        this_nobs = dict_apply(nobs, lambda x: x[:,:1,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        pred = self.model(nobs_features)

        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        loss_dict = {'bc_loss': loss.item()}

        return loss, loss_dict
