from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import pointnet2_utils
from pcdet.models.model_utils.mlp import MLP
from pcdet.models.model_utils.weight_init import *

def to_radian(degree):
    return degree / 180.0 * np.pi

def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)
    Returns:
    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t((torch.t(Ib) * wb)) + torch.t((torch.t(Ic) * wc)) + torch.t((torch.t(Id) * wd))
    return ans

class Atten_Fusion_Conv(nn.Module):
    def __init__(self, inplane_I, inplanes_P):
        super(Atten_Fusion_Conv, self).__init__()

        self.Cross_Fusion = CrossViewTransATT(query_dim=inplanes_P, key_dim=inplanes_P, proj_dim=inplanes_P // 8)
        self.conv = nn.Conv1d(in_channels = inplanes_P, out_channels= inplanes_P, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(inplanes_P)

        init_type = 'kaiming_uniform'
        self.init_weights(init_type)    #for ped

    def init_weights(self, init_type):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                if init_type == 'kaiming_uniform':
                    kaiming_init(m, distribution='uniform')
                elif init_type == 'kaiming_normal':
                    kaiming_init(m, distribution='normal')
                elif init_type == 'xavier':
                    xavier_init(m)
                elif init_type == 'caffe2_xavier':
                    caffe2_xavier_init(m)

    def forward(self, cyv_features, point_features):

        fusion_features = self.Cross_Fusion(cyv_features, point_features)
        fusion_features = F.relu(self.bn(self.conv(fusion_features)))
        fused_features = torch.cat([point_features, fusion_features], dim=1)

        return fused_features

class BidirectionalTransFusion(nn.Module):
    def __init__(self, num_filter):
        super(BidirectionalTransFusion, self).__init__()

        self.point_fc1 = nn.Sequential(
            nn.Conv1d(num_filter, num_filter, 1, stride=1, bias=False),
            nn.BatchNorm1d(num_filter),
            nn.ReLU()
        )

        self.trans_fc1 = nn.Sequential(
            nn.Conv1d(num_filter, num_filter, 1, stride=1, bias=False),
            nn.BatchNorm1d(num_filter),
            nn.ReLU()
        )

        self.point_fc2 = nn.Sequential(
            nn.Conv1d(num_filter, num_filter, 1, stride=1, bias=False),
            nn.BatchNorm1d(num_filter),
        )

        self.trans_fc2 = nn.Sequential(
            nn.Conv1d(num_filter, num_filter, 1, stride=1, bias=False),
            nn.BatchNorm1d(num_filter),
        )

        self.point_relu = nn.ReLU()
        self.trans_relu = nn.ReLU()

        self.point_att_path = nn.Sequential(
            nn.Conv1d(num_filter, num_filter, 1, stride=1),
            nn.BatchNorm1d(num_filter),
            nn.ReLU(),
        )

        self.trans_att_path = nn.Sequential(
            nn.Conv1d(num_filter, num_filter, 1, stride=1),
            nn.BatchNorm1d(num_filter),
            nn.ReLU(),
        )
        init_type = 'kaiming_uniform'
        self.init_weights(init_type)    #for ped

    def init_weights(self, init_type):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                if init_type == 'kaiming_uniform':
                    kaiming_init(m, distribution='uniform')
                elif init_type == 'kaiming_normal':
                    kaiming_init(m, distribution='normal')
                elif init_type == 'xavier':
                    xavier_init(m)
                elif init_type == 'caffe2_xavier':
                    caffe2_xavier_init(m)

    def forward(self, point_features, trans_features):
        point_features_v1 = self.point_fc1(point_features)
        trans_features_v1 = self.trans_fc1(trans_features)

        att_point_to_trans = torch.sigmoid(self.point_att_path(point_features_v1))
        att_trans_to_point = torch.sigmoid(self.trans_att_path(trans_features_v1))

        point_features_v2 = self.point_fc2(point_features_v1)
        trans_features_v2 = self.trans_fc2(trans_features_v1)

        point_fusion = self.point_relu(point_features_v1 + att_trans_to_point * point_features_v2)
        trans_fusion = self.trans_relu(trans_features_v1 + att_point_to_trans * trans_features_v2)

        fusion_features = torch.cat([point_fusion, trans_fusion], dim=1)

        return fusion_features

class CrossViewTransATT(nn.Module):
    def __init__(self, query_dim=64, key_dim=64, proj_dim=8):
        super(CrossViewTransATT, self).__init__()

        self.query_conv = nn.Conv1d(
            in_channels=query_dim, out_channels=proj_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(
            in_channels=key_dim, out_channels=proj_dim, kernel_size=1)
        self.value_conv = nn.Conv1d(
            in_channels=key_dim, out_channels=proj_dim, kernel_size=1)

        self.conv = nn.Conv1d(in_channels=proj_dim, out_channels=query_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_x, ref_x):
        """
        Args:
            query_x (torch.Tensor): the query feature map
            ref_x (torch.Tensor): the reference feature map
        """
        batch_size, C, N = ref_x.size()
        proj_query = self.query_conv(query_x).view(batch_size, -1, N)  # B x C x N

        proj_key = self.key_conv(ref_x).view(batch_size, -1, N).permute(0, 2, 1).contiguous()  # B x N x C

        proj_value = self.value_conv(ref_x).view(batch_size, -1, N)  # B x C x N

        weights = torch.bmm(proj_query, proj_key)  # transpose B*C*C

        att_map = self.softmax(weights)
        z = torch.bmm(att_map.permute(0, 2, 1).contiguous(), proj_value)
        output = query_x + self.conv(z)

        return output

class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def calc_square_dist(self, a, b, norm=True):
        """
        Calculating square distance between a and b
        a: [bs, n, c]
        b: [bs, m, c]
        """
        n = a.shape[1]
        m = b.shape[1]
        num_channel = a.shape[-1]
        a_square = a.unsqueeze(dim=2)  # [bs, n, 1, c]
        b_square = b.unsqueeze(dim=1)  # [bs, 1, m, c]
        a_square = torch.sum(a_square * a_square, dim=-1)  # [bs, n, 1]
        b_square = torch.sum(b_square * b_square, dim=-1)  # [bs, 1, m]
        a_square = a_square.repeat((1, 1, m))  # [bs, n, m]
        b_square = b_square.repeat((1, n, 1))  # [bs, n, m]

        coor = torch.matmul(a, b.transpose(1, 2))  # [bs, n, m]

        if norm:
            dist = a_square + b_square - 2.0 * coor  # [bs, npoint, ndataset]
            # dist = torch.sqrt(dist)
        else:
            dist = a_square + b_square - 2 * coor
            # dist = torch.sqrt(dist)
        return dist

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                pointnet2_utils.farthest_point_sample(xyz, self.npoint)
            ).transpose(1, 2).contiguous() if self.npoint is not None else None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True,
                 use_xyz: bool = True, pool_method='max_pool'):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))

        self.pool_method = pool_method


class PointnetSAModuleMSG_WithSampling(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with specific downsampling and multiscale grouping """

    def __init__(self, *,
                 npoint_list: List[int],
                 sample_range_list: List[int],
                 sample_type_list: List[int],
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],                 
                 use_xyz: bool = True,
                 dilated_group=False,
                 trans_enable=False,
                 density: bool = False,
                 cyv_aug=False,
                 cyv_features: List[int],
                 stride: List[int],
                 pool_method='max_pool',
                 aggregation_mlp: List[int],
                 confidence_mlp: List[int],
                 num_features: List[int],
                 num_head: List[int],
                 num_hidden_features: List[int],
                 dropout: List[float],
                 num_layer: List[int],
                 point_cloud_range,
                 num_class):
        """
        :param npoint_list: list of int, number of samples for every sampling type
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling type
        :param sample_type_list: list of str, list of used sampling type, d-fps or f-fps
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_group: whether to use dilated group
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        :param num_class: int, class for process
        """
        super().__init__()
        self.sample_type_list = sample_type_list
        self.sample_range_list = sample_range_list
        self.dilated_group = dilated_group
        self.point_cloud_range = point_cloud_range
        self.cyv_aug = cyv_aug
        self.density = density
        self.trans_enable = trans_enable

        assert len(radii) == len(nsamples) == len(mlps)

        self.position_aware = nn.ModuleList()
        self.transformer_local = nn.ModuleList()
        self.trans_fusion = nn.ModuleList()

        self.num_features = num_features
        self.num_head = num_head
        self.num_hidden_features = num_hidden_features
        self.dropout = dropout
        self.num_layer = num_layer

        self.cyv_features = cyv_features
        self.stride = stride

        self.z_max = 1.0
        self.z_min = -2.5
        self.z_range = self.z_max - self.z_min
        self.append_far = False
        self.fov_left = to_radian(45.0)  # field of view left in rad
        self.fov_right = to_radian(-45.0)  # field of view right in rad
        self.fov_horizontal = abs(self.fov_left) + abs(self.fov_right)  # get field of view horizontal in rad
        self.frontal_axis = 'X'
        self.num_cols = 512
        self.num_rows = 48

        self.npoint_list = npoint_list
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        if self.cyv_aug:
            for k in range(len(self.cyv_features)):
                self.fuse3d_mlp = nn.Sequential(
                    nn.Conv1d(cyv_features[k] * 4, cyv_features[k] * 2, kernel_size=1, bias=False),
                    nn.BatchNorm1d(cyv_features[k] * 2),
                    nn.ReLU()
                )
                self.cyv_mlp = MLP(in_channel=cyv_features[k], conv_channels=(cyv_features[k] * 2,), bias=False)
                self.fusion_block = Atten_Fusion_Conv(inplane_I=cyv_features[k] * 2, inplanes_P=cyv_features[k] * 2)
                self.stride = stride[k]

        if self.trans_enable:
            for k in range(len(self.num_features)):
                encoder_layers_local = nn.TransformerEncoderLayer(self.num_features[k], self.num_head[k],
                                                                  self.num_hidden_features[k], self.dropout[k])
                self.transformer_local.append(nn.TransformerEncoder(encoder_layers_local, self.num_layer[k]))
                # encoder_norm = nn.LayerNorm(self.num_features[k])
                # self.transformer_local.append(nn.TransformerEncoder(encoder_layers_local, self.num_layer[k], encoder_norm))

        out_channels = 0
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            if self.dilated_group:
                if i == 0:
                    min_radius = 0.
                else:
                    min_radius = radii[i-1]
                self.groupers.append(
                    pointnet2_utils.QueryDilatedAndGroup(
                        radius, min_radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )
            else:
                self.groupers.append(
                    pointnet2_utils.QueryAndGroup(
                        radius, nsample, use_xyz=use_xyz, density=density)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )
            mlp_spec = mlps[i]
            mlp_spec2 = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            shared_mlps_pos = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1],
                              kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            if self.trans_enable:
                for k in range(len(mlp_spec2)-3):
                    shared_mlps_pos.extend([
                        nn.Conv2d(3, mlp_spec2[k + 1], kernel_size=1, bias=False),
                        nn.BatchNorm2d(mlp_spec[k + 1]),
                        nn.ReLU(),
                        nn.Conv2d(mlp_spec2[k + 1], mlp_spec2[k + 3], kernel_size=1, bias=False)
                    ])
                    self.position_aware.append(nn.Sequential(*shared_mlps_pos))
            self.mlps.append(nn.Sequential(*shared_mlps))
            out_channels += mlp_spec[-1]

        self.pool_method = pool_method

        if (aggregation_mlp is not None) and (len(aggregation_mlp) != 0) and (len(self.mlps) > 0):
            shared_mlp = []
            for k in range(len(aggregation_mlp)):
                if self.trans_enable:
                    self.trans_fusion = BidirectionalTransFusion(num_filter=out_channels)
                    shared_mlp.extend([
                        nn.Conv1d(out_channels * 2,       #out_channels * 2
                                  aggregation_mlp[k], kernel_size=1, bias=False),
                        nn.BatchNorm1d(aggregation_mlp[k]),
                        nn.ReLU()
                    ])
                else:
                    shared_mlp.extend([
                        nn.Conv1d(out_channels,
                                  aggregation_mlp[k], kernel_size=1, bias=False),
                        nn.BatchNorm1d(aggregation_mlp[k]),
                        nn.ReLU()
                    ])
                out_channels = aggregation_mlp[k]
            self.aggregation_layer = nn.Sequential(*shared_mlp)
        else:
            self.aggregation_layer = None

        if (confidence_mlp is not None) and (len(confidence_mlp) != 0):
            shared_mlp = []
            for k in range(len(confidence_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels,
                              confidence_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(confidence_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = confidence_mlp[k]
            shared_mlp.append(
                nn.Conv1d(out_channels, num_class, kernel_size=1, bias=True),
            )
            self.confidence_layers = nn.Sequential(*shared_mlp)
        else:
            self.confidence_layers = None

    def get_cols(self, points, truncate=False):
        """ Returns the column indices for unfolding point cloud """
        # for frontal axis = y
        if self.frontal_axis == 'Y':
            xs = points[:, :, 1]
            ys = -points[:, :, 0]
        else:
            xs = points[:, :, 0]
            ys = points[:, :, 1]

        yaw = -torch.atan2(ys, (xs + 1e-8))

        proj_x = (yaw + abs(self.fov_left)) / self.fov_horizontal  # in [0.0, 1.0]

        proj_x *= self.num_cols

        mask = (proj_x >= 0) & (proj_x < self.num_cols)

        if truncate:
            proj_x = torch.maximum(proj_x, torch.zeros_like(proj_x))
            proj_x = torch.minimum(proj_x, torch.ones_like(proj_x) * (self.num_cols - 1))

        return proj_x, mask

    def get_rows(self, points, truncate=False):
        """ Returns the row indices for unfolding point cloud """
        zs = points[:, :, 2]
        proj_y = 1 - (zs - self.z_min) / self.z_range  # in [0.0, 1.0]
        proj_y *= self.num_rows

        mask = (proj_y >= 0) & (proj_y < self.num_rows)
        if truncate:
            proj_y = torch.maximum(proj_y, torch.zeros_like(proj_y))
            proj_y = torch.minimum(proj_y, torch.ones_like(proj_y) * (self.num_rows - 1))

        return proj_y, mask

    def interpolate_from_cyv_features(self, keypoints, cyv_features, cyv_stride):
        batch_size, C, H, W = cyv_features.shape
        coords_batch_x, mask_x = self.get_cols(keypoints)
        coords_batch_y, mask_y = self.get_rows(keypoints)

        mask_all = mask_x & mask_y
        coords_batch_x = coords_batch_x / cyv_stride
        coords_batch_y = coords_batch_y / cyv_stride

        point_cyv_features_list = []
        for k in range(batch_size):
            valid_mask = mask_all[k].flatten()
            cur_x_idxs = coords_batch_x[k]
            cur_y_idxs = coords_batch_y[k]
            cur_cyv_features = cyv_features[k].permute(1, 2, 0)
            cur_point_cyv_features = torch.zeros((len(valid_mask), cur_cyv_features.shape[-1]),
                                                 dtype=cur_cyv_features.dtype, device=cur_cyv_features.device)
            point_cyv_features = bilinear_interpolate_torch(cur_cyv_features, cur_x_idxs[valid_mask == 1], cur_y_idxs[valid_mask == 1])
            cur_point_cyv_features[valid_mask == 1, :] = point_cyv_features
            point_cyv_features_list.append(cur_point_cyv_features.unsqueeze(dim=0))
        point_cyv_features = torch.cat(point_cyv_features_list, dim=0)

        return point_cyv_features

    def forward(self, batch_dict, xyz: torch.Tensor, features: torch.Tensor = None, cls_features: torch.Tensor = None,
                new_xyz=None, ctr_xyz=None):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param cls_features: (B, N, num_class) tensor of the descriptors of the the confidence (classification) features 
        :param new_xyz: (B, M, 3) tensor of the xyz coordinates of the sampled points
        "param ctr_xyz: tensor of the xyz coordinates of the centers 
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
            cls_features: (B, npoint, num_class) tensor of confidence (classification) features
        """
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous() 
        sampled_idx_list = []
        attended_features_list = []

        if ctr_xyz is None:
            last_sample_end_index = 0
            
            for i in range(len(self.sample_type_list)):
                sample_type = self.sample_type_list[i]
                sample_range = self.sample_range_list[i]
                npoint = self.npoint_list[i]

                if npoint <= 0:
                    continue
                if sample_range == -1: #全部
                    xyz_tmp = xyz[:, last_sample_end_index:, :]
                    feature_tmp = features.transpose(1, 2)[:, last_sample_end_index:, :].contiguous()  
                    cls_features_tmp = cls_features[:, last_sample_end_index:, :] if cls_features is not None else None 
                else:
                    xyz_tmp = xyz[:, last_sample_end_index:sample_range, :].contiguous()
                    feature_tmp = features.transpose(1, 2)[:, last_sample_end_index:sample_range, :]
                    cls_features_tmp = cls_features[:, last_sample_end_index:sample_range, :] if cls_features is not None else None 
                    last_sample_end_index += sample_range

                if xyz_tmp.shape[1] <= npoint: # No downsampling
                    sample_idx = torch.arange(xyz_tmp.shape[1], device=xyz_tmp.device, dtype=torch.int32) * torch.ones(xyz_tmp.shape[0], xyz_tmp.shape[1], device=xyz_tmp.device, dtype=torch.int32)

                elif ('cls' in sample_type) or ('ctr' in sample_type):
                    cls_features_max, class_pred = cls_features_tmp.max(dim=-1)
                    score_pred = torch.sigmoid(cls_features_max) # B,N
                    score_picked, sample_idx = torch.topk(score_pred, npoint, dim=-1)           
                    sample_idx = sample_idx.int()

                elif 'D-FPS' in sample_type or 'DFS' in sample_type:
                    sample_idx = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)

                elif 'F-FPS' in sample_type or 'FFS' in sample_type:
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = self.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    sample_idx = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)

                elif sample_type == 'FS':
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = self.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    sample_idx_1 = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                    sample_idx_2 = pointnet2_utils.furthest_point_sample(xyz_tmp, npoint)
                    sample_idx = torch.cat([sample_idx_1, sample_idx_2], dim=-1)  # [bs, npoint * 2]
                elif 'Rand' in sample_type:
                    sample_idx = torch.randperm(xyz_tmp.shape[1],device=xyz_tmp.device)[None, :npoint].int().repeat(xyz_tmp.shape[0], 1)
                elif sample_type == 'ds_FPS' or sample_type == 'ds-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for i in range(len(xyz_tmp)):
                        per_xyz = xyz_tmp[i]
                        radii = per_xyz.norm(dim=-1) -5 
                        storted_radii, indince = radii.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1 ,3)

                        per_idx_div = indince.view(part_num,-1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div ,dim=0)
                    idx_div = torch.cat(idx_div ,dim=0)
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, (npoint//part_num))

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):                    
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)
                    sample_idx = index.reshape(xyz.shape[0], npoint).int()

                elif sample_type == 'ry_FPS' or sample_type == 'ry-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for i in range(len(xyz_tmp)):
                        per_xyz = xyz_tmp[i]
                        ry = torch.atan(per_xyz[:,0]/per_xyz[:,1])
                        storted_ry, indince = ry.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1 ,3)

                        per_idx_div = indince.view(part_num,-1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div ,dim=0)
                    idx_div = torch.cat(idx_div ,dim=0)
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, (npoint//part_num))

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):                    
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)

                    sample_idx = index.reshape(xyz.shape[0], npoint).int()

                sampled_idx_list.append(sample_idx)

            sampled_idx_list = torch.cat(sampled_idx_list, dim=-1) 
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sampled_idx_list).transpose(1, 2).contiguous()

        else:
            new_xyz = ctr_xyz

        if len(self.groupers) > 0:
            for i in range(len(self.groupers)):
                new_features, group_xyz, group_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
                new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
                if self.pool_method == 'max_pool':
                    new_features = F.max_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                elif self.pool_method == 'avg_pool':
                    new_features = F.avg_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                else:
                    raise NotImplementedError

                if self.trans_enable:
                    position_encoding = self.position_aware[i](group_xyz)
                    input_features = group_features + position_encoding
                    B, D, np, ns = input_features.shape
                    input_features = input_features.permute(0,2,1,3).reshape(-1,D,ns).permute(2,0,1)
                    attended_features = self.transformer_local[i](input_features).permute(1,2,0).reshape(B,np,D,ns).transpose(1,2)

                    output_features = F.max_pool2d(attended_features, kernel_size=[1,attended_features.size(3)])
                    output_features = output_features.squeeze(-1)

                new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
                new_features_list.append(new_features)

                if self.trans_enable:
                    attended_features_list.append(output_features)
                    fuse_features = torch.cat(attended_features_list, dim=1)

            new_features = torch.cat(new_features_list, dim=1)

        if self.trans_enable:
            new_features = self.trans_fusion(new_features, fuse_features)

        if self.aggregation_layer is not None:
            new_features = self.aggregation_layer(new_features)
        else:
            new_features = pointnet2_utils.gather_operation(features, sampled_idx_list).contiguous()

        if self.cyv_aug:
            point_cyv_features = self.interpolate_from_cyv_features(new_xyz,
                                                                    batch_dict['fv_features_{}x'.format(self.stride)],
                                                                    cyv_stride=self.stride)
            cylindrical_features = point_cyv_features.permute(0, 2, 1)
            cylindrical_features = self.cyv_mlp(cylindrical_features)
            new_features = self.fusion_block(cylindrical_features, new_features)
            new_features = self.fuse3d_mlp(new_features)

        if self.confidence_layers is not None:
            cls_features = self.confidence_layers(new_features).transpose(1, 2)
        else:
            cls_features = None

        return new_xyz, new_features, cls_features

class Vote_layer(nn.Module):
    """ Light voting module with limitation"""
    def __init__(self, mlp_list, pre_channel, max_translate_range):
        super().__init__()
        self.mlp_list = mlp_list
        if len(mlp_list) > 0:
            for i in range(len(mlp_list)):
                shared_mlps = []

                shared_mlps.extend([
                    nn.Conv1d(pre_channel, mlp_list[i], kernel_size=1, bias=False),
                    nn.BatchNorm1d(mlp_list[i]),
                    nn.ReLU()
                ])
                pre_channel = mlp_list[i]
            self.mlp_modules = nn.Sequential(*shared_mlps)
        else:
            self.mlp_modules = None

        self.ctr_reg = nn.Conv1d(pre_channel, 3, kernel_size=1)
        self.max_offset_limit = torch.tensor(max_translate_range).float() if max_translate_range is not None else None
       
    def forward(self, xyz, features):
        xyz_select = xyz
        features_select = features

        if self.mlp_modules is not None: 
            new_features = self.mlp_modules(features_select) #([4, 256, 256]) ->([4, 128, 256])            
        else:
            # new_features = new_features
            new_features = features
        
        ctr_offsets = self.ctr_reg(new_features) #[4, 128, 256]) -> ([4, 3, 256])

        ctr_offsets = ctr_offsets.transpose(1, 2)#([4, 256, 3])
        feat_offets = ctr_offsets[..., 3:]
        new_features = feat_offets
        ctr_offsets = ctr_offsets[..., :3]
        
        if self.max_offset_limit is not None:
            max_offset_limit = self.max_offset_limit.view(1, 1, 3)            
            max_offset_limit = self.max_offset_limit.repeat((xyz_select.shape[0], xyz_select.shape[1], 1)).to(xyz_select.device) #([4, 256, 3])
      
            limited_ctr_offsets = torch.where(ctr_offsets > max_offset_limit, max_offset_limit, ctr_offsets)
            min_offset_limit = -1 * max_offset_limit
            limited_ctr_offsets = torch.where(limited_ctr_offsets < min_offset_limit, min_offset_limit, limited_ctr_offsets)
            vote_xyz = xyz_select + limited_ctr_offsets
        else:
            vote_xyz = xyz_select + ctr_offsets

        return vote_xyz, new_features, xyz_select, ctr_offsets


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None,
                 bn: bool = True, use_xyz: bool = True, pool_method='max_pool'):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__(
            mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz,
            pool_method=pool_method
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()

        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp[k + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)

if __name__ == "__main__":
    pass
