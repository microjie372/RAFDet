import copy

import numpy as np
import torch
from torch.nn import functional as F

from ...utils.common_utils import check_numpy_to_torch
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
class PointProjection(object):
    """Project 3d points to 2d plane
    
    Args:
        project_cfg (dict): projection configuration
    """

    def __init__(self, project_cfg):
        proj_type = project_cfg['TYPE']
        self.proj_type = proj_type
        self.remove_empty_bboxes=project_cfg.get('REMOVE_EMPTY_BBOXES',False)
        self.append_far=project_cfg.get('APPEND_FAR',False)
        if proj_type == 'spherical':
            self.projector = SphericalProjector(project_cfg)
        elif proj_type == 'cylindrical':
            self.projector = CylindricalProjector(project_cfg)
        elif proj_type == 'bev':
            self.projector = BEVProjector(project_cfg)
        else:
            raise ValueError("Unknown projection type: {}".format(proj_type))

    def project(self, points,data_dict):
        """Call function to project 3d points to 2d plane

        Args:
            points (torch.Tensor): N x 4 input point cloud 

        Returns:
            dict: Results after projection
        """

        points = torch.from_numpy(points)
        results = self.projector.do_projection(points, data_dict)

        if self.remove_empty_bboxes and ('gt_boxes' in data_dict):
            gt_bboxes= data_dict['gt_boxes']
            num_boxes = len(gt_bboxes)
            if 'gt_boxes_mask' in data_dict:
                gt_boxes_mask=data_dict['gt_boxes_mask']
            else:
                gt_boxes_mask = np.array([True] * num_boxes, dtype=np.bool_)
            # point_masks =roiaware_pool3d_utils.points_in_boxes_gpu(points.unsqueeze(0)[:,:,:3].cuda(),
            #                                                        gt_bboxes_torch.unsqueeze(0).cuda()).squeeze(0).cpu().numpy()
            new_points = results["points_img"].permute(1,2,0).contiguous()
            new_mask = results["proj_masks"]
            new_points = new_points[new_mask][:,:3]
            gt_bboxes_expand=copy.deepcopy(gt_bboxes[:,:7])
            gt_bboxes_expand[:,3:5]+=0.20
            new_point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(new_points.numpy(),
                                                                        gt_bboxes_expand)

            for i in range(num_boxes):
                box_point_num_after = new_points[new_point_masks[i]>0].shape[0]
                if box_point_num_after == 0:
                    gt_boxes_mask[i] = False

            data_dict['gt_boxes']=gt_bboxes[gt_boxes_mask]
        if self.proj_type=="bev":
            data_dict['points_img_bev'] = results['points_img'].numpy()
            data_dict['proj_masks_bev'] = results['proj_masks'].numpy()
        else:
            data_dict['points_img'] = results['points_img'].numpy()
            data_dict['proj_masks'] = results['proj_masks'].numpy()
            if self.append_far:
                data_dict['points_img_far'] = results['points_img_far'].numpy()
                data_dict['proj_masks_far'] = results['proj_masks_far'].numpy()

class ProjectionBase(object):

    def __init__(self, cfg):
        self.cfg = cfg
        
        self.num_cols = cfg.get('NUM_COLS', 512)
        self.num_rows = cfg.get('NUM_ROWS', 48)
        self.frontal_axis = cfg.get('FRONTAL_AXIS', 'X') # KITTI 

    def xyz_to_plane(self, points_xyz):
        """Project 3d points to 2d plane
        Args:
            points_xyz: N * 3
        Returns:
            points_plane: N * 3
        """
        pass

    def plane_to_xyz(self, points_plane):
        """Project 2d points to 3d plane
        Args:
            points_plane: N * 3
        Returns:
            points_xyz: N * 3
        """
        pass
    
    @staticmethod
    def to_radian(degree):
        return degree / 180.0 * np.pi
   
    @staticmethod
    def get_point_clusters(points_img, proj_masks, kernel_size=4, stride=4, sub_mean=True):
        """Unfold points_img into point clusters by sliding window
        
        Args:
            points_img: (C, H, W) projected points
            proj_masks: (H, W) masks of the points_img
            kernel_size: (4 x 4) size of the sliding window
            stride: stride of the sliding window
            sub_mean: if True, each point substracts mean of each group
        Returns:
            points_out: (ngroup, npoints_per_group, dim) clustered points
        """
        points_img, is_numpy = check_numpy_to_torch(points_img)
        proj_masks, is_numpy = check_numpy_to_torch(proj_masks)
        
        points_img = points_img.unsqueeze(0) # B, H, W, C
        points_img = points_img.permute(0, 3, 1, 2) # B, C, H, W
        B, C, H, W = points_img.shape

        points = F.unfold(points_img, kernel_size=kernel_size, stride=stride)
        npoints_per_group = kernel_size ** 2
        points = points.view(B, C, npoints_per_group, -1) # B, C, npoints_per_group, ngroup
        points = points.permute(0, 3, 2, 1) # B, ngroup, npoints_per_group, C
        
        if sub_mean:
            points_sum = torch.sum(points, dim=2, keepdim=True) # B, ngroup, 1, C

            proj_masks = proj_masks.unsqueeze(0).unsqueeze(0) # B, 1, H, W
            masks = F.unfold(proj_masks, kernel_size=kernel_size, stride=stride)
            masks = masks.view(B, npoints_per_group, -1) # B, npoints_per_group, ngroup
            masks_sum = torch.sum(masks, dim=1) # B, ngroup
            masks_sum = torch.clamp_min(masks_sum, min=1.0).type_as(points_sum)

            points_mean = points_sum / masks_sum[:, :, None, None] # B, ngroup, 1, 1
            
            cluster = points - points_mean # B, ngroup, npoints_per_group, C
            points_out = torch.cat([points, cluster], dim= -1) # B, ngroup, npoints_per_group, 2 * C
            masks_out = masks.permute(0, 2, 1) # B, ngroup, npoints_per_group
            points_out *= masks_out[..., None].float()
        else:
            points_out = points
        
        points_out = points_out.squeeze(0) # ngroup, npoints_per_group, dim
        return points_out.numpy() if is_numpy else points_out
    

    def do_projection(self, points,data_dict):
        """ 
        Scan unfolding of point cloud of shape N x 4
        This functions performs a cylindrical or Spherical projection of a 3D point cloud given in cartesian coordinates.

        Args:
            points (torch.Tensor): input point cloud of shape Nx4, 4 = [x, y, z, intensity]
        Returns:
            output (dict): projected image and mask
        """
        output = {}
        dim = points.shape[1]
        assert dim >= 3
        rot_offset = data_dict.get("noise_rotation",0)
        self.fov_right += rot_offset
        self.fov_left += rot_offset
        proj_col, mask_col = self.get_cols(points)
        proj_row, mask_row = self.get_rows(points)

        mask_all = mask_col & mask_row
        
        # skip invalid points
        proj_col = proj_col[mask_all > 0]
        proj_row = proj_row[mask_all > 0]
        points = points[mask_all > 0]

        # Get depth of all points for ordering.
        points_depth = self.get_depth(points)
        
        # Get the indices in order of decreasing depth.
        indices = torch.arange(points_depth.shape[0])
        order = torch.argsort(points_depth, descending=True)

        if self.append_far:
            indices_far = torch.arange(points_depth.shape[0])
            order_far = torch.argsort(points_depth, descending=False)
            indices_far = indices_far[order_far]
            proj_col_far = proj_col[order_far].long()
            proj_row_far = proj_row[order_far].long()

        indices = indices[order]
        proj_col = proj_col[order].long()
        proj_row = proj_row[order].long()

        # Project the points.
        points_img = -1.0 * points.new_ones(self.num_rows, self.num_cols, 5)
        points_img[proj_row, proj_col, :4] = points[order, :4]
        points_img[proj_row, proj_col, -1] = points_depth[order]
        output["points_img"] = points_img

        if self.append_far:
            points_img_far = -1.0 * points.new_ones(self.num_rows, self.num_cols, dim + 1)
            points_img_far[proj_row_far, proj_col_far, :dim] = points[order_far]
            points_img_far[proj_row_far, proj_col_far, -1] = points_depth[order_far]
            output["points_img_far"]=points_img_far

        proj_point_indices = -1.0 * points.new_ones(self.num_rows, self.num_cols)
        proj_point_indices[proj_row, proj_col] = indices.float()
        # output["proj_point_indices"] = proj_point_indices
        proj_masks = (proj_point_indices != -1)
        output["proj_masks"] = proj_masks
        if self.append_far:
            proj_point_indices_far = -1.0 * points.new_ones(self.num_rows, self.num_cols)
            proj_point_indices_far[proj_row_far, proj_col_far] = indices_far.float()
            proj_masks_far = (proj_point_indices_far != -1)
            output["proj_masks_far"] = proj_masks_far

        points_img *= proj_masks[..., None].float()
        output['points_img'] = points_img.permute(2, 0, 1).contiguous()  # C, H, W

        if self.append_far:
            points_img_far*=proj_masks_far[...,None]
            output['points_img_far'] = points_img_far.permute(2, 0, 1).contiguous()  # C, H, W
        self.fov_right -= rot_offset
        self.fov_left -= rot_offset

        # for debug
        if False:
            import cv2
            img = points_img.numpy()
            img_show = 255 * (1 - img[..., 4])
            # img_show = 255 * proj_masks.float().numpy()
            img_show = img_show.astype(np.uint8)
            img_show = np.repeat(img_show[..., None], 3, axis=2)
            cv2.imwrite("/tmp/points_img.png", img_show)
            #cv2.imshow('debug', )
            #cv2.waitKey()
            exit(0)
        return output

class BEVProjector(ProjectionBase):

    def __init__(self, cfg):
        super().__init__(cfg)

        point_cloud_range = np.array(cfg['POINT_CLOUD_RANGE'], dtype=np.float32)
        
        self.x_min = point_cloud_range[0]
        self.y_min = point_cloud_range[1]
        # self.z_min = point_cloud_range[2]
        self.x_max = point_cloud_range[3]
        self.y_max = point_cloud_range[4]
        self.append_xy=cfg.get('APPEND_XY',False)
        voxel_size = cfg['VOXEL_SIZE']
        grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size)

        # for compitability with point pillars
        # BEV map: B * C * ny * nx -> ny = num_rows (H), nx = num_cols (W)

        self.num_cols = int(grid_size[0] + 0.5) # lidar y
        self.num_rows = int(grid_size[1] + 0.5) # lidar x
        # breakpoint()
    
    def get_rows(self, points, truncate=False):

        points, is_numpy = check_numpy_to_torch(points)
       
        ys = points[:, 1]
        
        proj_y = (ys - self.y_min) / (self.y_max - self.y_min)  # in [0.0, 1.0]

        proj_y *= self.num_rows
       
        mask = (proj_y >= 0) & (proj_y < self.num_rows)

        if truncate:
            proj_y = torch.maximum(proj_y, torch.zeros_like(proj_y))
            proj_y = torch.minimum(proj_y, torch.ones_like(proj_y) * (self.num_rows - 1))
        
        if is_numpy:
            proj_y = proj_y.numpy()
            mask = mask.numpy()

        return proj_y, mask
    
    def get_cols(self, points, truncate=False):

        points, is_numpy = check_numpy_to_torch(points)
       
        xs = points[:, 0]
        
        proj_x = (xs - self.x_min) / (self.x_max - self.x_min)  # in [0.0, 1.0]
        proj_x *= self.num_cols

        mask = (proj_x >= 0) & (proj_x < self.num_cols)

        if truncate:
            proj_x = torch.maximum(proj_x, torch.zeros_like(proj_x))
            proj_x = torch.minimum(proj_x, torch.ones_like(proj_x) * (self.num_cols - 1))

        if is_numpy:
            proj_x = proj_x.numpy()
            mask = mask.numpy()

        return proj_x, mask

    def get_projected_features(self, points, features, with_raw_features=False):

        num_points, dim_points = points.shape[:2]
        num_features, dim_features = features.shape[:2]
        dim_output = dim_features
        assert dim_points >= 3
        assert num_points == num_features

        # Get the indices of the rows and columns to project to.
        proj_col, mask_col = self.get_cols(points)
        proj_row, mask_row = self.get_rows(points)
        
        mask = mask_col & mask_row
        points = points[mask]
        features = features[mask]
        proj_col = proj_col[mask].long()
        proj_row = proj_row[mask].long()
        
        if with_raw_features:
            dim_output += dim_points
            features = torch.cat([points, features], dim=-1)

        indices = proj_row * self.num_cols + proj_col

        # ny * nx * dim
        proj_features = points.new_zeros(self.num_rows * self.num_cols, dim_output)
        # number of points per grid
        npoints_per_grid = points.new_zeros(self.num_rows * self.num_cols, 1)

        proj_features.index_add_(0, indices, features)
        npoints_per_grid.index_add_(0, indices, points.new_ones(indices.shape[0], 1))
        npoints_per_grid = torch.clamp_min(npoints_per_grid, min=1.0).type_as(npoints_per_grid)
        proj_features = proj_features / npoints_per_grid

        proj_features = proj_features.view(self.num_rows, self.num_cols, dim_output)
        proj_features = proj_features.permute(2, 0, 1).contiguous() # dim, h, w -> dim, ny, nx
        return proj_features

    def get_projected_features_v2(self, voxel_features, coors, batch_size):
        """Scatter features of single sample.

               Args:
                   voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
                   coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).
                       The first column indicates the sample ID.
                   batch_size (int): Number of samples in the current batch.
               """
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                voxel_features.shape[-1],
                self.num_cols * self.num_rows,
                dtype=voxel_features.dtype,
                device=voxel_features.device)
            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * self.num_cols + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, voxel_features.shape[-1], self.num_rows,
                                         self.num_cols)

        return batch_canvas

    def do_projection(self, points,data_dict):
        """ 
        Scan unfolding of point cloud of shape [N, (x,y,z)]

        Args:
            points (torch.Tensor): shape=(N, 4), x, y, z, intensity
        Returns:
            output (dict): projected image and mask
        """
        output = {}

        dim = points.shape[1]
        assert dim >= 3

        # Get the indices of the rows and columns to project to.
        proj_col, mask_col = self.get_cols(points)
        proj_row, mask_row = self.get_rows(points)
        
        mask_all = mask_col & mask_row
        
        # skip invalid points
        proj_col = proj_col[mask_all > 0]
        proj_row = proj_row[mask_all > 0]
        points = points[mask_all > 0]

        # Get depth (z value) of all points for ordering.
        points_depth = points[:, 2]


        # Get the indices in order of increasing depth.
        indices = np.arange(points_depth.shape[0])
        order = torch.argsort(points_depth, descending=False)

        indices = indices[order]
        points = points[order]
        proj_col = proj_col[order].long()
        proj_row = proj_row[order].long()
        points_range = torch.norm(points[:, :3], p=2, dim=1)
        
        unique_coords, inverse, unique_counts = torch.unique(torch.hstack([proj_row[:,None], proj_col[:,None]]),
                dim=0, return_inverse=True, return_counts=True)
        # https://github.com/pytorch/pytorch/issues/36748
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        unique_indices = inverse.new_empty(unique_coords.size(0)).scatter_(0, inverse, perm)
        
        # normalized density
        density = torch.minimum(torch.tensor(1.0), torch.log(unique_counts + 1.0) / torch.log(torch.tensor(64.0)))
        
        proj_col = proj_col[unique_indices] 
        proj_row = proj_row[unique_indices]
        intensity = points[unique_indices, 3]
        elevation = points[unique_indices, 2]
        points_x=points[unique_indices,0]
        points_y=points[unique_indices,1]
        points_range=points_range[unique_indices]
        # Project the points.
        points_img = -1.0 * points.new_ones(self.num_rows, self.num_cols, 3)
        if self.append_xy:
            points_img = -1.0 * points.new_ones(self.num_rows, self.num_cols, 6)
        points_img[proj_row, proj_col, 0] = intensity # intensity
        points_img[proj_row, proj_col, 1] = elevation # height z
        points_img[proj_row, proj_col, 2] = density # density
        if self.append_xy:
            points_img[proj_row,proj_col, 3]=points_x
            points_img[proj_row,proj_col,4]=points_y
            points_img[proj_row,proj_col,5]=points_range
        output["points_img"] = points_img

        proj_point_indices = -1.0 * points.new_ones(self.num_rows, self.num_cols)
        proj_point_indices[proj_row, proj_col] = points.new_tensor(indices[unique_indices])
        proj_masks = (proj_point_indices != -1)
        output["proj_masks"] = proj_masks 

        points_img *= proj_masks[..., None].float()
        
        # for debug
        if False:
            import cv2
            img_show = points_img.numpy()
            img_show = (255 * (1 - img_show)).astype(np.uint8)
            print(img_show.shape)
            cv2.imwrite('/tmp/debug_bev.png', img_show)
            #cv2.imshow('debug_bev', img_show)
            #cv2.waitKey()
            exit(0)

        output['points_img'] = points_img.permute(2, 0, 1).contiguous() # C, H, W
        return output

class SphericalProjector(ProjectionBase):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.fov_up = self.to_radian(cfg.get('FOV_UP', 3.0))  # field of view up in rad
        self.fov_down = self.to_radian(cfg.get('FOV_DOWN', -25.0))  # field of view down in rad
        self.fov_vertical = abs(self.fov_down) + abs(self.fov_up)  # get field of view vertical in rad
        
        self.fov_left = self.to_radian(cfg.get('FOV_LEFT', 45.0)) # field of view left in rad
        self.fov_right = self.to_radian(cfg.get('FOV_RIGHT', -45.0)) # field of view right in rad
        self.fov_horizontal = abs(self.fov_left) + abs(self.fov_right)  # get field of view horizontal in rad
        
    def xyz_to_plane(self, points):
        points_xyz, is_numpy = check_numpy_to_torch(points)
        proj_x, mask_x = self.get_cols(points_xyz)
        proj_y, mask_y = self.get_rows(points_xyz)

        mask = mask_x & mask_y

        depth = self.get_depth(points_xyz)
        points_plane = torch.cat([proj_x[:, None], proj_y[:, None], depth[:, None]], dim=-1)
        
        if is_numpy:
            points_plane = points_plane.numpy()
            mask = mask.numpy()

        return points_plane, mask

    def plane_to_xyz(self, points_plane):
        points, is_numpy = check_numpy_to_torch(points_plane)
        
        proj_x = points[:, 0].float() / self.num_cols
        proj_y = points[:, 1].float() / self.num_rows
        depth = points[:, 2]

        # proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov_vertical  # in [0.0, 1.0]
        pitch = (1.0 - proj_y) * self.fov_vertical - abs(self.fov_down)
        
        # pitch = torch.arcsin(zs / (depth + 1e-8))
        zs = depth * torch.sin(pitch)
        
        # proj_x = (yaw + abs(self.fov_left)) / self.fov_horizontal  # in [0.0, 1.0]
        yaw = proj_x * self.fov_horizontal - abs(self.fov_left)
        yaw *= -1.0
        
        dxy = depth * torch.cos(pitch)
        xs = dxy * torch.cos(yaw)
        ys = dxy * torch.sin(yaw)

        points_xyz = torch.cat([xs[:,None], ys[:,None], zs[:,None]], dim=1)
        return points_xyz.numpy() if is_numpy else points_xyz

    def get_depth(self, points_xyz):
        points, is_numpy = check_numpy_to_torch(points_xyz)
        points_depth = torch.norm(points[:, :3], p=2, dim=1)
        return points_depth.numpy() if is_numpy else points_depth

    def get_cols(self, points, truncate=False):
        """ Returns the column indices for unfolding point cloud """

        points, is_numpy = check_numpy_to_torch(points)
       
        # for frontal axis = y
        if self.frontal_axis == 'Y':
            xs = points[:, 1]
            ys = -points[:, 0]
        else:
            xs = points[:, 0]
            ys = points[:, 1]
        
        yaw = -torch.atan2(ys, (xs + 1e-8))

        proj_x = (yaw + abs(self.fov_left)) / self.fov_horizontal  # in [0.0, 1.0]

        proj_x *= self.num_cols
        
        mask = (proj_x >= 0) & (proj_x < self.num_cols)

        if truncate:
            proj_x = torch.maximum(proj_x, torch.zeros_like(proj_x))
            proj_x = torch.minimum(proj_x, torch.ones_like(proj_x) * (self.num_cols - 1))

        if is_numpy:
            proj_x = proj_x.numpy()
            mask = mask.numpy()
            
        return proj_x, mask

    def get_rows(self, points, truncate=False):
        """ Returns the row indices for unfolding point cloud """
        points, is_numpy = check_numpy_to_torch(points)
        
        zs = points[:, 2]
        depth = self.get_depth(points)
        pitch = torch.arcsin(zs / (depth + 1e-8))

        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov_vertical  # in [0.0, 1.0]

        proj_y *= self.num_rows
        
        mask = (proj_y >=0) & (proj_y < self.num_rows)

        if truncate:
            proj_y = torch.maximum(proj_y, torch.zeros_like(proj_y))
            proj_y = torch.minimum(proj_y, torch.ones_like(proj_y) * (self.num_rows - 1))

        if is_numpy:
            proj_y = proj_y.numpy()
            mask = mask.numpy()
        
        return proj_y, mask

class CylindricalProjector(ProjectionBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.z_max = cfg.get('Z_MAX', 1.0)
        self.z_min = cfg.get('Z_MIN', -2.5)
        self.z_range = self.z_max - self.z_min
        self.append_far=cfg.get('APPEND_FAR',False)
        self.fov_left = self.to_radian(cfg.get('FOV_LEFT', 45.0)) # field of view left in rad
        self.fov_right = self.to_radian(cfg.get('FOV_RIGHT', -45.0)) # field of view right in rad
        self.fov_horizontal = abs(self.fov_left) + abs(self.fov_right)  # get field of view horizontal in rad

    def xyz_to_plane(self, points):
        points_xyz, is_numpy = check_numpy_to_torch(points)
        xs, mask_x = self.get_cols(points_xyz)
        ys, mask_y = self.get_rows(points_xyz)
        
        mask = mask_x & mask_y

        depth = self.get_depth(points_xyz)
        points_plane = torch.cat([xs[:, None], ys[:, None], depth[:, None]],dim=-1)
        
        if is_numpy:
            points_plane = points_plane.numpy()
            mask = mask.numpy()

        return points_plane, mask

    def plane_to_xyz(self, points_plane):
        points, is_numpy = check_numpy_to_torch(points_plane)
       
        depth = points[:, 2]
        proj_x = points[:, 0] / self.num_cols # in [0.0, 1.0]
        proj_y = points[:, 1] / self.num_rows # in [0.0, 1.0]

        # proj_x = (yaw + abs(self.fov_left)) / self.fov_horizontal  # in [0.0, 1.0]
        yaw = proj_x * self.fov_horizontal - abs(self.fov_left)
        yaw *= -1.0
        
        xs = depth * torch.cos(yaw)
        ys = depth * torch.sin(yaw)

        # proj_y = 1 - (zs - self.z_min) / self.z_range # in [0.0, 1.0]
        zs = (1 - proj_y) * self.z_range + self.z_min

        points_lidar = torch.cat([xs[:,None], ys[:,None], zs[:,None]], dim=1)
        return points_lidar

    def get_depth(self, points):
        points, is_numpy = check_numpy_to_torch(points)
        points_depth = torch.norm(points[:, :2], p=2, dim=1)
        return points_depth.numpy() if is_numpy else points_depth

    def get_cols(self, points, truncate=False):
        """ Returns the column indices for unfolding point cloud """

        points, is_numpy = check_numpy_to_torch(points)

        # for frontal axis = y
        if self.frontal_axis == 'Y':
            xs = points[:, 1]
            ys = -points[:, 0]
        else:
            xs = points[:, 0]
            ys = points[:, 1]
        
        yaw = -torch.atan2(ys, (xs + 1e-8))

        proj_x = (yaw + abs(self.fov_left)) / self.fov_horizontal  # in [0.0, 1.0]

        proj_x *= self.num_cols
        
        mask = (proj_x >= 0) & (proj_x < self.num_cols)
       
        if truncate:
            proj_x = torch.maximum(proj_x, torch.zeros_like(proj_x))
            proj_x = torch.minimum(proj_x, torch.ones_like(proj_x) * (self.num_cols - 1))
            
        if is_numpy:
            proj_x = proj_x.numpy()
            mask = mask.numpy()

        return proj_x, mask

    def get_rows(self, points, truncate=False):
        """ Returns the row indices for unfolding point cloud """
        points, is_numpy = check_numpy_to_torch(points)
        
        zs = points[:, 2]
        proj_y = 1 - (zs - self.z_min) / self.z_range # in [0.0, 1.0]
        proj_y *= self.num_rows
        
        mask = (proj_y >= 0) & (proj_y < self.num_rows)
        if truncate:
            proj_y = torch.maximum(proj_y, torch.zeros_like(proj_y))
            proj_y = torch.minimum(proj_y, torch.ones_like(proj_y) * (self.num_rows - 1))

        if is_numpy:
            proj_y = proj_y.numpy()
            mask = mask.numpy()

        return proj_y, mask

