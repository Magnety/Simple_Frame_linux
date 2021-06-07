import torch
from torch import nn
import numpy as np
import random
from simple_frame.utilities.to_torch import maybe_to_torch, to_cuda

from simple_frame.byol.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, \
    rotate_coords_2d, rotate_coords_3d, scale_coords, resize_segmentation, resize_multichannel_image, \
    elastic_deform_coordinates_2
class RandCrop_tu(nn.Module):
    def __init__(self,image_size,crop_size):
        super(RandCrop_tu, self).__init__()
        self.image_size = image_size
        self.crop_size = crop_size
    def forward(self, x):
        b,c,d,w,h = x.shape[0],x.shape[1],x.shape[2],x.shape[3],x.shape[4]
        out = torch.zeros((b,c,self.crop_size[0],self.crop_size[1],self.crop_size[2]), device=x.device)
        for _b in range(b):
            w_seed = random.randint(20, 54)
            #print("w_seed:",w_seed)

            h_seed = random.randint(20, 54)
            out[_b,:,:,:,:] = x[_b,:,:,w_seed:w_seed+64,h_seed:h_seed+64]
        return out

class Gaussiannoise_tu(nn.Module):
    def __init__(self,image_size,SNR):
        super(Gaussiannoise_tu, self).__init__()
        self.image_size = image_size
        self.snr = SNR
    def forward(self, x):
        b,c,d,w,h = x.shape[0],x.shape[1],x.shape[2],x.shape[3],x.shape[4]
        out = torch.zeros((b,c,d,w,h), device=x.device)
        for _b in range(b):
            noise = torch.randn((c,d,w,h),device=x.device)
            noise = noise-torch.mean(noise)
            signal_power = torch.linalg.norm(x[_b,:,:,:,:])**2/(c*d*w*h)
            noise_variance = signal_power/np.power(10,(self.snr/10))
            noise = (torch.sqrt(noise_variance)/torch.std(noise))*noise
            out[_b,:,:,:,:] = x[_b,:,:,:,:]+ noise
        return out

class Mirror_tu(nn.Module):
    def __init__(self,image_size):
        super(Mirror_tu, self).__init__()
        self.image_size = image_size
    def forward(self, x):
        b,c,d,w,h = x.shape[0],x.shape[1],x.shape[2],x.shape[3],x.shape[4]
        x_np = x.cpu().numpy()
        out_np = np.zeros((b,c,d,w,h))
        #out = torch.zeros((b,c,d,w,h), device=x.device)
        for _b in range(b):
            x_flag = np.random.uniform()
            y_flag = np.random.uniform()
            if x_flag>0.5:
                out_np[_b,:,:,:,:] = x_np[_b,:,:,:,::-1]
            if y_flag>0.5:
                out_np[_b, :, :, :, :] = out_np[_b,:,:,::-1,:]
        out = torch.from_numpy(out_np)
        out = out.type(torch.HalfTensor)
        out = out.to(x.device)

        return out

class Spatial_tansform_tu(nn.Module):
    def __init__(self,image_size, patch_center_dist_from_border=30,
                    do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                    do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                    do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                    random_crop=True, p_el_per_sample=0.5,
                    p_scale_per_sample=0.5, p_rot_per_sample=0.5, independent_scale_for_each_axis=False,
                    p_rot_per_axis: float = 0.5, p_independent_scale_per_axis: int = 1):
        super(Spatial_tansform_tu, self).__init__()
        self.image_size = image_size
        self.patch_center_dist_from_border=patch_center_dist_from_border
        self.do_elastic_deform=do_elastic_deform
        self.alpha=alpha
        self.sigma=sigma
        self.do_rotation=do_rotation
        self.angle_x=angle_x
        self.angle_y=angle_y
        self.angle_z= angle_z
        self.do_scale=do_scale
        self.scale=scale
        self.border_mode_data=border_mode_data
        self.border_cval_data=border_cval_data
        self.order_data=order_data
        self.random_crop=random_crop
        self.p_el_per_sample=p_el_per_sample
        self.p_scale_per_sample=p_scale_per_sample
        self.p_rot_per_sample=p_rot_per_sample
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.p_rot_per_axis=p_rot_per_axis
        self.p_independent_scale_per_axis=p_independent_scale_per_axis
    def forward(self, x):
        b,c,d,w,h = x.shape[0],x.shape[1],x.shape[2],x.shape[3],x.shape[4]
        x_np = x.cpu().numpy()
        out_np = np.zeros((b,c,d,w,h))
        for _b in range(b):
            coords = create_zero_centered_coordinate_mesh((d,w,h))
            modified_coords = False
            if self.do_elastic_deform and np.random.uniform()<self.p_el_per_sample:
                a = np.random.uniform(self.alpha[0],self.alpha[1])
                s = np.random.uniform(self.sigma[0],self.sigma[1])
                coords = elastic_deform_coordinates(coords,a,s)
                modified_coords = True
            if self.do_rotation and np.random.uniform()<self.p_rot_per_sample:
                if np.random.uniform()<=self.p_rot_per_axis:
                    a_x = np.random.uniform(self.angle_x[0],self.angle_x[1])
                else:
                    a_x = 0
                if np.random.uniform()<=self.p_rot_per_axis:
                    a_y = np.random.uniform(self.angle_y[0],self.angle_y[1])
                else:
                    a_y = 0
                if np.random.uniform()<=self.p_rot_per_axis:
                    a_z = np.random.uniform(self.angle_z[0],self.angle_z[1])
                else:
                    a_z = 0
                coords = rotate_coords_3d(coords,a_x,a_y,a_z)
                modified_coords = True
            if self.do_scale and np.random.uniform()<self.p_scale_per_sample:
                if self.scale[0]<1:
                    sc = np.random.uniform(self.scale[0],1)
                else:
                    sc = np.random.uniform(max(self.scale[0],1),self.scale[1])
                coords = scale_coords(coords,sc)
                modified_coords = True
            if modified_coords:
                out_np[_b,0,:,:,:] = interpolate_img(x_np[_b,0],coords,self.order_data,self.border_mode_data,cval=self.border_cval_data)
            out = torch.from_numpy(out_np)
            out = out.type(torch.HalfTensor)
            out = out.to(x.device)
        return out
