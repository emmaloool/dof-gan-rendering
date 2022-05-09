import numpy as np
from skimage import io, color

import copy, os, time, math
from scipy import interpolate, signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from cp_hw2 import XYZ2lRGB, lRGB2XYZ
from cv2 import GaussianBlur
import torch

from scipy import misc

'''
    Warp depth map (1x64x64) to create a depth map for each aperture view
    for a (5x5) square aperture: D(x + u D(x))

    Note that the output channel dimension is the flattened aperture coordinates
    
    Input
    -----
        D: B x 1 x 64 x 64

    Output
    ------
        D_warp: B x 25 x 64 x 64    (flat_uv x H X W)
'''
def warp_depthmap(D):
                                
    # batch_size = D.shape[0]

    # D = D.reshape((batch_size,64,64))  # reshape into 2D depth map
    # D_warp = torch.zeros((batch_size, 25, 64, 64))

    # for y in range(64):         # y: vertical aperture coordinate
    #     for x in range(64):     # x: horizontal aperture coordinate
    #         D_xy = D[:,y,x]
    #         for v in range(5):                  # v: vertical aperture coordinate
    #             for u in range(5):              # u: horizontal aperture coordinate
    #                 flat_uv = v*5+u
    #                 y_p, x_p = (torch.clamp(y + (v*D_xy).int(), 0, 63), torch.clamp(x + (u*D_xy).int(), 0, 63))
    #                 for i in range(batch_size):
    #                     D_warp[i, flat_uv, y, x] = D[i, y_p[i], x_p[i]]
                                
    # return D_warp
    
    # initialize
    batch_size = D.shape[0]
    device = D.device
    # 1. get u and v values
    uv_range = torch.arange(0, 5, device=device)
    vs, us = torch.meshgrid(uv_range, uv_range)
    # 2. offset so that aperture center is 0
    vs = vs - 2
    us = us - 2
    # 3. broadcast u and v values for each depth pixel
    vs = vs.reshape(1, 25, 1, 1).repeat(batch_size, 1, 64, 64)
    us = us.reshape(1, 25, 1, 1).repeat(batch_size, 1, 64, 64)
    # 4. get pixel indices
    xy_range = torch.arange(0, 64, device=device)
    ys, xs = torch.meshgrid(xy_range, xy_range)
    # 5. broadcast x and y values for each depth pixel
    ys = ys.reshape(1, 1, 64, 64).repeat(batch_size, 25, 1, 1)
    xs = xs.reshape(1, 1, 64, 64).repeat(batch_size, 25, 1, 1)
    # 6. get x_p and y_p for each depth pixel
    yps = torch.clamp((ys + vs * D), 0, 63).long()
    xps = torch.clamp((xs + us * D), 0, 63).long()
    # 7. combine x_p and y_p to get offset for each depth subview
    ps = yps * 64 + xps # BS x 25 x x 64 x 64

    # Indexing D_warp from D
    # 1. set up indices for flattened D_warp and D
    bs_arr = torch.arange(0, batch_size, device=device).long().reshape(batch_size, 1, 1, 1).repeat(1, 25, 64, 64).flatten()
    ap_arr = torch.arange(0, 25, device=device).long().reshape(1, 25, 1, 1).repeat(batch_size, 1, 64, 64).flatten()
    xy_arr = ps.flatten()
    inds = bs_arr * (25 * 64 * 64) + ap_arr * (64 * 64) + xy_arr
    # 2. copy D to the same shape as D_warp
    Ds = D.repeat(1, 25, 1, 1)
    # 3. index D_warp
    D_warp_flat = Ds.flatten()[inds]
    # 4. reshape D_warp and return
    D_warp = D_warp_flat.reshape(batch_size, 25, 64, 64)
    return D_warp



'''
    Render a lightfield image, using depth map for each view in the lightfield M
    and deep DoF image D 

    Input
    -----
        M: B x 25 x 64 x 64, learned depth map for all aperture subviews
        I_g: B x 3 x 64 x 64, deep DoF image

    Output
    ------
        I_s: B x 3 x 64 x64, shallow DoF image
'''
def Render(M, I_g):
    I_s = torch.zeros_like(I_g)
    batch_size = I_s.shape[0]

    # Integrate over aperture subviews
    # for v in range(5):
    #     for u in range(5):
    #         if (v == 0 and u == 0) or (v == 4 and u == 0) or (v == 0 and u == 4) or (v == 4 and u == 4):
    #             continue    # skip corners for circle aperture
    #         for y in range(64):
    #             for x in range(64):
    #                 flat_uv = v*5+u
    #                 M_xu = M[:, flat_uv, y, x]
    #                 y_p, x_p = (torch.clamp(y + (v*M_xu).int(), 0, 63), torch.clamp(x + (u*M_xu).int(), 0, 63))
    #                 for i in range(batch_size):
    #                     I_s[i,:,y,x] += I_g[i, :, y_p[i], x_p[i]]

    # initialize
    device = I_g.device
    batch_size = I_s.shape[0]

    # 1. get u and v values
    uv_range = torch.arange(0, 5, device=device)
    vs, us = torch.meshgrid(uv_range, uv_range)
    # 2. offset so that aperture center is 0
    vs = vs - 2
    us = us - 2
    # 3. broadcast u and v values for each image pixel
    vs = vs.reshape(1, 25, 1, 1, 1).repeat(batch_size, 1, 3, 64, 64)
    us = us.reshape(1, 25, 1, 1, 1).repeat(batch_size, 1, 3, 64, 64)
    # 4. get pixel indices
    xy_range = torch.arange(0, 64, device=device)
    ys, xs = torch.meshgrid(xy_range, xy_range)
    # 5. broadcast x and y values for each image pixel
    ys = ys.reshape(1, 1, 1, 64, 64).repeat(batch_size, 25, 3, 1, 1)
    xs = xs.reshape(1, 1, 1, 64, 64).repeat(batch_size, 25, 3, 1, 1)
    # 6. broadcast M for each image pixel
    M = M.reshape(batch_size, 25, 1, 64, 64).repeat(1, 1, 3, 1, 1)
    # 7. get x_p and y_p for each image pixel
    yps = torch.clamp((ys + vs * M), 0, 63).long()
    xps = torch.clamp((xs + us * M), 0, 63).long()
    # 8. combine x_p and y_p to get offset for each image
    ps = yps * 64 + xps # BS x 25 x 3 x 64 x 64

    # index I_g
    # 1. construct array for indices of flattened I_g
    bs_arr = torch.arange(0, batch_size, device=device).long().reshape(batch_size, 1, 1, 1, 1).repeat(1, 25, 3, 64, 64).flatten()
    ap_arr = torch.arange(0, 25, device=device).long().reshape(1, 25, 1, 1, 1).repeat(batch_size, 1, 3, 64, 64).flatten()
    co_arr = torch.arange(0, 3, device=device).long().reshape(1, 1, 3, 1, 1).repeat(batch_size, 25, 1, 64, 64).flatten()
    xy_arr = ps.flatten()
    inds = bs_arr * (25 * 3 * 64 * 64) + ap_arr * (3 * 64 * 64) + co_arr * (64 * 64) + xy_arr
    inds = torch.clamp(inds, 0, batch_size * 25 * 3 * 64 * 64 - 1)
    # 2. expand I_g
    I_g_exp = I_g.reshape(batch_size, 1, 3, 64, 64).repeat(1, 25, 1, 1, 1)
    # 3. index I_g
    I_s = I_g_exp.flatten()[inds]
    I_s = I_s.reshape(batch_size, 25, 3, 64, 64)
    # 4. construct aperture circle
    # hardcoded
    aperture_range = torch.ones_like(M)
    aperture_range[:,0,:,:,:] = 0
    aperture_range[:,4,:,:,:] = 0
    aperture_range[:,20,:,:,:] = 0
    aperture_range[:,24,:,:,:] = 0
    # 5. mask I_s
    I_s = I_s * aperture_range
    # 6. integrate over aperture range
    I_s = torch.sum(I_s, 1) / 21.

    return I_s
















    
    return I_s




# ---------------------------- manual testing ----------------------------

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray 
    
# testWarp is the same as warp_depthmap, but supports images that aren't 64x64 lol
def testWarp(D):
    orig_shape = D.shape
    D_warp = np.zeros((25, orig_shape[0], orig_shape[1]))

    for y in range(orig_shape[0]):         # y: vertical aperture coordinate
        for x in range(orig_shape[1]):     # x: horizontal aperture coordinate
            print(y,x)
            D_xy = 1.0/D[y][x]
            if D_xy == np.inf: D_xy = 0
            for v in range(5):                  # v: vertical aperture coordinate
                for u in range(5):              # u: horizontal aperture coordinate
                    flat_uv = v*5+u
                    y_p, x_p = (np.clip(y + int(v*D_xy), 0, orig_shape[0]-1), np.clip(x + int(u*D_xy), 0, orig_shape[1]-1))
                    D_warp[flat_uv][y][x] = D[y_p][x_p]

    return D_warp


# test = rgb2gray(io.imread("depth.jpeg"))
# D_warp = testWarp(test)
# D_warp = (D_warp - np.min(D_warp) )/ (np.max(D_warp) - np.min(D_warp) )

# print("done")

# # # ------------ create subaperture views ------------
# VIEWS = np.zeros((5*test.shape[0], 5*test.shape[1]))
# for v in range(5):
#     for u in range(5):
#         flat_uv = v*5+u
#         print(flat_uv)
#         for y in range(test.shape[0]):
#             for x in range(test.shape[1]):
#                 VIEWS[v*test.shape[0] + y][u*test.shape[1] + x] = D_warp[flat_uv][y][x]

# io.imsave("collage.jpeg", VIEWS)




