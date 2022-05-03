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
    # ------------ single channel version ---------------
    # D = D.reshape((64,64))  # reshape into 2D depth map
    # D_warp = np.zeros((25, 64, 64))

    # for y in range(64):         # y: vertical aperture coordinate
    #     for x in range(64):     # x: horizontal aperture coordinate
    #         D_xy = D[y][x]
    #         for v in range(5):                  # v: vertical aperture coordinate
    #             for u in range(5):              # u: horizontal aperture coordinate
    #                 flat_uv = v*5+u
    #                 y_p, x_p = (np.clip(y + int(v*D_xy), 0, 63), np.clip(x + int(u*D_xy), 0, 63))
    #                 D_warp[flat_uv][y][x] = D[y_p][x_p]
                                
    batch_size = D.shape[0]

    D = D.reshape((batch_size,64,64))  # reshape into 2D depth map
    D_warp = torch.zeros((batch_size, 25, 64, 64))

    for y in range(64):         # y: vertical aperture coordinate
        for x in range(64):     # x: horizontal aperture coordinate
            D_xy = D[:,y,x]
            for v in range(5):                  # v: vertical aperture coordinate
                for u in range(5):              # u: horizontal aperture coordinate
                    flat_uv = v*5+u
                    y_p, x_p = (torch.clamp(y + (v*D_xy).int(), 0, 63), torch.clamp(x + (u*D_xy).int(), 0, 63))
                    for i in range(batch_size):
                        D_warp[i, flat_uv, y, x] = D[i, y_p[i], x_p[i]]
                                
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
    for v in range(5):
        for u in range(5):
            if (v == 0 and u == 0) or (v == 4 and u == 0) or (v == 0 and u == 4) or (v == 4 and u == 4):
                continue    # skip corners for circle aperture
            for y in range(64):
                for x in range(64):
                    flat_uv = v*5+u
                    M_xu = M[:, flat_uv, y, x]
                    y_p, x_p = (torch.clamp(y + (v*M_xu).int(), 0, 63), torch.clamp(x + (u*M_xu).int(), 0, 63))
                    for i in range(batch_size):
                        I_s[i,:,y,x] += I_g[i, :, y_p[i], x_p[i]]
    return I_s


# a = torch.randn(16,25,64,64)
# b = torch.randn(16,3,64,64)
# Render(a, b)

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray 

def testWarp(D):
    # D = D.reshape((64,64))  # reshape into 2D depth map
    orig_shape = D.shape
    D_warp = np.zeros((25, orig_shape[0], orig_shape[1]))

    for y in range(orig_shape[0]):         # y: vertical aperture coordinate
        for x in range(orig_shape[1]):     # x: horizontal aperture coordinate
            print(y,x)
            D_xy = D[y][x]
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

# # ------------ create subaperture views ------------
# VIEWS = np.zeros((5*test.shape[0], 5*test.shape[1]))
# for v in range(5):
#     for u in range(5):
#         flat_uv = v*5+u
#         print(flat_uv)
#         for y in range(test.shape[0]):
#             for x in range(test.shape[1]):
#                 VIEWS[v*test.shape[0] + y][u*test.shape[1] + x] = D_warp[flat_uv][y][x]

# io.imsave("collage.jpeg", VIEWS)


# io.imsave("blah.jpeg", rgb2gray(test))




