from skimage import io
import numpy as np
from scipy import interpolate as interp
import helpers

def import_nerf_lf(path, bsize=16, offset=0):
    shape_test = io.imread(path + '0_0.jpg')
    s, t, c = np.shape(shape_test)

    L = np.zeros((bsize, bsize, s, t, c))
    
    for u in range(offset, offset+bsize):
        for v in range(offset, offset+bsize):
            subimg = io.imread(path + str(u) + '_' + str(v) + '.jpg')
            L[v-offset, u-offset, :, :, :] = subimg
            # M = rearrange(L) / 255.
            # io.imshow(M)
            # io.show()
    
    return L

def rearrange(L):
    #get shape of lightfield
    shape = np.shape(L)
    bh = shape[0]
    bw = shape[1]
    lh = shape[2]
    lw = shape[3]
    c = shape[4]

    #get shape of mosaic
    mh = lh * bh
    mw = lw * bw

    #generate mosaic image
    M = np.zeros((mh, mw, c))
    for u in range(bh):
        for v in range(bw):
            #find block in mosaic image
            iu = lh * u
            iv = lw * v
            #copy block from lightfield image to mosaic image
            M[iu:iu+lh, iv:iv+lw, :] = L[u, v, :, :, :]
    
    return M

#get interpolation functions - one for each RGB channel
def get_interp_fns(L):
    #dictionary of functions mapped to (u, v) pairs
    fns = {}

    shape = np.shape(L)
    bh = shape[0]
    bw = shape[1]
    ih = shape[2]
    iw = shape[3]
    c =  shape[4]

    for u in range(bh):
        for v in range(bw):
            #separate channels
            sub_img = L[u, v, :, :, :]
            r, g, b = helpers.get_channels(sub_img)
            
            #perform shift using interpolate
            x = np.arange(iw)
            y = np.arange(ih)

            fr = interp.interp2d(x, y, r)
            fg = interp.interp2d(x, y, g)
            fb = interp.interp2d(x, y, b)

            f_rgb = {'r': fr, 'g': fg, 'b': fb}
            fns[(u, v)] = f_rgb

    return fns

#refocus using interpolation functions
#aperture from 2-16, smaller D is closer focus
def refocus_fn(L, fns, d, aperture=-1, amode="none"):
    shape = np.shape(L)
    bh = shape[0]
    bw = shape[1]
    ih = shape[2]
    iw = shape[3]
    c =  shape[4]

    shift_img = np.zeros((ih, iw, c))

    #get real offsets based on lenslet size
    lensletSize = bh
    maxUV = (lensletSize - 1) / 2.

    if (aperture == -1 or aperture >= bh):
        urange = np.arange(bh)
        vrange = np.arange(bw)
    elif (amode == "rect"):
        urange = np.arange(maxUV - aperture / 2, maxUV + aperture / 2 + 1)
        vrange = urange
    elif (amode == "circ"):
        fullrange = np.arange(bh)
        urange = np.empty(0)
        vrange = np.empty(0)
        for u in fullrange:
            for v in fullrange:
                uoffset = u - maxUV
                voffset = v - maxUV
                if (np.sqrt(uoffset ** 2 + voffset ** 2) <= aperture / 2):
                    urange = np.append(urange, u)
                    vrange = np.append(vrange, v)
        
        if (len(urange) == 0 or len(vrange) == 0):
            urange = np.arange(bh)
            vrange = np.arange(bw)
    else:
        urange = np.arange(bh)
        vrange = np.arange(bw)
        
    for u in urange.astype(np.int32):
        for v in vrange.astype(np.int32):
            #perform shift using interpolate
            x = np.arange(iw)
            y = np.arange(ih)

            f_rgbs = fns[(u, v)]
            fr = f_rgbs['r']
            fg = f_rgbs['g']
            fb = f_rgbs['b']

            dv = v - maxUV
            du = u - maxUV

            sx = x + (d * dv)
            sy = y - (d * du)

            sr = fr(sx, sy)
            sg = fg(sx, sy)
            sb = fb(sx, sy)

            shift_img += helpers.stack_channels(sr, sg, sb)
    
    shift_img /= (np.size(urange) * np.size(vrange))
    return shift_img

L = import_nerf_lf('lf/', bsize=5) / 255.
fns = get_interp_fns(L)

M = rearrange(L)
io.imsave('lf_mosaic.jpg', M)

# different focus distances
img = refocus_fn(L, fns, d=-6, aperture=-1, amode="circ")
for d in range(-5, -1):
        img = np.append(img, refocus_fn(L, fns, d=d, aperture=-1, amode="circ"), axis=1)

io.imshow(img)
io.show()
io.imsave('lf_focal_dists.jpg', img)

img = refocus_fn(L, fns, d=-4.5, aperture=5, amode='circ')
io.imsave('lf_example.jpg', img)
for a in range(4, 0, -1):
    img = np.append(img, refocus_fn(L, fns, d=-4, aperture=a, amode='circ'), axis=1)

io.imshow(img)
io.show()
io.imsave('lf_apertures.jpg', img)


    