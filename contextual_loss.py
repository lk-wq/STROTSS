import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import utils_go
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss


def pairwise_distances_sq_l2_s(x, y):
    x_norm = (x.float() ** 2).sum(1).view(-1, 1).half()
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y.float() ** 2).sum(1).view(1, -1).half()

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return (torch.clamp(dist.float(), 1e-5, 1e5).float() / x.size(1)).half()


def pairwise_distances_cos_s(x, y):
    x_norm = torch.sqrt((x.float() ** 2).sum(1).view(-1, 1)).half()
    y_t = torch.transpose(y, 0, 1)

    y_norm = torch.sqrt((y.float() ** 2).sum(1).view(1, -1)).half()

    dist = 1. - (torch.mm(x / x_norm, y_t / y_norm))

    return dist.half()


def pairwise_distances_sq_l2(x, y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return (torch.clamp(dist.float(), 1e-5, 1e5) / x.size(1))


def pairwise_distances_cos(x, y):
    x_norm = torch.sqrt((x ** 2).sum(1).view(-1, 1))
    y_t = torch.transpose(y, 0, 1)

    y_norm = torch.sqrt((y ** 2).sum(1).view(1, -1))

    dist = 1. - torch.mm(x, y_t) / x_norm / y_norm

    return dist


def pairwise_distances_cos2(x, y):
    x = x.squeeze()
    y = y.squeeze()
    x_norm = torch.sqrt((x ** 2).sum(1).view(-1, 1))
    y_t = torch.transpose(y, 0, 1)
    y_norm = torch.sqrt((y ** 2).sum(1).view(1, -1))

    dist = 1. - torch.mm(x, y_t) / x_norm / y_norm

    # print("dist size : ", dist.unsqueeze(0).size())

    return dist.unsqueeze(0)


def pairwise_distances_cos_and_l2(x, y):
    x = x.squeeze()
    y = y.squeeze()
    x_norm = torch.sqrt((x ** 2).sum(1).view(-1, 1))
    y_t = torch.transpose(y, 0, 1)
    y_norm = torch.sqrt((y ** 2).sum(1).view(1, -1))

    dist = 1. - torch.mm(x, y_t) / x_norm / y_norm

    dist_l2_pre = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    dist_l2 = (torch.clamp(dist_l2_pre.float(), 1e-5, 1e5) / x.size(1))

    return (dist + dist_l2).unsqueeze(0)


def get_DMat(scl, X, Y, h=1.0, cb=0, splits=[128 * 3 + 256 * 3 + 512 * 4], cos_d=True):
    n = X.size(0)
    m = Y.size(0)
    M = utils_go.match_device(X, Variable(torch.zeros(n, m)) )
       
    
    if scl == 1:
        pre_init = 0
        init = 1
        second = 10
    elif scl == 2:
        pre_init = 0

        init = 1
        second = 10
    elif scl == 3:
        pre_init = 0

        init = 3
        second = 10
    elif scl == 4:
        pre_init = 0

        init = 2
        second = 11
    elif scl == 5:
        pre_init = 0

        init = 3
        second = 13
    elif scl == 6:
        pre_init = 0

        init = 5
        second = 14
    elif scl == 7:
        pre_init = 0

        init = 3
        second = 13
    elif scl == 8:
        pre_init = 0

        init = 2
        second = 13
    elif scl == 9:
        pre_init = 0

        init = 2
        second = 13
    elif scl == 10:
        pre_init = 0

        init = 0
        second = 11
    elif scl == 11:
        pre_init = 5

        init = 7
        second = 8

    if 1:
        cb = 0
        ce = 0
        for i in range(len(splits)):
            #print("i",i, M.get_device())
                
            if cos_d:

                ce = cb + splits[i]
                if len(splits) > 1:
                    if i>= pre_init and i < init:
                        M = utils_go.to_device(M) + pairwise_distances_cos(utils_go.to_device(X[:, cb:ce]),utils_go.to_device( Y[:, cb:ce]) )
                    elif i >= init and i < second:
                        M = utils_go.to_device2(M) + pairwise_distances_cos(utils_go.to_device2(X[:, cb:ce]),utils_go.to_device2( Y[:, cb:ce]) )
                    elif i >= second:
                        M = utils_go.to_device3(M) + pairwise_distances_cos(utils_go.to_device3(X[:, cb:ce]),utils_go.to_device3( Y[:, cb:ce]) )
                    else:
                        M = utils_go.to_device1(M) + pairwise_distances_cos(utils_go.to_device1(X[:, cb:ce]),utils_go.to_device1( Y[:, cb:ce]) )
                else:
                    M = M + pairwise_distances_cos(X[:, cb:ce], Y[:, cb:ce])
                cb = ce

            else:

                ce = cb + splits[i]

                M = M + torch.sqrt(pairwise_distances_sq_l2(X[:, cb:ce], Y[:, cb:ce]))

                cb = ce

    return M


def viz_d(zx, coords):
    viz = zx[0][:, :1, :, :].clone() * 0.

    for i in range(coords.shape[0]):
        vizt = zx[0][:, :1, :, :].clone() * 0.

        for z in zx:
            cx = int(coords[i, 0] * z.size(2))
            cy = int(coords[i, 1] * z.size(3))

            anch = z[:, :, cx:cx + 1, cy:cy + 1]
            x_norm = torch.sqrt((z ** 2).sum(1, keepdim=True))
            y_norm = torch.sqrt((anch ** 2).sum(1, keepdim=True))
            dz = torch.sum(z * anch, 1, keepdim=True) / x_norm / y_norm
            vizt = vizt + F.upsample(dz, (viz.size(2), viz.size(3)), mode='bilinear') * z.size(1)

        viz = torch.max(viz, vizt / torch.max(vizt))

    vis_o = viz.clone()
    viz = viz.data.cpu().numpy()[0, 0, :, :] / len(zx)
    return vis_o


def remd_loss_OT(long_side, cut, scl, X1, Y1, h=None, cos_d=True,
                 splits=[3 + 64 + 64 + 128 + 128 + 256 + 256 + 256 + 512 + 512],
                 return_mat=False):
    d = X1.size(1)

    if d == 3:
        X1 = utils_go.rgb_to_yuv_pc(X1.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
        Y1 = utils_go.rgb_to_yuv_pc(Y1.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
    else:
        X1 = X1.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
        Y1 = Y1.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

    loss_out = 0
    if d == 3:
        loss1 = SamplesLoss(loss="sinkhorn", p=2, blur=.01, cost=pairwise_distances_cos_and_l2)
    else:
        loss1 = SamplesLoss(loss="sinkhorn", p=2, blur=.01, cost=pairwise_distances_cos2)

    split_start = 0
    for i in splits:
        split_end = split_start + i
        Xnew = X1[:, split_start:split_end]
        Ynew = Y1[:, split_start:split_end]
        if d == 3:

            loss_out_1 = utils_go.to_device1(
                loss1(utils_go.to_device(Xnew.unsqueeze(0)), utils_go.to_device(Ynew.unsqueeze(0))))
            loss_out += loss_out_1  # + loss_out_2/2

        else:

            loss_out_1 = utils_go.to_device1(
                loss1(utils_go.to_device(Xnew.unsqueeze(0)), utils_go.to_device(Ynew.unsqueeze(0))))
            loss_out += loss_out_1  # + loss_out_2/2

        split_start = split_end
    # print("loss out : " , loss_out)
    return loss_out.unsqueeze(0)


def remd_loss(long_side, cut, scl, X, Y, h=None, cos_d=True,
              splits=[3 + 64 + 64 + 128 + 128 + 256 + 256 + 256 + 512 + 512], return_mat=False):
    d = X.size(1)

    if d == 3:

        X = utils_go.rgb_to_yuv_pc(X.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
        Y = utils_go.rgb_to_yuv_pc(Y.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)

    else:

        X = X.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
        Y = Y.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

    # Relaxed EMD
    CX_M = get_DMat(scl, X, Y, 1., cos_d=True, splits=splits)

    if return_mat:
        return CX_M

    if d == 3:
        CX_M = CX_M + get_DMat(scl, X, Y, 1., cos_d=False, splits=splits)

    m1, m1_inds = CX_M.min(1)
    m2, m2_inds = CX_M.min(0)
    if m1.mean() > m2.mean():
        used_style_feats = Y[m1_inds, :]
    else:
        used_style_feats = Y

    remd = torch.max(m1.mean(), m2.mean())

    return remd, used_style_feats

def remd_lossg(long_side, cut, scl, X, Y, h=None, cos_d=True,
               splits=[3 + 64 + 64 + 128 + 128 + 256 + 256 + 256 + 512 + 512], return_mat=False):
    d = X.size(1)

    if d == 3:
        X = utils_go.rgb_to_yuv_pc(X.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
        Y = utils_go.rgb_to_yuv_pc(Y.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)

    else:

        X = X.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
        Y = Y.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

    # Relaxed EMD
    if 1:
        CX_M = get_DMat(scl, X, Y, 1., cos_d=True, splits=splits)
    else:
        CX_M = get_DMat_small(scl, X, Y, 1., cos_d=True, splits=splits)

    if return_mat:
        return CX_M

    if d == 3:
        if 1:
            CX_M = CX_M + get_DMat(scl, X, Y, 1., cos_d=False, splits=splits)

        else:
            CX_M = CX_M + get_DMat_small(scl, X, Y, 1., cos_d=False, splits=splits)

    m1, m1_inds = CX_M.min(1)
    m2, m2_inds = CX_M.min(0)
    if m1.mean() > m2.mean():
        # print("y size : ", Y.size())
        used_style_feats = Y[m1_inds, :]
    else:
        used_style_feats = Y

    remd = torch.max(m1.mean(), m2.mean())

    return remd, used_style_feats

def remd_loss_g(X, Y, GX, GY, h=1.0, splits=[3 + 64 + 64 + 128 + 128 + 256 + 256 + 256 + 512 + 512]):
    d = X.size(1)

    if d == 3:
        X = utils_go.rgb_to_yuv_pc(X.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
        Y = utils_go.rgb_to_yuv_pc(Y.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
        GX = utils_go.rgb_to_yuv_pc(GX.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
        GY = utils_go.rgb_to_yuv_pc(GY.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)

    else:
        X = X.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
        Y = Y.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
        GX = GX.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
        GY = GY.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

    c1 = 10000.
    c2 = 1.

    CX_M = get_DMat(scl, X, Y, 1., cos_d=True, splits=splits)

    if d == 3:
        CX_M = CX_M + get_DMat(scl, X, Y, 1., cos_d=False, splits=splits)

    CX_M_2 = get_DMat(scl, GX, GY, 1., cos_d=True, splits=splits) + get_DMat(scl, GX, GY, 1., cos_d=False,
                                                                             splits=splits)  # CX_M[i:,i:].clone()
    for i in range(GX.size(0) - 1):
        CX_M_2[(i + 1):, i] = CX_M_2[(i + 1):, i] * 1000.
        CX_M_2[i, (i + 1):] = CX_M_2[i, (i + 1):] * 1000.

    m1, m1_inds = CX_M.min(1)
    m2, m2_inds = CX_M.min(0)
    m2, min_inds = torch.topk(m2, m1.size(0), largest=False)

    if m1.mean() > m2.mean():
        used_style_feats = Y[m1_inds, :]
    else:
        used_style_feats = Y[min_inds, :]

    m12, _ = CX_M_2.min(1)
    m22, _ = CX_M_2.min(0)

    used_style_feats = Y[m1_inds, :]
    remd = torch.max(m1.mean() * h, m2.mean()) + c2 * torch.max(m12.mean() * h, m22.mean())

    return remd, used_style_feats

def moment_loss(scl, X, Y, moments=[1, 2]):
    d = X.size(1)
    ell = 0.

    Xo = X.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
    Yo = Y.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

    # splits = [Xo.size(1)]

    if scl <= 3:
        splits = [3] + [96] + [168] + [336] + [1008] * 2 + [1008] * 3 + [1008] + [1344] + [2016] * 6
    elif scl == 11:
        splits = [3] + [96] + [168] + [336] + [1008] * 2 + [1008] * 3 + [1008] + [1344] + [2016] * 6 + [2688] + [
            4032] * 4
    else:
        splits = [3] + [96] + [168] + [336] + [1008] * 2 + [1008] * 3 + [1008] + [1344] + [2016] * 6 + [2688] + [
            4032] * 3
    
    cb = 0
    ce = 0
    for i in range(len(splits)):
        ce = cb + splits[i]
        
        #print("in moment loss", i)
        
        if scl == 6:
            pre_init = 0
            init = 6
            second = 100
            third = 100

        elif scl == 7:
            second = 100
            third = 100

            pre_init = 0
            init = 6
        elif scl == 8:
            second = 100
            third = 100

            pre_init = 0
            init = 0
        elif scl == 9:
            second = 100
            third = 100

            pre_init = 0
            init = 0
        elif scl == 10:
            second = 100
            third = 100

            pre_init = 0
            init = 0
        elif scl == 11:
            pre_init = 0
            init = 10
            second = 22
            third = 100

        else:
            pre_init = 0
            init = 1000
            
        if i>= pre_init and i < init:
            X = utils_go.to_device(Xo[:, cb:ce])
            Y = utils_go.to_device(Yo[:, cb:ce])
            func = utils_go.to_device
        #elif i >= init and i < second:
        #    X = utils_go.to_device2(Xo[:, cb:ce])
        #    Y = utils_go.to_device2(Yo[:, cb:ce])
        elif i >= init and i < second:
            X = utils_go.to_device3(Xo[:, cb:ce])
            Y = utils_go.to_device3(Yo[:, cb:ce])
            func = utils_go.to_device
        elif i >= second and i < third:
            X = utils_go.to_device1(Xo[:, cb:ce])
            Y = utils_go.to_device1(Yo[:, cb:ce])
            func = utils_go.to_device
        elif i >= third:
            X = utils_go.to_device2(Xo[:, cb:ce])
            Y = utils_go.to_device2(Yo[:, cb:ce])
            func = utils_go.to_device
            
            
        #else:
        #    X = utils_go.to_device1(Xo[:, cb:ce])
        #    Y = utils_go.to_device1(Yo[:, cb:ce])

        cb = ce

        mu_x = torch.mean(X, 0, keepdim=True)
        mu_y = torch.mean(Y, 0, keepdim=True)
        mu_d = torch.abs(mu_x - mu_y).mean()

        if 1 in moments:
            ell = ell + func(mu_d)

        if 2 in moments:
            sig_x = (torch.mm((X - mu_x).transpose(0, 1), (X - mu_x)) / X.size(0))
            sig_y = (torch.mm((Y - mu_y).transpose(0, 1), (Y - mu_y)) / Y.size(0))

            sig_d = torch.abs(sig_x - sig_y).mean()
            ell = ell + func(sig_d)

    return ell / len(splits)

def moment_loss_g(X, Y, GX, moments=[1, 2]):
    d = X.size(1)
    ell = 0.

    Xo = X.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
    Yo = Y.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
    GXo = GX.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

    betas = torch.pow(get_DMat(scl, Xo, GXo), 1)
    betas, _ = torch.max(betas, 1)
    betas = betas.unsqueeze(1).detach()
    betas = betas * torch.ge(betas, 0.2).float()

    splits = [Xo.size(1)]
    cb = 0
    ce = 0
    for i in range(len(splits)):
        ce = cb + splits[i]
        X = Xo[:, cb:ce]
        Y = Yo[:, cb:ce]
        cb = ce

        mu_x = torch.sum(betas * X, 0, keepdim=True) / torch.sum(betas)
        mu_y = torch.mean(Y, 0, keepdim=True)
        mu_d = torch.abs(mu_x - mu_y).mean()

        if 1 in moments:
            ell = ell + mu_d

        if 2 in moments:
            sig_x = torch.mm(((betas * X - mu_x)).transpose(0, 1), (betas * X - mu_x)) / torch.sum(torch.pow(betas, 2))
            sig_y = torch.mm((Y - mu_y).transpose(0, 1), (Y - mu_y)) / Y.size(0)

            sig_d = torch.abs(sig_x - sig_y).mean()
            ell = ell + sig_d

    return ell


def dp_loss(long_side, cut, scl, X, Y):
    d = X.size(1)

    X = X.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
    Y = Y.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

    Xc = X[:, -2:]
    Y = Y[:, :-2]
    X = X[:, :-2]

    if 0:
        dM = torch.exp(-2. * get_DMat(scl, Xc, Xc, 1., cos_d=False).float()).half()
        dM = dM / dM.sum(0, keepdim=True).detach() * dM.size(0)

    else:
        dM = 1.

    if 1:

        Mx = get_DMat(scl, X, X, 1., cos_d=True, splits=[X.size(1)])
        # print("Mx sum : ", Mx.sum(0, keepdim=True).max(0) )
        Mx = Mx / Mx.sum(0, keepdim=True)

        My = get_DMat(scl, Y, Y, 1., cos_d=True, splits=[X.size(1)])
        My = My / My.sum(0, keepdim=True)

        d = torch.abs(dM * (Mx - My)).mean() * X.size(0)

        return d
    else:
        Mx = get_DMat_small(scl, X, X, 1., cos_d=True, splits=[X.size(1)])
        Mx = Mx / Mx.sum(0, keepdim=True)

        My = get_DMat_small(scl, Y, Y, 1., cos_d=True, splits=[X.size(1)])
        My = My / My.sum(0, keepdim=True)

        d = torch.abs(dM * (Mx - My)).mean() * X.size(0)

        return d

def dp_loss_g(X, Y, GX):
    d = X.size(1)

    X = X.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
    Y = Y.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
    GX = GX.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

    betas, _ = torch.max(torch.pow(get_DMat(scl, X, GX), 1), 1)
    betas = betas.unsqueeze(1).detach()
    betas = torch.matmul(betas, betas.transpose(0, 1))

    Mx = get_DMat(scl, X, X, 1., splits=[X.size(1)])
    Mx = Mx / Mx.sum(0, keepdim=True)

    My = get_DMat(scl, Y, Y, 1., splits=[X.size(1)])
    My = My / My.sum(0, keepdim=True)

    d = torch.abs(betas * (Mx - My)).sum(0).mean()

    return d
