import math

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from .contextual_loss import *
from . import utils

use_random=True

class objective_class():

    def __init__(self, objective='remd_dp'):
import math

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from contextual_loss import *
import utils_go

'''
class Depth_Prepare(nn.Module):
    def __init__(self):
        super(Depth_Prepare, self).__init__()
        self.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, input):
        print("size of input is : " , input.size())

        out = F.upsample(input, size=(228, 304), mode='bilinear', align_corners=False)
        out = self.normalizer(out)
        #out = torch.cat([torch_img, torch.ones_like(torch_img[:,0,:,:]).unsqueeze(0)],dim=1)
        print("out size is : ", out.size())
        return out

DP = Depth_Prepare()
depth_model = define_model(is_resnet=True, is_densenet=False, is_senet=False)
depth_model = torch.nn.DataParallel(depth_model).cuda()
depth_model.load_state_dict(torch.load('/content/pretrained_model/model_resnet'))
'''


class objective_class():

    def __init__(self, objective='remd_dp'):

        self.z_dist = utils_go.to_device1(torch.zeros(1))

        self.rand_ixx = {}
        self.rand_ixy = {}
        self.rand_iy = {}

        if objective == 'remd_dp':
            self.eval = self.gen_remd_dp_objective

        if objective == 'remd_dp_g':
            self.eval = self.gen_remd_dp_objective_guided

    def gen_remd_dp_objective_guided(self, long_side, cut, scl, z_x_content, z_x_style, z_c, z_s, gz, d_ell,
                                     d_weight=10.0,
                                     content_weight=4.0,
                                     moment_weight=1.0, style_loss_func=remd_loss, content_loss_func=dp_loss, h=1.0):

        ## Extract Random Subset of Features from Stylized Image & Content Image ##
        # (Choose Features from Same locations in Stylized Image & Content Image) #
        final_loss = 0.
        for ri in range(len(self.rand_ixx.keys())):
            '''
            x = new_img.clone()
            image = DP(x)
            output = depth_model(image)
            content_DP = DP(new_content)

            depth = depth_model(content_DP)

            ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().cuda()
            ones = torch.autograd.Variable(ones)

            depth_grad = get_gradient(depth)
            output_grad = get_gradient(output)
            depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
            depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
            output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
            output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

            depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
            output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

            # depth_normal = F.normalize(depth_normal, p=2, dim=1)
            # output_normal = F.normalize(output_normal, p=2, dim=1)

            # loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()
            # loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
            # loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
            # loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()

            loss_depth = (torch.abs(output - depth) + 0.5).mean()
            loss_dx = (torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
            loss_dy = (torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
            loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()

            loss_content_depth = (loss_depth + loss_normal + (loss_dx + loss_dy)) * 10
            # loss_ok = loss.clone().detach()
            # loss_content_depth = loss_content_depth.clamp(-loss_ok*0.01, loss_ok*0.05)
            # loss_content_depth += 1000 * mse(depth_img, content_depth_img)

            print("loss_content_depth : ", loss_content_depth)
            '''
            # print("this is self.rand_ixx.keys() : " , self.rand_ixx.keys() )
            # print("self.rand_ixx : " , self.rand_ixx[0])
            # print("z_x_style : ", z_x_style[0].size())
            # print("z_x_content : " , z_x_content[0].size())
            xx, xy, yx = self.get_feature_inds(scl, ri=ri)
            # print("this is ri : " , ri)
            # print("thi is self.get_feature_inds 1 : " , len(xx) )
            # print("thi is self.get_feature_inds 2 : " , len(xy) )
            # print("thi is self.get_feature_inds 3 : " , len(yx) )
            # print("z_x length : " , len(z_x))
            # print("z_c length : " , len(z_c))
            # print("this is z_x : " , z_x[0].size())
            # print("this is z_c : " , z_c[0].size())
            # for i in range(0,  len(z_x_content) ):
            #    print("z x content : " , z_x_content[i].size())
            # for i in z_x_content:
            #    print("slot size : " , i.size())

            if 1:
                x_st, c_st = self.spatial_feature_extract(z_x_content, z_c, xx, xy, scl)
            else:
                x_st, c_st = self.spatial_feature_extract_small(z_x_content, z_c, xx, xy, scl)

            if gz.sum() > 0.:
                gxx, gxy = self.get_feature_inds_g()
                gx_st, gc_st = self.spatial_feature_extract(z_x_content, z_c, gxx, gxy)

            d = z_s[ri][0].size(1)
            # print("this is d : " , d)
            z_st = z_s[ri][0].view(1, d, -1, 1)

            ## Compute Content Loss ##

            fm = 3 + 2 * 64 + 128 * 2 + 256 * 3 + 512 * 2
            if scl in [10,11]:
                ell_content = utils_go.to_device(content_loss_func(long_side, cut, scl, utils_go.to_device2(x_st[:, :, :, :]), utils_go.to_device2(c_st[:, :, :, :])))
                
            else:

                ell_content = utils_go.to_device(content_loss_func(long_side, cut, scl, utils_go.to_device1(x_st[:, :, :, :]), utils_go.to_device1(c_st[:, :, :, :])))


            print("ec", ell_content)


            ## Compute Style Loss ##

            fm = 2619

            # START INDEX SELECTION 2

            if scl <= 3:
                channel_list = [3] + [96] + [168] + [336] + [1008] * 2 + [1008] * 3 + [1008] + [1344] + [2016] * 6

            elif scl == 11:
                channel_list = [3] + [96] + [168] + [336] + [1008] * 2 + [1008] * 3 + [1008] + [1344] + [2016] * 6 + [
                    2688] + [4032] * 4

            else:
                channel_list = [3] + [96] + [168] + [336] + [1008] * 2 + [1008] * 3 + [1008] + [1344] + [2016] * 6 + [
                    2688] + [4032] * 3

            # END INDEX SELECTION 2

            if scl == 1:
                remd_loss = \
                utils_go.to_device(remd_loss_OT(long_side, cut, scl, utils_go.to_device2(x_st[:, :-2, :, :]), utils_go.to_device2(z_st), utils_go.to_device2(self.z_dist), splits=channel_list)[0] / len(channel_list))
            else:
                remd_loss = \
                utils_go.to_device(style_loss_func(long_side, cut, scl, utils_go.to_device1(x_st[:, :-2, :, :]), utils_go.to_device1(z_st), utils_go.to_device1(self.z_dist), splits=channel_list)[0] / len(channel_list))

            print("remd_loss", remd_loss)

            if gz.sum() > 0.:
                for j in range(gz.size(2)):
                    remd_loss += \
                        style_loss_func(scl, gx_st[:, :-2, j:(j + 1), :], gz[:, :, j:(j + 1), :], self.z_dist[:1] * 0.)[
                            0] / gz.size(2)

            ### Compute Moment Loss (constrains magnitude of features ###
            moment_ell = 0.
            if gz.sum() > 0.:
                moment_ell = moment_loss(torch.cat([x_st, gx_st], 2)[:, :-2, :, :], torch.cat([z_st, gz], 2),
                                         moments=[1, 2])
            else:
                # print("z_st style : " , )
                if scl == 4:
                    moment_ell = utils_go.to_device(moment_loss(scl, utils_go.to_device(x_st[:, :-2, :, :]), utils_go.to_device(z_st), moments=[1, 2]))
                elif scl == 5:
                    moment_ell = utils_go.to_device(moment_loss(scl, utils_go.to_device(x_st[:, :-2, :, :]), utils_go.to_device(z_st), moments=[1, 2]))
                elif scl == 6:
                    moment_ell = utils_go.to_device(moment_loss(scl, utils_go.to_device(x_st[:, :-2, :, :]), utils_go.to_device(z_st), moments=[1, 2]))
                elif scl == 7:
                    moment_ell = utils_go.to_device(moment_loss(scl, utils_go.to_device(x_st[:, :-2, :, :]), utils_go.to_device(z_st), moments=[1, 2]))
                elif scl == 8:
                    moment_ell = utils_go.to_device(moment_loss(scl, utils_go.to_device(x_st[:, :-2, :, :]), utils_go.to_device(z_st), moments=[1, 2]))
                elif scl == 9:
                    moment_ell = utils_go.to_device(moment_loss(scl, utils_go.to_device3(x_st[:, :-2, :, :]), utils_go.to_device3(z_st), moments=[1, 2]))
                elif scl == 10:
                    moment_ell = utils_go.to_device(moment_loss(scl, utils_go.to_device3(x_st[:, :-2, :, :]), utils_go.to_device3(z_st), moments=[1, 2]))
                elif scl == 11:
                    moment_ell = utils_go.to_device(moment_loss(scl, utils_go.to_device3(x_st[:, :-2, :, :]), utils_go.to_device3(z_st), moments=[1, 2]))

                else:
                    moment_ell = utils_go.to_device(moment_loss(scl, utils_go.to_device3(x_st[:, :-2, :, :]), utils_go.to_device3(z_st), moments=[1, 2]))
                print("me 1", moment_ell)

            ### Add Pallette Matching Loss ###
            content_weight_frac = 1. / max(content_weight, 1.)
            if scl == 1:
                moment_ell_1 = content_weight_frac * \
                               utils_go.to_device(remd_loss_OT(long_side, cut, scl, utils_go.to_device3(x_st[:, :3, :, :]), utils_go.to_device3(z_st[:, :3, :, :]), utils_go.to_device3(self.z_dist),
                                            splits=[3])[0])
            elif scl == 4:
                moment_ell_1 = content_weight_frac * \
                               utils_go.to_device(style_loss_func(long_side, cut, scl, utils_go.to_device(x_st[:, :3, :, :]), utils_go.to_device(z_st[:, :3, :, :]), utils_go.to_device(self.z_dist), splits=[3])[0])
            elif scl == 5:
                moment_ell_1 = content_weight_frac * \
                               utils_go.to_device(style_loss_func(long_side, cut, scl, utils_go.to_device(x_st[:, :3, :, :]), utils_go.to_device(z_st[:, :3, :, :]), utils_go.to_device(self.z_dist), splits=[3])[0])
            elif scl == 6:
                moment_ell_1 = content_weight_frac * \
                               utils_go.to_device(style_loss_func(long_side, cut, scl, utils_go.to_device(x_st[:, :3, :, :]), utils_go.to_device(z_st[:, :3, :, :]), utils_go.to_device(self.z_dist), splits=[3])[0])
            elif scl == 7:
                moment_ell_1 = content_weight_frac * \
                               utils_go.to_device(style_loss_func(long_side, cut, scl, utils_go.to_device(x_st[:, :3, :, :]), utils_go.to_device(z_st[:, :3, :, :]), utils_go.to_device(self.z_dist), splits=[3])[0])
            elif scl == 8:
                moment_ell_1 = content_weight_frac * \
                               utils_go.to_device(style_loss_func(long_side, cut, scl, utils_go.to_device(x_st[:, :3, :, :]), utils_go.to_device(z_st[:, :3, :, :]), utils_go.to_device(self.z_dist), splits=[3])[0])
            elif scl == 9:
                moment_ell_1 = content_weight_frac * \
                               utils_go.to_device(style_loss_func(long_side, cut, scl, utils_go.to_device(x_st[:, :3, :, :]), utils_go.to_device(z_st[:, :3, :, :]), utils_go.to_device(self.z_dist), splits=[3])[0])
            elif scl == 10:
                moment_ell_1 = content_weight_frac * \
                               utils_go.to_device(style_loss_func(long_side, cut, scl, utils_go.to_device3(x_st[:, :3, :, :]), utils_go.to_device3(z_st[:, :3, :, :]), utils_go.to_device3(self.z_dist), splits=[3])[0])
            elif scl == 11:
                moment_ell_1 = content_weight_frac * \
                               utils_go.to_device(style_loss_func(long_side, cut, scl, utils_go.to_device3(x_st[:, :3, :, :]), utils_go.to_device3(z_st[:, :3, :, :]), utils_go.to_device3(self.z_dist), splits=[3])[0])

            else:
                moment_ell_1 = content_weight_frac * \
                               utils_go.to_device(style_loss_func(long_side, cut, scl, utils_go.to_device3(x_st[:, :3, :, :]), utils_go.to_device3(z_st[:, :3, :, :]), utils_go.to_device3(self.z_dist), splits=[3])[0])

            print("me 2", moment_ell_1)
            moment_ell += moment_ell_1.squeeze()

            ### Combine Terms and Normalize ###
            ell_style = utils_go.to_device(remd_loss) + moment_weight * utils_go.to_device(moment_ell)
            print("ell style location ", ell_style.get_device())
            style_weight = 1.0 + moment_weight

            final_loss += (content_weight * utils_go.to_device(ell_content) + ell_style) / (content_weight + style_weight)

            print("ls ", final_loss)

        return final_loss / len(self.rand_ixx.keys())

    def init_inds(self, z_x, z_s_all, r, ri):

        const = (128 ** 2) * 4

        z_s = z_s_all[ri]

        try:
            temp = self.rand_ixx[ri]
        except:
            self.rand_ixx[ri] = []
            self.rand_ixy[ri] = []
            self.rand_iy[ri] = []

        for i in range(len(z_s)):

            d = z_s[i].size(1)
            z_st = z_s[i].view(1, d, -1, 1)
            x_st = z_x[i]

            big_size = x_st.size(3) * x_st.size(2)

            # global use_random
            if 1:
                stride_x = int(max(math.floor(math.sqrt(big_size // const)), 1))
                offset_x = np.random.randint(stride_x)

                stride_y = int(max(math.ceil(math.sqrt(big_size // const)), 1))
                offset_y = np.random.randint(stride_y)
            else:
                stride_x = int(max(math.floor(math.sqrt(big_size // const)), 1))
                offset_x = stride_x // 2

                stride_y = int(max(math.ceil(math.sqrt(big_size // const)), 1))
                offset_y = stride_y // 2

            region_mask = r  # .flatten()

            xx, xy = np.meshgrid(np.array(range(x_st.size(2)))[offset_x::stride_x],
                                 np.array(range(x_st.size(3)))[offset_y::stride_y])

            xx = np.expand_dims(xx.flatten(), 1)
            xy = np.expand_dims(xy.flatten(), 1)
            xc = np.concatenate([xx, xy], 1)

            try:
                xc = xc[region_mask[xy[:, 0], xx[:, 0]], :]
            except:
                region_mask = region_mask[:, :]
                xc = xc[region_mask[xy[:, 0], xx[:, 0]], :]

            self.rand_ixx[ri].append(xc[:, 0])
            self.rand_ixy[ri].append(xc[:, 1])

            zx = np.array(range(z_st.size(2))).astype(np.int32)

            self.rand_iy[ri].append(zx)

    def init_g_inds(self, coords, x_im):

        self.g_ixx = (coords[:, 0] * x_im.size(2)).astype(np.int64)
        self.g_ixy = (coords[:, 1] * x_im.size(3)).astype(np.int64)

    def spatial_feature_extract(self, z_x, z_c, xx, xy, scl):

        l2 = []
        l3 = []

        for i in range(len(z_x)):

            temp = z_x[i]
            temp2 = z_c[i]

            if i > 0 and z_x[i - 1].size(2) > z_x[i].size(2):
                xx = xx / 2.0
                xy = xy / 2.0

            #   print("temp : " , temp.size())
            #   print("temp2  :" , temp2.size())
            xxm = np.floor(xx).astype(np.float32)
            xxr = xx - xxm

            xym = np.floor(xy).astype(np.float32)
            xyr = xy - xym

            # print("temp2 size : " , temp2.size() )
            # print("temp size : ", temp.size())
            if scl == 11:
                w00 = utils_go.to_device2(torch.from_numpy((1. - xxr) * (1. - xyr))).float().unsqueeze(0).unsqueeze(
                    1).unsqueeze(3)
                w01 = utils_go.to_device2(torch.from_numpy((1. - xxr) * xyr)).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)
                w10 = utils_go.to_device2(torch.from_numpy(xxr * (1. - xyr))).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)
                w11 = utils_go.to_device2(torch.from_numpy(xxr * xyr)).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)
            else:
                w00 = utils_go.to_device1(torch.from_numpy((1. - xxr) * (1. - xyr))).float().unsqueeze(0).unsqueeze(
                    1).unsqueeze(3)
                w01 = utils_go.to_device1(torch.from_numpy((1. - xxr) * xyr)).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)
                w10 = utils_go.to_device1(torch.from_numpy(xxr * (1. - xyr))).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)
                w11 = utils_go.to_device1(torch.from_numpy(xxr * xyr)).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)

            xxm = np.clip(xxm.astype(np.int32), 0, temp.size(2) - 1)
            xym = np.clip(xym.astype(np.int32), 0, temp.size(3) - 1)

            s00 = xxm * temp.size(3) + xym
            s01 = xxm * temp.size(3) + np.clip(xym + 1, 0, temp.size(3) - 1)
            s10 = np.clip(xxm + 1, 0, temp.size(2) - 1) * temp.size(3) + (xym)
            s11 = np.clip(xxm + 1, 0, temp.size(2) - 1) * temp.size(3) + np.clip(xym + 1, 0, temp.size(3) - 1)

            temp = temp.view(1, temp.size(1), temp.size(2) * temp.size(3), 1)

            temp = temp[:, :, s00, :].mul_(w00).add_(temp[:, :, s01, :].mul_(w01)).add_(
                temp[:, :, s10, :].mul_(w10)).add_(temp[:, :, s11, :].mul_(w11))

            temp2 = temp2.view(1, temp2.size(1), temp2.size(2) * temp2.size(3), 1)

            temp2 = temp2[:, :, s00, :].mul_(w00).add_(temp2[:, :, s01, :].mul_(w01)).add_(
                temp2[:, :, s10, :].mul_(w10)).add_(temp2[:, :, s11, :].mul_(w11))
            # print("size of temp : " , temp.size())
            l2.append(temp)
            l3.append(temp2)

        if scl >= 10:
            x_st = utils_go.to_device2(torch.cat([li.contiguous() for li in l2], 1).float())
            c_st = utils_go.to_device2(torch.cat([li.contiguous() for li in l3], 1).float())

            xx = utils_go.to_device2(torch.from_numpy(xx)).view(1, 1, x_st.size(2), 1).float()
            yy = utils_go.to_device2(torch.from_numpy(xy)).view(1, 1, x_st.size(2), 1).float()

            x_st = torch.cat([x_st, xx, yy], 1)
            c_st = torch.cat([c_st, xx, yy], 1)

            return x_st, c_st
        else:
            x_st = utils_go.to_device1(torch.cat([li.contiguous() for li in l2], 1).float())
            c_st = utils_go.to_device1(torch.cat([li.contiguous() for li in l3], 1).float())

            xx = utils_go.to_device1(torch.from_numpy(xx)).view(1, 1, x_st.size(2), 1).float()
            yy = utils_go.to_device1(torch.from_numpy(xy)).view(1, 1, x_st.size(2), 1).float()

            x_st = torch.cat([x_st, xx, yy], 1)
            c_st = torch.cat([c_st, xx, yy], 1)

            return x_st, c_st
           

    def shuffle_feature_inds(self, i=0):

        for ri in self.rand_ixx.keys():
            np.random.shuffle(self.rand_ixx[ri][i])
            np.random.shuffle(self.rand_ixy[ri][i])
            np.random.shuffle(self.rand_iy[ri][i])

    def get_feature_inds(self, scl=1, ri=0, i=0, cnt=4 * (32 ** 2)):

        if scl == 1:
            cnt = 5000
        elif scl == 2:
            cnt = 128*128//2
        elif scl == 3:
            cnt = (256*256)//16
        elif scl == 4:
            cnt = (384*384)//32
        elif scl == 5:
            cnt = (512*512)//40
        elif scl == 6:
            cnt = (768*768)//84
        elif scl == 7:
            cnt = (1024*1024)//180
        elif scl == 8:
            cnt = (1280*1280)//300
        elif scl == 9:
            cnt = (1536*1536)//520
        elif scl == 10:
            cnt = (1856*1856)//680
        elif scl == 11:
            cnt = (2048*2048)//1400
        elif scl == 12:
            cnt = 1 * (32 ** 2)

        print("cnt---------->", cnt, scl)
        xx = self.rand_ixx[ri][i][:cnt]
        xy = self.rand_ixy[ri][i][:cnt]
        yx = self.rand_iy[ri][i][:cnt]

        return xx, xy, yx

    def get_feature_inds_g(self, i=0, cnt=(32 ** 2) // 8):

        xx = self.g_ixx
        xy = self.g_ixy

        return xx, xy
        self.z_dist = torch.zeros(1).cuda()

        self.rand_ixx = {}
        self.rand_ixy = {}
        self.rand_iy = {}

        if objective == 'remd_dp':
            self.eval = self.gen_remd_dp_objective

        if objective == 'remd_dp_g':
            self.eval = self.gen_remd_dp_objective_guided

    def gen_remd_dp_objective_guided(self, z_x, z_c, z_s, gz, d_ell, d_weight=10.0, content_weight=4.0, moment_weight=1.0, style_loss_func=remd_loss, content_loss_func=dp_loss, h=1.0):


        ## Extract Random Subset of Features from Stylized Image & Content Image ##
        # (Choose Features from Same locations in Stylized Image & Content Image) #
        final_loss = 0.
        for ri in range(len(self.rand_ixx.keys())):
            xx, xy, yx = self.get_feature_inds(ri=ri)
            x_st, c_st = self.spatial_feature_extract(z_x, z_c, xx, xy)

            if gz.sum() > 0.:
                gxx, gxy = self.get_feature_inds_g()
                gx_st, gc_st = self.spatial_feature_extract(z_x, z_c, gxx, gxy)

            ## Reshape Features from Style Distribution ##
            d = z_s[ri][0].size(1)
            z_st = z_s[ri][0].view(1,d,-1,1)


            ## Compute Content Loss ##

            fm = 3+2*64+128*2+256*3+512*2

            ell_content = content_loss_func(x_st[:,:,:,:], c_st[:,:,:,:])


            ## Compute Style Loss ##
            remd_loss, used_style_feats = style_loss_func(x_st[:,:fm,:,:], z_st[:,:fm,:,:], self.z_dist, splits=[fm])

            if gz.sum() > 0.:
                for j in range(gz.size(2)):
                    remd_loss += style_loss_func(gx_st[:,:-2,j:(j+1),:], gz[:,:,j:(j+1),:],self.z_dist[:1]*0.)[0]/gz.size(2)

            ### Compute Moment Loss (constrains magnitude of features ###
            moment_ell = 0.
            if gz.sum() > 0.:
                moment_ell = moment_loss(torch.cat([x_st,gx_st],2)[:,:-2,:,:],torch.cat([z_st,gz],2),moments=[1,2])
            else:
                moment_ell = moment_loss(x_st[:,:-2,:,:],z_st,moments=[1,2])

            ### Add Pallette Matching Loss ###                                        
            content_weight_frac = 1./max(content_weight,1.)
            moment_ell += content_weight_frac*style_loss_func(x_st[:,:3,:,:], z_st[:,:3,:,:],self.z_dist,splits=[3])[0]


            ### Combine Terms and Normalize ###
            ell_style = remd_loss+moment_weight*moment_ell
            style_weight = 1.0 + moment_weight
            final_loss += (content_weight*ell_content+ell_style)/(content_weight+style_weight)
        
        return final_loss/len(self.rand_ixx.keys())


    def init_inds(self, z_x, z_s_all, r, ri):

        const = 128**2


        z_s = z_s_all[ri]


        try:
            temp = self.rand_ixx[ri]
        except:
            self.rand_ixx[ri]= []
            self.rand_ixy[ri]= []
            self.rand_iy[ri]= []

        for i in range(len(z_s)):

            d = z_s[i].size(1)
            z_st = z_s[i].view(1,d,-1,1)
            x_st = z_x[i]


            big_size = x_st.size(3)*x_st.size(2)

            global use_random
            if use_random:
                stride_x = int(max(math.floor(math.sqrt(big_size//const)),1))
                offset_x = np.random.randint(stride_x)
                
                stride_y = int(max(math.ceil(math.sqrt(big_size//const)),1))
                offset_y = np.random.randint(stride_y)
            else:
                stride_x = int(max(math.floor(math.sqrt(big_size//const)),1))
                offset_x = stride_x//2
                
                stride_y = int(max(math.ceil(math.sqrt(big_size//const)),1))
                offset_y = stride_y//2

            region_mask = r#.flatten()

            xx,xy = np.meshgrid(np.array(range(x_st.size(2)))[offset_x::stride_x], np.array(range(x_st.size(3)))[offset_y::stride_y] )
            
            xx = np.expand_dims(xx.flatten(),1)
            xy = np.expand_dims(xy.flatten(),1)
            xc = np.concatenate([xx,xy],1)

            try:
                xc = xc[region_mask[xy[:,0],xx[:,0]],:]
            except:
                region_mask = region_mask[:,:]
                xc = xc[region_mask[xy[:,0],xx[:,0]],:]

            self.rand_ixx[ri].append(xc[:,0])
            self.rand_ixy[ri].append(xc[:,1])

            zx = np.array(range(z_st.size(2))).astype(np.int32)

            self.rand_iy[ri].append(zx)

    def init_g_inds(self, coords, x_im):

        self.g_ixx = (coords[:,0]*x_im.size(2)).astype(np.int64)
        self.g_ixy = (coords[:,1]*x_im.size(3)).astype(np.int64)
   

    def spatial_feature_extract(self, z_x, z_c, xx, xy):

        l2 = []
        l3 = []

        for i in range(len(z_x)):

            temp = z_x[i]
            temp2 = z_c[i]

            if i>0 and z_x[i-1].size(2) > z_x[i].size(2):
                xx = xx/2.0
                xy = xy/2.0

            xxm = np.floor(xx).astype(np.float32)
            xxr = xx - xxm

            xym = np.floor(xy).astype(np.float32)
            xyr = xy - xym

            w00 = utils.to_device(torch.from_numpy((1.-xxr)*(1.-xyr))).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)
            w01 = utils.to_device(torch.from_numpy((1.-xxr)*xyr)).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)
            w10 = utils.to_device(torch.from_numpy(xxr*(1.-xyr))).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)
            w11 = utils.to_device(torch.from_numpy(xxr*xyr)).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)


            xxm = np.clip(xxm.astype(np.int32),0,temp.size(2)-1)
            xym = np.clip(xym.astype(np.int32),0,temp.size(3)-1)

            s00 = xxm*temp.size(3)+xym
            s01 = xxm*temp.size(3)+np.clip(xym+1,0,temp.size(3)-1)
            s10 = np.clip(xxm+1,0,temp.size(2)-1)*temp.size(3)+(xym)
            s11 = np.clip(xxm+1,0,temp.size(2)-1)*temp.size(3)+np.clip(xym+1,0,temp.size(3)-1)


            temp = temp.view(1,temp.size(1),temp.size(2)*temp.size(3),1)
            temp = temp[:,:,s00,:].mul_(w00).add_(temp[:,:,s01,:].mul_(w01)).add_(temp[:,:,s10,:].mul_(w10)).add_(temp[:,:,s11,:].mul_(w11))
            

            temp2 = temp2.view(1,temp2.size(1),temp2.size(2)*temp2.size(3),1)
            temp2 = temp2[:,:,s00,:].mul_(w00).add_(temp2[:,:,s01,:].mul_(w01)).add_(temp2[:,:,s10,:].mul_(w10)).add_(temp2[:,:,s11,:].mul_(w11))

            l2.append(temp)
            l3.append(temp2)

        x_st = torch.cat([li.contiguous() for li in l2],1)
        c_st = torch.cat([li.contiguous() for li in l3],1)


        xx = torch.from_numpy(xx).cuda().view(1,1,x_st.size(2),1).float()
        yy = torch.from_numpy(xy).cuda().view(1,1,x_st.size(2),1).float()


        x_st = torch.cat([x_st,xx,yy],1)
        c_st = torch.cat([c_st,xx,yy],1)

        return x_st, c_st




    def shuffle_feature_inds(self, i=0):
        global use_random

        if use_random:
            for ri in self.rand_ixx.keys():
                np.random.shuffle(self.rand_ixx[ri][i])
                np.random.shuffle(self.rand_ixy[ri][i])
                np.random.shuffle(self.rand_iy[ri][i])


    def get_feature_inds(self,ri=0, i=0, cnt=32**2):

        global use_random

        if use_random:
            xx = self.rand_ixx[ri][i][:cnt]
            xy = self.rand_ixy[ri][i][:cnt]
            yx = self.rand_iy[ri][i][:cnt]
        else:
            xx = self.rand_ixx[ri][i][::(self.rand_ixx[ri][i].shape[0]//cnt)]
            xy = self.rand_ixy[ri][i][::(self.rand_ixy[ri][i].shape[0]//cnt)]
            yx =  self.rand_iy[ri][i][::(self.rand_iy[ri][i].shape[0]//cnt)]

        return xx, xy, yx
    
    def get_feature_inds_g(self, i=0, cnt=32**2):

        xx = self.g_ixx
        xy = self.g_ixy

        return xx, xy
