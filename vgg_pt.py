from collections import namedtuple
import random
import ssl

import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torchvision import models
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context
use_random = True


class Vgg16_pt(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16_pt, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.vgg_layers = vgg_pretrained_features
        self.slice1 = torch.nn.Sequential()
        self.slice2 =from collections import namedtuple
import random
import ssl

import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torchvision import models
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

import torch.nn as nn

import utils_go
from utils_go import *


class Hook_Get_Content(nn.Module):
    def __init__(self):
        super(Hook_Get_Content, self).__init__()
        self.content_hook = None
        self.hook_input = None
        self.mode = None
        self.crit = nn.L1Loss()
        self.go = False
        self.channels = 0

    def hook(self, module, input, output):
        if self.mode == 'content-capture':
            #  print("content capture")
            self.rand = torch.randint(0, output[0].size(1), (output[0].size(1) // 4,)).cuda()

            self.content_hook = torch.index_select(output[0], 0, self.rand)

            self.channels = self.content_hook.size(0)

            if len(self.content_hook.size()) < 4:
                self.content_hook = self.content_hook.clone().unsqueeze(0)

            self.go = True

        if self.mode == 'stylized-capture':
            self.hook_input = torch.index_select(output[0], 0, self.rand)

            if len(self.hook_input.size()) < 4:
                self.hook_input = self.hook_input.clone().unsqueeze(0)


class Nas(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Nas, self).__init__()
        # vgg_pretrained_features = models.vgg16(pretrained=True).features
        model = NASNetALarge(num_classes=1001)
        
        #change directory from which to load the weights of Nasnet large to fit your file system
        model.load_state_dict(torch.load('nasnet_no_lin.pth'))

        model.eval()

        new_model = model

        self.new = new_model
        
        for param in self.new.parameters():
            param.requires_grad = False
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.inds = range(11)
        self.layer_list = ['conv0', 'cell_stem_0', 'cell_stem_1', 'cell_0', 'cell_1', 'cell_2', 'cell_3', 'cell_4',
                           'cell_5',
                           'reduction_cell_0', 'cell_6', 'cell_7', 'cell_8', 'cell_9', 'cell_10', 'cell_11',
                           'reduction_cell_1',
                           'cell_12', 'cell_13', 'cell_14', 'cell_15', 'cell_16', 'cell_17']

        self.layer_style = self.layer_list[:-10]

        self.layer_content = self.layer_list[:-14]

        self.gpu_5 = []  # 5 corr. 512 px

        self.gpu_4 = []  # 5 corr. 512 px

        self.gpu_8 = [ 'cell_12', 'cell_13', 'cell_14', 'cell_15', 'cell_16', 'cell_17']
        self.gpu_9 =[ 'cell_11', 'reduction_cell_1', 'cell_12',  'cell_13' , 'cell_14', 'cell_15', 'cell_16', 'cell_17']
        self.gpu_10 = ['cell_10', 'cell_11', 'reduction_cell_1', 'cell_12', 'cell_13', 'cell_14', 'cell_15', 'cell_16',
                       'cell_17']

        self.gpu_11 = [ 'cell_5', 'cell_6', 'cell_7', 'cell_8', 'cell_9', 'cell_10', 'cell_11', 'reduction_cell_1', 'cell_12', 'cell_13',
                       'cell_14', 'cell_15', 'cell_16', 'cell_17']
        self.gpu_12 = ['reduction_cell_0', 'cell_6', 'cell_7', 'cell_8', 'cell_9', 'cell_10', 'cell_11',
                       'reduction_cell_1', 'cell_12', 'cell_13', 'cell_14', 'cell_15', 'cell_16', 'cell_17']

    def forward_base(self, name, scl, X, rand):
        inds = self.inds
        print(
        "NAME IS ----------------------------------------------------------------------------------------------------------------> ",
        name)
        if 0:
            x = utils_go.to_device1(X)
            l2 = [utils_go.to_device1(X)]
            self.new = utils_go.to_device1(self.new)
        else:
            x = X
            l2 = [X]

        if scl in [10] and name == 'utils':
            l3 = [utils_go.to_device2(X)]
        elif scl == 11:
            l3 = [utils_go.to_device2(X)]
        else:
            l3 = [utils_go.to_device1(X)]
        count = 0
        
        gpu_2_list = []
        
        if scl <= 3:
            layer_list_in_use = self.layer_list[:16]
        elif scl == 11:
            layer_list_in_use = self.layer_list[:21]

        else:
            layer_list_in_use = self.layer_list[:20]


        for i in layer_list_in_use:
            if i in ['conv0', 'cell_stem_0']:
                x = getattr(self.new, i).forward(x)
            elif i == 'cell_stem_1':
                x = getattr(self.new, i).forward(l2[-2], x)

            else:
                # print("else ok", i)
                # print("x " , x.size())
                # print("l2[-2]", l2[-2].size())

                if scl < 8:
                    gpu_1_list = self.gpu_5
                elif scl == 8:
                    gpu_1_list = self.gpu_8
                elif scl == 9:
                    gpu_1_list = self.gpu_9
                elif scl == 10:
                    gpu_1_list = self.gpu_10
                    gpu_2_list = gpu_1_list[-1]

                elif scl == 11:
                    gpu_1_list = self.gpu_11
                    gpu_2_list = gpu_1_list[-2] #gpu_1_list[-1]

                else:
                    gpu_1_list = self.gpu_12
                if (i in gpu_1_list):
                    if i in gpu_2_list and scl in [10,11] and name == 'utils':
                        x = utils_go.to_device2(getattr(self.new, i)).forward(utils_go.to_device2(x),
                                                                              utils_go.to_device2(l2[-2]))
                    else:
                        x = utils_go.to_device1(getattr(self.new, i)).forward(utils_go.to_device1(x),
                                                                              utils_go.to_device1(l2[-2]))
                else:
                    x = getattr(utils_go.to_device(self.new), i).forward(utils_go.to_device(x),
                                                                         utils_go.to_device(l2[-2]))


            if (name == 'content' or name == 'content for style'):
                if scl == 1:
                    if i in self.layer_list[:16]:
                        # print("rand : " , rand)
                        # print("x size : ", x.size())
                        # print("rand count : " , rand[count].max())
                        l3.append(utils_go.to_device1(x))
                elif scl == 2:
                    if i in self.layer_list[:16]:
                        # print("rand : " , rand)
                        # print("x size : ", x.size())
                        # print("rand count : " , rand[count].max())
                        l3.append(utils_go.to_device1(x))
                elif scl == 3:
                    if i in self.layer_list[:16]:
                        # print("rand : " , rand)
                        # print("x size : ", x.size())
                        # print("rand count : " , rand[count].max())
                        l3.append(utils_go.to_device1(x))
                elif scl >= 4 and scl <8:
                    if i in self.layer_list[:16]:
                        # print("rand : " , rand)
                        # print("x size : ", x.size())
                        # print("rand count : " , rand[count].max())
                        if scl == 10 and name == 'utils':
                            l3.append(utils_go.to_device2(x))                            
                        else:
                            l3.append(utils_go.to_device1(x))

                elif scl == 11:
                    if i in self.layer_list[:21]:
                        # print("rand : " , rand)
                        # print("x size : ", x.size())
                        # print("rand count : " , rand[count].max())
                        if scl == 11:
                            #print("in the if vgg --------------->")

                            l3.append(utils_go.to_device2(x))                            
                        else:
                            l3.append(utils_go.to_device1(x))

                else:
                    if i in self.layer_list[:20]:
                        # print("rand : " , rand)
                        # print("x size : ", x.size())
                        # print("rand count : " , rand[count].max())
                        if scl == 10 and name == 'utils':
                            #print("in the if vgg --------------->")

                            l3.append(utils_go.to_device2(x))                            
                        else:
                            l3.append(utils_go.to_device1(x))
            else:
                if scl == 1:
                    if i in self.layer_list[:16]:
                        # print("rand : " , rand)
                        # print("x size : ", x.size())
                        # print("rand count : " , rand[count].max())
                        # print("x type pre select : " , x.type() )
                        # print("x type post select : ", torch.index_select(x,1,rand[count].cuda() ).type())
                        if scl == 10 and name == 'utils':
                            l3.append(utils_go.to_device2(x))                            
                        else:
                            l3.append(utils_go.to_device1(x))
                elif scl == 2:
                    if i in self.layer_list[:16]:
                        # print("rand : " , rand)
                        # print("x size : ", x.size())
                        # print("rand count : " , rand[count].max())
                        if scl == 10 and name == 'utils':
                            l3.append(utils_go.to_device2(x))                            
                        else:
                            l3.append(utils_go.to_device1(x))
                elif scl == 3:
                    if i in self.layer_list[:16]:
                        # print("rand : " , rand)
                        # print("x size : ", x.size())
                        # print("rand count : " , rand[count].max())
                        if scl == 10 and name == 'utils':
                            l3.append(utils_go.to_device2(x))                            
                        else:
                            l3.append(utils_go.to_device1(x))
                elif scl >= 4 and scl <8:
                    if i in self.layer_list[:16]:
                        # print("rand : " , rand)
                        # print("x size : ", x.size())
                        # print("rand count : " , rand[count].max())
                        if scl == 10 and name == 'utils':
                            l3.append(utils_go.to_device2(x))                            
                        else:
                            l3.append(utils_go.to_device1(x))

                elif scl == 11:
                    if i in self.layer_list[:21]:
                        # print("rand : " , rand)
                        # print("x size : ", x.size())
                        # print("rand count : " , rand[count].max())
                        if scl == 11 and name != 'utils':
                            #print("in the if vgg --------------->")

                            l3.append(utils_go.to_device2(x))                            
                        else:
                            l3.append(utils_go.to_device1(x))
                else:
                    if i in self.layer_list[:20]:
                        # print("rand : " , rand)
                        # print("x size : ", x.size())
                        # print("rand count : " , rand[count].max())
                        if scl == 10 and name == 'utils':
                            l3.append(utils_go.to_device2(x))                            
                        else:
                            l3.append(utils_go.to_device1(x))
            l2.append(x)
            if count > 3:
                l2.pop(0)
            count += 1
        return l3

    def forward(self, scl, X, rand, name):

        inds = self.inds

        x = X
        l2 = self.forward_base(name, scl, X, rand)
        out2 = l2

        return out2

    def forward_cat_large(self, scl, X, r, inds=[1, 3, 5, 8, 11], rand=True, samps=100, forward_func=None,
                          rand_whoa=None, name=None):

        if not forward_func:
            forward_func = self.forward


        x = X

        out2 = forward_func(scl, X, rand_whoa, name)

        try:
            r = r[:, :, 0]
        except:
            pass

        if r.max() < 0.1:
            region_mask = np.greater(r.flatten() + 1., 0.5)
        else:
            region_mask = np.greater(r.flatten(), 0.5)

        xx, xy = np.meshgrid(np.array(range(x.size(2))), np.array(range(x.size(3))))
        xx = np.expand_dims(xx.flatten(), 1)
        xy = np.expand_dims(xy.flatten(), 1)
        xc = np.concatenate([xx, xy], 1)

        xc = xc[region_mask, :]

        np.random.shuffle(xc)

        const2 = min(samps, xc.shape[0])

        xx = xc[:const2, 0]
        yy = xc[:const2, 1]

        temp = X
        temp_list = [temp[:, :, xx[j], yy[j]].unsqueeze(2).unsqueeze(3) for j in range(const2)]
        temp = torch.cat(temp_list, 2)

        l2 = []
        for i in range(len(out2)):

            temp = out2[i]


            if i > 0 and out2[i].size(2) < out2[i - 1].size(2):
                xx = xx / 2.0
                yy = yy / 2.0

            xx = np.clip(xx, 0, temp.size(2) - 1).astype(np.int32)
            yy = np.clip(yy, 0, temp.size(3) - 1).astype(np.int32)

            temp_list = [temp[:, :, xx[j], yy[j]].unsqueeze(2).unsqueeze(3) for j in range(const2)]

            temp = torch.cat(temp_list, 2)

            l2.append(temp.clone().detach())

        out2 = [torch.cat([li.contiguous() for li in l2], 1)]

        return out2

    def forward_diff(self, X, inds=[1, 3, 5, 8, 11], rand=True):

        inds = self.inds
        l2 = self.forward_base(X, inds, rand)

        out2 = [l2[i].contiguous() for i in inds]

        for i in range(len(out2)):
            temp = out2[i]
            temp2 = F.pad(temp, (2, 2, 0, 0), value=1.)
            temp3 = F.pad(temp, (0, 0, 2, 2), value=1.)
            out2[i] = torch.cat(
                [temp, temp2[:, :, :, 4:], temp2[:, :, :, :-4], temp3[:, :, 4:, :], temp3[:, :, :-4, :]], 1)

        return out2


class Nas_Content(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Nas_Content, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.vgg_layers = vgg_pretrained_features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.new = new_model
        for param in self.new.parameters():
            param.requires_grad = False

        for x in range(1):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.inds = range(11)

        self.layer_list = ['conv0', 'cell_stem_0', 'cell_stem_1', 'cell_0', 'cell_1', 'cell_2', 'cell_3', 'cell_4',
                           'cell_5',
                           'reduction_cell_0', 'cell_6', 'cell_7', 'cell_8', 'cell_9', 'cell_10', 'cell_11',
                           'reduction_cell_1',
                           'cell_12', 'cell_13', 'cell_14', 'cell_15', 'cell_16', 'cell_17']

        self.layer_acquire = self.layer_list[2:]

    def forward_base(self, X, rand):
        inds = self.inds

        x = X
        l2 = [X]
        l3 = [X]
        count = 0
        for i in self.layer_list:
            try:
                x = getattr(self.new, i).forward(x)  # [:,:,1:-1,1:-1]
            except:
                pass
            try:
                if i == 'cell_stem_1':

                    x = getattr(self.new, i).forward(l2[-2], x)
                else:

                    x = getattr(self.new, i).forward(x, l2[-2])

            except:

                pass

            if i in self.layer_acquire:
                l3.append(torch.index_select(x, 1, rand[count].cuda()))
            l2.append(x)
            count += 1

        return l3

    def forward(self, X, rand):

        inds = self.inds

        x = X
        l2 = self.forward_base(X, rand)
        out2 = l2

        return out2

    def forward_cat(self, X, r, inds=[1, 3, 5, 8, 11], rand=True, samps=100, forward_func=None, rand_whoa=None):

        if not forward_func:
            forward_func = self.forward

        x = X
        out2 = forward_func(X, rand_whoa)

        try:
            r = r[:, :, 0]
        except:
            pass

        if r.max() < 0.1:
            region_mask = np.greater(r.flatten() + 1., 0.5)
        else:
            region_mask = np.greater(r.flatten(), 0.5)

        xx, xy = np.meshgrid(np.array(range(x.size(2))), np.array(range(x.size(3))))
        xx = np.expand_dims(xx.flatten(), 1)
        xy = np.expand_dims(xy.flatten(), 1)
        xc = np.concatenate([xx, xy], 1)

        xc = xc[region_mask, :]

        np.random.shuffle(xc)

        const2 = min(samps, xc.shape[0])

        xx = xc[:const2, 0]
        yy = xc[:const2, 1]

        temp = X
        temp_list = [temp[:, :, xx[j], yy[j]].unsqueeze(2).unsqueeze(3) for j in range(const2)]
        temp = torch.cat(temp_list, 2)

        l2 = []
        for i in range(len(out2)):

            temp = out2[i]

            if i > 0 and out2[i].size(2) < out2[i - 1].size(2):
                xx = xx / 2.0
                yy = yy / 2.0

            xx = np.clip(xx, 0, temp.size(2) - 1).astype(np.int32)
            yy = np.clip(yy, 0, temp.size(3) - 1).astype(np.int32)

            temp_list = [temp[:, :, xx[j], yy[j]].unsqueeze(2).unsqueeze(3) for j in range(const2)]

            temp = torch.cat(temp_list, 2)

            l2.append(temp.clone().detach())

        out2 = [torch.cat([li.contiguous() for li in l2], 1)]

        return out2

    def forward_diff(self, X, inds=[1, 3, 5, 8, 11], rand=True):

        inds = self.inds
        l2 = self.forward_base(X, inds, rand)

        out2 = [l2[i].contiguous() for i in inds]

        for i in range(len(out2)):
            temp = out2[i]
            temp2 = F.pad(temp, (2, 2, 0, 0), value=1.)
            temp3 = F.pad(temp, (0, 0, 2, 2), value=1.)
            out2[i] = torch.cat(
                [temp, temp2[:, :, :, 4:], temp2[:, :, :, :-4], temp3[:, :, 4:, :], temp3[:, :, :-4, :]], 1)

        return out2


class Nasnet(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Nasnet, self).__init__()
        self.nas = new_model

        if not requires_grad:
            for param in self.nas.parameters():
                param.requires_grad = False

        self.inds = range(11)

        self.layer_list = ['conv0', 'cell_stem_0', 'cell_stem_1', 'cell_0', 'cell_1', 'cell_2', 'cell_3', 'cell_4',
                           'cell_5',
                           'reduction_cell_0', 'cell_6', 'cell_7', 'cell_8', 'cell_9', 'cell_10', 'cell_11',
                           'reduction_cell_1',
                           'cell_12', 'cell_13', 'cell_14', 'cell_15', 'cell_16', 'cell_17']

        self.layer_acquire = self.layer_list[:-8]

    def forward_base(self, X, rand):
        inds = self.inds

        x = X
        l2 = [X]
        for i in self.layer_list:
            try:
                x = getattr(self.nas, i).forward(x)  # [:,:,1:-1,1:-1]
            except:
                pass
            if i in self.layer_acquire:
                l2.append(x)

        return l2

    def forward(self, X, inds=[1, 3, 5, 8, 11], rand=True):

        inds = self.inds

        x = X
        l2 = self.forward_base(X, rand)
        out2 = l2

        return out2

    def forward_cat(self, X, r, inds=[1, 3, 5, 8, 11], rand=True, samps=100, forward_func=None):

        if not forward_func:
            forward_func = self.forward

        x = X
        out2 = forward_func(X, rand)

        print("initial r size : ", r.shape)

        try:
            r = r[:, :, 0]
        except:
            pass

        print("r size after try/except : ", r.shape)

        if r.max() < 0.1:
            region_mask = np.greater(r.flatten() + 1., 0.5)
        else:
            region_mask = np.greater(r.flatten(), 0.5)

        print("region_mask shape : ", region_mask.shape)

        print("x shape : ", x.shape)

        xx, xy = np.meshgrid(np.array(range(x.size(2))), np.array(range(x.size(3))))

        print("xx : ", xx.shape)
        print("xy : ", xy.shape)
        xx = np.expand_dims(xx.flatten(), 1)

        print("xx post : ", xx.shape)
        xy = np.expand_dims(xy.flatten(), 1)

        print("xy post : ", xy.shape)
        xc = np.concatenate([xx, xy], 1)

        print("xc size : ", xc.shape)

        xc = xc[region_mask, :]

        print("xc post : ", xc.shape)

        np.random.shuffle(xc)

        const2 = min(samps, xc.shape[0])

        print("const2 : ", const2)

        xx = xc[:const2, 0]
        yy = xc[:const2, 1]

        print("xx 3 : ", xx.shape)
        print("yy 3 : ", yy.shape)

        temp = X
        temp_list = [temp[:, :, xx[j], yy[j]].unsqueeze(2).unsqueeze(3) for j in range(const2)]
        temp = torch.cat(temp_list, 2)

        l2 = []
        for i in range(len(out2)):

            temp = out2[i]

            if i > 0 and out2[i].size(2) < out2[i - 1].size(2):
                xx = xx / 2.0
                yy = yy / 2.0

            xx = np.clip(xx, 0, temp.size(2) - 1).astype(np.int32)
            yy = np.clip(yy, 0, temp.size(3) - 1).astype(np.int32)

            temp_list = [temp[:, :, xx[j], yy[j]].unsqueeze(2).unsqueeze(3) for j in range(const2)]
            temp = torch.cat(temp_list, 2)

            l2.append(temp.clone().detach())

        out2 = [torch.cat([li.contiguous() for li in l2], 1)]

        return out2

    def forward_diff(self, X, inds=[1, 3, 5, 8, 11], rand=True):

        inds = self.inds
        l2 = self.forward_base(X, inds, rand)

        out2 = [l2[i].contiguous() for i in inds]

        for i in range(len(out2)):
            temp = out2[i]
            temp2 = F.pad(temp, (2, 2, 0, 0), value=1.)
            temp3 = F.pad(temp, (0, 0, 2, 2), value=1.)
            out2[i] = torch.cat(
                [temp, temp2[:, :, :, 4:], temp2[:, :, :, :-4], temp3[:, :, 4:, :], temp3[:, :, :-4, :]], 1)

        return out2


import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torch.nn as nn
import torch

pretrained_settings = {
    'nasnetalarge': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth',
            'input_space': 'RGB',
            'input_size': [3, 331, 331],  # resize 354
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
        'imagenet+background': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth',
            'input_space': 'RGB',
            'input_size': [3, 331, 331],  # resize 354
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1001
        }
    }
}


class MaxPoolPad(nn.Module):

    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:]
        return x


class AvgPoolPad(nn.Module):

    def __init__(self, stride=2, padding=1):
        super(AvgPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.AvgPool2d(3, stride=stride, padding=padding, count_include_pad=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:]
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, dw_kernel,
                                          stride=dw_stride,
                                          padding=dw_padding,
                                          bias=bias,
                                          groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(BranchSeparables, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, in_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(in_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesStem(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(out_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesReduction(BranchSeparables):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, z_padding=1, bias=False):
        BranchSeparables.__init__(self, in_channels, out_channels, kernel_size, stride, padding, bias)
        self.padding = nn.ZeroPad2d((z_padding, 0, z_padding, 0))

    def forward(self, x):
        x = self.relu(x)
        x = self.padding(x)
        x = self.separable_1(x)
        x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellStem0(nn.Module):
    def __init__(self, stem_filters, num_filters=42):
        super(CellStem0, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(self.stem_filters, self.num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps=0.001, momentum=0.1, affine=True))

        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.num_filters, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparablesStem(self.stem_filters, self.num_filters, 7, 2, 3, bias=False)

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(self.stem_filters, self.num_filters, 7, 2, 3, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_right = BranchSeparablesStem(self.stem_filters, self.num_filters, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.num_filters, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class CellStem1(nn.Module):

    def __init__(self, stem_filters, num_filters):
        super(CellStem1, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(2 * self.num_filters, self.num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps=0.001, momentum=0.1, affine=True))

        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(self.stem_filters, self.num_filters // 2, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(self.stem_filters, self.num_filters // 2, 1, stride=1, bias=False))

        self.final_path_bn = nn.BatchNorm2d(self.num_filters, eps=0.001, momentum=0.1, affine=True)

        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.num_filters, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(self.num_filters, self.num_filters, 7, 2, 3, bias=False)

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(self.num_filters, self.num_filters, 7, 2, 3, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(self.num_filters, self.num_filters, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.num_filters, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)

        x_relu = self.relu(x_conv0)
        # path 1
        x_path1 = self.path_1(x_relu)
        # path 2
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        # final path
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))

        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class FirstCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))

        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))

        self.final_path_bn = nn.BatchNorm2d(out_channels_left * 2, eps=0.001, momentum=0.1, affine=True)

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

        self.comb_iter_1_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_relu = self.relu(x_prev)
        # path 1
        x_path1 = self.path_1(x_relu)
        # path 2
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        # final path
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))

        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1, bias=False)

        self.comb_iter_1_left = BranchSeparables(out_channels_left, out_channels_left, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell0(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))

        self.comb_iter_0_left = BranchSeparablesReduction(out_channels_right, out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 7, 2, 3, bias=False)

        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 7, 2, 3, bias=False)

        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparablesReduction(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell1(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, bias=False)

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NASNetALarge(nn.Module):
    """NASNetALarge (6 @ 4032) """

    def __init__(self, num_classes=1001, stem_filters=96, penultimate_filters=4032, filters_multiplier=2):
        super(NASNetALarge, self).__init__()
        self.num_classes = num_classes
        self.stem_filters = stem_filters
        self.penultimate_filters = penultimate_filters
        self.filters_multiplier = filters_multiplier

        filters = self.penultimate_filters // 24
        # 24 is default value for the architecture

        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels=self.stem_filters, kernel_size=3, padding=0,
                                                stride=2,
                                                bias=False))
        self.conv0.add_module('bn', nn.BatchNorm2d(self.stem_filters, eps=0.001, momentum=0.1, affine=True))

        self.conv0 = self.conv0.to(device='cuda:0')

        self.cell_stem_0 = CellStem0(self.stem_filters, num_filters=filters // (filters_multiplier ** 2)).to(
            device='cuda:0')
        self.cell_stem_1 = CellStem1(self.stem_filters, num_filters=filters // filters_multiplier).to(device='cuda:0')

        self.cell_0 = FirstCell(in_channels_left=filters, out_channels_left=filters // 2,
                                in_channels_right=2 * filters, out_channels_right=filters).to(device='cuda:0')
        self.cell_1 = NormalCell(in_channels_left=2 * filters, out_channels_left=filters,
                                 in_channels_right=6 * filters, out_channels_right=filters).to(device='cuda:0')
        self.cell_2 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters,
                                 in_channels_right=6 * filters, out_channels_right=filters).to(device='cuda:0')
        self.cell_3 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters,
                                 in_channels_right=6 * filters, out_channels_right=filters).to(device='cuda:0')
        self.cell_4 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters,
                                 in_channels_right=6 * filters, out_channels_right=filters).to(device='cuda:0')
        self.cell_5 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters,
                                 in_channels_right=6 * filters, out_channels_right=filters).to(device='cuda:0')

        self.reduction_cell_0 = ReductionCell0(in_channels_left=6 * filters, out_channels_left=2 * filters,
                                               in_channels_right=6 * filters, out_channels_right=2 * filters).to(
            device='cuda:1')

        self.cell_6 = FirstCell(in_channels_left=6 * filters, out_channels_left=filters,
                                in_channels_right=8 * filters, out_channels_right=2 * filters).to(device='cuda:1')
        self.cell_7 = NormalCell(in_channels_left=8 * filters, out_channels_left=2 * filters,
                                 in_channels_right=12 * filters, out_channels_right=2 * filters).to(device='cuda:1')
        self.cell_8 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters,
                                 in_channels_right=12 * filters, out_channels_right=2 * filters).to(device='cuda:1')
        self.cell_9 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters,
                                 in_channels_right=12 * filters, out_channels_right=2 * filters).to(device='cuda:1')
        self.cell_10 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters,
                                  in_channels_right=12 * filters, out_channels_right=2 * filters).to(device='cuda:1')
        self.cell_11 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters,
                                  in_channels_right=12 * filters, out_channels_right=2 * filters).to(device='cuda:1')

        self.reduction_cell_1 = ReductionCell1(in_channels_left=12 * filters, out_channels_left=4 * filters,
                                               in_channels_right=12 * filters, out_channels_right=4 * filters).to(
            device='cuda:1')

        self.cell_12 = FirstCell(in_channels_left=12 * filters, out_channels_left=2 * filters,
                                 in_channels_right=16 * filters, out_channels_right=4 * filters).to(device='cuda:1')
        self.cell_13 = NormalCell(in_channels_left=16 * filters, out_channels_left=4 * filters,
                                  in_channels_right=24 * filters, out_channels_right=4 * filters).to(device='cuda:1')
        self.cell_14 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters,
                                  in_channels_right=24 * filters, out_channels_right=4 * filters).to(device='cuda:1')
        self.cell_15 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters,
                                  in_channels_right=24 * filters, out_channels_right=4 * filters).to(device='cuda:1')
        self.cell_16 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters,
                                  in_channels_right=24 * filters, out_channels_right=4 * filters).to(device='cuda:1')
        self.cell_17 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters,
                                  in_channels_right=24 * filters, out_channels_right=4 * filters).to(device='cuda:1')

        # self.relu = nn.ReLU()
        # self.avg_pool = nn.AvgPool2d(11, stride=1, padding=0)
        # self.dropout = nn.Dropout()
        # self.last_linear = nn.Linear(24 * filters, self.num_classes)

    def features(self, input):
        x_conv0 = self.conv0(input)
        #        print("x_conv0 : ", x_conv0.size())
        x_stem_0 = self.cell_stem_0(x_conv0)
        #        print("x_stem_0 : ", x_stem_0.size())
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)

        #        print("x_stem_1 : ", x_stem_1.size())

        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        #        print("x_cell_0 : ", x_cell_0.size())
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        #        print("x_cell_1 : ", x_cell_1.size())

        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        #        print("x_cell_2 : ", x_cell_2.size())

        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        #        print("x_cell_3 : ", x_cell_3.size())

        x_cell_4 = self.cell_4(x_cell_3, x_cell_2)
        #        print("x_cell_4 : ", x_cell_4.size())

        x_cell_5 = self.cell_5(x_cell_4, x_cell_3)
        #        print("x_cell_5 : ", x_cell_5.size())

        x_reduction_cell_0 = self.reduction_cell_0(x_cell_5, x_cell_4)
        #        print("x_reduction_cell_0 : ", x_reduction_cell_0.size())

        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_4)
        #        print("x_cell_6 : ", x_cell_6.size())
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        #        print("x_cell_7 : ", x_cell_7.size())

        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        #        print("x_cell_8 : ", x_cell_8.size())

        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        #        print("x_cell_9 : ", x_cell_9.size())

        x_cell_10 = self.cell_10(x_cell_9, x_cell_8)
        #        print("x_cell_10 : ", x_cell_10.size())

        x_cell_11 = self.cell_11(x_cell_10, x_cell_9)
        #        print("x_cell_11 : ", x_cell_11.size())

        x_reduction_cell_1 = self.reduction_cell_1(x_cell_11, x_cell_10)
        #        print("x_reduction_cell_1 : ", x_reduction_cell_1.size())

        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_10)
        #        print("x_cell_12 : ", x_cell_12.size())

        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        #        print("x_cell_13 : ", x_cell_13.size())

        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        #        print("x_cell_14 : ", x_cell_14.size())

        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        #        print("x_cell_15 : ", x_cell_15.size())

        x_cell_16 = self.cell_16(x_cell_15, x_cell_14)
        #        print("x_cell_16 : ", x_cell_16.size())

        x_cell_17 = self.cell_17(x_cell_16, x_cell_15)
        #        print("x_cell_17 : ", x_cell_17.size())

        return x_cell_17

    # def logits(self, features):
    #    x = self.relu(features)
    #    x = self.avg_pool(x)
    #    x = x.view(x.size(0), -1)
    #    x = self.dropout(x)
    #    x = self.last_linear(x)
    #    return x

    def forward(self, input):
        x = self.features(input)
        # x = self.logits(x)
        return x


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.classifier = nn.Sequential(
        #    nn.Linear(512 * 7 * 7, 4096),
        #    nn.ReLU(True),
        #    nn.Dropout(),
        #    nn.Linear(4096, 4096),
        #    nn.ReLU(True),
        #    nn.Dropout(),
        #    nn.Linear(4096, num_classes),
        # )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)




