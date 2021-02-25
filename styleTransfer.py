import time
import math
import sys

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from imageio import imread, imwrite

from st_helper import *
import utils_go
from utils_go import *

import sys

from skimage.transform import match_histograms
# from linear_color_transfer import match_color

import scipy.misc as spi
from skimage import io, transform, img_as_float
from skimage.io import imsave
from PIL import Image
from numpy import eye
from matplotlib.pyplot import imread

import sys
import argparse, time, os
import imageio
import subprocess

import matplotlib

import os
from collections import OrderedDict
from datetime import datetime
import json


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def parse(opt_path):
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    opt['timestamp'] = get_timestamp()
    scale = opt['scale']
    rgb_range = opt['rgb_range']

    # export CUDA_VISIBLE_DEVICES
    if torch.cuda.is_available():
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        print('===> Export CUDA_VISIBLE_DEVICES = [' + gpu_list + ']')
    else:
        print('===> CPU mode is set (NOTE: GPU is recommended)')

    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['scale'] = scale
        dataset['rgb_range'] = rgb_range

    # for network initialize
    opt['networks']['scale'] = opt['scale']
    network_opt = opt['networks']

    config_str = '%s_in%df%d_x%d' % (network_opt['which_model'].upper(), network_opt['in_channels'],
                                     network_opt['num_features'], opt['scale'])
    exp_path = os.path.join(os.getcwd(), 'experiments', config_str)

    if opt['is_train'] and opt['solver']['pretrain']:
        if 'pretrained_path' not in list(opt['solver'].keys()): raise ValueError(
            "[Error] The 'pretrained_path' does not declarate in *.json")
        exp_path = os.path.dirname(os.path.dirname(opt['solver']['pretrained_path']))
        if opt['solver']['pretrain'] == 'finetune': exp_path += '_finetune'

    exp_path = os.path.relpath(exp_path)

    path_opt = OrderedDict()
    path_opt['exp_root'] = exp_path
    path_opt['epochs'] = os.path.join(exp_path, 'epochs')
    path_opt['visual'] = os.path.join(exp_path, 'visual')
    path_opt['records'] = os.path.join(exp_path, 'records')
    opt['path'] = path_opt

    if opt['is_train']:
        # create folders
        if opt['solver']['pretrain'] == 'resume':
            opt = dict_to_nonedict(opt)
        else:
            util.mkdir_and_rename(opt['path']['exp_root'])  # rename old experiments if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'exp_root'))
            save(opt)
            opt = dict_to_nonedict(opt)

        print("===> Experimental DIR: [%s]" % exp_path)

    return opt


def save(opt):
    dump_dir = opt['path']['exp_root']
    dump_path = os.path.join(dump_dir, 'options.json')
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    return int(result)

def run_st(content_path, style_path, content_weight, content_iterate, use_iterate, max_scl, coords, use_guidance,
           regions, output_path='./output.png'):
    smll_sz = 64

    start = time.time()

    content_weight_freeze = content_weight

    content_im_big = utils_go.to_device(
        Variable(load_path_for_pytorch(content_path, 512, force_scale=True).unsqueeze(0)))

    counter = 0

    # We use scl_list to control how big the images scale up in addition to controlling how much we subsample the content and style images
    
    scl_list = [1] + [2] + [3] + [4] + [5] + [6] + [7]*1 + [8]*1 + [9]*1 + [10]*1 + [11]*1 # + [6]*2 + [7]*2 +[8]*2 + [9]*2 +[10]*2 + [11]*2  #[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 5, 5, 5, 6, 6, 6, 7, 7, 7] # 2,3,4,5,3,4,5,4,5]

    for j, scl in enumerate(scl_list):
        print("counter is : ", counter)
        content_weight = content_weight_freeze * 2.0 ** (max_scl - int(scl))
        
        # size_list lists the sizes of images we scale to. The largest size is 2048 x 2048, smallest is 64 x 64
        size_list = [64] + [128] + [256] + [384] + [512] + [768] + [1024]*1 + [1280]*1  + [1536]*1 + [1856]*1 + [2048]*1 # + [768]*2 + [1024]*2 +[1280]*2 + [1536]*2 +[1856]*2 + [2048]*2
        long_side = size_list[j]  # smll_sz * (2 ** (int(scl) - 1))
        lr = 2e-3

        content_im = utils_go.to_device(
            Variable(load_path_for_pytorch(content_path, long_side, force_scale=True).unsqueeze(0))).half()
        content_im_mean = utils_go.to_device(
            Variable(load_path_for_pytorch(style_path, long_side, force_scale=True).unsqueeze(0))).mean(2,
                                                                                                        keepdim=True).mean(
            3, keepdim=True).half()

        ### Compute bottom level of laplaccian pyramid for content image at current scale ###
        lap = content_im.clone() - F.upsample(
            F.upsample(content_im, (content_im.size(2) // 2, content_im.size(3) // 2), mode='bilinear'),
            (content_im.size(2), content_im.size(3)), mode='bilinear')

        ### Initialize by zeroing out all but highest and lowest levels of Laplaccian Pyramid ###
        if scl == 1:
            if use_iterate and counter == 0:
                stylized_im = utils_go.to_device(
                    Variable(load_path_for_pytorch(content_iterate, smll_sz, force_scale=True).unsqueeze(0))) + lap
            elif not use_iterate and counter == 0:
                stylized_im = Variable(content_im_mean + lap)
            else:
                stylized_im = F.upsample(stylized_im.clone(), (content_im.size(2), content_im.size(3)),
                                         mode='bilinear') + lap

        ### Otherwise bilinearly upsample previous scales output and add back bottom level of Laplaccian pyramid for current scale of content image ###

        if long_side == 128:
            if use_iterate and counter == 0:
                stylized_im = utils_go.to_device(
                    Variable(load_path_for_pytorch(content_iterate, long_side, force_scale=True).unsqueeze(0)))
            else:
                stylized_im = F.upsample(stylized_im.clone(), (content_im.size(2), content_im.size(3)),
                                         mode='bilinear') + lap
        elif long_side == 256:
            if use_iterate and counter == 0:
                stylized_im = utils_go.to_device(
                    Variable(load_path_for_pytorch(content_iterate, long_side, force_scale=True).unsqueeze(0))).half() + lap.half()
            else:
                stylized_im = F.upsample(stylized_im.clone(), (content_im.size(2), content_im.size(3)),
                                         mode='bilinear') + lap
        elif long_side == 384:
            if use_iterate and counter == 0:
                stylized_im = utils_go.to_device(
                    Variable(load_path_for_pytorch(content_iterate, long_side, force_scale=True).unsqueeze(0))).half() + lap.half()
            else:
                stylized_im = F.upsample(stylized_im.clone(), (content_im.size(2), content_im.size(3)),
                                         mode='bilinear').half() + lap.half()
        elif long_side == 512:
            if use_iterate and counter == 0:
                stylized_im = utils_go.to_device(
                    Variable(load_path_for_pytorch(content_iterate, long_side, force_scale=True).unsqueeze(0))).half() + lap.half()
            else:

                stylized_im = F.upsample(stylized_im.clone(), (content_im.size(2), content_im.size(3)),
                                         mode='bilinear').half() + lap.half()
        elif long_side == 768:
            if use_iterate and counter == 0:
                stylized_im = utils_go.to_device(
                    Variable(load_path_for_pytorch(content_iterate, long_side, force_scale=True).unsqueeze(0))) + lap
            else:

                stylized_im = F.upsample(stylized_im.clone(), (content_im.size(2), content_im.size(3)),
                                         mode='bilinear').half() + lap.half()

        elif long_side > 768:
            if use_iterate and counter == 0:
                stylized_im = utils_go.to_device(
                    Variable(load_path_for_pytorch(content_iterate, long_side, force_scale=True).unsqueeze(0)))
            else:
                stylized_im = F.upsample(stylized_im.clone(), (content_im.size(2), content_im.size(3)), mode='bilinear')

        if long_side == 768:
            lr = (0.75e-3)/2

        if long_side == 512:
            lr = 1e-3
        if long_side == 1024:
            lr = (0.5e-3)/2
            content_weight = 0.75
        if long_side > 1024:
            lr = (0.375e-3)/2
            content_weight = 0.75
        if long_side == 384:
            content_weight = 2.25
        if long_side == 768:
            content_weight = 1.5
        if long_side == 1536:
            content_weight = 0.5

        if long_side >= 2048:
            lr = (0.25e-3)/2
            content_weight = 0.375

        ### Style Transfer at this scale ###
        if long_side < 1536:

            stylized_im, final_loss = style_transfer_large(counter, stylized_im, content_im, style_path, output_path,
                                                           scl, long_side, 0.,
                                                           use_guidance=use_guidance, coords=coords,
                                                           content_weight=content_weight, lr=lr, regions=regions)
        else:
            stylized_im, final_loss = style_transfer_large(counter, stylized_im, content_im, style_path, output_path,
                                                           scl, long_side, 0.,
                                                           use_guidance=use_guidance, coords=coords,
                                                           content_weight=content_weight, lr=lr, regions=regions)


        canvas = torch.clamp(stylized_im, -0.5, 0.5)[0].data.cpu().numpy().transpose(1, 2, 0)

        # imwrite('/content/super-resolution/output.png', canvas)

        ### Decrease Content Weight for next scale ###
        content_weight = content_weight_freeze * 2.0 ** (max_scl - scl)

        print("content_weight_freeze : ", content_weight_freeze)

        counter += 1

    print("Finished in: ", int(time.time() - start), 'Seconds')
    print('Final Loss:', final_loss)

    # canvas = aug_canvas(stylized_im, scl)

    canvas = torch.clamp(stylized_im[0], -0.5, 0.5).data.cpu().numpy().transpose(1, 2, 0)
    imwrite(output_path, canvas)

    print("hist matching start")
    print("hist matching end")

    return final_loss, stylized_im


if __name__ == '__main__':

    ### Parse Command Line Arguments ###
    content_path = sys.argv[1]
    style_path = sys.argv[2]
    content_weight = float(sys.argv[3])
    
    #the use_iterate command enables you to start scaling the size of an intermediate image instead of starting from scratch
    use_iterate = 'iterate' in sys.argv
    
    print("sys.argv : ", sys.argv, type(sys.argv))
    super_res = 'super_res' in sys.argv
    if use_iterate:
        content_iterate = sys.argv[4]
    else:
        content_iterate = None

    max_scl = 5

    use_guidance_region = '-gr' in sys.argv
    use_guidance_points = False

    use_gpu = not ('-cpu' in sys.argv)
    utils_go.use_gpu = use_gpu

    paths = glob(style_path + '*')
    losses = []
    ims = []

    ### Preprocess User Guidance if Required ###
    coords = 0.
    if use_guidance_region:
        i = sys.argv.index('-gr')
        regions = utils_go.extract_regions(sys.argv[i + 1], sys.argv[i + 2])
    else:
        try:
            regions = [[imread(content_path)[:, :, 0] * 0. + 1.], [imread(style_path)[:, :, 0] * 0. + 1.]]
        except:
            regions = [[imread(content_path)[:, :] * 0. + 1.], [imread(style_path)[:, :] * 0. + 1.]]

    ### Style Transfer and save output ###
    loss, canvas = run_st(content_path, style_path, content_weight, content_iterate, use_iterate, max_scl, coords,
                          use_guidance_points, regions)

