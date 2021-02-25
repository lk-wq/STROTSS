import time
import math
import sys
from glob import glob
import shutil

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from imageio import imread, imwrite

import torch.nn as nn

import utils_go
from utils_go import *
from vgg_pt import *
from pyr_lap import *
from stylize_objectives import objective_class

from apex import amp

from pathlib import Path

from tqdm import tqdm
from helperbot import (
    TriangularLR, BaseBot, WeightDecayOptimizerWrapper,
    LearningRateSchedulerCallback
)
from helperbot.metrics import SoftmaxAccuracy
from apex import amp
from apex.optimizers import FusedAdam

max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

import subprocess


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

def style_transfer(counter, stylized_im, content_im, style_path, output_path, scl, long_side, mask,
                   content_weight=0., use_guidance=False, regions=0, coords=0, lr=2e-3):
    start = time.time()
    torch.cuda.empty_cache()

    # Neural style transfer is performed iteratively. We control the number of iterations depending on the scale of the image. You can set higher or lower numbers of iterations
    # here depending on how much compute you have.
    REPORT_INTERVAL = 100
    RESAMPLE_FREQ = 1
    RESAMPLE_INCREASE_FREQ = 150
    
    save_ind = 0
    counter = counter

    if long_side == 64:
        inner_iter = 1
        MAX_ITER = 109
    if long_side == 128:
        inner_iter = 2
        MAX_ITER = 109
    if long_side == 256:
        inner_iter = 3
        MAX_ITER = 109
    if long_side == 384:
        inner_iter = 4
        MAX_ITER = 109
    if long_side == 512:
        inner_iter = 4
        MAX_ITER = 200
    if long_side == 768:
        inner_iter = 4
        MAX_ITER = 150  # probably 150 on pass 1
    if long_side == 1024:
        inner_iter = 4
        MAX_ITER = 200
    if long_side == 1280:
        inner_iter = 4
        MAX_ITER = 200
    if long_side == 1536:
        inner_iter = 1
        MAX_ITER = 200
    if long_side == 1856:
        inner_iter = 1
        MAX_ITER = 200
    if long_side == 2048:
        inner_iter = 1
        MAX_ITER = 209

    cut = 10000000

    use_pyr = True
    torch.backends.cudnn.enabled = True
    temp_name = './' + output_path.split('/')[-1].split('.')[0] + '_temp.png'
    string_scl = str(scl)
    output_path = output_path.split('/')[-1].split('.')[0] + string_scl + '.png'

    ### Keep track of current output image for GUI ###
    canvas = aug_canvas(stylized_im, scl)
    imwrite(temp_name, canvas)
    shutil.move(temp_name, output_path)

    #### Define feature extractor ###
    cnn1 = (Nas())    
    
    #### Optimize over laplaccian pyramid instead of pixels directly ####
    if use_pyr:
        s_pyr = dec_lap_pyr(stylized_im, 5)
        s_pyr = [Variable(li.data, requires_grad=True) for li in s_pyr]
    else:
        s_pyr = [Variable(stylized_im.data, requires_grad=True)]
        
    ### Define Optimizer ###
    optimizer = optim.RMSprop(s_pyr, lr=lr)

    # You can choose to train in half precision here or not. Half precision doesn't generally produce images that look as good but can scale to larger images.
    cnn1, optimizer = amp.initialize(
        cnn1, optimizer, opt_level='O2', keep_batchnorm_fp32=False, loss_scale='dynamic',
        cast_model_outputs=torch.float16
    )

    phi1 = lambda scl, x, rand, name: cnn1.forward(scl, x, rand, name)
    phi12 = lambda scl, x, name: cnn1.forward_cat(scl, x, name=name)

    paths = glob(style_path + '*')[::3]

    ### Create Objective Object ###
    objective_wrapper = 0
    objective_wrapper = objective_class(objective='remd_dp_g')

    ### Extract guidance features if required ###
    gs = np.array([0.])
    if use_guidance:
        gs = load_style_guidance(phi, style_path, coords[:, 2:], scale=long_side)


    ### Randomly choose spatial locations to extract features from ###
    with torch.no_grad():
        z_c = phi1(scl, content_im, [], 'content')
        #   print("content im size : " , content_im.size(), content_im.max(), content_im.min())
        z_s = load_style_folder(scl, phi12, paths, scale=long_side)

    for j in range(MAX_ITER):

        if use_pyr:
            stylized_im = syn_lap_pyr(s_pyr)
        else:
            stylized_im = s_pyr[0]

        z_x_style = phi1(scl, stylized_im, [], 'stylized')

        for i in range(0, inner_iter):
            ell = 0.
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            
            # Based on the scale of the image we sub-sample points from the content image
            with torch.no_grad():
                if j == 0 or j % (RESAMPLE_FREQ * 1) == 0:
                    z_s_all = []
                    for ri in range(len(regions[1])):
                        if long_side == 64:
                            z_s_input = selection_subsampling(z_s, paths, regions, ri, samps=5000, scale=long_side)
                        elif long_side == 128:
                            z_s_input = selection_subsampling(z_s, paths, regions, ri, samps=5000, scale=long_side)
                        elif long_side == 256:
                            z_s_input = selection_subsampling(z_s, paths, regions, ri, samps=5000, scale=long_side)
                        elif long_side == 384:
                            z_s_input = selection_subsampling(z_s, paths, regions, ri, samps=5000, scale=long_side)
                        elif long_side == 512:
                            z_s_input = selection_subsampling(z_s, paths, regions, ri, samps=5000, scale=long_side)
                        elif long_side == 768:
                            z_s_input = selection_subsampling(z_s, paths, regions, ri, samps=5000, scale=long_side)
                        elif long_side == 1024:
                            z_s_input = selection_subsampling(z_s, paths, regions, ri, samps=4500, scale=long_side)
                        elif long_side == 1280:
                            z_s_input = selection_subsampling(z_s, paths, regions, ri, samps=int(3.0 * (32 ** 2)),
                                                              scale=long_side)
                        elif long_side == 1536:
                            z_s_input = selection_subsampling(z_s, paths, regions, ri, samps=int(4.0 * (32 ** 2)),
                                                              scale=long_side)
                        elif long_side == 1856:
                            z_s_input = selection_subsampling(z_s, paths, regions, ri, samps=5 * (32 ** 2),
                                                              scale=long_side)
                        elif long_side == 2048:
                            z_s_input = selection_subsampling(z_s, paths, regions, ri, samps=5 * (32 ** 2),
                                                              scale=long_side)
                    z_s_all = [z_s_input]
                    torch.cuda.empty_cache()

                ## Dramatically Resample Large Set of Spatial Locations ##
                if j == 0 or j % (RESAMPLE_FREQ * 1) == 0:
                    for ri in range(len(regions[0])):

                        r_temp = regions[0][ri]
                        r_temp = torch.from_numpy(r_temp).unsqueeze(0).unsqueeze(0).contiguous()
                        r = F.upsample(r_temp, (stylized_im.size(3), stylized_im.size(2)), mode='bilinear')[0, 0, :,
                            :].numpy()

                        if r.max() < 0.1:
                            r = np.greater(r + 1., 0.5)
                        else:
                            r = np.greater(r, 0.5)

                        objective_wrapper.init_inds(z_c, z_s_all, r, ri)

                torch.cuda.empty_cache()

                ## Subsample spatial locations to compute loss over ##
                if j == 0 or j % RESAMPLE_FREQ == 0:
                    objective_wrapper.shuffle_feature_inds()
                torch.cuda.empty_cache()

            ell += objective_wrapper.eval(long_side, cut, scl, z_x_style, z_x_style, z_c, z_s_all, gs, 0.,
                                          content_weight=content_weight, moment_weight=1.0) / inner_iter
            torch.cuda.empty_cache()

        with amp.scale_loss(ell, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        ## Periodically save output image for GUI ##
        if (j + 1) % 10 == 0:
            canvas = aug_canvas(stylized_im, scl)
            imwrite(temp_name, canvas)
            shutil.move(temp_name, output_path)
            print("tensor loss : ", ell)

        ### Periodically Report Loss and Save Current Image ###
        if (j + 1) % REPORT_INTERVAL == 0:
            save_ind += 1

        print("iteration : ", j)
        print("counter : ", counter)
        torch.cuda.empty_cache()
    end = time.time()
    print("scale time is :  ----------------------------------------------> ", end - start)
    return stylized_im, ell

# we ha
def style_transfer_large(counter, stylized_im, content_im, style_path, output_path, scl, long_side, mask,
                         content_weight=0., use_guidance=False, regions=0, coords=0, lr=2e-3):
    start = time.time()
    REPORT_INTERVAL = 100
    RESAMPLE_FREQ = 1
    RESAMPLE_INCREASE_FREQ = 150
    if scl == 1:
        MAX_ITER = 17
    if scl == 2:
        MAX_ITER = 17
    save_ind = 0
    counter = counter

    if scl == 3:
        MAX_ITER = 17

    if scl == 4:
        MAX_ITER = 17

    if counter == 0:
        MAX_ITER = 50

    if counter > 9:
        MAX_ITER = 150
    else:
        MAX_ITER = 100
        
    MAX_ITER = 200
    if scl >= 6:
        MAX_ITER = 150
    MAX_ITER = 400

    use_pyr = True
    torch.backends.cudnn.enabled = True
    temp_name = './' + output_path.split('/')[-1].split('.')[0] + '_temp.png'
    string_scl = str(scl) + '_' + str(counter)
    output_path = output_path.split('/')[-1].split('.')[0] + string_scl + '.png'

    ### Keep track of current output image for GUI ###
    canvas = aug_canvas(stylized_im, long_side)
    imwrite(temp_name, canvas)
    shutil.move(temp_name, output_path)

    #### Define feature extractor ###
    # if scl == 1:
    #    cnn = utils.to_device(Vgg16_pt_small())
    # else:
    # cnn = utils.to_device(Vgg16_pt())

    cnn1 = utils_go.to_device(Nas())

    # mode = 'content-capture'

    # phi = lambda x, rand: cnn.forward(x,rand)

    # phi = lambda scl, x : cnn.forward(scl,x)

    #### Optimize over laplaccian pyramid instead of pixels directly ####

    ### Define Optimizer ###
    if use_pyr:
        s_pyr = dec_lap_pyr(stylized_im, 5)
        s_pyr = [Variable(li.data, requires_grad=True) for li in s_pyr]
    else:
        s_pyr = [Variable(stylized_im.data, requires_grad=True)]

    optimizer = optim.RMSprop(s_pyr, lr=lr)

    cnn1, optimizer = amp.initialize(
        cnn1, optimizer, opt_level='O2', keep_batchnorm_fp32=False, loss_scale="dynamic"
    )

    phi1 = lambda scl, x, rand, name: cnn1.forward(scl, x, rand, name)
    phi12 = lambda scl, x, y, z, rand, name: cnn1.forward_cat_large(scl, x, z, samps=y, forward_func=cnn1.forward,
                                                                    rand_whoa=rand, name=name)

    # phi_content = lambda scl, x: cnn2.forward(scl, x)

    ### Pre-Extract Content Features ###
    # z_c_for_style = phi1(content_im)

    ### Pre-Extract Style Features from a Folder###
    paths = glob(style_path + '*')[::3]

    ### Create Objective Object ###
    objective_wrapper = 0
    objective_wrapper = objective_class(objective='remd_dp_g')

    ### Extract guidance features if required ###
    gs = np.array([0.])
    if use_guidance:
        gs = load_style_guidance(phi, style_path, coords[:, 2:], scale=long_side)

    ### Randomly choose spatial locations to extract features from ###
    if use_pyr:
        stylized_im = syn_lap_pyr(s_pyr)
    else:
        stylized_im = s_pyr[0]

    # z_c = phi_content(scl, content_im)
    with torch.no_grad():
        z_c = phi1(scl, content_im, [], 'content')
        #   print("content im size : " , content_im.size(), content_im.max(), content_im.min())

        z_s_all = []
        for ri in range(len(regions[1])):
            if scl == 1:
                z_s, style_ims = load_style_folder_large(scl, phi12, [], paths, regions, ri, n_samps=-1,
                                                         subsamps=1000, scale=long_side, inner=5)
            elif scl == 2:
                z_s, style_ims = load_style_folder_large(scl, phi12, [], paths, regions, ri, n_samps=-1,
                                                         subsamps=3176, scale=long_side, inner=5)
            elif scl == 3:
                z_s, style_ims = load_style_folder_large(scl, phi12, [], paths, regions, ri, n_samps=-1,
                                                         subsamps=8092, scale=long_side, inner=5)
            elif scl == 4:
                z_s, style_ims = load_style_folder_large(scl, phi12, [], paths, regions, ri, n_samps=-1,
                                                         subsamps=4508, scale=long_side, inner=6)
            elif scl == 5:
                z_s, style_ims = load_style_folder_large(scl, phi12, [], paths, regions, ri, n_samps=-1,
                                                         subsamps=4000, scale=long_side, inner=6)
            elif scl == 6:
                z_s, style_ims = load_style_folder_large(scl, phi12, [], paths, regions, ri, n_samps=-1,
                                                         subsamps=4100, scale=long_side, inner=7)
            elif scl == 7:
                z_s, style_ims = load_style_folder_large(scl, phi12, [], paths, regions, ri, n_samps=-1,
                                                         subsamps=4100, scale=long_side, inner=8)
            elif scl == 8:
                z_s, style_ims = load_style_folder_large(scl, phi12, [], paths, regions, ri, n_samps=-1,
                                                         subsamps=4400, scale=long_side, inner=7)
            elif scl == 9:
                z_s, style_ims = load_style_folder_large(scl, phi12, [], paths, regions, ri, n_samps=-1,
                                                         subsamps=4400, scale=long_side, inner=6)
            elif scl == 10:
                z_s, style_ims = load_style_folder_large(scl, phi12, [], paths, regions, ri, n_samps=-1,
                                                         subsamps=3850, scale=long_side, inner=6)
            elif scl == 11:
                z_s, style_ims = load_style_folder_large(scl, phi12, [], paths, regions, ri, n_samps=-1,
                                                         subsamps=3200, scale=long_side, inner=6)
            elif scl == 12:
                z_s, style_ims = load_style_folder_large(scl, phi12, [], paths, regions, ri, n_samps=-1,
                                                         subsamps=1000, scale=long_side, inner=4)
            else:
                z_s, style_ims = load_style_folder_large(scl, phi12, [], paths, regions, ri, n_samps=-1,
                                                         subsamps=1000, scale=long_side, inner=5)

            z_s_all.append(z_s)
        torch.cuda.empty_cache()

        for ri in range(len(regions[0])):

            r_temp = regions[0][ri]
            # print("r_temp sizE :" , r_temp.shape)
            r_temp = torch.from_numpy(r_temp).unsqueeze(0).unsqueeze(0).contiguous()
            r = F.upsample(r_temp, (stylized_im.size(3), stylized_im.size(2)), mode='bilinear')[0, 0, :, :].numpy()
            # print("r_temp sizE 2 :" , r_temp.shape)
            # print("s i sizE :" , stylized_im.shape)

            if r.max() < 0.1:
                r = np.greater(r + 1., 0.5)
            else:
                r = np.greater(r, 0.5)

            objective_wrapper.init_inds(z_c, z_s_all, r, ri)
        torch.cuda.empty_cache()

    for i in range(MAX_ITER):

        ### zero out gradients and compute output image from pyramid ##
        optimizer.zero_grad()
        if use_pyr:
            stylized_im = syn_lap_pyr(s_pyr)
        else:
            stylized_im = s_pyr[0]

        mode = 'stylized-capture'
        torch.cuda.empty_cache()

        # START INDEX SELECTION 2

        style_index_list = []

        z_x_style = phi1(scl, stylized_im, style_index_list, 'stylized')

        torch.cuda.empty_cache()

        # END INDEX SELECTION 2
        with torch.no_grad():

            ## Dramatically Resample Large Set of Spatial Locations ##
            if i == 0 or i % (1 * 10) == 0:
                for ri in range(len(regions[0])):

                    r_temp = regions[0][ri]
                    r_temp = torch.from_numpy(r_temp).unsqueeze(0).unsqueeze(0).contiguous()
                    r = F.upsample(r_temp, (stylized_im.size(3), stylized_im.size(2)), mode='bilinear')[0, 0, :,
                        :].numpy()

                    if r.max() < 0.1:
                        r = np.greater(r + 1., 0.5)
                    else:
                        r = np.greater(r, 0.5)

                    objective_wrapper.init_inds(z_c, z_s_all, r, ri)

            torch.cuda.empty_cache()

            ## Subsample spatial locations to compute loss over ##
            if i == 0 or i % 1 == 0:
                objective_wrapper.shuffle_feature_inds()

        ## Extract Features from Current Output
        # z_x_content = phi1(scl, stylized_im, style_index_list, 'content for style')
        # z_x_content = phi_content(scl, stylized_im)

        torch.cuda.empty_cache()

        # print("stylized im size : " , stylized_im.size())

        ## Compute Objective and take gradient step ##
        torch.cuda.empty_cache()

        cut = 1000000

        ell = objective_wrapper.eval(long_side, cut, scl, z_x_style, z_x_style, z_c, z_s_all, gs, 0.,
                                     content_weight=content_weight, moment_weight=1.0)

        # with amp.scale_loss(ell, optimizer) as scaled_loss:
        #    scaled_loss.backward()

        with amp.scale_loss(ell, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()

        ## Periodically save output image for GUI ##
        if (i + 1) % 100 == 0:
            canvas = aug_canvas(stylized_im, long_side)
            imwrite(temp_name, canvas)
            shutil.move(temp_name, output_path)
            print("tensor loss : ", ell)

        ### Periodically Report Loss and Save Current Image ###
        if (i + 1) % REPORT_INTERVAL == 0:
            print((i + 1), ell)
            save_ind += 1

        print("iteration : ", i)
        print("counter : ", counter)
        torch.cuda.empty_cache()
    end = time.time()
    print("scale time is :  ----------------------------------------------> ", end - start)
    return stylized_im, ell

