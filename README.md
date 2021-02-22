# Style Transfer by Relaxed Optimal Transport and Self-Similarity (STROTSS)
This repo is a fork of https://github.com/nkolkin13/STROTSS 

The code in this repo is designed to run on a 4 V100 GPUs. Compared with the original repo the most significant change we make is that we replace the VGG backbone with Nasnet-Large in 'vgg_pt.py'. To accomplish this we parallelize the Nasnet model and loss functions in 'vgg_pt.py' and 'contextual_loss.py', respectively, to support training on 4 GPUs. 

The code is also reconfigured to support training on larger images of size 2048x2048--and so there are alterations to 'styleTransfer.py' and 'st_helper.py' to support this additional image scaling relative to the original repo. 

We also introduced a layer-wise Relaxed Optimal Transport in 'contextual_loss.py' which seems to substantially improve results.

Output images can be seen here: https://mjemison.com/deep-style
