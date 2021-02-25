# Style Transfer by Relaxed Optimal Transport and Self-Similarity (STROTSS)
This repo is a fork of https://github.com/nkolkin13/STROTSS 

The code in this repo is designed to run on a 4 V100 GPUs. We observe that the expensive loss function in the original repo establishes a 'compute-quality' tradeoff. This means that we can increase the quality of the images by increasing the proportion of the images sampled before evaluation by the loss functions.

We also replace the VGG backbone with Nasnet-Large in 'vgg_pt.py'. To accomplish this we parallelize the Nasnet model and loss functions in 'vgg_pt.py' and 'contextual_loss.py', respectively, to support training on 4 GPUs. 

The code is also reconfigured to support training on larger images of size 2048x2048--and so there are alterations to 'styleTransfer.py' and 'st_helper.py' to support this additional image scaling relative to the original repo. 

We also introduced a layer-wise Relaxed Optimal Transport in 'contextual_loss.py' which seems to substantially improve results.

Output images can be seen here: https://mjemison.com/deep-style
