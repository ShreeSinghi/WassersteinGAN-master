from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import json

# python image_to_video_model.py --n_frames=5 --netG="C:\Users\singh\Downloads\WassersteinGAN-master\bruh2\netG_epoch_8.pth" --netD="C:\Users\singh\Downloads\WassersteinGAN-master\bruh2\netD_epoch_8.pth" --n_extra_layers=2 --experiment=bruh2
#used float16 training but it was slow

import models.dcgan as dcgan
import models.mlp as mlp


parser = argparse.ArgumentParser()
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--n_frames', type=int, default=5, help='Number of video frames')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')

opt = parser.parse_args()
print(opt)

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
n_extra_layers = int(opt.n_extra_layers)
n_frames = int(opt.n_frames)


netG = dcgan.DCGAN_G(opt.imageSize, nz, nc*n_frames, ngf, 1, n_extra_layers)

g_dict = torch.load(opt.netG)
final_key = list(filter(lambda x: x.endswith(":convt.weight") and x.startswith("main.final:"), g_dict.keys()))[0]
weights = torch.cat([g_dict[final_key]]*n_frames, dim=1)

del g_dict[final_key]
g_dict[f'main.final:{ngf}-{nc*n_frames}:convt.weight'] = weights

netG.load_state_dict(g_dict)

netD = dcgan.DCGAN_D(opt.imageSize, nz, nc*n_frames, ndf, 1, n_extra_layers)

d_dict = torch.load(opt.netD)
first_key = list(filter(lambda x: x.endswith(":conv.weight") and x.startswith("main.initial:"), d_dict.keys()))[0]
weights = torch.cat([d_dict[first_key]]*n_frames, dim=1)/n_frames

del d_dict[first_key]
d_dict[f'main.initial:{nc*n_frames}-{ndf}:conv.weight'] = weights

netD.load_state_dict(d_dict)

torch.save(netG.state_dict(), f'{opt.experiment}/modified_G.pth')
torch.save(netD.state_dict(), f'{opt.experiment}/modified_D.pth')