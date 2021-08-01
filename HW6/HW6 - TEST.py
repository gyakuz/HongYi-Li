import random
import os
import glob
import time
import datetime
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from qqdm.notebook import qqdm
import torch
import numpy as np
from sagan_models import Generator, Discriminator
from utils import *


workspace_dir = '.'


config = {
    'batch_size': 32,
    'early_stop': 300,
    'z_dim': 128,
    'lambda_gp':10,
    'total_step':1000000,
    'save_path': './model.ckpt'

}
batch_size = 32


g_lr = 0.0001
d_lr = 0.0004
imsize=64
g_conv_dim=64
n_epoch = 50 # 50
n_critic = 5 # 5
clip_value = 0.01

num=0

# Model
G = Generator(batch_size,imsize, config['z_dim'], g_conv_dim).cuda()

G.load_state_dict(torch.load(os.path.join( '.', 'G.pth')))



for e in range(10):
  z_sample = Variable(torch.randn(100, config['z_dim'])).cuda()
  fake_images, _, _ = G(z_sample)
  for image in fake_images:
    save_image(image,
               os.path.join('./face3', '{}.png'.format(num+ 1)))
    num=num+1
