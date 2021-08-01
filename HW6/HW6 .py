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

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(2021)

class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        # 1. Load the image
        img = torchvision.io.read_image(fname)
        # 2. Resize and normalize the images using torchvision.
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples


def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(fnames, transform)
    return dataset

dataset = get_dataset(os.path.join(workspace_dir, 'faces'))



def load_pretrained_model(self):
    self.G.load_state_dict(torch.load(os.path.join(
        self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
    self.D.load_state_dict(torch.load(os.path.join(
        self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
    print('loaded trained models (step: {})..!'.format(self.pretrained_model))


config = {
    'batch_size': 32,
    'early_stop': 300,
    'z_dim': 128,
    'lambda_gp':10,
    'total_step':1000000,
    'save_path': './model.ckpt'

}
batch_size = 32

z_sample = Variable(torch.randn(64, config['z_dim'])).cuda()
g_lr = 0.0001
d_lr = 0.0004
imsize=64
g_conv_dim=64
d_conv_dim=64
""" Medium: WGAN, 50 epoch, n_critic=5, clip_value=0.01 """
n_epoch = 50 # 50
n_critic = 5 # 5
clip_value = 0.01

log_dir = os.path.join(workspace_dir, 'logs')
ckpt_dir = os.path.join(workspace_dir, 'checkpoints')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# Model
G = Generator(batch_size,imsize, config['z_dim'], g_conv_dim).cuda()
D = Discriminator(batch_size,imsize, d_conv_dim).cuda()
G.load_state_dict(torch.load(os.path.join( '.', 'G.pth')))
D.load_state_dict(torch.load(os.path.join(  '.', 'D.pth')))
G.train()
D.train()

# Loss
criterion = nn.CrossEntropyLoss()

# DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)



def train(dataloader, D,G, config,z_sample):

 start_time = time.time()
 opt_D = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=d_lr, betas=(0.0, 0.9))
 opt_G = torch.optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), lr=g_lr, betas=(0.0, 0.9))

 for e, epoch in enumerate(range(n_epoch)):
    #progress_bar = qqdm(dataloader)
    D.train()
    G.train()
    print(e)
    for data in dataloader:
        imgs = data

        imgs = imgs.cuda()

        bs = imgs.size(0)

        # ============================================
        #  Train D
        # ============================================
        z = Variable(torch.randn(bs, config['z_dim'])).cuda()
        r_imgs = Variable(imgs).cuda()
        f_imgs,gf1,gf2 = G(z)

        """ Medium: Use WGAN Loss. """

        # Model forwarding
        r_logit,dr1,dr2 = D(r_imgs.detach())
        f_logit,df1,df2 = D(f_imgs.detach())

        # Compute the loss for the discriminator.
        d_loss_real = - torch.mean(r_logit)
        d_loss_fake = f_logit.mean()

        loss_D = d_loss_real + d_loss_fake


        # Model backwarding
        opt_D.zero_grad()
        opt_G.zero_grad()
        loss_D.backward()

        # Update the discriminator.
        opt_D.step()

        """ Compute gradient penalty """
        alpha = torch.rand(r_imgs.size(0), 1, 1, 1).cuda().expand_as(r_imgs)
        interpolated = Variable(alpha * r_imgs.data + (1 - alpha) * f_imgs.data, requires_grad=True)
        out, _, _ = D(interpolated)

        grad = torch.autograd.grad(outputs=out,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(out.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        loss_D = config['lambda_gp'] * d_loss_gp

        opt_D.zero_grad()
        opt_G.zero_grad()
        loss_D.backward()
        opt_D.step()
        ''''''
        # ============================================
        #  Train G
        # ============================================

        # Generate some fake images.
        z = Variable(torch.randn(bs, config['z_dim'])).cuda()
        f_imgs ,_,_ = G(z)

        # Model forwarding
        f_logit,_,_ = D(f_imgs)
        g_loss_fake= - f_logit.mean()
        opt_D.zero_grad()
        opt_G.zero_grad()

        # Model backwarding
        G.zero_grad()
        g_loss_fake.backward()

        # Update the generator.
        opt_G.step()



    G.eval()
    '''
        if (e + 1) % 5 == 0:
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
              " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}".
              format(elapsed, e + 1, config['total_step'], (e + 1),
                     config['total_step'], d_loss_real.item()[0],
                     G.attn1.gamma.mean().data[0], G.attn2.gamma.mean().data[0]))
    '''
    if (e + 1) % 1 == 0:
            fake_images, _, _ = G(z_sample)
            save_image(denorm(fake_images.data),
                   os.path.join('.', '{}_fake.png'.format(e + 1)))



    G.train()

    if (e + 1) % 1 == 0:
            torch.save(G.state_dict(), os.path.join('.', 'G.pth'))
            torch.save(D.state_dict(), os.path.join('.', 'D.pth'))
torch.cuda.empty_cache()
train(dataloader, D,G, config,z_sample)