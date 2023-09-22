import os
import time
import math
from numbers import Number
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

import lib.utils as utils
from lib.utils import AverageMeter
from models.VAE_models import *

def train_VAE_idaa(train_loader, args, DEVICE):
    # parse command line arguments

    torch.cuda.set_device(DEVICE)

    vae = vae_idaa(z_dim=args.latent_dim, dataset=args.dataset).to(DEVICE)

    # setup the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - epoch / args.num_epochs)

    # training loop
    train_loader = train_loader[0]
    dataset_size = len(train_loader.dataset)
    num_iterations = len(train_loader) * args.num_epochs
    iteration = 0

    while iteration < num_iterations:
        vae.train()
        loss_avg = AverageMeter()
        loss_rec = AverageMeter()
        loss_kl = AverageMeter()
        for i, x in enumerate(train_loader):
            x = x[0].type(torch.cuda.FloatTensor)
            iteration += 1
            bs = x.shape[0]
            optimizer.zero_grad()
            # transfer to GPU
            x = x.cuda()
            # wrap the mini-batch in a PyTorch Variable
            x = Variable(x)
            _, gx, mu, logvar = vae(x)
            optimizer.zero_grad()
            l_rec = F.mse_loss(x, gx)
            l_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            l_kl /= bs * 3 * args.latent_dim
            loss = l_rec + 0.1 * l_kl
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.data.item(), bs)
            loss_rec.update(l_rec.data.item(), bs)
            loss_kl.update(l_kl.data.item(), bs)
        scheduler.step()

    vae.eval()
    filename = os.path.join(args.save, 'vae_idaa.pth')
    torch.save(vae.state_dict(), filename)
    return vae


##########################################

class LinearWalk(nn.Module):
    def __init__(self, dim_z, reduction_ratio=1.0):
        super(LinearWalk, self).__init__()
        self.walker = nn.Sequential(
            nn.Linear(dim_z, int(dim_z / reduction_ratio)),
            nn.Linear(int(dim_z / reduction_ratio), dim_z)
        )
        # weight initialization: default

    def forward(self, input):
        return self.walker(input)
    

################# InfoMin ####################

class ViewLearner(nn.Module):
    def __init__(self, dataset):
        super(ViewLearner,self).__init__()

        self.conv = nn.Conv2d(1, 1, kernel_size=(1,1))
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(1,1))
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: input tensor of shape (batch_size, channels, height, width)
        out = self.conv(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.relu(out)
        return out   