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

import lib.dist as dist
import lib.utils as utils
#import lib.datasets as dset
from lib.flows import FactorialNormalizingFlow
from models.VAE_models import *

from elbo_decomposition import elbo_decomposition
from plot_latent_vs_true import * 


class VAE(nn.Module):
    def __init__(self, z_dim, args=None, use_cuda=False, prior_dist=dist.Normal(), q_dist=dist.Normal(),
                 include_mutinfo=True, tcvae=False, mss=False):
        super(VAE, self).__init__()

        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.args = args
        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb = 0
        self.beta = 5
        self.mss = mss
        self.x_dist = dist.Bernoulli()

        # Model-specific
        # distribution family of p(z)
        self.prior_dist = prior_dist
        self.q_dist = q_dist
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

        # create the encoder and decoder networks
        if self.args.dataset == 'ucihar':
            self.encoder = ConvEncoder(z_dim * self.q_dist.nparams)
            self.decoder = ConvDecoder(z_dim)
        elif self.args.dataset == 'shar':
            self.encoder = ConvEncoder_shar(z_dim * self.q_dist.nparams)
            self.decoder = ConvDecoder_shar(z_dim)
        elif self.args.dataset == 'usc' or self.args.dataset == 'hhar':
            self.encoder = ConvEncoder_usc(z_dim * self.q_dist.nparams)
            self.decoder = ConvDecoder_usc(z_dim)       
        elif self.args.dataset == 'ieee_small' or self.args.dataset == 'ieee_big' or self.args.dataset == 'dalia':
            self.encoder = ConvEncoder_ieeesmall(z_dim * self.q_dist.nparams)
            self.decoder = ConvDecoder_ieeesmall(z_dim)        
        elif self.args.dataset == 'ecg':
            self.encoder = ConvEncoder_ecg(z_dim * self.q_dist.nparams)
            self.decoder = ConvDecoder_ecg(z_dim)                       

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # samples from the model p(x|z)p(z)
    def model_sample(self, batch_size=1):
        # sample from prior (value will be sampled by guide when computing the ELBO)
        prior_params = self._get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        # decode the latent code z
        x_params = self.decoder.forward(zs)
        return x_params

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
        x = x.view(x.size(0), 1, x.size(2), x.size(3)) if len(x.shape) == 4 else x.view(x.size(0), 1, x.size(1), x.size(2)) 
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        return zs, z_params

    def decode(self, z):
        dummy = self.decoder.forward(z)
        x_params = dummy.view(z.size(0), 1, dummy.size(2), dummy.size(3))
        xs = self.x_dist.sample(params=x_params)
        return xs, x_params

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        zs, z_params = self.encode(x)
        xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

    def elbo(self, x, dataset_size):
        # log p(x|z) + log p(z) - log q(z|x)
        batch_size, sequence_length, channels = x.size(0), x.size(1), x.size(2)
        x = x.view(batch_size, 1, sequence_length, channels)
        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)

        elbo = logpx + logpz - logqz_condx
        if self.beta == 1 and self.include_mutinfo and self.lamb == 0:
            return elbo, elbo.detach()

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
        )

        if not self.mss:
            # minibatch weighted sampling
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        else:
            # minibatch stratified sampling
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqz.data))
            logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)

        if not self.tcvae:
            if self.include_mutinfo:
                modified_elbo = logpx - self.beta * (
                    (logqz_condx - logpz) -
                    self.lamb * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = logpx - self.beta * (
                    (logqz - logqz_prodmarginals) +
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
        else:
            if self.include_mutinfo:
                modified_elbo = logpx - \
                    (logqz_condx - logqz) - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
            else:
                modified_elbo = logpx - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)

        return modified_elbo, elbo.detach()


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


win_samples = None
win_test_reco = None
win_latent_walk = None
win_train_elbo = None


def plot_elbo(train_elbo, vis):
    global win_train_elbo
    win_train_elbo = vis.line(torch.Tensor(train_elbo), opts={'markers': True}, win=win_train_elbo)


def anneal_kl(args, vae, iteration):
    if args.dataset == 'ucihar':
        warmup_iter = 2000

    if args.lambda_anneal:
        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
    else:
        vae.lamb = 0
    if args.beta_anneal:
        vae.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
    else:
        vae.beta = args.beta

def setup_the_VAE(args):
    # setup the VAE
    if args.dist == 'normal':
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
    elif args.dist == 'laplace':
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
    elif args.dist == 'flow':
        prior_dist = FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
        q_dist = dist.Normal()
    return prior_dist, q_dist

def train_VAE(train_loader, args, DEVICE):
    # parse command line arguments

    torch.cuda.set_device(DEVICE)

    # setup the VAE
    prior_dist, q_dist = setup_the_VAE(args)

    vae = VAE(z_dim=args.latent_dim, args=args, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist,
        include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, mss=args.mss)

    # setup the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)

    # # setup visdom for visualization
    # if args.visdom:
    #     vis = visdom.Visdom(env=args.save, port=4500)

    train_elbo = []

    # training loop
    train_loader = train_loader[0]
    dataset_size = len(train_loader.dataset)
    num_iterations = len(train_loader) * args.num_epochs
    iteration = 0
    # initialize loss accumulator
    elbo_running_mean = utils.RunningAverageMeter()
    while iteration < num_iterations:
        for i, x in enumerate(train_loader):
            #import pdb;pdb.set_trace();
            x = x[0].type(torch.cuda.FloatTensor)
            iteration += 1
            batch_time = time.time()
            vae.train()
            anneal_kl(args, vae, iteration)
            optimizer.zero_grad()
            # transfer to GPU
            x = x.cuda()
            # wrap the mini-batch in a PyTorch Variable
            x = Variable(x)
            # do ELBO gradient and accumulate loss
            obj, elbo = vae.elbo(x, dataset_size)
            if utils.isnan(obj).any():
                raise ValueError('NaN spotted in objective.')
            obj.mean().mul(-1).backward()
            elbo_running_mean.update(elbo.mean())
            optimizer.step()

            # report training diagnostics
            if iteration % args.log_freq == 0:
                train_elbo.append(elbo_running_mean.avg)
                print('[iteration %03d] time: %.2f \tbeta %.2f \tlambda %.2f training ELBO: %.4f (%.4f)' % (
                    iteration, time.time() - batch_time, vae.beta, vae.lamb,
                    elbo_running_mean.val, elbo_running_mean.avg))

                vae.eval()

                # # plot training and test ELBOs
                # if args.visdom:
                #     display_samples(vae, x, vis)
                #     plot_elbo(train_elbo, vis)

                utils.save_checkpoint({
                    'state_dict': vae.state_dict(),
                    'args': args}, args.save, 0)
                #eval('plot_vs_gt_' + args.dataset)(vae, train_loader.dataset, os.path.join(args.save, 'gt_vs_latent_{:05d}.png'.format(iteration)))

    # Report statistics after training
    vae.eval()
    utils.save_checkpoint({
        'state_dict': vae.state_dict(),
        'args': args}, args.save, 0)
    dataset_loader = DataLoader(train_loader.dataset, batch_size=1000, num_workers=1, shuffle=False)
    logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = \
        elbo_decomposition(vae, dataset_loader)
    torch.save({
        'logpx': logpx,
        'dependence': dependence,
        'information': information,
        'dimwise_kl': dimwise_kl,
        'analytical_cond_kl': analytical_cond_kl,
        'marginal_entropies': marginal_entropies,
        'joint_entropy': joint_entropy
    }, os.path.join(args.save, 'elbo_decomposition.pth'))

    #eval('plot_vs_gt_' + args.dataset)(vae, dataset_loader.dataset, os.path.join(args.save, 'gt_vs_latent.png'))
    return vae
