import torch
import torch.nn as nn
import numpy as np
import os
import pickle as cp
from augmentations import gen_aug
from utils import tsne, mds, _logger
import time
from models.frameworks import *
from models.backbones import *
from models.loss import *
from new_augmentations import *
from vae_quant import setup_the_VAE, VAE
from vae_IDAA import *
from plot_latent_vs_true import * 
from sklearn.metrics import roc_auc_score
from data_preprocess import data_preprocess_ucihar
from data_preprocess import data_preprocess_shar
from data_preprocess import data_preprocess_hhar
from data_preprocess import data_preprocess_usc
from data_preprocess import data_preprocess_ieee_small
from data_preprocess import data_preprocess_ieee_big
from data_preprocess import data_preprocess_dalia
from data_preprocess import data_preprocess_ecg

from sklearn.metrics import f1_score
from scipy.special import softmax
import seaborn as sns
import fitlog
from copy import deepcopy

# create directory for saving models and plots
global model_dir_name
model_dir_name = 'results'
if not os.path.exists(model_dir_name):
    os.makedirs(model_dir_name)
global plot_dir_name
plot_dir_name = 'plot'
if not os.path.exists(plot_dir_name):
    os.makedirs(plot_dir_name)


def setup_dataloaders(args):
    if args.dataset == 'ucihar':
        args.n_feature = 9
        args.len_sw = 128
        args.n_class = 6
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain == '0'
        train_loaders, val_loader, test_loader = data_preprocess_ucihar.prep_ucihar(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int( args.len_sw * 0.5))
        # train_loaders[0].dataset.samples.shape --> To check shape
    if args.dataset == 'usc':
        args.n_feature = 6
        args.len_sw = 100
        args.n_class = 12
        train_loaders, val_loader, test_loader = data_preprocess_usc.prep_usc(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int( args.len_sw * 0.5))
    if args.dataset == 'shar':
        args.n_feature = 3
        args.len_sw = 151
        args.n_class = 17
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain == '1'
        train_loaders, val_loader, test_loader = data_preprocess_shar.prep_shar(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int(args.len_sw * 0.5))
    if args.dataset == 'ieee_small':
        args.n_feature = 1
        args.len_sw = 200
        args.n_class = 180 # 30 -- 210 bpm
        train_loaders, val_loader, test_loader = data_preprocess_ieee_small.prep_ieeesmall(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int( args.len_sw * 0.5))
    if args.dataset == 'ieee_big':
        args.n_feature = 1
        args.len_sw = 200
        args.n_class = 180 # 30 -- 210 bpm
        train_loaders, val_loader, test_loader = data_preprocess_ieee_big.prep_ieeebig(args)     
    if args.dataset == 'dalia':
        args.n_feature = 1
        args.len_sw = 200
        args.n_class = 180 # 30 -- 210 bpm
        train_loaders, val_loader, test_loader = data_preprocess_dalia.prep_dalia(args)              
    if args.dataset == 'ecg':
        args.n_feature = 4
        args.len_sw = 1000
        n_class = 4 if args.target_domain == '3' else 9
        setattr(args, 'n_class', n_class)
        train_loaders, val_loader, test_loader = data_preprocess_ecg.prep_ecg(args)              
    if args.dataset == 'hhar':
        args.n_feature = 6
        args.len_sw = 100
        args.n_class = 6
        source_domain = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'] if args.cases == 'subject_large' else ['a', 'b', 'c', 'd']
        # source_domain.remove(args.target_domain)
        train_loaders, val_loader, test_loader = data_preprocess_hhar.prep_hhar(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int(args.len_sw * 0.5),
                                                                                device=args.device,
                                                                                train_user=source_domain,
                                                                                test_user=args.target_domain)
    return train_loaders, val_loader, test_loader


def setup_linclf(args, DEVICE, bb_dim):
    '''
    @param bb_dim: output dimension of the backbone network
    @return: a linear classifier
    '''
    classifier = Classifier(bb_dim=bb_dim, n_classes=args.n_class)
    classifier.classifier.weight.data.normal_(mean=0.0, std=0.01)
    classifier.classifier.bias.data.zero_()
    classifier = classifier.to(DEVICE)
    return classifier


def setup_model_optm(args, DEVICE, classifier=True):
    # set up backbone network
    if args.backbone == 'FCN':
        backbone = FCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=True)
    elif args.backbone == 'DCL':
        backbone = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=True)
    elif args.backbone == 'LSTM':
        backbone = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=True)
    elif args.backbone == 'AE':
        backbone = AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128, backbone=True)
    elif args.backbone == 'CNN_AE':
        backbone = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=True)
    elif args.backbone == 'Transformer':
        backbone = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=True)
    else:
        NotImplementedError

    # set up model and optimizers
    if args.framework in ['byol', 'simsiam']:
        model = BYOL(DEVICE, backbone, window_size=args.len_sw, n_channels=args.n_feature, projection_size=args.p,
                     projection_hidden_size=args.phid, moving_average=args.EMA)
        optimizer1 = torch.optim.Adam(model.online_encoder.parameters(),
                                      args.lr,
                                      weight_decay=args.weight_decay)
        optimizer2 = torch.optim.Adam(model.online_predictor.parameters(),
                                      args.lr * args.lr_mul,
                                      weight_decay=args.weight_decay)
        optimizers = [optimizer1, optimizer2]
    elif args.framework == 'simclr':
        model = SimCLR(backbone=backbone, dim=args.p)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizers = [optimizer]
    elif args.framework == 'nnclr':
        model = NNCLR(backbone=backbone, dim=args.p, pred_dim=args.phid)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizers = [optimizer]
    elif args.framework == 'tstcc':
        model = TSTCC(backbone=backbone, DEVICE=DEVICE, temp_unit=args.temp_unit, tc_hidden=100)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        optimizers = [optimizer]

    else:
        NotImplementedError

    model = model.to(DEVICE)

    # set up linear classfier
    if classifier:
        bb_dim = backbone.out_dim
        classifier = setup_linclf(args, DEVICE, bb_dim)
        return model, classifier, optimizers

    else:
        return model, optimizers


def delete_files(args):
    for epoch in range(args.n_epoch):
        model_dir = model_dir_name + '/pretrain_' + args.model_name + str(epoch) + '.pt'
        if os.path.isfile(model_dir):
            os.remove(model_dir)

        cls_dir = model_dir_name + '/lincls_' + args.model_name + str(epoch) + '.pt'
        if os.path.isfile(cls_dir):
            os.remove(cls_dir)


def setup(args, DEVICE):
    # set up default hyper-parameters
    if args.framework == 'byol':
        args.weight_decay = 1.5e-6
    if args.framework == 'simsiam':
        args.weight_decay = 1e-4
        args.EMA = 0.0
        args.lr_mul = 1.0
    if args.framework in ['simclr', 'nnclr']:
        args.criterion = 'NTXent'
        args.weight_decay = 1e-6
    if args.framework == 'tstcc':
        args.criterion = 'NTXent'
        args.backbone = 'FCN'
        args.weight_decay = 3e-4

    model, classifier, optimizers = setup_model_optm(args, DEVICE, classifier=True)

    # loss fn
    if args.criterion == 'cos_sim':
        criterion = nn.CosineSimilarity(dim=1)
    elif args.criterion == 'NTXent':
        if args.framework == 'tstcc':
            criterion = NTXentLoss(DEVICE, args.batch_size, temperature=0.2)
        else:
            criterion = NTXentLoss(DEVICE, args.batch_size, temperature=0.1)

    args.model_name = 'try_scheduler_' + args.framework + '_pretrain_' + args.dataset + '_eps' + str(args.n_epoch) + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) \
                      + '_aug1' + args.aug1 + '_aug2' + args.aug2 + '_dim-pdim' + str(args.p) + '-' + str(args.phid) \
                      + '_EMA' + str(args.EMA) + '_criterion_' + args.criterion + '_lambda1_' + str(args.lambda1) + '_lambda2_' + str(args.lambda2) + '_tempunit_' + args.temp_unit

    # log
    if os.path.isdir(args.logdir) == False:
        os.makedirs(args.logdir)
    log_file_name = os.path.join(args.logdir, args.model_name + f".log")
    logger = _logger(log_file_name)
    #logger.debug(args)

    # fitlog
    fitlog.set_log_dir(args.logdir)
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)

    criterion_cls = nn.CrossEntropyLoss()
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=args.lr_cls)

    schedulers = []
    for optimizer in optimizers:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=0)
        schedulers.append(scheduler)

    global nn_replacer
    nn_replacer = None
    if args.framework == 'nnclr':
        nn_replacer = NNMemoryBankModule(size=args.mmb_size)

    global recon
    recon = None
    if args.backbone in ['AE', 'CNN_AE']:
        recon = nn.MSELoss()

    return model, optimizers, schedulers, criterion, logger, fitlog, classifier, criterion_cls, optimizer_cls

def load_vae(args, DEVICE):
    prior_dist, q_dist = setup_the_VAE(args)
    vae = VAE(z_dim=args.latent_dim, args=args, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist,
    include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, mss=args.mss).to(DEVICE)
    vae_model = torch.load(args.save+'/checkpt-0000.pth')
    vae.load_state_dict(vae_model['state_dict'])
    vae.eval()
    return vae

def gen_adv(model, vae, x_i, criterion, args , DEVICE):
    eps = 0.20
    vae.train()
    x_i = x_i.to(DEVICE).float()
    _, z_i = model(x_i, x_i)
    z_i = Variable(z_i.data, requires_grad=True)
    with torch.no_grad():
        z, gx, _, _ = vae(x_i)
    variable_bottle = Variable(z.detach(), requires_grad=True)
    adv_gx = vae(variable_bottle, decode=True)
    adv_gx = adv_gx if len(adv_gx.shape) == 3 else torch.unsqueeze(adv_gx,2)

    x_j_adv = adv_gx + (x_i - gx).detach()
    _, z_j_adv = model(x_j_adv, x_j_adv)
    tmp_loss = criterion(z_i, z_j_adv)

    tmp_loss.backward()

    with torch.no_grad():
        sign_grad = variable_bottle.grad.data.sign()
        variable_bottle.data = variable_bottle.data + eps * sign_grad
        adv_gx = vae(variable_bottle, True)
        adv_gx = adv_gx if len(adv_gx.shape) == 3 else torch.unsqueeze(adv_gx,2)
        x_j_adv = adv_gx + (x_i - gx).detach()
    x_j_adv.detach()
    x_j_adv.requires_grad = False
    return x_j_adv

def add_gauss_latent(sample, args, DEVICE):
    vae = load_vae(args, DEVICE)
    sample = sample.to(DEVICE).float()
    zs, z_params = vae.encode(sample)
    noise_tensor = torch.zeros(zs.shape)
    noise_tensor = nn.init.trunc_normal_(noise_tensor, mean=0.0, std=0.1, a=-2.0, b=2.0).to(DEVICE) # Get values from truncated Normal distribution
    zs_added = zs + noise_tensor
    xs, _ = vae.decode(zs_added)
    #xs, x_params, zs, z_params = vae.reconstruct_img(sample)
    return torch.squeeze(xs,1)

def interpolate_in_latent(sample, args, inds, similarities, DEVICE):
    chosen_index = np.random.binomial(1,0.7)
    vae, filename = vae_idaa(z_dim=args.latent_dim, dataset=args.dataset).to(DEVICE), os.path.join(args.save, 'vae_idaa.pth')
    vae.load_state_dict(torch.load(filename))

    sample = sample.to(DEVICE).float()
    #zs, z_params = vae.encode(sample)
    zs, gx, _, _ = vae(sample)
    index = torch.randperm(sample.size(0))
    mixed_zs = torch.empty(zs.shape, dtype=torch.float32)
    #mixing_coeff = ((0.7 - 1) * torch.rand(1) + 1).to(DEVICE)
    mixing_coeff = mixing_coefficient_set_for_each(similarities)
    index = torch.randperm(sample.size(0))
    index0 = torch.arange(256)

    mixing_coeff = mixing_coeff[index,index0].to(DEVICE)
    mixed_zs = zs * mixing_coeff[:, None] + (1 - mixing_coeff[:,None]) * zs[index]
    #xs, _ = vae.decode(mixed_zs.to(DEVICE))
    xs = vae(mixed_zs, True)
    xs = xs if len(xs.shape) == 3 else torch.unsqueeze(xs,2)
    return xs

def calculate_similarity_latents(args, sample, DEVICE):
    vae = load_vae(args, DEVICE)
    qz_params = vae.encoder.forward(sample.to(DEVICE).float()).view(sample.size(0), args.latent_dim, vae.q_dist.nparams).data
    latent_values = vae.q_dist.sample(params=qz_params)
    a_norm = latent_values / latent_values.norm(dim=1)[:, None]
    b_norm = latent_values / latent_values.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0,1))
    res = res.fill_diagonal_(0) # Make diagonals to 0
    return res

def calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=None, nn_replacer=None, view_learner=None):
    if args.VAE:
        similarities = calculate_similarity_latents(args, sample, DEVICE)
        out1, inds1 = torch.topk(similarities,3)
        out, inds = torch.max(similarities,dim=1)
        aug_sample1 = gen_aug(sample, args.aug1)
        aug_sample2 = gen_new_aug_2(gen_aug(sample, args.aug2), args, inds, out, DEVICE, similarities)
    elif args.VanillaMixup: 
        aug_sample1, aug_sample2 = gen_aug(sample, args.aug1), vanilla_mix_up(gen_aug(sample, args.aug2))
    elif args.BestMixup:
        similarities = calculate_similarity_latents(args, sample, DEVICE)
        aug_sample1, aug_sample2 = gen_aug(sample, args.aug1), best_mix_up(sample, args, similarities, DEVICE)
    elif args.GeoMix:
        aug_sample1, aug_sample2 = sample, vanilla_mix_up_geo(sample)        
    elif args.DACL: 
        aug_sample1, aug_sample2 = gen_aug(sample, args.aug1), gen_aug(sample, args.aug2)
    elif args.IDAA:
        vae, filename = vae_idaa(z_dim=args.latent_dim, dataset=args.dataset).to(DEVICE), os.path.join(args.save, 'vae_idaa.pth')
        vae.load_state_dict(torch.load(filename))
        aug_sample1 = gen_aug(sample, args.aug1)
        aug_sample2 = sample
        aug_sample_adv = gen_adv(model, vae, gen_aug(sample, args.aug2), criterion, args, DEVICE) if model.training else aug_sample2
    elif args.GaussLatent:
        aug_sample1, aug_sample2 = gen_aug(sample, args.aug1), gen_aug(add_gauss_latent(sample, args, DEVICE).cpu(), args.aug2)
    elif args.dim_mixing:
        aug_sample1, aug_sample2 = gen_aug(sample, args.aug1), gen_aug(sample, args.aug2) 
    elif args.InfoMin:
        aug_sample1 = torch.squeeze(view_learner(torch.unsqueeze(sample,1).to(DEVICE).float()),1)
        aug_sample2 = torch.squeeze(view_learner(torch.unsqueeze(gen_aug(sample, 'noise'),1).to(DEVICE).float()),1)        
    elif args.Randomfftmix:
        aug_sample1, aug_sample2 = gen_aug(sample, args.aug1), gen_new_aug(gen_aug(sample, args.aug2), args, DEVICE)  
    elif args.ablation_2:
        similarities = calculate_similarity_latents(args, sample, DEVICE)  
        aug_sample1, aug_sample2 = gen_aug(sample, args.aug1), gen_new_aug_3_ablation(gen_aug(sample, args.aug2), args, DEVICE, similarities)
    elif args.opposite_phase:
        similarities = calculate_similarity_latents(args, sample, DEVICE)  
        aug_sample1, aug_sample2 = gen_aug(sample, args.aug1), opposite_phase(gen_aug(sample, args.aug2), args, DEVICE, similarities)
    elif args.ablation_tfc:
        aug_sample1, aug_sample2 = gen_aug(sample, args.aug1), gen_new_aug_4_comparison(gen_aug(sample, args.aug2), args, DEVICE)  
    elif args.STAug:
        aug_sample1, aug_sample2 = gen_aug(sample, args.aug1), vanilla_mix_up(STAug(gen_aug(sample, args.aug2), args, DEVICE))
    elif args.CutMix:
        aug_sample1, aug_sample2 = gen_aug(sample, args.aug1), cut_mix(gen_aug(sample, args.aug2), alpha=1)  
    elif args.SpecMix:
        aug_sample1, aug_sample2 = gen_aug(sample, args.aug1), spec_mix(gen_aug(sample, args.aug2))
    elif args.BinaryMix:
        aug_sample1, aug_sample2 = gen_aug(sample, args.aug1), vanilla_mix_up_binary(gen_aug(sample, args.aug2))
    elif args.BestGeoMix:
        aug_sample1, aug_sample2 = gen_aug(sample, args.aug1), best_mix_up_geo(gen_aug(sample, args.aug2), alpha=1) 
    else:
        aug_sample1 = gen_aug(sample, args.aug1) # Shape --> (Batch_size, number of inputs, Channel size)
        aug_sample2 = gen_aug(sample, args.aug2)

    aug_sample1, aug_sample2, target = aug_sample1.to(DEVICE).float(), aug_sample2.to(DEVICE).float(), target.to(
        DEVICE).long()
    if args.framework in ['byol', 'simsiam']:
        assert args.criterion == 'cos_sim'
    if args.framework in ['tstcc', 'simclr', 'nnclr']:
        assert args.criterion == 'NTXent'
    if args.framework in ['byol', 'simsiam', 'nnclr']:
        if args.backbone in ['AE', 'CNN_AE']:
            x1_encoded, x2_encoded, p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            recon_loss = recon(aug_sample1, x1_encoded) + recon(aug_sample2, x2_encoded)
        elif args.DACL:
            p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2, DACL_training=args.DACL)
        else:
            p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
        if args.framework == 'nnclr':
            z1 = nn_replacer(z1, update=False)
            z2 = nn_replacer(z2, update=True)
        if args.criterion == 'cos_sim':
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        elif args.criterion == 'NTXent':
            loss = (criterion(p1, z2) + criterion(p2, z1)) * 0.5
        if args.backbone in ['AE', 'CNN_AE']:
            loss = loss * args.lambda1 + recon_loss * args.lambda2
    if args.framework == 'simclr':
        if args.backbone in ['AE', 'CNN_AE']:
            x1_encoded, x2_encoded, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            recon_loss = recon(aug_sample1, x1_encoded) + recon(aug_sample2, x2_encoded)
        elif args.DACL: # Mixing representations (h) via intermediate layers
            z1, z2 = model(x1=aug_sample1, x2=aug_sample2, DACL_training=args.DACL)
        else:
            z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
        if args.dim_mixing: # Feature extrapolation
            loss = criterion(z1, z2, args.dim_mixing)
        elif args.IDAA:
            aug_sample_adv = aug_sample_adv.to(DEVICE).float()
            loss_og = criterion(z1, z2) 
            z1, z2_adv = model(x1=aug_sample1, x2=aug_sample_adv)
            loss_adv = criterion(z1, z2_adv) 
            loss = loss = loss_og + args.alpha * loss_adv     
        else:
            loss = criterion(z1, z2) 

        if args.backbone in ['AE', 'CNN_AE']:
            loss = loss * args.lambda1 + recon_loss * args.lambda2
    if args.framework == 'tstcc':
        nce1, nce2, p1, p2 = model(x1=aug_sample1, x2=aug_sample2, DACL_training=args.DACL)
        tmp_loss = nce1 + nce2
        ctx_loss = criterion(p1, p2)
        loss = tmp_loss * args.lambda1 + ctx_loss * args.lambda2
    return loss


def train(train_loaders, val_loader, model, logger, fitlog, DEVICE, optimizers, schedulers, criterion, args):
    best_model = None
    if args.IDAA:
        filename = os.path.join(args.save, 'vae_idaa.pth')
        vae = vae_idaa(z_dim=args.latent_dim, dataset=args.dataset).to(DEVICE)
        if not os.path.exists(filename):
            vae_model = train_VAE_idaa(train_loaders, args, DEVICE)
        else: 
            vae.load_state_dict(torch.load(filename))
     
    if args.COPGEN:
        nn_walker = LinearWalk(args.latent_dim).to(DEVICE)
        optimizer_walk = optim.Adam(nn_walker.parameters(), lr=0.00001,
                            betas=(0.5, 0.999))
        vae = load_vae(args, DEVICE)

    if args.InfoMin:
        filename = os.path.join(args.save, 'view_learner.pth')
        view_learner = ViewLearner(dataset=args.dataset).to(DEVICE)
        optimizer_view_learner = optim.Adam(view_learner.parameters(), lr=0.001)        

    if args.plot_tsne:
        vae = load_vae(args, DEVICE)
        save_file = 'train' + str(args.dataset) + str(args.target_domain)
        plot_vs_gt_ucihar(vae, args, train_loaders, DEVICE, save_file, z_inds=None)

    min_val_loss = 1e8
    for epoch in range(args.n_epoch):
        #logger.debug(f'\nEpoch : {epoch}')
        total_loss = 0
        n_batches = 0
        model.train()
        for i, train_loader in enumerate(train_loaders):
            for idx, (sample, target, domain) in enumerate(train_loader):
                for optimizer in optimizers:
                    optimizer.zero_grad()
                if sample.size(0) != args.batch_size:
                    continue
                n_batches += 1
                if args.COPGEN:
                    z, _ = vae.encode(sample.to(DEVICE).float())
                    z_new = z = nn_walker(z)
                    img_new, _ = vae.decode(z_new)
                    sample = torch.squeeze(img_new)

                if args.InfoMin:
                    loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer, view_learner=view_learner)
                else:
                    loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)

                total_loss += loss.item()
                # if args.InfoMin:
                #     loss.backward(retain_graph=True)
                # else:
                #     loss.backward() 
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                if args.framework in ['byol', 'simsiam']:
                    model.update_moving_average()
                if args.COPGEN:
                    optimizer_walk.zero_grad()
                    loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
                    loss_walker = - loss
                    loss_walker.backward()
                    optimizer_walk.step()
                if args.InfoMin:
                    optimizer_view_learner.zero_grad()
                    loss_view = - calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer, view_learner=view_learner)
                    loss_view.backward()
                    optimizer_view_learner.step()                    

        fitlog.add_loss(optimizers[0].param_groups[0]['lr'], name="learning rate", step=epoch)
        for scheduler in schedulers:
            scheduler.step()

        # save model
        model_dir = model_dir_name + '/pretrain_' + args.model_name + str(epoch) + '.pt'
        #print('Saving model at {} epoch to {}'.format(epoch, model_dir))
        torch.save({'model_state_dict': model.state_dict()}, model_dir)
        #logger.debug(f'Train Loss     : {total_loss / n_batches:.4f}')
        fitlog.add_loss(total_loss / n_batches, name="pretrain training loss", step=epoch)

        if args.cases in ['subject', 'subject_large']:
            with torch.no_grad():
                best_model = copy.deepcopy(model.state_dict())
        else:
            with torch.no_grad():
                model.eval()
                total_loss = 0
                n_batches = 0
                for idx, (sample, target, domain) in enumerate(val_loader):
                    if sample.size(0) != args.batch_size:
                        continue
                    n_batches += 1
                    loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
                    total_loss += loss.item()
                if total_loss <= min_val_loss:
                    min_val_loss = total_loss
                    best_model = copy.deepcopy(model.state_dict())
                logger.debug(f'Val Loss     : {total_loss / n_batches:.4f}')
                fitlog.add_loss(total_loss / n_batches, name="pretrain validation loss", step=epoch)
    return best_model


def test(test_loader, best_model, logger, fitlog, DEVICE, criterion, args): # Test the pre-trained model without fine-tuning on the downstream task
    model, _ = setup_model_optm(args, DEVICE, classifier=False)
    model.load_state_dict(best_model)
    # with torch.no_grad():
    #     model.eval()
    #     total_loss = 0
    #     n_batches = 0
    #     for idx, (sample, target, domain) in enumerate(test_loader):
    #         # import pdb;pdb.set_trace();
    #         # if sample.size(0) != args.batch_size:
    #         #     continue
    #         # n_batches += 1
    #         loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
    #         total_loss += loss.item()
    #     logger.debug(f'Test Loss     : {total_loss:.4f}')
    #     fitlog.add_best_metric({"dev": {"pretrain test loss": total_loss}})

    return model


def lock_backbone(model, args):
    for name, param in model.named_parameters():
        param.requires_grad = False

    if args.framework in ['simsiam', 'byol']:
        trained_backbone = model.online_encoder.net
    elif args.framework in ['simclr', 'nnclr', 'tstcc']:
        trained_backbone = model.encoder
    else:
        NotImplementedError

    return trained_backbone


def calculate_lincls_output(sample, target, trained_backbone, classifier, criterion):
    _, feat = trained_backbone(sample)
    if len(feat.shape) == 3:
        feat = feat.reshape(feat.shape[0], -1)
    output = classifier(feat)
    # import pdb;pdb.set_trace(); -> Check if classifier has only one linear layer.
    loss = criterion(output, target)
    _, predicted = torch.max(output.data, 1)
    return loss, predicted, feat, output


def train_lincls(train_loaders, val_loader, trained_backbone, classifier, logger, fitlog, DEVICE, optimizer, criterion, args):
    best_lincls = None
    min_val_loss = 1e8
    
    # if args.plot_tsne:
    #     import pdb;pdb.set_trace();
    #     plot_vs_gt_usc(vae, train_loaders.dataset, 'train', z_inds=None)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=0)

    for epoch in range(args.n_epoch):
        classifier.train()
        #logger.debug(f'\nEpoch : {epoch}')
        total_loss = 0
        total = 0
        correct = 0
        for i, train_loader in enumerate(train_loaders):
            for idx, (sample, target, domain) in enumerate(train_loader):
                #import pdb;pdb.set_trace();
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                loss, predicted, _ , _= calculate_lincls_output(sample, target, trained_backbone, classifier, criterion)
                total_loss += loss.item()
                total += target.size(0)
                correct += (predicted == target).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # save model
        model_dir = model_dir_name + '/lincls_' + args.model_name + str(epoch) + '.pt'
        torch.save({'trained_backbone': trained_backbone.state_dict(), 'classifier': classifier.state_dict()}, model_dir)

        if args.scheduler:
            scheduler.step()

        if args.cases in ['subject', 'subject_large']:
            with torch.no_grad():
                best_lincls = copy.deepcopy(classifier.state_dict())
        else:
            with torch.no_grad():
                classifier.eval()
                total_loss = 0
                total = 0
                correct = 0
                for idx, (sample, target, domain) in enumerate(val_loader):
                    sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                    loss, predicted, _ , _ = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion)
                    total_loss += loss.item()
                    total += target.size(0)
                    correct += (predicted == target).sum()
                acc_val = float(correct) * 100.0 / total
                if total_loss <= min_val_loss:
                    min_val_loss = total_loss
                    best_lincls = copy.deepcopy(classifier.state_dict())
    return best_lincls


def test_lincls(test_loader, trained_backbone, best_lincls, logger, fitlog, DEVICE, criterion, args, plt=False):  # Test the fine-tuned model
    classifier = setup_linclf(args, DEVICE, trained_backbone.out_dim)
    classifier.load_state_dict(best_lincls)
    total_loss = 0
    total = 0
    correct = 0
    confusion_matrix = torch.zeros(args.n_class, args.n_class)
    feats = None
    trgs = np.array([])
    preds = np.array([])
    otp = np.array([])
    with torch.no_grad():
        classifier.eval()
        for idx, (sample, target, domain) in enumerate(test_loader):
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            loss, predicted, feat, output = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion)
            total_loss += loss.item()
            if feats is None:
                feats = feat
            else:
                feats = torch.cat((feats, feat), 0)
            trgs = np.append(trgs, target.data.cpu().numpy())
            preds = np.append(preds, predicted.data.cpu().numpy())
            otp = np.vstack((otp, output.data.cpu().numpy())) if otp.size != 0 else output.data.cpu().numpy()
            for t, p in zip(target.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            total += target.size(0)
            correct += (predicted == target).sum()
        acc_test = float(correct) * 100.0 / total

        miF = f1_score(trgs, preds, average='micro') * 100
        maF = f1_score(trgs, preds, average='weighted') * 100

        if args.dataset == 'ieee_small' or args.dataset =='ieee_big' or args.dataset == 'dalia':
            acc_test = np.sqrt(np.mean((trgs-preds)**2))
            maF = np.mean(np.abs(trgs-preds))
        if args.dataset == 'ecg':
            otp1 =  softmax(otp,axis=1)
            maF = roc_auc_score(trgs, otp1, multi_class='ovo')

        print(f'epoch test loss     : {total_loss:.4f}, test acc     : {acc_test:.4f}, miF     : {miF:.4f}, maF     : {maF:.4f}')

        fitlog.add_best_metric({"dev": {"Test Loss": total_loss}})
        fitlog.add_best_metric({"dev": {"Test Acc": acc_test}})
        fitlog.add_best_metric({"dev": {"miF": miF}})
        fitlog.add_best_metric({"dev": {"maF": maF}})
    
    if plt == True:
        tsne(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_tsne.png')
        mds(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_mds.png')
        sns_plot = sns.heatmap(confusion_matrix, cmap='Blues', annot=True)
        sns_plot.get_figure().savefig(plot_dir_name + '/' + args.model_name + '_confmatrix.png')
        print('plots saved to ', plot_dir_name)

    return acc_test,maF