# encoding=utf-8
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix
from models.backbones import *
from models.loss import *
from trainer import *
import torch
import torch.nn as nn
import argparse
from datetime import datetime
import pickle
import numpy as np
import os
import logging
import sys
from data_preprocess.data_preprocess_utils import normalize
from vae_quant import setup_the_VAE, VAE
from scipy import signal
from copy import deepcopy
import fitlog
from utils import tsne, mds, _logger
from new_augmentations import vanilla_mixup_sup, cutmix_sup
# fitlog.debug()


parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--cuda', default=0, type=int, help='cuda device ID, 0/1')
# hyperparameter
parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=60, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_cls', type=float, default=1e-3, help='learning rate for linear classifier')
parser.add_argument('--VAE', action='store_true')
parser.add_argument('--VanillaMixup', action='store_true')
parser.add_argument('--BinaryMix', action='store_true')
parser.add_argument('--Cutmix', action='store_true')
parser.add_argument('--Magmix', action='store_true')
# dataset
parser.add_argument('--dataset', type=str, default='ucihar', choices=['oppor', 'ucihar', 'shar', 'hhar', 'usc', 'ieee_small', 'ieee_big', 'dalia', 'ecg'], help='name of dataset')
parser.add_argument('--n_feature', type=int, default=77, help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=30, help='length of sliding window')
parser.add_argument('--n_class', type=int, default=18, help='number of class')
parser.add_argument('--cases', type=str, default='subject_val', choices=['random', 'subject', 'subject_large', 'cross_device', 'joint_device'], help='name of scenarios')
parser.add_argument('--split_ratio', type=float, default=0.2, help='split ratio of test/val: train(0.64), val(0.16), test(0.2)')
parser.add_argument('--target_domain', type=str, default='0', help='the target domain, [0 to 29] for ucihar, '
                                                                   '[1,2,3,5,6,9,11,13,14,15,16,17,19,20,21,22,23,24,25,29] for shar, '
                                                                   '[a-i] for hhar')

# backbone model
parser.add_argument('--backbone', type=str, default='DCL', choices=['FCN', 'DCL', 'LSTM', 'AE', 'CNN_AE', 'Transformer'], help='name of framework')

# log
parser.add_argument('--logdir', type=str, default='log/', help='log directory')

# AE & CNN_AE
parser.add_argument('--lambda1', type=float, default=1.0, help='weight for reconstruction loss when backbone in [AE, CNN_AE]')

# python main_supervised_baseline.py --dataset 'ecg' --cuda 1

# hhar
parser.add_argument('--device', type=str, default='Phones', choices=['Phones', 'Watch'], help='data of which device to use (random case); data of which device to be used as training data (cross-device case, data from the other device as test data)')


# VAE
parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
parser.add_argument('-n', '--num-epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('-b', '--batch-size', default=2048, type=int, help='batch size')
parser.add_argument('-l', '--learning-rate', default=1e-3, type=float, help='learning rate')
parser.add_argument('-z', '--latent_dim', default=10, type=int, help='size of latent dimension')
parser.add_argument('--beta', default=5, type=float, help='ELBO penalty term')
parser.add_argument('--tcvae', action='store_true')
parser.add_argument('--exclude-mutinfo', action='store_true')
parser.add_argument('--beta-anneal', action='store_true')
parser.add_argument('--lambda-anneal', action='store_true')
parser.add_argument('--mss', action='store_true', help='use the improved minibatch estimator')
parser.add_argument('--conv', action='store_true')
# parser.add_argument('--visdom', action='store_true', help='whether plotting in visdom is desired')
parser.add_argument('--save', type=str, default='test3')
parser.add_argument('--log_freq', default=200, type=int, help='num iterations per log')


parser.add_argument('--mean', type=float, default=1, help='Mean of Gaussian')
parser.add_argument('--std', type=float, default=0.1, help='std of Gaussian')
parser.add_argument('--low_limit', type=float, default=0.7, help='low limit of Gaussian')
parser.add_argument('--high_limit', type=float, default=1, help='high limit of Gaussian')
parser.add_argument('--alpha', default=0.2, type=float, help='beta term')

############### Parser done ################


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    if args.VAE:
        loss = (lam * nn.CrossEntropyLoss(reduction='none')(pred, y_a) + (1 - lam) * nn.CrossEntropyLoss(reduction='none')(pred, y_b)).sum()/y_a.size(0)
    else:
        loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    return loss

def train(args, train_loaders, val_loader, model, DEVICE, optimizer, criterion, save_dir='results/'):
    min_val_loss = 1e8
    for epoch in range(args.n_epoch):
        #logger.debug(f'\nEpoch : {epoch}')

        train_loss = 0
        n_batches = 0
        total = 0
        correct = 0
        model.train()
        for loader_idx, train_loader in enumerate(train_loaders):
            for idx, (sample, target, domain) in enumerate(train_loader):
                if args.VAE:
                    sample, target, lam, target_2 = gen_new_aug_3_ablation_sup(sample, args, DEVICE, target, alpha=args.alpha)
                elif args.VanillaMixup: 
                    sample, target, lam, target_2 = vanilla_mixup_sup(sample, target, alpha=args.alpha)
                elif args.Cutmix:
                    sample, target, lam, target_2 = cutmix_sup(sample, target, alpha=args.alpha)
                elif args.BinaryMix:
                    sample, target, lam, target_2 = binary_mixup_sup(sample, target, alpha=args.alpha)
                elif args.Magmix:
                    sample, target, lam, target_2 = mag_mixup_sup(sample, args, DEVICE, target, alpha=args.alpha)

                n_batches += 1
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                if args.backbone[-2:] == 'AE':
                    out, x_decoded = model(sample)
                else:
                    out, _ = model(sample)
                loss = mixup_criterion(criterion, out, target, target_2.to(DEVICE).long(), lam.to(DEVICE).float()) if args.VAE or args.VanillaMixup or args.Cutmix else criterion(out, target)
                if args.backbone[-2:] == 'AE':
                    # print(loss.item(), nn.MSELoss()(sample, x_decoded).item())
                    loss += nn.MSELoss()(sample, x_decoded) * args.lambda1
                
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # _, predicted = torch.max(out.data, 1)
                # total += target.size(0)
                # correct += (predicted == target).sum()
        #acc_train = float(correct) * 100.0 / total
        # fitlog.add_loss(train_loss / n_batches, name="Train Loss", step=epoch)
        # fitlog.add_metric({"dev": {"Train Acc": acc_train}}, step=epoch)
        # logger.debug(f'Train Loss     : {train_loss / n_batches:.4f}\t | \tTrain Accuracy     : {acc_train:2.4f}\n')

        if val_loader is None:
            best_model = deepcopy(model.state_dict())
            model_dir = save_dir + args.model_name + '.pt'
            # print('Saving models at {} epoch to {}'.format(epoch, model_dir))
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_dir)
        else:
            with torch.no_grad():
                model.eval()
                val_loss = 0
                n_batches = 0
                total = 0
                correct = 0
                for idx, (sample, target, domain) in enumerate(val_loader):
                    n_batches += 1
                    sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                    if args.backbone[-2:] == 'AE':
                        out, x_decoded = model(sample)
                    else:
                        out, _ = model(sample)
                    loss = criterion(out, target)
                    if args.backbone[-2:] == 'AE':
                        loss += nn.MSELoss()(sample, x_decoded) * args.lambda1
                    val_loss += loss.item()
                    _, predicted = torch.max(out.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum()
                acc_val = float(correct) * 100.0 / total
                # fitlog.add_loss(val_loss / n_batches, name="Val Loss", step=epoch)
                # fitlog.add_metric({"dev": {"Val Acc": acc_val}}, step=epoch)
                # logger.debug(f'Val Loss     : {val_loss / n_batches:.4f}\t | \tVal Accuracy     : {acc_val:2.4f}\n')
                # import pdb;pdb.set_trace();
                if val_loss <= min_val_loss:
                    min_val_loss = val_loss
                    best_model = deepcopy(model.state_dict())
                    # print('update')
                    model_dir = save_dir + args.model_name + '.pt'
                    # print('Saving models at {} epoch to {}'.format(epoch, model_dir))
                    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_dir)

    return best_model

def test(test_loader, model, DEVICE, criterion, plt=False):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        n_batches = 0
        total = 0
        correct = 0
        feats = None
        prds = None
        trgs = None
        otp = np.array([])
        confusion_matrix = torch.zeros(args.n_class, args.n_class)
        for idx, (sample, target, domain) in enumerate(test_loader):
            n_batches += 1
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            out, features = model(sample)
            loss = criterion(out, target)
            total_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()
            otp = np.vstack((otp, out.data.cpu().numpy())) if otp.size != 0 else out.data.cpu().numpy()
            if prds is None:
                prds = predicted
                trgs = target
                feats = features[:, :]
            else:
                prds = torch.cat((prds, predicted))
                trgs = torch.cat((trgs, target))
                feats = torch.cat((feats, features), 0)

        acc_test = float(correct) * 100.0 / total
        maF = f1_score(trgs.cpu().numpy(), prds.cpu().numpy(), average='weighted') * 100
        
    #logger.debug(f'Test Loss     : {total_loss / n_batches:.4f}\t | \tTest Accuracy     : {acc_test:2.4f}\n')
    #for t, p in zip(trgs.view(-1), prds.view(-1)):
        #confusion_matrix[t.long(), p.long()] += 1
    #logger.debug(confusion_matrix)
    #logger.debug(confusion_matrix.diag() / confusion_matrix.sum(1))
    #fitlog.add_hyper(confusion_matrix, name='conf_mat')
    if args.dataset == 'ieee_small' or args.dataset =='ieee_big' or args.dataset == 'dalia':
        acc_test = np.sqrt(torch.mean(((trgs-prds)**2).float()).cpu())
        maF = torch.mean((torch.abs(trgs-prds)).float()).cpu()
        print(f'RMSE : {acc_test:.4f}, MSE acc     : {maF:.4f}')
    elif args.dataset == 'ecg':
        otp1 =  softmax(otp,axis=1)
        maF = roc_auc_score(trgs.cpu(), otp1, multi_class='ovo')
        print(f'MAF : {maF:.4f}')
    if plt == True:
        tsne(feats, trgs, domain=None, save_dir=plot_dir_name + args.model_name + '_tsne.png')
        mds(feats, trgs, domain=None, save_dir=plot_dir_name + args.model_name + 'mds.png')
        sns_plot = sns.heatmap(confusion_matrix, cmap='Blues', annot=True)
        sns_plot.get_figure().savefig(plot_dir_name + args.model_name + '_confmatrix.png')
    return acc_test, maF

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_sup(args):
    set_seed(10)
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('device:', DEVICE, 'dataset:', args.dataset)

    train_loaders, val_loader, test_loader = setup_dataloaders(args)

    if args.backbone == 'FCN':
        model = FCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=False)
    elif args.backbone == 'DCL':
        model = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=False)
    elif args.backbone == 'LSTM':
        model = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=False)
    elif args.backbone == 'AE':
        model = AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128, backbone=False)
    elif args.backbone == 'CNN_AE':
        model = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=False)
    elif args.backbone == 'Transformer':
        model = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)
    else:
        NotImplementedError

    model = model.to(DEVICE)

    args.model_name = args.backbone + '_'+args.dataset + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) + '_sw' + str(args.len_sw)

    save_dir = 'results/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # log
    if os.path.isdir(args.logdir) == False:
        os.makedirs(args.logdir)
    log_file_name = os.path.join(args.logdir, args.model_name + f".log")
    #logger = _logger(log_file_name)
    #logger.debug(args)

    # fitlog
    fitlog.set_log_dir(args.logdir)
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)

    criterion = nn.CrossEntropyLoss()

    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, args.lr)

    best_model = train(args, train_loaders, val_loader, model, DEVICE, optimizer, criterion)

    if args.backbone == 'FCN':
        model_test = FCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=False)
    elif args.backbone == 'DCL':
        model_test = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=False)
    elif args.backbone == 'LSTM':
        model_test = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=False)
    elif args.backbone == 'AE':
        model_test = AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128, backbone=False)
    elif args.backbone == 'CNN_AE':
        model_test = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=False)
    elif args.backbone == 'Transformer':
        model_test = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)
    else:
        NotImplementedError

    model_test.load_state_dict(best_model)
    model_test = model_test.to(DEVICE)
    acc,mf1 = test(test_loader, model_test, DEVICE, criterion, plt=False)
    return acc,mf1
    #training_end = datetime.now()
    #training_time = training_end - training_start
    #logger.debug(f"Training time is : {training_time}")


def set_domains(args):
    args = parser.parse_args()
    if args.dataset == 'shar':
        domain = [1, 2, 3, 5]
    elif args.dataset == 'ucihar':
        domain = [0, 1, 2, 3, 4]
    elif args.dataset == 'usc':
        domain = [10, 11, 12, 13]
    elif args.dataset == 'ieee_small':
        domain = [0, 1, 2, 3, 4]
    elif args.dataset == 'ieee_big':
        domain = [17, 18, 19, 20, 21]
    elif args.dataset == 'dalia':
        domain = [0, 1, 2, 3, 4]     
    elif args.dataset == 'ecg':
        domain = [1, 3]
    elif args.dataset == 'hhar':
        domain = ['a', 'b', 'c', 'd']
    return domain

if __name__ == '__main__':
    args = parser.parse_args()
    domain = set_domains(args)
    all_metrics = []
    for k in domain:
        setattr(args, 'target_domain', str(k))
        setattr(args, 'save', args.dataset + str(k))
        setattr(args, 'cases', 'subject_val')
        # if args.dataset == 'hhar':
        #     setattr(args, 'cases', 'subject')
        # else:
        #     setattr(args, 'cases', 'subject_large')
        mif,maf = train_sup(args)
        all_metrics.append([mif,maf])
    values = np.array(all_metrics)
    mean = np.mean(values,0)
    print('Mean Acc: {}, Mean F1: {}'.format(mean[0],mean[1]))