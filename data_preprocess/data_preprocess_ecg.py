'''
Data Pre-processing on ecg dataset.

'''

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import scipy.io
import pickle as cp
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split
from data_preprocess.base_loader import base_loader


def load_domain_data(domain_idx):
    str_folder = './data/ECG_Data/'
    file = open(str_folder + 'ECG_data.pkl', 'rb')
    data = cp.load(file)
    data = data['whole_dataset']
    data = np.asarray(data)
    domain_idx = int(domain_idx)
    X = data[domain_idx,0]
    y = np.squeeze(data[domain_idx,1]) - 1 if np.min(data[domain_idx,1]) != 0 else np.squeeze(data[domain_idx,1])
    d = np.full(y.shape, domain_idx, dtype=int)
    return X, y, d

class data_loader_ecg(base_loader):
    def __init__(self, samples, labels, domains):
        super(data_loader_ecg, self).__init__(samples, labels, domains)

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return np.squeeze(np.transpose(sample, (1, 0, 2)),0), target, domain


def prep_domains_ecg_subject(args):
    # todo: make the domain IDs as arguments or a function with args to select the IDs (default, customized, small, etc)
    source_domain_list = ['0','1'] if args.target_domain == '1' else ['2', '3']
    
    source_domain_list.remove(args.target_domain)

    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        x, y, d = load_domain_data(source_domain)

        x = np.transpose(x.reshape((-1, 1, 1000, 4)), (0, 2, 1, 3))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    unique_y, counts_y = np.unique(y_win_all, return_counts=True)
    weights = 100.0 / torch.Tensor(counts_y)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_all, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights), replacement=True)

    data_set = data_loader_ecg(x_win_all, y_win_all, d_win_all)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    source_loaders = [source_loader]

    # target domain data prep
    x, y, d = load_domain_data(args.target_domain)

    x = np.transpose(x.reshape((-1, 1, 1000, 4)), (0, 2, 1, 3))

    data_set = data_loader_ecg(x, y, d)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    return source_loaders, None, target_loader

def prep_domains_ecg_subject_large(args):
    source_domain_list = ['0', '1', '2', '3']
    source_domain_list.remove(args.target_domain)
    
    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        x, y, d = load_domain_data(source_domain)
        x = np.transpose(x.reshape((-1, 1, 1000, 4)), (0, 2, 1, 3))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    unique_y, counts_y = np.unique(y_win_all, return_counts=True)
    weights = 100.0 / torch.Tensor(counts_y)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_all, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    data_set = data_loader_ecg(x_win_all, y_win_all, d_win_all)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=None)
    source_loaders = [source_loader]

    # target domain data prep
    x, y, d = load_domain_data(args.target_domain)

    x = np.transpose(x.reshape((-1, 1, 1000, 4)), (0, 2, 1, 3))

    data_set = data_loader_ecg(x, y, d)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)
    return source_loaders, None, target_loader

def prep_domains_ecg_sp(args):
    source_domain_list = ['0','1'] if args.target_domain == '1' else ['2', '3']

    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    n_train, n_test, split_ratio = [], 0, 0.0

    for source_domain in source_domain_list:
        x_win, y_win, d_win = load_domain_data(source_domain)

        x_win = np.transpose(x_win.reshape((-1, 1, 1000, 4)), (0, 2, 1, 3))

        x_win_all = np.concatenate((x_win_all, x_win), axis=0) if x_win_all.size else x_win
        y_win_all = np.concatenate((y_win_all, y_win), axis=0) if y_win_all.size else y_win
        d_win_all = np.concatenate((d_win_all, d_win), axis=0) if d_win_all.size else d_win
        n_train.append(x_win.shape[0])

    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_split(x_win_all, y_win_all, d_win_all, split_ratio=args.split_ratio)

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_set_r = data_loader_ecg(x_win_train, y_win_train, d_win_train)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    val_set_r = data_loader_ecg(x_win_val, y_win_val, d_win_val)
    val_loader_r = DataLoader(val_set_r, batch_size=args.batch_size, shuffle=False)
    test_set_r = data_loader_ecg(x_win_test, y_win_test, d_win_test)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size, shuffle=False)

    return [train_loader_r], val_loader_r, test_loader_r

def prep_ecg(args):
    if args.cases == 'subject':
        return prep_domains_ecg_subject(args)
    elif args.cases == 'subject_large':
        return prep_domains_ecg_subject_large(args)
    elif args.cases == 'subject_val':
        return prep_domains_ecg_sp(args)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'

