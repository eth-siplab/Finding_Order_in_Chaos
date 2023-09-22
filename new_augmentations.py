import numpy as np
import torch
import scipy
import emd
import random



def gen_new_aug(sample, args, DEVICE):
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    index = torch.randperm(sample.size(0))
    mixing_coeff = (0.9 - 1) * torch.rand(1) + 1  
    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * mixing_coeff + (1 - mixing_coeff) * abs_fft[index]
    z =  torch.polar(mixed_abs, phase_fft) # Go back to fft
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time

def gen_new_aug_2(sample, args, inds, out, DEVICE, similarities):
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    inds = torch.randperm(sample.size(0))
    mixing_coeff = mixing_coefficient_set_for_each(similarities, inds, args) 
    coeffs = mixing_coeff.squeeze()

    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * coeffs[:, None, None] + (1 - coeffs[:, None, None]) * abs_fft[inds]
    mixed_phase = phase_mix(phase_fft, inds, similarities)
    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time

def gen_new_aug_3_ablation(sample, args, DEVICE, similarities): # Apply proposed mixup but use random coeffs instead of similarity based
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    inds = torch.randperm(sample.size(0))
    coeffs = torch.ones(sample.shape[0])
    coeffs = torch.nn.init.trunc_normal_(coeffs,1,0.1,0.9,1)

    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * coeffs[:, None, None] + (1 - coeffs[:, None, None]) * abs_fft[inds]
    dtheta, sign = phase_mix_2(phase_fft, inds)
    mixed_phase = phase_fft + (1-coeffs[:, None, None]) * torch.abs(dtheta) * sign
    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time

def gen_new_aug_4_comparison(sample, args, DEVICE): # Apply random phase changes but keep amplitude the same
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    inds = torch.randperm(sample.size(0))
    coeffs = torch.ones(sample.shape[0])
    coeffs = torch.nn.init.trunc_normal_(coeffs,1,0.1,0.9,1)

    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * coeffs[:, None, None] + (1 - coeffs[:, None, None]) * abs_fft[inds]
    dtheta, sign = phase_mix_2(phase_fft, inds)
    mixed_phase = phase_fft + (1-coeffs[:, None, None]) * torch.abs(dtheta) * sign
    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time

def opposite_phase(sample, args, DEVICE, similarities): # Show the importance of phase interpolations
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    inds = torch.randperm(sample.size(0))
    coeffs = torch.ones(sample.shape[0])
    coeffs = torch.nn.init.trunc_normal_(coeffs,1,0.1,0.9,1)

    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * coeffs[:, None, None] + (1 - coeffs[:, None, None]) * abs_fft[inds]
    dtheta, sign = phase_mix_2(phase_fft, inds)
    mixed_phase = phase_fft + (1-coeffs[:, None, None]) * torch.abs(dtheta) * -sign
    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time

def STAug(sample, args, DEVICE): # Comparison for Spectral and Time Augmentation
    sample = sample.detach().cpu().numpy()
    for i in range(sample.shape[0]): # For each sample in the batch
        for k in range(sample.shape[2]): # For each channel in time-series
            current_imf = emd.sift.sift(sample[i,:,k])
            w = np.random.uniform(0, 2, current_imf.shape[1])
            weighted_imfs = current_imf * w[None,:]
            s_prime = np.sum(weighted_imfs,axis=1)
            sample[i,:,k] = s_prime
    return torch.from_numpy(sample).float()


def vanilla_mix_up(sample):
    mixing_coeff = (0.75 - 1) * torch.rand(1) + 1   
    # Permute batch index for mixing
    index = torch.randperm(sample.size(0))
    # Mix the data
    mixed_data = mixing_coeff * sample + (1 - mixing_coeff) * sample[index]
    return mixed_data

def vanilla_mix_up_geo(sample):
    mixing_coeff = (0.7 - 1) * torch.rand(1) + 1
    # Permute batch index for mixing
    index = torch.randperm(sample.size(0))
    # Mix the data
    mixed_data = sample**mixing_coeff * sample[index]**(1 - mixing_coeff)
    return mixed_data

def vanilla_mix_up_binary(sample):
    alpha=0.8
    lam = torch.empty(sample.shape).uniform_(alpha, 1)
    mask = torch.empty(sample.shape).bernoulli_(lam)
    x_shuffle = sample[torch.randperm(sample.shape[0])]
    x_mixup = sample * mask + x_shuffle * (1 - mask)
    return x_mixup

def best_mix_up(sample, args, similarities, DEVICE): # Choose coeffs from best, but apply linear (Vanilla mixup) --- Ablation
    index = torch.randperm(sample.size(0))
    coeffs = mixing_coefficient_set_for_each(similarities, index, args) 
    coeffs = coeffs.squeeze()
    # Mix the data
    mixed_data = coeffs[:, None, None] * sample + (1 - coeffs[:, None, None]) * sample[index]
    return mixed_data

def best_mix_up_geo(sample, args, inds, out):
    mixed_samples = torch.empty(sample.shape, dtype=torch.float64)
    for idx, ind in enumerate(inds):
        mixing_coeff = (0.7 - 1) * torch.rand(1) + 1
        mixed_samples[idx,:,:] = sample[idx,:,:]**mixing_coeff *  sample[ind,:,:]**(1 - mixing_coeff)
    return mixed_samples


def mixing_coefficient_set(out):
    mixing_coefficient = torch.ones(out.shape).to(out.device)
    for idx, ind in enumerate(out):
        if ind > 0.7:
            mixing_coefficient[idx] = (0.7 - 1) * torch.rand(1) + 1
        else:
            torch.nn.init.trunc_normal_(mixing_coefficient[idx],0.85,out[idx],0.7,1)
    return mixing_coefficient

def mixing_coefficient_set_for_each(similarities, inds, args):
    threshold = 0.8
    mixing_coefficient = torch.ones(similarities.shape)
    similarities = similarities.cpu()
    distances = torch.gather(similarities,0,inds.unsqueeze(1)).cpu().numpy()
    
    mixing_coefficient = torch.ones(similarities.shape)
    distances[distances>threshold] = (0.7 - 1) * torch.rand(1) + 1
    mixing_coefficient = torch.ones(distances.shape)
    mixing_coefficient = torch.nn.init.trunc_normal_(mixing_coefficient,args.mean, args.std, args.low_limit,args.high_limit) 
    # mixing_coefficient = torch.nn.init.trunc_normal_(mixing_coefficient,0.9,0.2,0.7,1) --> Example
    distances[distances<=threshold] = mixing_coefficient[distances<=threshold]
    distances = torch.from_numpy(distances)
    return distances


def spec_mix(samples):
    batch_size, alpha = samples.size(0), 1
    indices = torch.randperm(batch_size)
    lam = (0.1 - 0.4) * torch.rand(1) + 0.4
    for i in range(samples.size(2)):
        current_channel = samples[:,:,i]
        current_channel_stft = torch.stft(current_channel,samples.size(1),return_complex=True)
        shuffled_data = current_channel_stft[indices, :, :]
        cut_len = int(lam * current_channel_stft.size(1))
        cut_start = np.random.randint(0, current_channel_stft.size(1) - cut_len + 1)
        current_channel_stft[:, cut_start:cut_start+cut_len, :] = shuffled_data[:, cut_start:cut_start+cut_len, :]
        samples[:,:,i] = torch.istft(current_channel_stft, n_fft=samples.size(1),length=samples.size(1))
    return samples


def cut_mix(data,alpha=2):
    batch_size = data.size(0)
    indices = torch.randperm(batch_size)
    shuffled_data = data[indices]

    lam = (0.1 - 0.3) * torch.rand(1) + 0.3

    cut_len = int(lam * data.size(1))
    cut_start = np.random.randint(0, data.size(1) - cut_len + 1)

    data[:, cut_start:cut_start+cut_len] = shuffled_data[:, cut_start:cut_start+cut_len]
    return data

def phase_mix(phase_fft, inds, similarities):
    phase_difference = phase_fft - phase_fft[inds]
    dtheta = phase_difference % (2 * torch.pi)

    dtheta[dtheta > torch.pi] -= 2 * torch.pi
    clockwise = dtheta > 0
    sign = torch.where(clockwise, -1, 1)
    coeffs = torch.squeeze(mixing_coefficient_set_for_each_phase(similarities, inds))
    mixed_phase = phase_fft
    mixed_phase = phase_fft + (1-coeffs[:, None, None]) * torch.abs(dtheta) * sign
    return mixed_phase

def phase_mix_2(phase_fft, inds):
    phase_difference = phase_fft - phase_fft[inds]
    dtheta = phase_difference % (2 * torch.pi)

    dtheta[dtheta > torch.pi] -= 2 * torch.pi
    clockwise = dtheta > 0
    locs = torch.where(torch.abs(phase_difference) > torch.pi, -1, 1)
    sign = torch.where(clockwise, -1, 1)
    return dtheta, sign

def mixing_coefficient_set_for_each_phase(similarities, inds):
    threshold = 0.8
    mixing_coefficient = torch.ones(similarities.shape)
    similarities = similarities.cpu()
    distances = torch.gather(similarities,0,inds.unsqueeze(1)).cpu().numpy()
    
    mixing_coefficient = torch.ones(similarities.shape)
    distances[distances>threshold] = (0.9 - 1) * torch.rand(1) + 1
    mixing_coefficient = torch.ones(distances.shape)
    mixing_coefficient = torch.nn.init.trunc_normal_(mixing_coefficient,1,0.1,0.9,1) 
    # mixing_coefficient = torch.nn.init.trunc_normal_(mixing_coefficient,0.9,0.2,0.7,1)
    distances[distances<=threshold] = mixing_coefficient[distances<=threshold]
    distances = torch.from_numpy(distances)
    return distances

def check_max_not_selected(max_indices, indices, abs_fft):
    for i in range(len(max_indices)):
        while indices[i] == max_indices[i].item():
            #np.random.shuffle(indices)
            indices = np.random.choice(np.ceil(abs_fft.size(1)/2).astype(int),abs_fft.size(2)) 
    return indices


######################################### For Supervised Learning Paradigm #########################################

def vanilla_mixup_sup(sample, target, alpha=0.3):
    size_of_batch = sample.size(0)
    # Choose quarters of the batch to mix
    indices = torch.randperm(size_of_batch)
    m = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
    mixing_coeff = m.sample()  
    # Mix the data
    mixed_data = mixing_coeff * sample + (1 - mixing_coeff) * sample[indices]
    return mixed_data, target, mixing_coeff, target[indices]


def gen_new_aug_3_ablation_sup(sample, args, DEVICE, target, alpha=0.2): 
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    inds = torch.randperm(sample.size(0))
    m = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
    coeffs = m.sample()

    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * coeffs + (1 - coeffs) * abs_fft[inds]

    dtheta, sign = phase_mix_2(phase_fft, inds)
    dtheta2, sign2 = phase_mix_2(phase_fft[inds], torch.linspace(0,63,64,dtype=inds.dtype))
    #mixed_phase = phase_fft if coeffs > 0.5 else phase_fft[inds]
    phase_coeff = (0.9 - 1) * torch.rand(1) + 1
    mixed_phase = phase_fft + (1-phase_coeff) * torch.abs(dtheta) * sign if coeffs > 0.5 else phase_fft[inds] + (1-phase_coeff) * torch.abs(dtheta2) * sign2
    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time, target, coeffs, target[inds]

def cutmix_sup(data, target, alpha=1.):
    batch_size = data.size(0)
    indices = torch.randperm(batch_size)
    shuffled_data = data[indices]

    m = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
    lam = m.sample()

    cut_len = int(lam * data.size(1))
    cut_start = np.random.randint(0, data.size(1) - cut_len + 1)

    data[:, cut_start:cut_start+cut_len] = shuffled_data[:, cut_start:cut_start+cut_len]
    return data, target, lam, target[indices]

def binary_mixup_sup(sample, target, alpha=0.2):
    lam = torch.empty(sample.shape).uniform_(alpha, 1)
    mask = torch.empty(sample.shape).bernoulli_(lam)
    indices = torch.randperm(sample.shape[0])
    x_shuffle = sample[indices]
    x_mixup = sample * mask + x_shuffle * (1 - mask)
    return x_mixup, target, lam, target[indices]


def gen_new_aug_2_sup(sample, args, inds, out, DEVICE, similarities, target):
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    inds = torch.randperm(sample.size(0))
    mixing_coeff = mixing_coefficient_set_for_each(similarities, inds, args) 
    coeffs = mixing_coeff.squeeze()

    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * coeffs[:, None, None] + (1 - coeffs[:, None, None]) * abs_fft[inds]
    mixed_phase = phase_mix(phase_fft, inds, similarities)
    #z =  torch.polar(mixed_abs, torch.angle(fftsamples)) # Go back to fft
    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time, target, coeffs, target[inds]


def mag_mixup_sup(sample, args, DEVICE, target, alpha=0.2): 
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    index = torch.randperm(sample.size(0))
    m = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
    coeffs = m.sample()
    abs_fft = torch.abs(fftsamples)
    phase_fft, phase_fft2 = torch.angle(fftsamples), torch.angle(fftsamples[index])
    mixed_abs = abs_fft * coeffs + (1 - coeffs) * abs_fft[index] 
    z =  torch.polar(mixed_abs, phase_fft) if coeffs > 0.5 else torch.polar(mixed_abs, phase_fft2)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    #value = torch.roll(value,5,1)
    return mixed_samples_time, target, coeffs, target[index]