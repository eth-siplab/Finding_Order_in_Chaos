import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import numpy as np
from utils import tsne, mds
from torch.autograd import Variable
from torch.utils.data import DataLoader
import brewer2mpl
import seaborn as sns
import pandas as pd

bmap = brewer2mpl.get_map('Set1', 'qualitative', 3)
colors = bmap.mpl_colors

plt.style.use('ggplot')

VAR_THRESHOLD = 1e-2


def plot_vs_gt_shapes(vae, shapes_dataset, save, z_inds=None):
    dataset_loader = DataLoader(shapes_dataset, batch_size=1000, num_workers=1, shuffle=False)

    N = len(dataset_loader.dataset)  # number of data samples
    K = vae.z_dim                    # number of latent variables
    nparams = vae.q_dist.nparams
    vae.eval()

    # print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    for xs in dataset_loader:
        batch_size = xs.size(0)
        xs = Variable(xs.view(batch_size, 1, 64, 64).cuda(), volatile=True)
        qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
        n += batch_size

    qz_params = qz_params.view(3, 6, 40, 32, 32, K, nparams)

    # z_j is inactive if Var_x(E[z_j|x]) < eps.
    qz_means = qz_params[:, :, :, :, :, :, 0]
    var = torch.std(qz_means.contiguous().view(N, K), dim=0).pow(2)
    active_units = torch.arange(0, K)[var > VAR_THRESHOLD].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))
    n_active = len(active_units)
    print('Number of active units: {}/{}'.format(n_active, vae.z_dim))

    if z_inds is None:
        z_inds = active_units

    # subplots where subplot[i, j] is gt_i vs. z_j
    mean_scale = qz_means.mean(2).mean(2).mean(2)  # (shape, scale, latent)
    mean_rotation = qz_means.mean(1).mean(2).mean(2)  # (shape, rotation, latent)
    mean_pos = qz_means.mean(0).mean(0).mean(0)  # (pos_x, pos_y, latent)

    fig = plt.figure(figsize=(3, len(z_inds)))  # default is (8,6)
    gs = gridspec.GridSpec(len(z_inds), 3)
    gs.update(wspace=0, hspace=0)  # set the spacing between axes.

    vmin_pos = torch.min(mean_pos)
    vmax_pos = torch.max(mean_pos)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[i * 3])
        ax.imshow(mean_pos[:, :, j].numpy(), cmap=plt.get_cmap('coolwarm'), vmin=vmin_pos, vmax=vmax_pos)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(r'$z_' + str(j) + r'$')
        if i == len(z_inds) - 1:
            ax.set_xlabel(r'pos')

    vmin_scale = torch.min(mean_scale)
    vmax_scale = torch.max(mean_scale)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[1 + i * 3])
        ax.plot(mean_scale[0, :, j].numpy(), color=colors[2])
        ax.plot(mean_scale[1, :, j].numpy(), color=colors[0])
        ax.plot(mean_scale[2, :, j].numpy(), color=colors[1])
        ax.set_ylim([vmin_scale, vmax_scale])
        ax.set_xticks([])
        ax.set_yticks([])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == len(z_inds) - 1:
            ax.set_xlabel(r'scale')

    vmin_rotation = torch.min(mean_rotation)
    vmax_rotation = torch.max(mean_rotation)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[2 + i * 3])
        ax.plot(mean_rotation[0, :, j].numpy(), color=colors[2])
        ax.plot(mean_rotation[1, :, j].numpy(), color=colors[0])
        ax.plot(mean_rotation[2, :, j].numpy(), color=colors[1])
        ax.set_ylim([vmin_rotation, vmax_rotation])
        ax.set_xticks([])
        ax.set_yticks([])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == len(z_inds) - 1:
            ax.set_xlabel(r'rotation')

    fig.text(0.5, 0.03, 'Ground Truth', ha='center')
    fig.text(0.01, 0.5, 'Learned Latent Variables ', va='center', rotation='vertical')
    plt.savefig(save)
    plt.close()


def plot_vs_gt_usc(vae, dataset, save, z_inds=None):
    dataset_loader = DataLoader(dataset, batch_size=1000, num_workers=1, shuffle=False)

    N = len(dataset_loader.dataset)  # number of data samples
    K = vae.z_dim                    # number of latent variables
    nparams = vae.q_dist.nparams
    vae.eval()

    # print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)
    qz_activity_labels = torch.Tensor(N)
    n = 0
    with torch.no_grad():
        for xs in dataset_loader:
            samples = xs[0].type(torch.cuda.FloatTensor)
            import pdb;pdb.set_trace();
            batch_size = samples.size(0)
            samples = Variable(samples.view(batch_size, 1, 100, 6).cuda())
            qz_params[n:n + batch_size] = vae.encoder.forward(samples).view(batch_size, vae.z_dim, nparams).data
            qz_activity_labels[n:n + batch_size] = xs[1].type(torch.cuda.FloatTensor)
            n += batch_size
    
    latent_values = vae.q_dist.sample(params=qz_params)
    tsne(latent_values,qz_activity_labels,'test10_usc_tsne')
    import pdb;pdb.set_trace();



def plot_vs_gt_ucihar(vae, args, train_loaders, DEVICE, save, z_inds=None):
    N = len(train_loaders)  # number of data samples
    K = vae.z_dim                    # number of latent variables
    nparams = vae.q_dist.nparams
    vae.eval()
    # print('Computing q(z|x) distributions.')
    qz_params = torch.zeros(1, K, nparams).to(DEVICE)
    qz_activity_labels = torch.zeros(N).to(DEVICE)
    n = 0
    with torch.no_grad():
        for i, train_loader in enumerate(train_loaders):
            for idx, (samples, target, domain) in enumerate(train_loader):
                samples = samples.to(DEVICE).float()
                target = target.to(DEVICE).float()
                #import pdb;pdb.set_trace();
                batch_size = samples.size(0)
                samples = Variable(samples.view(batch_size, 1, samples.size(1), samples.size(2)))
                qz_params = torch.cat((qz_params, vae.encoder.forward(samples).view(batch_size, vae.z_dim, nparams).data),0)
                qz_activity_labels = torch.hstack((qz_activity_labels,target))
                n += batch_size
    latent_values = vae.q_dist.sample(params=qz_params[1:,:,:])
    similarities = similarity_latents(args, latent_values, DEVICE, vae)
    class_distances = count_class_labels(qz_activity_labels[1:],similarities, args)
    tsne(latent_values,qz_activity_labels[1:], save)
    #mds(latent_values,qz_activity_labels,'test1_v2_mds')

def similarity_latents(args, latent_values, DEVICE, vae):
    a_norm = latent_values / latent_values.norm(dim=1)[:, None]
    b_norm = latent_values / latent_values.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0,1))
    res = res.fill_diagonal_(0) # Make diagonals to 0
    return res

def count_class_labels(labels,distances, args):
    # compute distances within each class
    class_distances, other_distances = np.zeros([1,]), np.zeros([1,])
    #for i in range(labels.size(0) + 1): 
    for i in range(200):        
        indices = (labels[i] == labels).nonzero(as_tuple=True)[0]
        other_indices = (labels[i] != labels).nonzero(as_tuple=True)[0]           
        class_distances = np.hstack((class_distances,distances[i,indices].cpu()))
        other_distances = np.hstack((other_distances,distances[i,other_indices].cpu()))
    combined_array = np.vstack((other_distances[0:len(class_distances)],class_distances,))
    ###
    combined_array = np.round(combined_array,decimals=3)
    df = pd.DataFrame(combined_array.T, columns = ['Inter-Class','Intra-Class'])
    hist1 = sns.kdeplot(data=df, hue_order=["Inter-Class", "Intra-Class"], fill=True, common_norm=False, palette="crest", alpha=.6, linewidth=1.5)
    #hist1 = sns.kdeplot(data=df, hue_order=["Inter-Class", "Intra-Class"], multiple="stack")
    hist1.set_xlabel("Cosine Distance",fontdict= { 'fontsize': 13, 'weight':'bold','color': 'black'})
    hist1.set_ylabel("Frequency",fontdict={'fontsize': 13, 'weight':'bold','color': 'black'})
    fig = hist1.get_figure()
    filename = args.dataset + '-300dpi.svgz'
    fig.savefig('max_edit.pdf',dpi=300)
    #import pdb;pdb.set_trace();    
    return class_distances

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpt', required=True)
    parser.add_argument('-zs', type=str, default=None)
    parser.add_argument('-gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
    parser.add_argument('-save', type=str, default='latent_vs_gt.pdf')
    parser.add_argument('-elbo_decomp', action='store_true')
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('--target_domain', type=str, default='0')    
    args = parser.parse_args()
    DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    from elbo_decomposition import elbo_decomposition
    import lib.dist as dist
    import lib.flows as flows
    from vae_quant import VAE, setup_data_loaders
    
    def load_model_and_dataset(checkpt_filename):
        print('Loading model and dataset.')
        checkpt = torch.load(checkpt_filename, map_location=lambda storage, loc: storage)
        args = checkpt['args']
        state_dict = checkpt['state_dict']

        # model
        if not hasattr(args, 'dist') or args.dist == 'normal':
            prior_dist = dist.Normal()
            q_dist = dist.Normal()
        elif args.dist == 'laplace':
            prior_dist = dist.Laplace()
            q_dist = dist.Laplace()
        elif args.dist == 'flow':
            prior_dist = flows.FactorialNormalizingFlow(dim=args.latent_dim, nsteps=4)
            q_dist = dist.Normal()
        vae = VAE(z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist, conv=args.conv)
        vae.load_state_dict(state_dict, strict=False)

        # dataset loader
        loader = setup_data_loaders(args)
        return vae, loader, args

    z_inds = list(map(int, args.zs.split(','))) if args.zs is not None else None
    torch.cuda.set_device(args.gpu)
    vae, dataset_loader, cpargs = load_model_and_dataset(args.checkpt)
    if args.elbo_decomp:
        elbo_decomposition(vae, dataset_loader)
    eval('plot_vs_gt_' + cpargs.dataset)(vae, dataset_loader.dataset, args.save, z_inds)