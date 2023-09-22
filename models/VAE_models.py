import torch.nn as nn
import torch


################# UCIHAR ####################

class ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, (9, 3), stride=(2,1))  
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (7, 3), stride=(2,1))  
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 3), stride=(3,1))  
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (5, 3), stride=(2,1))  
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 512, (2, 1), stride=(1,1))
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, 1, 128, 9)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        z = self.conv_z(x).view(x.size(0), self.output_dim)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, (2, 9), stride=(1, 1), padding=0)  # 2 x 9
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 128, (4, 1), stride=(6,1), padding=(1,0))  # 8 x 9
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, (4, 1), stride=(2,1), padding=(1,0))  # 16 x 9
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, (4, 1), stride=(2,1), padding=(1,0))  # 32 x 9
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, (4, 1), stride=(2,1), padding=(1,0))  # 64 x 9
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, (4, 1), stride=(2,1), padding=(1,0)) # 128 x 9

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        mu_img = self.conv_final(x)
        return mu_img

################# IDAA ####################

class vae_idaa(nn.Module):
    def __init__(self, z_dim, dataset):
        super(vae_idaa,self).__init__()

        if dataset == 'ucihar':
            self.encoder = ConvEncoder(z_dim)
            self.decoder = ConvDecoder(z_dim)
        elif dataset == 'shar':
            self.encoder = ConvEncoder_shar(z_dim)
            self.decoder = ConvDecoder_shar(z_dim)
        elif dataset == 'usc' or dataset == 'hhar':
            self.encoder = ConvEncoder_usc(z_dim)
            self.decoder = ConvDecoder_usc(z_dim)
        elif dataset == 'ieee_small' or dataset == 'ieee_big' or dataset == 'dalia':
            self.encoder = ConvEncoder_ieeesmall(z_dim)
            self.decoder = ConvDecoder_ieeesmall(z_dim)
        elif dataset == 'ecg':
            self.encoder = ConvEncoder_ecg(z_dim)
            self.decoder = ConvDecoder_ecg(z_dim)

        self.zdim = z_dim
        self.bn = nn.BatchNorm2d(1)
        self.fc11 = nn.Linear(z_dim, z_dim)
        self.fc12 = nn.Linear(z_dim, z_dim)
        self.fc21 = nn.Linear(z_dim, z_dim)      

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.zdim)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.zdim)
        h3 = self.decoder(z)
        return h3

    def forward(self, x, decode=False):
        if decode:
            z_projected = self.fc21(x)
            gx = self.decode(z_projected)
            gx = self.bn(gx)
            gx = torch.squeeze(gx)
            return gx
        else:
            _, mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            z_projected = self.fc21(z)
            gx = self.decode(z_projected)
            gx = self.bn(gx)
            gx = torch.squeeze(gx,1)
        return z, gx, mu, logvar        


################# IDAA ####################

class view_learner(nn.Module):
    def __init__(self, dataset):
        super(view_learner,self).__init__()

        self.conv = nn.Conv2d(1, 1, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: input tensor of shape (batch_size, channels, height, width)
        out = self.conv(x)
        out = self.relu(out)
        return out        
    
################# SHAR ####################


class ConvEncoder_shar(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_shar, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, (9, 2), stride=(2,1))  
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (7, 2), stride=(2,1))  
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 1), stride=(3,1))  
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (5, 1), stride=(2,1))  
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 512, (3, 1), stride=(1,1))
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, 1, 151, 3)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        z = self.conv_z(x).view(x.size(0), self.output_dim)
        return z


class ConvDecoder_shar(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder_shar, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, (3, 3), stride=(1, 1), padding=0)  # 2 x 9
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 128, (3, 1), stride=(4,1), padding=(1,0))  # 8 x 9
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, (3, 1), stride=(2,1), padding=(1,0))  # 16 x 9
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, (5, 1), stride=(2,1), padding=(1,0))  # 32 x 9
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, (7, 1), stride=(2,1), padding=(1,0))  # 64 x 9
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, (9, 1), stride=(2,1), padding=(1,0)) # 128 x 9

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):      
        x = z.view(z.size(0), z.size(1), 1, 1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        mu_img = self.conv_final(x)
        return mu_img
    

#################################################


class ConvEncoder_usc(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_usc, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, (9, 2), stride=(2,1))  
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (7, 2), stride=(2,1))  
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 2), stride=(2,1))  
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (5, 2), stride=(2,1))  
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 512, (2, 2), stride=(1,1))
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, 1, 100, 6)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        z = self.conv_z(x).view(x.size(0), self.output_dim)
        return z


class ConvDecoder_usc(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder_usc, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, (2, 6), stride=(1, 1), padding=0)  # 2 x 6
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 128, (6, 1), stride=(2,1), padding=(1,0))  # 8 x 9
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, (4, 1), stride=(2,1), padding=(1,0))  # 16 x 9
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, (5, 1), stride=(2,1), padding=(1,0))  # 32 x 9
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, (4, 1), stride=(2,1), padding=(1,0))  # 64 x 9
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, (4, 1), stride=(2,1), padding=(1,0)) # 128 x 9

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)    
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        mu_img = self.conv_final(x)
        return mu_img
    

#################################################


class ConvEncoder_ieeesmall(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_ieeesmall, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, (13, 1), stride=(2,1))  
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (9, 1), stride=(2,1))  
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (9, 1), stride=(2,1))  
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (7, 1), stride=(2,1))  
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 512, (5, 1), stride=(2,1))
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, 1, 200, 1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        z = self.conv_z(x).view(x.size(0), self.output_dim)
        return z


class ConvDecoder_ieeesmall(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder_ieeesmall, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, (6, 1), stride=(1, 1), padding=0)  # 2 x 6
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 128, (4, 1), stride=(2,1), padding=(1,0))  # 8 x 9
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, (5, 1), stride=(2,1), padding=(1,0))  # 16 x 9
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, (4, 1), stride=(2,1), padding=(1,0))  # 32 x 9
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, (4, 1), stride=(2,1), padding=(1,0))  # 64 x 9
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, (4, 1), stride=(2,1), padding=(1,0)) # 128 x 9

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)  # Batch, Latent, 1, 1
        x = self.act(self.bn1(self.conv1(x)))   # Batch, 512, 6, 1
        x = self.act(self.bn2(self.conv2(x)))   # Batch, 128, 12, 1
        x = self.act(self.bn3(self.conv3(x)))   # Batch, 64, 25, 1
        x = self.act(self.bn4(self.conv4(x)))   # Batch, 32, 50, 1
        x = self.act(self.bn5(self.conv5(x)))   # Batch, 32, 100, 1
        mu_img = self.conv_final(x)             # Batch, 32, 200, 1
        return mu_img
    


#################################################


class ConvEncoder_ecg(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_ecg, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, (12, 2), stride=(3,1))  
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (10, 2), stride=(3,1))  
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (8, 2), stride=(3,1))  
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (8, 1), stride=(3,1))  
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 512, (7, 1), stride=(3,1))
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, 1, 1000, 4)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        z = self.conv_z(x).view(x.size(0), self.output_dim)
        return z


class ConvDecoder_ecg(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder_ecg, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, (4, 4), stride=(1, 1), padding=0)  # 4 x 4
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 128, (5, 1), stride=(3,1), padding=(1,0))  # 12 x 4
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, (5, 1), stride=(3,1), padding=(1,0))  # 36 x 4
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, (6, 1), stride=(3,1), padding=(1,0))  # 109 x 4
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, (9, 1), stride=(3,1), padding=(1,0))  # 331 x 4
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, (12, 1), stride=(3,1), padding=(1,0)) # 1000 x 4

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):      
        x = z.view(z.size(0), z.size(1), 1, 1)        
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        mu_img = self.conv_final(x)
        return mu_img
    