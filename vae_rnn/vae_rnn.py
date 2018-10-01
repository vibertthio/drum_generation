import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


'''
Constants
'''
NUM_EPOCHS = 10
BATCH_SIZE = 256
LR = 0.005
NUM_FEATURES = 9 # original: 128, trimmed: 47 (34 - 81)
SEQ_LEN = 96
NUM_BARS = 1
NUM_BEATS_PER_BAR = 4
NUM_DIRECTIONS = 2
GRU_HIDDEN_SIZE = 16
LINEAR_HIDDEN_SIZE = [64, 32]

ACTIVATION = 'relu'
activation_function = torch.relu
activation_function_out = torch.relu


'''
check the GPU usage
'''
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if use_cuda:
    print('run on GPU')
else:
    print('run on CPU')


class Encoder(torch.nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.gru = torch.nn.GRU(
            input_size=NUM_FEATURES,
            num_layers=1,
            hidden_size=GRU_HIDDEN_SIZE,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        
        self.gru_out_dim = SEQ_LEN * GRU_HIDDEN_SIZE * NUM_DIRECTIONS
        self.bn0 = torch.nn.BatchNorm1d(self.gru_out_dim)
        self.linear0 = torch.nn.Linear(
            self.gru_out_dim,
            LINEAR_HIDDEN_SIZE[0])
        self.bn1 = torch.nn.BatchNorm1d(LINEAR_HIDDEN_SIZE[0])

    def forward(self, x):
        # print(type(x))
        # print(x)
        
        x, _ = self.gru(x, None)
        x = x.contiguous().view(
            BATCH_SIZE,
            self.gru_out_dim)
        x = self.bn0(x)
        x = activation_function(self.linear0(x))
        x = self.bn1(x)
        
        return x


class Decoder(torch.nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.beat = 12

        self.gru_in_dim = SEQ_LEN * NUM_FEATURES
        self.linear0 = torch.nn.Linear(
            LINEAR_HIDDEN_SIZE[1],
            self.gru_in_dim)
        self.bn0 = torch.nn.BatchNorm1d(self.gru_in_dim)

        self.bn1 = torch.nn.BatchNorm1d(SEQ_LEN)
        self.linear1 = torch.nn.Linear(
            NUM_FEATURES,
            NUM_FEATURES)
        self.bn2 = torch.nn.BatchNorm1d(12)

        self.gru = torch.nn.GRU(
            input_size=NUM_FEATURES,
            num_layers=1,
            hidden_size=NUM_FEATURES,
            batch_first=True,
            bidirectional=False,
        )
    def forward(self, x):
        melody = torch.zeros((x.shape[0], SEQ_LEN, NUM_FEATURES)).to(device)

        x = activation_function_out(self.bn0(self.linear0(x)))
        hn = torch.zeros(
            1,
            BATCH_SIZE, NUM_FEATURES
        ).cuda()
        
        x = x.contiguous().view(
            BATCH_SIZE,
            SEQ_LEN,
            NUM_FEATURES)

        for i in range(8):
            x, hn = self.gru(x, hn)
            x = self.bn1(x)
            out = self.bn2(self.linear1(x[:,:12,:]))
            melody[:,12*i:(12*i+12),:] = torch.sigmoid(out)

        melody = activation_function_out(melody)
        return melody


class VAE(torch.nn.Module):


    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(
            LINEAR_HIDDEN_SIZE[0],
            LINEAR_HIDDEN_SIZE[1])
        self._enc_log_sigma = torch.nn.Linear(
            LINEAR_HIDDEN_SIZE[0],
            LINEAR_HIDDEN_SIZE[1])

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(
            np.random.normal(0, 1, size=sigma.size())
        ).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False).to(device)

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        output = self.decoder(z)

        return output



def elbo(recon_tracks, tracks, mu, sigma):
    """
    Args:
        recon_x: generating images
        x: origin images
        mu: latent mean
        logvar: latent log variance
    """
    BCE = F.binary_cross_entropy(
        recon_tracks,
        tracks,
        reduction='sum',
    )
    KLD = 0.5 * torch.sum(mu * mu + sigma.exp() - sigma - 1)
    return BCE + KLD


def plot_track(track, cmap='Blues', single=True, bres=3):
    t = track
    if t.shape[1] != 128:
        t = np.append(np.zeros(((t.shape[0]), 34)), t, axis=1)
        t = np.append(t, np.zeros((t.shape[0], 47)), axis=1)
    pr = Track(pianoroll=t)

    fig, axs = pr.plot(
        xtick='beat',
        ytick='octave',
        yticklabel='number',
        beat_resolution=bres,
        cmap=cmap
    )
    y = axs.set_ylim(34, 81) # C0 - C2
    if single:
        x = axs.set_xlim(0, BAR_DIVISION)
    plt.show()