from torchvision.utils import save_image
import torch.nn as nn
import torch

from pyro.distributions import Normal, Bernoulli
from pyro.infer import SVI, JitTrace_ELBO  # , Trace_ELBO
from pyro.optim import Adam
import pyro

from tqdm import tqdm
import os

OUT = 'results'


class Decoder(nn.Module):
  def __init__(self, z_dim, h_dim, i_dim=784):
    super(Decoder, self).__init__()
    self.linear1 = nn.Linear(z_dim, h_dim)
    self.linear2 = nn.Linear(h_dim, i_dim)
    self.softplus = nn.Softplus()

  def forward(self, z):
    h = self.softplus(self.linear1(z))
    loc_img = torch.sigmoid(self.linear2(h))
    return loc_img


class Encoder(nn.Module):
  def __init__(self, z_dim, h_dim, i_dim=784):
    super(Encoder, self).__init__()
    self.linear1 = nn.Linear(i_dim, h_dim)
    self.linear21 = nn.Linear(h_dim, z_dim)
    self.linear22 = nn.Linear(h_dim, z_dim)
    self.softplus = nn.Softplus()

  def forward(self, x):
    h = self.softplus(self.linear1(x))
    loc_z = self.linear21(h)
    scale_z = torch.exp(self.linear22(h))
    # assert not torch.isnan(scale_z).any()
    return loc_z, scale_z


class VAE(nn.Module):
  def __init__(self, z_dim=50, h_dim=400, results='results/', name='vae'):
    super(VAE, self).__init__()
    # create the encoder and decoder networks
    self.encoder = Encoder(z_dim, h_dim)
    self.decoder = Decoder(z_dim, h_dim)
    self.z_dim = z_dim
    self.h_dim = h_dim
    self.save_path = os.path.join(results, f'{name}.params')

  def model(self, x):
    pyro.module("decoder", self.decoder)
    with pyro.plate("data", x.shape[0]):
      # setup hyperparameters of prior, z : (loc_z, scale_z)
      loc_z = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
      scale_z = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
      # sample from prior
      z = pyro.sample("latent", Normal(loc_z, scale_z).to_event(1))
      # decoder sampled latent code
      loc_img = self.decoder(z)
      # sample from Bernoulli with `p=loc_img`
      pyro.sample("obs", Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))

      return loc_img

  def guide(self, x):
    pyro.module("encoder", self.encoder)
    with pyro.plate("data", x.shape[0]):
      loc_z, scale_z = self.encoder(x)
      pyro.sample("latent", Normal(loc_z, scale_z).to_event(1))

  def reconstruct_image(self, x):
    loc_z, scale_z = self.encoder(x)
    z = Normal(loc_z, scale_z).sample()
    loc_img = self.decoder(z)
    return loc_img

  def fit(self, train_loader, test_loader, num_epochs=100, test_frequency=5, stop_at_loss=100.):
    # setup optimizer
    optimizer = Adam({ 'lr' : 1.0e-3 })
    # loss function
    elbo = JitTrace_ELBO()
    # setup svi
    svi = SVI(self.model, self.guide, optimizer, loss=elbo)
    for epoch in range(num_epochs):
      epoch_loss = torch.tensor([ svi.step(x.reshape(-1, 784))
        for x, _ in tqdm(train_loader) ]).mean().item() / train_loader.batch_size
      print(f'Epoch {epoch + 1} : Loss {epoch_loss}')

      if epoch and epoch % test_frequency == 0:
        # report test diagnostics
        test_loss = torch.tensor([ svi.step(x.reshape(-1, 784))
          for x, _ in tqdm(test_loader) ]).mean().item() / test_loader.batch_size
        print(f'*Test Loss* {test_loss}')
        # save to disk
        self.save()
        if int(test_loss) <= int(stop_at_loss):
          return self

  def save(self):
    torch.save(self.state_dict(), self.save_path)

  def load(self):
    self.load_state_dict(torch.load(self.save_path))

  def generate_samples(self, test_loader, n, save=True, sigma=1):
    z = Normal(torch.zeros(1, self.z_dim), sigma * torch.ones(1, self.z_dim)).sample([n])
    samples = self.decoder(z)
    if save:
      save_image(samples.reshape(-1, 1, 28, 28),
          os.path.join(OUT, 'vae_samples.png')
          )
    return samples
