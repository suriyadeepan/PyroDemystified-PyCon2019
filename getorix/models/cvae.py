from getorix.models.utils import one_hot

from torchvision.utils import save_image
import torch.nn as nn
import torch

from pyro.distributions import Normal, Bernoulli
from pyro.infer import SVI, JitTrace_ELBO
from pyro.optim import Adam
import pyro

from tqdm import tqdm
import os

OUT = 'results'


class Decoder(nn.Module):
  def __init__(self, z_dim, h_dim, n_classes=10, i_dim=784):
    super(Decoder, self).__init__()
    self.linear1 = nn.Linear(z_dim + n_classes, h_dim)
    self.linear2 = nn.Linear(h_dim, i_dim)
    self.softplus = nn.Softplus()

  def forward(self, z, y, is_one_hot=False):
    y = one_hot(y) if not is_one_hot else y
    input = torch.cat([z, y], axis=1)
    h = self.softplus(self.linear1(input))
    loc_img = torch.sigmoid(self.linear2(h))
    return loc_img


class Encoder(nn.Module):
  def __init__(self, z_dim, h_dim, n_classes=10, i_dim=784):
    super(Encoder, self).__init__()
    self.linear1 = nn.Linear(i_dim + n_classes, h_dim)
    self.linear21 = nn.Linear(h_dim, z_dim)
    self.linear22 = nn.Linear(h_dim, z_dim)
    self.softplus = nn.Softplus()

  def forward(self, x, y, is_one_hot=False):
    x = x.view(-1, 28 * 28)
    y = one_hot(y) if not is_one_hot else y
    input = torch.cat([x, y], axis=1)
    h = self.softplus(self.linear1(input))
    loc_z = self.linear21(h)
    scale_z = torch.exp(self.linear22(h))
    return loc_z, scale_z


class CVAE(nn.Module):
  def __init__(self, z_dim=50, h_dim=400, n_classes=10,
      results='results/', name='cvae'):
    super(CVAE, self).__init__()
    # create the encoder and decoder networks
    self.encoder = Encoder(z_dim, h_dim)
    self.decoder = Decoder(z_dim, h_dim)
    self.z_dim = z_dim
    self.h_dim = h_dim
    self.n_classes = n_classes
    self.save_path = os.path.join(results, f'{name}.params')

  def model(self, x, y):
    pyro.module("decoder", self.decoder)
    with pyro.plate("data", x.shape[0]):
      # setup hyperparameters of prior, z : (loc_z, scale_z)
      loc_z = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
      scale_z = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
      # sample from prior
      z = pyro.sample("latent", Normal(loc_z, scale_z).to_event(1))
      # decoder sampled latent code
      loc_img = self.decoder(z, y)
      # sample from Bernoulli with `p=loc_img`
      pyro.sample("obs", Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))

      return loc_img

  def guide(self, x, y):
    pyro.module("encoder", self.encoder)
    with pyro.plate("data", x.shape[0]):
      loc_z, scale_z = self.encoder(x, y)
      pyro.sample("latent", Normal(loc_z, scale_z).to_event(1))

  def reconstruct_image(self, x, y):
    loc_z, scale_z = self.encoder(x, y)
    z = Normal(loc_z, scale_z).sample()
    loc_img = self.decoder(z, y)
    return loc_img

  def fit(self, train_loader, test_loader, num_epochs=30, test_frequency=5, stop_at_loss=100.):
    # setup optimizer
    optimizer = Adam({ 'lr' : 1.0e-3 })
    # loss function
    elbo = JitTrace_ELBO()
    # setup svi
    svi = SVI(self.model, self.guide, optimizer, loss=elbo)
    for epoch in range(num_epochs):
      epoch_loss = torch.tensor([ svi.step(x, y)
        for x, y in tqdm(train_loader) ]).mean().item() / train_loader.batch_size
      print(f'Epoch {epoch + 1} : Loss {epoch_loss}')

      if epoch and epoch % test_frequency == 0:
        # report test diagnostics
        test_loss = torch.tensor([ svi.step(x, y)
          for x, y in tqdm(test_loader) ]).mean().item() / test_loader.batch_size
        print(f'*Test Loss* {test_loss}')
        # save to disk
        self.save()
        if int(test_loss) <= int(stop_at_loss):
          return self

  def save(self):
    torch.save(self.state_dict(), self.save_path)

  def load(self):
    self.load_state_dict(torch.load(self.save_path))

  def generate_samples(self, test_loader, n, save=True, sigma=0.3, c=None):
    c = torch.zeros(n, ).random_(0, self.n_classes).long() if c is None else c
    z = Normal(torch.zeros(self.z_dim), sigma * torch.ones(self.z_dim)).rsample([n, ])
    samples = self.decoder(z, c)
    if save:
      save_image(samples.reshape(-1, 1, 28, 28),
          os.path.join(OUT, 'cvae_samples.png')
          )
