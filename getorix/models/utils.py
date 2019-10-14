import torch
import pyro.distributions as pdist
import torch.nn.functional as F
import pyro


def make_normal_prior(w):
  return pdist.Normal(torch.zeros_like(w), torch.ones_like(w))


def make_variational_params(name, w, act_fn=F.softplus):
  with torch.no_grad():
    loc = pyro.param(f'{name}_loc', torch.randn_like(w))
    scale = pyro.param(f'{name}_scale', torch.randn_like(w))
    return pdist.Normal(loc, act_fn(scale))


def one_hot(t, n_classes=10):
  if not isinstance(t, type(torch.tensor([0.6, 0.9]))):
    if not isinstance(t, type([0.6, 0.9])):
      t = [t]
    t = torch.tensor(t)
  t = t.reshape(-1, 1)
  t_one_hot = torch.zeros(t.size(0), n_classes)
  return t_one_hot.scatter(1, t.long(), 1)


def merge(x, y, n_classes=10, transform=True):
  batch_size = x.size(0)
  x = x.reshape(batch_size, -1)
  if len(y.size()) == 3 and len(x.size()) == 2:
    x_repeated = x.repeat(n_classes, x.size(0), x.size(1))
  if transform:
    y = one_hot(y, n_classes).float().reshape(-1, n_classes)
  return torch.cat([x_repeated, y], axis=-1)
