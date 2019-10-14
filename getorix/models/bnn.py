import torch.nn.functional as F
import torch.nn as nn
import torch

from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro

from tqdm import tqdm
import numpy as np
import os


def make_normal_prior(w):
  return Normal(torch.zeros_like(w), torch.ones_like(w))


def make_variational_params(name, w, act_fn=F.softplus):
  with torch.no_grad():
    loc = pyro.param(f'{name}_loc', torch.randn_like(w))
    scale = pyro.param(f'{name}_scale', torch.randn_like(w))
    return Normal(loc, act_fn(scale))


class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(NeuralNet, self).__init__()
    self.linear_1 = nn.Linear(input_size, hidden_size)
    self.linear_2 = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    x = F.relu(self.linear_1(x))
    x = self.linear_2(x)
    return x


class BayesianNeuralNet():

  def __init__(self, input_size, hidden_size, output_size,
      results='results/', name='bnn'):
    self.net = NeuralNet(input_size, hidden_size, output_size)
    self.results = results
    self.name = name
    self.save_path = os.path.join(results, f'{name}.params')

  def model(self, x, y):
    priors = {
        name : make_normal_prior(p) for name, p in self.net.named_parameters()
    }
    lifted_module = pyro.random_module("module", self.net, priors)
    lifted_reg_model = lifted_module()
    lhat = F.log_softmax(lifted_reg_model(x), dim=1)
    pyro.sample("y", Categorical(logits=lhat), obs=y)

  def guide(self, x, y):
    priors = {}
    for name, p in self.net.named_parameters():
      priors[name] = make_variational_params(name, p)
      pyro.sample(name, priors[name])

    lifted_module = pyro.random_module("module", self.net, priors)
    return lifted_module()

  def fit(self, data_loaders, num_epochs=10):
    train_loader, test_loader = data_loaders
    batch_size = train_loader.batch_size
    optimizer = Adam({'lr' : 0.0009})
    elbo = Trace_ELBO()
    svi = SVI(self.model, self.guide, optimizer, elbo)

    for epoch in range(num_epochs):
      loss = 0.
      iterations = 0
      for x, y in tqdm(train_loader):
        if x.size() == (batch_size, 1, 28, 28):
          loss += svi.step(x.reshape(batch_size, -1), y)
          iterations += 1

      print(epoch, ' Epoch loss : ', loss / (iterations * batch_size))

  def save(self):
    torch.save(self.net.state_dict(), self.save_path)

  def load(self):
    self.net.load_state_dict(torch.load(self.save_path))

  def run_ensemble(self, x, n):
    x = x.float()
    sampled_models = [self.guide(None, None) for _ in range(n)]
    yhats = [ model(x.reshape(-1, 28 * 28)) for model in sampled_models ]
    mean = torch.mean(torch.stack(yhats), 0)
    prediction = mean.argmax(1)
    hist = torch.stack(yhats).argmax(2).squeeze()
    hist = torch.histc(hist.float(), bins=10, min=0, max=9) / n
    return prediction, hist, hist[prediction] * 100.

  def evaluate(self, test_loader, n=20):
    correct = 0
    total = 0
    for j, data in enumerate(tqdm(test_loader)):
      images, labels = data
      prediction, _, _ = self.run_ensemble(images.view(-1, 28 * 28), n)
      total += labels.size(0)
      correct += (prediction.detach().numpy() == labels.numpy()).sum().item()

    return 100. * correct / total
