import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import pyro
import pyro.distributions as pdist
from pyro.optim import Adam
from pyro.infer import Trace_ELBO, SVI

from getorix.models.utils import make_normal_prior
from getorix.models.utils import make_variational_params

from tqdm import tqdm


class MuchLesserLeNet5(nn.Module):
  def __init__(self):
    super(MuchLesserLeNet5, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
        kernel_size=5, stride=1, padding=2, bias=True)
    # Max-pooling
    self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
    # Fully connected layer
    self.fc = nn.Linear(6 * 14 * 14, 10)

  def forward(self, x):
    batch_size = x.size(0)
    x = F.relu(self.conv1(x))  # C1
    x = self.max_pool_1(x)     # M1
    x = x.view(batch_size, -1)
    x = self.fc(x)
    return x


class LesserLeNet5(nn.Module):
  def __init__(self):
    super(LesserLeNet5, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
        kernel_size=5, stride=1, padding=2, bias=True)
    # Max-pooling
    self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
    # Convolution
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
        kernel_size=5, stride=1, padding=0, bias=True)
    # Max-pooling
    self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
    # Fully connected layer
    self.fc = nn.Linear(16 * 5 * 5, 10)

  def forward(self, x):
    batch_size = x.size(0)
    x = F.relu(self.conv1(x))  # C1
    x = self.max_pool_1(x)     # M1
    x = F.relu(self.conv2(x))  # C2
    x = self.max_pool_2(x)     # M2
    x = x.view(batch_size, -1)
    x = self.fc(x)
    return x


class LeNet5(nn.Module):
  def __init__(self):
    super(LeNet5, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
        kernel_size=5, stride=1, padding=2, bias=True)
    # Max-pooling
    self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
    # Convolution
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
        kernel_size=5, stride=1, padding=0, bias=True)
    # Max-pooling
    self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
    # Fully connected layer
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    batch_size = x.size(0)
    x = F.relu(self.conv1(x))  # C1
    x = self.max_pool_1(x)     # M1
    x = F.relu(self.conv2(x))  # C2
    x = self.max_pool_2(x)     # M2
    x = x.view(batch_size, -1)
    x = F.relu(self.fc1(x))    # FC1
    x = F.relu(self.fc2(x))    # FC2
    x = self.fc3(x)            # FC3
    return x


class BayesianLeNet5():

  def __init__(self, net, results='results/', name='blenet'):
    self.net = net
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
    pyro.sample("y", pdist.Categorical(logits=lhat), obs=y)

  def guide(self, x, y):
    priors = {}
    for name, p in self.net.named_parameters():
      priors[name] = make_variational_params(name, p)
      pyro.sample(name, priors[name])

    lifted_module = pyro.random_module("module", self.net, priors)
    return lifted_module()

  def test_fit(self):
    optimizer = Adam({'lr' : 0.0009})
    elbo = Trace_ELBO()
    svi = SVI(self.model, self.guide, optimizer, elbo)
    batch_size = 128
    loss = svi.step(
        torch.randn(batch_size, 1, 28, 28),
        2 * torch.ones(batch_size,)
        )
    assert loss, loss

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
          loss += svi.step(x, y)
          iterations += 1

      print(epoch, ' Epoch loss : ', loss / (iterations * batch_size))
      test_accuracy = self.evaluate(test_loader)
      print(f'*Test Accuracy : {test_accuracy}')

  def save(self):
    torch.save(self.net.state_dict(), self.save_path)

  def load(self):
    self.net.load_state_dict(torch.load(self.save_path))

  def run_ensemble(self, x, n):
    x = x.float()
    sampled_models = [self.guide(None, None) for _ in range(n)]
    yhats = [ model(x.reshape(-1, 1, 28, 28)) for model in sampled_models ]
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
      prediction, _, _ = self.run_ensemble(images.view(-1, 1, 28, 28), n)
      total += labels.size(0)
      correct += (prediction.detach().numpy() == labels.numpy()).sum().item()

    return 100. * correct / total
