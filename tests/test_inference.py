import pyro.distributions as pdist
import torch
import pyro

import pytest


@pytest.fixture
def normal_normal():
  def model(data):
    mu = pyro.sample('mu', pdist.Normal(0., 1.))
    sigma = pyro.sample('sigma', pdist.HalfCauchy(5.))
    with pyro.plate('observe_data'):
      pyro.sample('obs', pdist.Normal(mu, sigma), obs=data)
  return model


@pytest.fixture
def normal_data():
  return 3. * torch.randn([1000, ]) + 4.


def test_nuts_fit(normal_normal, normal_data):
  from getorix.inference import nuts_fit
  pyro.clear_param_store()
  trace, posteriors = nuts_fit(normal_normal,
      normal_data, num_samples=500, warmup_steps=200,
      var_names=['mu', 'sigma'])

  assert torch.abs(posteriors['mu'].mean - 4.) < 1.
  assert torch.abs(posteriors['sigma'].mean - 3.) < 1.


@pytest.fixture
def beta_bernoulli():
  def model(data):
    p = pyro.sample('p', pdist.Beta(2., 2.))  # `p` centered at 0.5
    with pyro.plate('observe_data'):
      pyro.sample('obs', pdist.Bernoulli(p), obs=data)
  return model


@pytest.fixture
def bernoulli_data():
  return pdist.Bernoulli(0.3).sample([10000, ])


@pytest.fixture
def beta_bernoulli_guide():
  def guide(data):
    a = pyro.param('a', torch.tensor(2.))
    b = pyro.param('b', torch.tensor(2.))
    pyro.sample('p', pdist.Beta(a, b))
  return guide


def test_variational_fit(beta_bernoulli, beta_bernoulli_guide, bernoulli_data):
  from getorix.inference import variational_fit
  pyro.clear_param_store()
  _, params = variational_fit(beta_bernoulli,
      beta_bernoulli_guide, bernoulli_data,
      num_epochs=10000, lr=0.01)
  # a, b = params['a'], params['b']


@pytest.fixture
def uniform_bernoulli():
  def model(data):
    p = pyro.sample('p', pdist.Uniform(0., 1.))  # `p` centered at 0.5
    with pyro.plate('observe_data'):
      pyro.sample('obs', pdist.Bernoulli(p), obs=data)
  return model


def test_auto_variational_fit(uniform_bernoulli, bernoulli_data):
  from getorix.inference import auto_variational_fit
  _, posterior = auto_variational_fit(
      uniform_bernoulli, bernoulli_data, num_epochs=1000)
  assert posterior.sample().size() == (1, )
