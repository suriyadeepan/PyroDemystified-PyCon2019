import pyro.distributions as pdist
import pyro

from torch.distributions import constraints
import torch


def linear(data):
  x, y = torch.unbind(data)

  alpha = pyro.sample('alpha', pdist.Normal(0., 1.))
  beta = pyro.sample('beta', pdist.Normal(1., 1.))
  epsilon = pyro.sample('epsilon', pdist.HalfCauchy(10.))

  mu = (alpha + beta * x)

  with pyro.plate('data'):
    pyro.sample('obs', pdist.Normal(mu, epsilon), obs=y)


def linear_guide(data):
  x, y = torch.unbind(data)
  loc_alpha = pyro.param('loc_alpha', torch.tensor(0.))
  scale_alpha = pyro.param('scale_alpha', torch.tensor(1.),
      constraint=constraints.positive)
  loc_beta = pyro.param('loc_beta', torch.tensor(1.))
  scale_beta = pyro.param('scale_beta', torch.tensor(10.),
      constraint=constraints.positive)
  scale_epsilon = pyro.param('scale_epsilon', torch.tensor(1.),
      constraint=constraints.positive)
  alpha = pyro.sample('alpha', pdist.Normal(loc_alpha, scale_alpha))
  beta = pyro.sample('beta', pdist.Normal(loc_beta, scale_beta))
  epsilon = pyro.sample('epsilon', pdist.HalfCauchy(scale_epsilon))


def polynomial(data):
  x, y = torch.unbind(data)
  alpha = pyro.sample("alpha", pdist.Normal(y.mean(), 1))
  beta_1 = pyro.sample("beta_1", pdist.Normal(2., 1.))
  beta_2 = pyro.sample("beta_2", pdist.Normal(2., 1.))
  epsilon = pyro.sample("epsilon", pdist.HalfCauchy(5.))
  mu = alpha + beta_1 * x + beta_2 * x ** 2
  with pyro.plate('data'):
    pyro.sample("y", pdist.Normal(mu, epsilon), obs=y)
