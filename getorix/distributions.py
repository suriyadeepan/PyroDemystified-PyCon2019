import pyro.distributions as pdist


def binomial(n, p):
  """ Number of successes in `n` trials """
  # trials are independent
  return (pdist.Bernoulli(p).sample([n, ]) == 1).sum()


def geometric(p, t=0):
  """ Number of failures until first success """
  x = pdist.Bernoulli(p).sample()
  if x == 1:
      return t
  else:
      return 1 + geometric(p, t + 1)
