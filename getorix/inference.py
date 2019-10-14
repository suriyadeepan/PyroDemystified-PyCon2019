from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.infer.mcmc import NUTS, MCMC
from pyro.infer import Trace_ELBO
import pyro

from tqdm import tqdm


def variational_fit(model, guide, data, num_epochs=2500, lr=0.001):
  """ Use Stochastic Variational Inference for inferring latent variables """
  svi = pyro.infer.SVI(model=model,
                   guide=guide,
                   optim=pyro.optim.Adam({'lr' : lr}),
                   loss=Trace_ELBO())
  losses = []
  for i in tqdm(range(num_epochs)):
    losses.append(svi.step(data))

  return losses, dict(pyro.get_param_store())


def auto_variational_fit(model, data, num_epochs=2500, lr=0.001):
  """ Use Stochastic Variational Inference for inferring latent variables """
  guide = AutoMultivariateNormal(model)
  svi = pyro.infer.SVI(model=model,
                   guide=guide,
                   optim=pyro.optim.Adam({'lr' : lr}),
                   loss=Trace_ELBO())
  losses = []
  for i in tqdm(range(num_epochs)):
    losses.append(svi.step(data))

  return losses, guide.get_posterior()


def nuts_fit(model, data, num_samples=None, warmup_steps=None,
    var_names=None):
  """ Use No U-turn Sampler for inferring latent variables """
  num_samples = int(0.8 * len(data)) if num_samples is None else num_samples
  warmup_steps = len(data) - num_samples if warmup_steps is None else warmup_steps
  nuts_kernel = NUTS(model, adapt_step_size=True)
  trace = MCMC(nuts_kernel, num_samples=num_samples,
      warmup_steps=warmup_steps).run(data)
  if var_names is None:
    posteriors = None
  else:
    posteriors = trace.marginal(var_names).empirical

  return trace, posteriors
