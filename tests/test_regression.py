import getorix.data as getdata
import getorix.regression as getreg
import getorix.inference as getinf

# import torch
import pyro


def test_linear():
  data = getdata.bap_lr()
  pyro.clear_param_store()
  trace, posterior = getinf.nuts_fit(
      getreg.linear, data, num_samples=100, warmup_steps=30)


def test_polynomial():
  data = getdata.anscombe()
  pyro.clear_param_store()
  losses, posterior = getinf.auto_variational_fit(
      getreg.polynomial, data, num_epochs=1000)
