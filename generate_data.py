import numpy as np
import pandas as pd
import os

DATA = 'data/'


def bap_linear_regression():
  np.random.seed(1)
  N = 100
  alpha_real = 2.5
  beta_real = 0.9
  eps_real = np.random.normal(0, 0.5, size=N)

  x = np.random.normal(10, 1, N)
  y_real = alpha_real + beta_real * x
  y = y_real + eps_real

  return pd.DataFrame({ 'x' : x, 'y' : y })


def bap_linear_regression_simple():
  np.random.seed(69)
  N = 100
  alpha_real = 2.5
  beta_real = 0.9
  # eps_real = np.random.normal(0, 0.5, size=N)

  x = np.random.normal(10, 1, N)
  y_real = alpha_real + beta_real * x
  # y = y_real + eps_real
  y = y_real

  return pd.DataFrame({ 'x' : x, 'y' : y })


__gen__ = {
    'bap_linear_regression' : bap_linear_regression,
    'bap_linear_regression_simple' : bap_linear_regression_simple,
    }

for name, fn in __gen__.items():
  path = os.path.join(DATA, f'{name}.csv')
  fn().to_csv(path)
  print(path)
