
def test_binomial():
  from getorix.distributions import binomial
  assert binomial(1000, 0.1) < binomial(1000, 0.5) < binomial(1000, 0.9)


def test_geometric():
  from getorix.distributions import geometric
  assert geometric(0.01) > geometric(0.99)
