import torch


def randint(shape, low=0, high=100):
  return torch.zeros(*shape).random_(low, high).int()
