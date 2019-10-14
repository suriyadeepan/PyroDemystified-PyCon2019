import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import pandas as pd

import torchvision.transforms.functional as TF
from PIL import Image
import os

DATA = 'data/'


def mnist(batch_size=128, one_hot=False):
    root = 'data/'
    download = True
    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans,
                           download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
        batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def anscombe(group=2):
  df_ans = pd.read_csv('data/anscombe.csv')
  groups = [ 'I', 'II', 'III', 'IV' ]
  df = df_ans[df_ans.group == groups[group - 1]]
  df.x = df.x - df.x.mean()
  return torch.stack(
      [ torch.tensor(df.x.values).float(), torch.tensor(df.y.values).float() ]
      )


def read_image_from_file(filepath):
  image = Image.open(filepath)
  x = TF.to_tensor(image)
  return x


def notmnist():
  notmnist_dir = os.path.join(DATA, 'notmnist')
  return torch.stack([
    read_image_from_file(os.path.join(notmnist_dir, f)).squeeze()
    for f in os.listdir(notmnist_dir) if '.png' in f ])


def bap_lr():
  df = pd.read_csv('data/bap_linear_regression.csv')
  return torch.stack(
      [ torch.tensor(df.x).float(), torch.tensor(df.y).float() ]
      )
