from getorix.data import mnist
from getorix.models.vae import VAE
from getorix.models.cvae import CVAE
from getorix.models.bnn import BayesianNeuralNet
from getorix.models.blenet import BayesianLeNet5
# from getorix.models.blenet import LesserLeNet5
from getorix.models.blenet import MuchLesserLeNet5

import pyro

DATA = 'data/'


def bnn_pipeline():
  # set batch size
  batch_size = 512
  # get data
  data_loaders = mnist(batch_size=batch_size)
  # instantiate Bayesian Neural Network
  bnn = BayesianNeuralNet(784, 1024, 10)
  # fit model
  bnn.fit(data_loaders, num_epochs=6)
  # save to disk
  bnn.save()


def vae_pipeline():
  # set batch size
  batch_size = 512
  # get data
  train_loader, test_loader = mnist(batch_size=batch_size)
  # instantiate Vanilla VAE
  vae = VAE(z_dim=10, h_dim=400)
  # fit model
  vae.fit(train_loader, test_loader)


def cvae_pipeline():
  # set batch size
  batch_size = 512
  # get data
  train_loader, test_loader = mnist(batch_size=batch_size)
  # instantiate Vanilla VAE
  vae = CVAE(z_dim=10, h_dim=400)
  # fit model
  vae.fit(train_loader, test_loader)


def blenet_pipeline():
  # set batch size
  batch_size = 256
  # get data
  data_loaders = mnist(batch_size=batch_size)
  train_loader, test_loader = data_loaders
  # instantiate lenet
  # lenet_small = LesserLeNet5()
  lenet_tiny = MuchLesserLeNet5()
  # instantiate Bayesian LeNet
  blenet = BayesianLeNet5(lenet_tiny)
  # run test on .fit() method
  # blenet.test_fit()
  # fit model
  blenet.fit(data_loaders, num_epochs=20)
  accuracy = blenet.evaluate(test_loader)
  print(f'Accuracy : {accuracy}')


if __name__ == '__main__':
  # clean up param store
  pyro.clear_param_store()
  bnn_pipeline()
