import torch
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

DEFAULT_COLOR = 'skyblue'
COLOR_GREY = 'grey'
COLORS = [ 'cyan', 'skyblue', 'magenta', 'blue', 'red', 'green', 'grey' ]


def hist(t, nbins=10, highlight=None):
  bar_plot = plt.bar(range(nbins), t, color=COLOR_GREY)
  if highlight:
    bar_plot[highlight.item()].set_color(DEFAULT_COLOR)
  plt.show(bar_plot)


def image(t, dim=28, grayscale=True):
  plt.axis('off')
  assert isinstance(t, type(torch.tensor([0.2, 0.3])))
  if not t.size(-1) == t.size(-2):
    t = t.view(-1, dim, dim)
  if len(t.size()) == 3:
    t = t.squeeze()
  cmap = 'gray' if grayscale else None
  plt.imshow(t.detach(), cmap=cmap)


def make_grid(t, ncols, dim):
  if not t.size(-1) == t.size(-2):
    t = t.view(-1, dim, dim)
  num_images = t.size(0)
  nrows = num_images // ncols
  print('nrows', nrows)
  rows = []
  for i in range(nrows):
    row = torch.cat(torch.unbind(t[i * ncols : (i + 1) * ncols]), axis=1)
    rows.append(row)
  return torch.cat(rows, axis=0)


def images(t, ncols, dim=28):
  plt.figure(figsize=(7, 7))
  plt.axis('off')
  plt.imshow(make_grid(t, ncols, dim=dim).detach(), cmap='gray')


def scatter(tx, ty, c=None):
  assert tx.size() == ty.size()
  assert len(tx.size()) == 1
  c = DEFAULT_COLOR if c is None else c
  plt.scatter(tx.detach(), ty.detach(), c=c)


def plot(t):
  if isinstance(t, type(torch.tensor([0.6, 0.9]))):
    t = t.detach()
  plt.plot(t, c=DEFAULT_COLOR)


def posterior(p, n=1000):
  posterior_samples = p.sample([n, ])
  if len(posterior_samples.size()) == 2:
    posterior_samples = posterior_samples.transpose(1, 0)
  else:
    posterior_samples = posterior_samples.view(1, -1)

  for i, samples in enumerate(posterior_samples):
    sns.distplot(samples, color=COLORS[-i])
    plt.figure()
