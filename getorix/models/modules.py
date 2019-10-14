import torch.nn as nn
import torch.nn.functional as F
import torch


class LatentVariableEncoder(nn.Module):
  def __init__(self, i_dim, h_dim, o_dim):
    super(LatentVariableEncoder, self).__init__()
    self.linear1 = nn.Linear(i_dim, h_dim)
    self.linear21 = nn.Linear(h_dim, o_dim)
    self.linear22 = nn.Linear(h_dim, o_dim)
    self.softplus = nn.Softplus()

  def forward(self, input):
    h = self.softplus(self.linear1(input))
    loc_z = self.linear21(h)
    scale_z = torch.exp(self.linear22(h))
    return loc_z, scale_z


class PartialVariableEncoder(nn.Module):
  def __init__(self, i_dim, h_dim, o_dim):
    super(PartialVariableEncoder, self).__init__()
    self.linear1 = nn.Linear(i_dim, h_dim)
    self.linear2 = nn.Linear(h_dim, o_dim)
    self.softplus = nn.Softplus()
    self.softmax = nn.Softmax()

  def forward(self, input):
    h = self.softplus(self.linear1(input))
    logits = self.linear2(h)
    y = F.softmax(logits, dim=1)
    return y


class ConditionalDecoder(nn.Module):
  def __init__(self, i_dim, h_dim, o_dim):
    super(ConditionalDecoder, self).__init__()
    self.linear1 = nn.Linear(i_dim, h_dim)
    self.linear2 = nn.Linear(h_dim, o_dim)
    self.softplus = nn.Softplus()

  def forward(self, input):
    h = self.softplus(self.linear1(input))
    loc_img = torch.sigmoid(self.linear2(h))
    return loc_img
