"""

Sequential/Variational dropout used for the input embedding lookup layer.

Input: (seq_len, batch, input_size)

"""

import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd.function import InplaceFunction


class AddNoise(InplaceFunction):

  def __init__(self, noise_type, sigma, i_src=None):
    super(AddNoise, self).__init__()
    self.noise_type = noise_type
    self.sigma = sigma
    self.i_src = i_src

  # if i_src is not None, then only generate mask for the i_src word
  def forward(self, input):
    output = input.clone()

    if self.sigma > 0:
      if self.i_src is None:
        self.noise = input.new().resize_(1, input.size(1), input.size(2))
        if self.noise_type == 'mul':
          self.noise.normal_(1, self.sigma)
          self.noise = self.noise.expand_as(input)
          output.mul_(self.noise)
        elif self.noise_type == 'add':
          self.noise.normal_(0, self.sigma)
          self.noise = self.noise.expand_as(input)
          output.add_(self.noise)
      else:
        self.noise = input.new().resize_as_(input)
        if self.noise_type == 'mul':
          self.noise.fill_(1)
          self.noise[self.i_src].normal_(1, self.sigma)
          output.mul_(self.noise)
        elif self.noise_type == 'add':
          self.noise.fill_(0)
          self.noise[self.i_src].normal_(0, self.sigma)
          output.add_(self.noise)

    return output

  def backward(self, grad_output):
    if self.sigma > 0:
      raise NotImplementedError
      return grad_output
    else:
      return grad_output
