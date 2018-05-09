"""

Sequential/Variational dropout used for the input embedding lookup layer.

Input: (seq_len, batch, input_size)

"""

import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd.function import InplaceFunction


class SeqDropout(InplaceFunction):

  def __init__(self, p=0.5, train=False, i_src=None):
    super(SeqDropout, self).__init__()
    if p < 0 or p > 1:
      raise ValueError("dropout probability has to be between 0 and 1, "
                       "but got {}".format(p))
    self.p = p
    self.train = train
    self.i_src = i_src

  # if i_src is not None, then only generate mask for the i_src word
  def forward(self, input):
    output = input.clone()

    if self.p > 0 and self.train:
      if self.i_src is None:
        self.noise = input.new().resize_(1, input.size(1), input.size(2))
        if self.p == 1:
          self.noise.fill_(0)
        else:
          self.noise.bernoulli_(1 - self.p).div_(1 - self.p)
        self.noise = self.noise.expand_as(input)
      else:
        self.noise = input.new().resize_as_(input).fill_(1)
        if self.p == 1:
          self.noise.fill_(0)
        else:
          self.noise[self.i_src].bernoulli_(1 - self.p).div_(1 - self.p)
      output.mul_(self.noise)

    return output

  def backward(self, grad_output):
    if self.p > 0 and self.train:
      return grad_output.mul(self.noise)
    else:
      return grad_output
