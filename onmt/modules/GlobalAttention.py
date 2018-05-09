"""
Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.


        H_1 H_2 H_3 ... H_n
          q   q   q       q
            |  |   |       |
              \ |   |      /
                      .....
                  \   |  /
                          a

Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.

    The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:

"""

import torch
import torch.nn as nn
import torch.nn.init


class GlobalAttention(nn.Module):

  def __init__(self, dim, opt):
    super(GlobalAttention, self).__init__()
    self.sm = nn.Softmax()
    self.linear_out = nn.Linear(dim * 2, dim, bias=False)
    self.tanh = nn.Tanh()
    self.mask = None
    self.att_type = opt.att_type if (opt is not None) and (
        'att_type' in opt) else 'bilinear'
    if self.att_type == 'bilinear':
      self.linear_in = nn.Linear(dim, dim, bias=False)
      torch.nn.init.eye(self.linear_in.weight.data)

  def applyMask(self, mask):
    self.mask = mask

  def forward(self, input, context):
    """
    input: batch x dim
    context: batch x sourceL x dim
    """

    if self.att_type == 'mlp':
      raise NotImplementedError
    else:
      if self.att_type == 'bilinear':
        targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1
      elif self.att_type == 'dot':
        targetT = input.unsqueeze(2)  # batch x dim x 1
      # Get attention
      attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL

    if self.mask is not None:
      attn.data.masked_fill_(self.mask, -float('inf'))
    attn = self.sm(attn)
    self.cache_attn = attn.data.clone()
    attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

    weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
    self.cache_weightedContext = weightedContext.data.clone()
    contextCombined = torch.cat((weightedContext, input), 1)

    contextOutput = self.tanh(self.linear_out(contextCombined))

    return contextOutput, attn
