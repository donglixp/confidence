import warnings
import torch
from torch.autograd import Function, NestedIOFunction, Variable
import torch.nn.functional as F


def FuncLinear(input, weight, bias=None):
  output = torch.mm(input, weight.t())
  if bias is not None:
    if input.dim() == 2:
      output += bias.expand_as(output)
    else:
      output += bias.unsqueeze(0).expand_as(output)
  return output


def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
  # save internal states in save_cell
  save_cell = {}
  save_cell['x'] = input.data.clone()

  hx, cx = hidden
  save_cell['h1'], save_cell['c1'] = hx.data.clone(), cx.data.clone()
  Wx, Wh1 = FuncLinear(input, w_ih, b_ih), FuncLinear(hx, w_hh, b_hh)
  gates = Wx + Wh1

  ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

  ingate = F.sigmoid(ingate)
  forgetgate = F.sigmoid(forgetgate)
  cellgate = F.tanh(cellgate)
  outgate = F.sigmoid(outgate)
  save_cell['i'], save_cell['f'], save_cell['g'], save_cell['o'] = ingate.data.clone(
  ), forgetgate.data.clone(), cellgate.data.clone(), outgate.data.clone()

  fc1, ig = (forgetgate * cx), (ingate * cellgate)
  save_cell['fc1'], save_cell['ig'] = fc1.data.clone(), ig.data.clone()
  cy = fc1 + ig
  save_cell['c'] = cy.data.clone()
  tanh_cy = F.tanh(cy)
  save_cell['tanh_c'] = tanh_cy.data.clone()
  hy = outgate * tanh_cy

  return (hy, cy), save_cell


def StackedRNN(inners, num_layers, lstm=True, dropout=0, train=False):
  num_directions = len(inners)
  total_layers = num_layers * num_directions

  def forward(input, hidden, weight):
    assert(len(weight) == total_layers)
    next_hidden = []

    hidden = list(zip(*hidden))

    save = []
    for i in range(num_layers):
      all_output = []
      for j, inner in enumerate(inners):
        l = i * num_directions + j

        hy, output, save_layer = inner(input, hidden[l], weight[l])
        next_hidden.append(hy)
        all_output.append(output)
        save.append(save_layer)

      input = torch.cat(all_output, input.dim() - 1)

      if dropout != 0 and i < num_layers - 1:
        input = F.dropout(input, p=dropout, training=train, inplace=False)

    next_h, next_c = zip(*next_hidden)
    next_hidden = (
        torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
        torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
    )

    return next_hidden, input, save

  return forward


def variable_recurrent_factory(batch_sizes):
  def fac(inner, reverse=False):
    if reverse:
      return VariableRecurrentReverse(batch_sizes, inner)
    else:
      return VariableRecurrent(batch_sizes, inner)
  return fac


def VariableRecurrent(batch_sizes, inner):
  def forward(input, hidden, weight):
    save_layer = []
    output = []
    input_offset = 0
    last_batch_size = batch_sizes[0]
    hiddens = []
    flat_hidden = not isinstance(hidden, tuple)
    if flat_hidden:
      hidden = (hidden,)
    for batch_size in batch_sizes:
      step_input = input[input_offset:input_offset + batch_size]
      input_offset += batch_size

      dec = last_batch_size - batch_size
      if dec > 0:
        hiddens.append(tuple(h[-dec:] for h in hidden))
        hidden = tuple(h[:-dec] for h in hidden)
      last_batch_size = batch_size

      if flat_hidden:
        hidden, save_cell = inner(step_input, hidden[0], *weight)
        hidden = (hidden,)
      else:
        hidden, save_cell = inner(step_input, hidden, *weight)

      output.append(hidden[0])
      save_layer.append(save_cell)
    hiddens.append(hidden)
    hiddens.reverse()

    hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
    assert hidden[0].size(0) == batch_sizes[0]
    if flat_hidden:
      hidden = hidden[0]
    output = torch.cat(output, 0)

    return hidden, output, save_layer

  return forward


def VariableRecurrentReverse(batch_sizes, inner):
  def forward(input, hidden, weight):
    output = []
    input_offset = input.size(0)
    last_batch_size = batch_sizes[-1]
    initial_hidden = hidden
    flat_hidden = not isinstance(hidden, tuple)
    if flat_hidden:
      hidden = (hidden,)
      initial_hidden = (initial_hidden,)
    hidden = tuple(h[:batch_sizes[-1]] for h in hidden)
    for batch_size in reversed(batch_sizes):
      inc = batch_size - last_batch_size
      if inc > 0:
        hidden = tuple(torch.cat((h, ih[last_batch_size:batch_size]), 0)
                       for h, ih in zip(hidden, initial_hidden))
      last_batch_size = batch_size
      step_input = input[input_offset - batch_size:input_offset]
      input_offset -= batch_size

      if flat_hidden:
        hidden = (inner(step_input, hidden[0], *weight),)
      else:
        hidden = inner(step_input, hidden, *weight)
      output.append(hidden[0])

    output.reverse()
    output = torch.cat(output, 0)
    if flat_hidden:
      hidden = hidden[0]
    return hidden, output

  return forward


def Recurrent(inner, reverse=False):
  def forward(input, hidden, weight):
    save_layer = []
    output = []
    steps = range(input.size(0) - 1, -1, -
                  1) if reverse else range(input.size(0))
    for i in steps:
      hidden, save_cell = inner(input[i], hidden, *weight)
      # hack to handle LSTM
      output.append(hidden[0] if isinstance(hidden, tuple) else hidden)
      save_layer.append(save_cell)

    if reverse:
      output.reverse()
    output = torch.cat(output, 0).view(input.size(0), *output[0].size())

    return hidden, output, save_layer

  return forward


def AutogradRNN(mode, input_size, hidden_size, num_layers=1, batch_first=False,
                dropout=0, train=True, bidirectional=False, batch_sizes=None,
                dropout_state=None, flat_weight=None):
  assert mode == 'LSTM'
  cell = LSTMCell

  if batch_sizes is None:
    rec_factory = Recurrent
  else:
    rec_factory = variable_recurrent_factory(batch_sizes)

  if bidirectional:
    layer = (rec_factory(cell), rec_factory(cell, reverse=True))
  else:
    layer = (rec_factory(cell),)

  func = StackedRNN(layer,
                    num_layers,
                    (mode == 'LSTM'),
                    dropout=dropout,
                    train=train)

  def forward(input, weight, hx=None):
    if batch_first and batch_sizes is None:
      input = input.transpose(0, 1)

    if hx is None:
      num_directions = 2 if bidirectional else 1
      max_batch_size = batch_sizes[0]
      hx = torch.autograd.Variable(input.data.new(
          num_layers * num_directions, max_batch_size, hidden_size).zero_())
      hx = (hx, hx)

    nexth, output, save = func(input, hx, weight)

    if batch_first and batch_sizes is None:
      output = output.transpose(0, 1)

    return output, nexth, save

  return forward


def FuncRNN(*args, **kwargs):
  def forward(input, *fargs, **fkwargs):
    func = AutogradRNN(*args, **kwargs)
    return func(input, *fargs, **fkwargs)

  return forward
