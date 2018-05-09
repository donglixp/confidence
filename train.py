from __future__ import division
import os
import onmt
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import json as js
from tensorboard_logger import Logger
from path import Path

parser = argparse.ArgumentParser(description='train.py')

# Data options
parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_dir', default='model',
                    help="Model save dir")

parser.add_argument('-train_from_state_dict', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model's state_dict.""")
parser.add_argument('-train_from', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")

# Model options
parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=200,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=200,
                    help='Word embedding sizes')
parser.add_argument('-input_feed', action='store_true',
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")
parser.add_argument('-att_type', default='dot',
                    help="""Attention type: [dot|bilinear|mlp]""")

# Optimization options
parser.add_argument('-encoder_type', default='text',
                    help="Type of encoder to use. Options are [text|img].")
parser.add_argument('-batch_size', type=int, default=32,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=60,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-optim', default='rmsprop',
                    help="Optimization method. [sgd|adagrad|adadelta|adam|rmsprop]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.4,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-dropout_in', action="store_true",
                    help='Apply dropout for word vectors.')
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")

# learning rate
parser.add_argument('-learning_rate', type=float, default=0.01,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1,
                    adadelta = 1, adam = 0.001""")
parser.add_argument('-alpha', type=float, default=0.95,
                    help="Optimization hyperparameter")
parser.add_argument('-learning_rate_decay', type=float, default=0.98,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=5,
                    help="""Start decaying every epoch after and including this epoch""")

# pretrained word vectors
parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")

# GPU
parser.add_argument('-gpus', default=[0], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")

parser.add_argument('-log_name', default='',
                    help="Name for tensorboard logging.")


def get_save_index(save_dir):
  save_index = 0
  while True:
    if Path(os.path.join(save_dir, 'run.%d' % (save_index,))).exists():
      save_index += 1
    else:
      break
  return save_index

opt = parser.parse_args()
opt.save_path = os.path.join(opt.save_dir, 'run.%d' %
                             (get_save_index(opt.save_dir),))
Path(opt.save_path).mkdir_p()
print(opt.save_path)
torch.manual_seed(123)
cuda.set_device(opt.gpus[0])
cuda.manual_seed(123)

print(opt)
js.dump(opt.__dict__, open(os.path.join(
    opt.save_path, 'opt.json'), 'w'), sort_keys=True, indent=2)


def NMTCriterion(vocabSize):
  weight = torch.ones(vocabSize)
  weight[onmt.Constants.PAD] = 0
  crit = nn.NLLLoss(weight, size_average=False)
  if opt.gpus:
    crit.cuda()
  return crit


def memoryEfficientLoss(outputs, targets, generator, crit, eval=False):
  # compute generations one piece at a time
  num_correct, loss = 0, 0
  outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

  batch_size = outputs.size(1)
  outputs_split = torch.split(outputs, opt.max_generator_batches)
  targets_split = torch.split(targets, opt.max_generator_batches)
  for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
    out_t = out_t.view(-1, out_t.size(2))
    scores_t = generator(out_t)
    loss_t = crit(scores_t, targ_t.view(-1))
    pred_t = scores_t.max(1)[1]
    num_correct_t = pred_t.data.eq(targ_t.data) \
                               .masked_select(
                                   targ_t.ne(onmt.Constants.PAD).data) \
                               .sum()
    num_correct += num_correct_t
    loss += loss_t.data[0]
    if not eval:
      loss_t.div(batch_size).backward()

  grad_output = None if outputs.grad is None else outputs.grad.data
  return loss, grad_output, num_correct


def eval(model, criterion, data):
  total_loss = 0
  total_words = 0
  total_num_correct = 0

  model.eval()
  for i in range(len(data)):
    # exclude original indices
    batch = data[i][:-1]
    outputs = model(batch)
    # exclude <s> from targets
    targets = batch[1][1:]
    loss, _, num_correct = memoryEfficientLoss(
        outputs, targets, model.generator, criterion, eval=True)
    total_loss += loss
    total_num_correct += num_correct
    total_words += targets.data.ne(onmt.Constants.PAD).sum()

  model.train()
  return total_loss / total_words, total_num_correct / total_words


def trainModel(model, trainData, validData, dataset, optim):
  logger = Logger(os.path.join(opt.save_path, 'tb'))
  iterations = 0

  print(model)
  model.train()
  # Define criterion of each GPU.
  criterion = NMTCriterion(dataset['dicts']['tgt'].size())
  start_time = time.time()

  for epoch in range(opt.start_epoch, opt.epochs + 1):
    print('')

    #  (1) train for one epoch on the training set
    if opt.extra_shuffle and epoch > opt.curriculum:
      trainData.shuffle()
    # Shuffle mini batch order.
    batchOrder = torch.randperm(len(trainData))

    total_loss, total_words, total_num_correct = 0, 0, 0
    report_loss, report_tgt_words = 0, 0
    report_src_words, report_num_correct = 0, 0
    start = time.time()
    for i in range(len(trainData)):
      iterations += 1

      batchIdx = batchOrder[i] if epoch > opt.curriculum else i
      # Exclude original indices.
      batch = trainData[batchIdx][:-1]

      model.zero_grad()
      outputs = model(batch)
      # Exclude <s> from targets.
      targets = batch[1][1:]
      loss, gradOutput, num_correct = memoryEfficientLoss(
          outputs, targets, model.generator, criterion)

      outputs.backward(gradOutput)

      # Update the parameters.
      optim.step()

      num_words = targets.data.ne(onmt.Constants.PAD).sum()
      report_loss += loss
      report_num_correct += num_correct
      report_tgt_words += num_words
      report_src_words += batch[0][1].data.sum()
      total_loss += loss
      total_num_correct += num_correct
      total_words += num_words
      if iterations % opt.log_interval == -1 % opt.log_interval:
        print(("Epoch %d, %d/%d; acc: %.2f; ppl: %.2f; %.0f src tok/s; %.0f tgt tok/s; %.0fs elapsed") %
              (epoch, i + 1, len(trainData),
               report_num_correct / report_tgt_words * 100.0,
               math.exp(report_loss / report_tgt_words),
               report_src_words / (time.time() - start),
               report_tgt_words / (time.time() - start),
               time.time() - start_time))
        # log to tensorboard
        logger.log_value("word_acc", float(report_num_correct) /
                         float(report_tgt_words), step=iterations)
        logger.log_value("ppl", math.exp(
            report_loss / report_tgt_words), step=iterations)

        report_loss, report_tgt_words = 0, 0
        report_src_words, report_num_correct = 0, 0
        start = time.time()
    train_loss, train_acc = total_loss / total_words, total_num_correct / total_words

    train_ppl = math.exp(min(train_loss, 100))
    print('Train perplexity: %g' % train_ppl)
    print('Train word accuracy: %g' % (train_acc * 100))

    #  (2) evaluate on the validation set
    valid_loss, valid_acc = eval(model, criterion, validData)
    valid_ppl = math.exp(min(valid_loss, 100))
    print('Validation perplexity: %g' % valid_ppl)
    print('Validation word accuracy: %g' % (valid_acc * 100))

    #  (3) update the learning rate
    optim.updateLearningRate(valid_ppl, epoch)

    model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                        else model.state_dict())
    model_state_dict = {k: v for k, v in model_state_dict.items()
                        if 'generator' not in k}
    generator_state_dict = (model.generator.module.state_dict()
                            if len(opt.gpus) > 1
                            else model.generator.state_dict())
    #  (4) drop a checkpoint
    checkpoint = {
        'model': model_state_dict,
        'generator': generator_state_dict,
        'dicts': dataset['dicts'],
        'opt': opt,
        'epoch': epoch,
        'optim': optim
    }
    if epoch % 5 == 0:
      torch.save(checkpoint, os.path.join(
          opt.save_path, 'm_%d_acc_%.2f.pt' % (epoch, 100.0 * valid_acc)))


def main():
  print("Loading data from '%s'" % opt.data)

  dataset = torch.load(opt.data)
  dict_checkpoint = (opt.train_from if opt.train_from
                     else opt.train_from_state_dict)
  if dict_checkpoint:
    print('Loading dicts from checkpoint at %s' % dict_checkpoint)
    checkpoint = torch.load(dict_checkpoint)
    dataset['dicts'] = checkpoint['dicts']

  trainData = onmt.Dataset(dataset['train']['src'],
                           dataset['train']['tgt'], opt.batch_size, opt.gpus,
                           data_type=dataset.get("type", "text"))
  validData = onmt.Dataset(dataset['valid']['src'],
                           dataset['valid']['tgt'], opt.batch_size, opt.gpus,
                           volatile=True,
                           data_type=dataset.get("type", "text"))

  dicts = dataset['dicts']
  print(' * vocabulary size. source = %d; target = %d' %
        (dicts['src'].size(), dicts['tgt'].size()))
  print(' * number of training sentences. %d' %
        len(dataset['train']['src']))
  print(' * maximum batch size. %d' % opt.batch_size)

  print('Building model...')

  if opt.encoder_type == "text":
    encoder = onmt.Models.Encoder(opt, dicts['src'])
  else:
    print("Unsupported encoder type %s" % (opt.encoder_type))

  decoder = onmt.Models.Decoder(opt, dicts['tgt'])

  generator = nn.Sequential(
      nn.Linear(opt.rnn_size, dicts['tgt'].size()),
      nn.LogSoftmax())

  model = onmt.Models.NMTModel(encoder, decoder)

  if opt.train_from:
    print('Loading model from checkpoint at %s' % opt.train_from)
    chk_model = checkpoint['model']
    generator_state_dict = chk_model.generator.state_dict()
    model_state_dict = {k: v for k, v in chk_model.state_dict().items()
                        if 'generator' not in k}
    model.load_state_dict(model_state_dict)
    generator.load_state_dict(generator_state_dict)
    opt.start_epoch = checkpoint['epoch'] + 1

  if opt.train_from_state_dict:
    print('Loading model from checkpoint at %s'
          % opt.train_from_state_dict)
    model.load_state_dict(checkpoint['model'])
    generator.load_state_dict(checkpoint['generator'])
    opt.start_epoch = checkpoint['epoch'] + 1

  if len(opt.gpus) >= 1:
    model.cuda()
    generator.cuda()
  else:
    model.cpu()
    generator.cpu()

  if len(opt.gpus) > 1:
    model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
    generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)

  model.generator = generator

  if not opt.train_from_state_dict and not opt.train_from:
    # for p in model.parameters():
    #   p.data.uniform_(-0.08, 0.08)
    encoder.load_pretrained_vectors(opt)
    decoder.load_pretrained_vectors(opt)

    optim = onmt.Optim(
        opt.optim, opt.learning_rate, opt.alpha, opt.max_grad_norm,
        lr_decay=opt.learning_rate_decay,
        start_decay_at=opt.start_decay_at
    )
  else:
    print('Loading optimizer from checkpoint:')
    optim = checkpoint['optim']
    print(optim)

  optim.set_parameters(model.parameters())

  if opt.train_from or opt.train_from_state_dict:
    optim.optimizer.load_state_dict(
        checkpoint['optim'].optimizer.state_dict())

  nParams = sum([p.nelement() for p in model.parameters()])
  print('* number of parameters: %d' % nParams)

  trainModel(model, trainData, validData, dataset, optim)


if __name__ == "__main__":
  main()
