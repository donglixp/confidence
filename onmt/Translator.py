import math
import onmt
import torch.nn as nn
import torch
from itertools import count
from torch.autograd import Variable
import torch.nn.functional as F
import onmt.modules as mod
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from utils.utils import add_eps, add_eps_, num_nan, trim_tensor_by_length


def get_noise_type(confidence_method_split):
    if 'add' in confidence_method_split:
        return 'add'
    elif 'mul' in confidence_method_split:
        return 'mul'


def recover_order(t, indices):
    return list(zip(*sorted(zip(t, indices), key=lambda x: x[-1])))[0]


def _var(t):
    # numerically stable
    return t.var(0)
    return t.pow(2).mean(0) - t.mean(0).pow(2)


class Translator(object):

    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch
        self.beam_accum = None

        checkpoint = torch.load(opt.model)

        model_opt = checkpoint['opt']
        self.src_dict = checkpoint['dicts']['src']
        self.tgt_dict = checkpoint['dicts']['tgt']
        self._type = model_opt.encoder_type \
            if "encoder_type" in model_opt else "text"

        encoder = onmt.Models.Encoder(model_opt, self.src_dict)

        decoder = onmt.Models.Decoder(model_opt, self.tgt_dict)
        model = onmt.Models.NMTModel(encoder, decoder)

        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, self.tgt_dict.size()),
            nn.LogSoftmax())

        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])

        if opt.cuda:
            model.cuda()
            generator.cuda()
        else:
            model.cpu()
            generator.cpu()

        model.generator = generator

        self.model = model
        self.model.eval()

    def reset_dropout_rate(self, dropout_rate):
        def _reset_dropout_rate(m):
            if isinstance(m, nn.Dropout):
                m.p = dropout_rate
            elif isinstance(m, onmt.Models.Encoder) or isinstance(m, onmt.Models.Decoder):
                m.dropout = dropout_rate
        self.model.apply(_reset_dropout_rate)

    def initBeamAccum(self):
        self.beam_accum = {
            "predicted_ids": [],
            "beam_parent_ids": [],
            "scores": [],
            "log_probs": []}

    def _getBatchSize(self, batch):
        return batch.size(1)

    def buildData(self, srcBatch, goldBatch):
        # This needs to be the same as preprocess.py.
        srcData = [self.src_dict.convertToIdx(
            b, onmt.Constants.UNK_WORD) for b in srcBatch]

        tgtData = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                                                  onmt.Constants.UNK_WORD,
                                                  onmt.Constants.BOS_WORD,
                                                  onmt.Constants.EOS_WORD) for b in goldBatch]

        return onmt.Dataset(srcData, tgtData, self.opt.batch_size,
                            self.opt.cuda, volatile=True,
                            data_type=self._type)

    def buildTargetTokens(self, pred, src, attn):
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == onmt.Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex[0]]
        return tokens

    def translateBatch(self, srcBatch, tgtBatch):
        beamSize = self.opt.beam_size

        #  (1) run the encoder on the src
        encStates, context = self.model.encoder(srcBatch)

        # Drop the lengths needed for encoder.
        srcBatch = srcBatch[0]
        batchSize = self._getBatchSize(srcBatch)

        rnnSize = context.size(2)
        encStates = (self.model._fix_enc_hidden(encStates[0]),
                     self.model._fix_enc_hidden(encStates[1]))

        decoder = self.model.decoder
        attentionLayer = decoder.attn

        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding
        padMask = srcBatch.data.eq(onmt.Constants.PAD).t()

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        goldScores = context.data.new(batchSize).zero_()
        if tgtBatch is not None:
            decStates = encStates
            decOut = self.model.make_init_decoder_output(context)
            attentionLayer.applyMask(padMask)
            initOutput = self.model.make_init_decoder_output(context)
            decOut, decStates, attn = self.model.decoder(
                tgtBatch[:-1], decStates, context, initOutput)
            for dec_t, tgt_t in zip(decOut, tgtBatch[1:].data):
                gen_t = self.model.generator.forward(dec_t)
                tgt_t = tgt_t.unsqueeze(1)
                scores = gen_t.data.gather(1, tgt_t)
                scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
                goldScores += scores

        #  (3) run the decoder to generate sentences, using beam search
        # Expand tensors for each beam.
        context = Variable(context.data.repeat(1, beamSize, 1))

        decStates = (Variable(encStates[0].data.repeat(1, beamSize, 1)),
                     Variable(encStates[1].data.repeat(1, beamSize, 1)))

        beam = [onmt.Beam(beamSize, self.opt.cuda) for k in range(batchSize)]

        decOut = self.model.make_init_decoder_output(context)

        padMask = srcBatch.data.eq(
            onmt.Constants.PAD).t() \
                               .unsqueeze(0) \
                               .repeat(beamSize, 1, 1)

        batchIdx = list(range(batchSize))
        remainingSents = batchSize
        for i in range(self.opt.max_sent_length):
            attentionLayer.applyMask(padMask)
            # Prepare decoder input.
            input = torch.stack([b.getCurrentState() for b in beam
                                 if not b.done]).t().contiguous().view(1, -1)
            decOut, decStates, attn = self.model.decoder(
                Variable(input, volatile=True), decStates, context, decOut)
            # decOut: 1 x (beam*batch) x numWords
            decOut = decOut.squeeze(0)
            out = self.model.generator.forward(decOut)

            # batch x beam x numWords
            wordLk = out.view(beamSize, remainingSents, -1) \
                        .transpose(0, 1).contiguous()
            attn = attn.view(beamSize, remainingSents, -1) \
                       .transpose(0, 1).contiguous()

            active = []
            for b in range(batchSize):
                if beam[b].done:
                    continue

                idx = batchIdx[b]
                if not beam[b].advance(wordLk.data[idx], attn.data[idx]):
                    active += [b]

                for decState in decStates:  # iterate over h, c
                    # layers x beam*sent x dim
                    sentStates = decState.view(-1, beamSize,
                                               remainingSents,
                                               decState.size(2))[:, :, idx]
                    sentStates.data.copy_(
                        sentStates.data.index_select(
                            1, beam[b].getCurrentOrigin()))

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return Variable(view.index_select(1, activeIdx)
                                .view(*newSize), volatile=True)

            decStates = (updateActive(decStates[0]),
                         updateActive(decStates[1]))
            decOut = updateActive(decOut)
            context = updateActive(context)
            padMask = padMask.index_select(1, activeIdx)

            remainingSents = len(active)

        #  (4) package everything up
        allHyp, allScores, allAttn = [], [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            hyps, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            allHyp += [hyps]
            valid_attn = srcBatch.data[:, b].ne(
                onmt.Constants.PAD).nonzero().squeeze(1)
            attn = [a.index_select(1, valid_attn) for a in attn]
            allAttn += [attn]

            if self.beam_accum:
                self.beam_accum["beam_parent_ids"].append(
                    [t.tolist()
                     for t in beam[b].prevKs])
                self.beam_accum["scores"].append([
                    ["%4f" % s for s in t.tolist()]
                    for t in beam[b].allScores][1:])
                self.beam_accum["predicted_ids"].append(
                    [[self.tgt_dict.getLabel(id)
                      for id in t.tolist()]
                     for t in beam[b].nextYs][1:])

        return allHyp, allScores, allAttn, goldScores

    def translate(self, srcBatch, goldBatch):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        src, tgt, indices = dataset[0]
        batchSize = self._getBatchSize(src[0])

        #  (2) translate
        pred, predScore, attn, goldScore = self.translateBatch(src, tgt)
        pred, predScore, attn, goldScore = list(zip(
            *sorted(zip(pred, predScore, attn, goldScore, indices),
                    key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        for b in range(batchSize):
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n])
                 for n in range(self.opt.n_best)]
            )

        return predBatch, predScore, goldScore, attn

    def conf_encode(self, srcBatch):
        #  (1) run the encoder on the src
        encStates, context = self.model.encoder(srcBatch)
        rnnSize = context.size(2)
        encStates = (self.model._fix_enc_hidden(encStates[0]),
                     self.model._fix_enc_hidden(encStates[1]))
        return encStates, context, rnnSize

    def drp_conf_once(self, srcBatch, tgtBatch, confidence_method, opt):
        confidence_method_split = confidence_method.split(':')
        #  (1) run the encoder on the src
        encStates, context, rnnSize = self.conf_encode(srcBatch)

        if 'bridge' in confidence_method_split:
            if 'drp' in confidence_method_split:
                # apply dropout for the encoding vector
                encStates = (F.dropout(encStates[0], opt.dropout_rate, True),
                             F.dropout(encStates[1], opt.dropout_rate, True))
            elif 'noise' in confidence_method_split:
                # add noise for the encoding vector
                noise_type = get_noise_type(confidence_method_split)
                encStates = (mod.AddNoise(noise_type, opt.noise_sigma)(encStates[0]),
                             mod.AddNoise(noise_type, opt.noise_sigma)(encStates[1]))

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        batchSize = self._getBatchSize(srcBatch[0])
        p_infer = []
        decStates = encStates
        decOut = self.model.make_init_decoder_output(context)
        self.model.decoder.attn.applyMask(
            srcBatch[0].data.eq(onmt.Constants.PAD).t())
        initOutput = self.model.make_init_decoder_output(context)
        decOut, decStates, attn = self.model.decoder(
            tgtBatch[:-1], decStates, context, initOutput)
        if ('classifier' in confidence_method_split) and ('noise' in confidence_method_split):
            # add noise for the classifier in decoder
            noise_type = get_noise_type(confidence_method_split)
            decOut = mod.AddNoise(noise_type, opt.noise_sigma)(decOut)
        for dec_t, tgt_t in zip(decOut, tgtBatch[1:].data):
            if ('classifier' in confidence_method_split) and ('drp' in confidence_method_split):
                # apply dropout for the classifier in decoder
                dec_t = F.dropout(dec_t, opt.dropout_rate, True)
            gen_t = self.model.generator.forward(dec_t)
            tgt_t = tgt_t.unsqueeze(1)
            if ('ent' in confidence_method_split) and ('forced_dec' in confidence_method_split):
                scores = gen_t.data.double().mul(gen_t.data.double().exp_()).sum(1)
            else:
                scores = gen_t.data.gather(1, tgt_t)
            scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
            p_infer.append(scores.view(-1))
        # -> (seq_len * batch_size)
        p_infer = torch.stack(p_infer)

        return p_infer

    def ent_conf_sample_once(self, srcBatch):
        #  (1) run the encoder on the src
        encStates, context, rnnSize = self.conf_encode(srcBatch)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        batchSize = self._getBatchSize(srcBatch[0])
        p_infer = context.data.new(batchSize).zero_()
        decStates = encStates
        decOut = self.model.make_init_decoder_output(context)
        input = self.tt.LongTensor(1, batchSize).fill_(onmt.Constants.BOS)
        eos_mask = self.tt.ByteTensor(batchSize).fill_(0)
        for i in xrange(self.opt.max_sent_length):
            self.model.decoder.attn.applyMask(
                srcBatch[0].data.eq(onmt.Constants.PAD).t())
            decOut, decStates, attn = self.model.decoder(
                Variable(input, volatile=True), decStates, context, decOut)
            decOut = decOut.squeeze(0)
            gen_t = self.model.generator.forward(decOut)
            tgt_t = gen_t.data.exp().multinomial(1)
            eos_mask = torch.max(eos_mask, tgt_t.eq(onmt.Constants.EOS))
            scores = gen_t.data.gather(1, tgt_t)
            scores.masked_fill_(eos_mask, 0)
            p_infer += scores
            # feed into the next time step
            input.copy_(tgt_t)

        return p_infer

    def beam_conf_once(self, srcBatch, tgtBatch, confidence_method, conf_n_best):
        beamSize = self.opt.beam_size
        confidence_method_split = confidence_method.split(':')
        #  (1) run the encoder on the src
        encStates, context, rnnSize = self.conf_encode(srcBatch)

        #  (3) run the decoder to generate sentences, using beam search
        # Expand tensors for each beam.
        batchSize = self._getBatchSize(srcBatch[0])
        context = Variable(context.data.repeat(1, beamSize, 1))

        decStates = (Variable(encStates[0].data.repeat(1, beamSize, 1)),
                     Variable(encStates[1].data.repeat(1, beamSize, 1)))

        beam = [onmt.Beam(beamSize, self.opt.cuda) for k in range(batchSize)]

        decOut = self.model.make_init_decoder_output(context)

        padMask = srcBatch[0].data.eq(
            onmt.Constants.PAD).t() \
                               .unsqueeze(0) \
                               .repeat(beamSize, 1, 1)

        batchIdx = list(range(batchSize))
        remainingSents = batchSize
        for i in range(self.opt.max_sent_length):
            self.model.decoder.attn.applyMask(padMask)
            # Prepare decoder input.
            input = torch.stack([b.getCurrentState() for b in beam
                                 if not b.done]).t().contiguous().view(1, -1)
            decOut, decStates, attn = self.model.decoder(
                Variable(input, volatile=True), decStates, context, decOut)
            # decOut: 1 x (beam*batch) x numWords
            decOut = decOut.squeeze(0)
            out = self.model.generator.forward(decOut)

            # batch x beam x numWords
            wordLk = out.view(beamSize, remainingSents, -1) \
                        .transpose(0, 1).contiguous()
            attn = attn.view(beamSize, remainingSents, -1) \
                       .transpose(0, 1).contiguous()

            active = []
            for b in range(batchSize):
                if beam[b].done:
                    continue

                idx = batchIdx[b]
                if not beam[b].advance(wordLk.data[idx], attn.data[idx]):
                    active += [b]

                for decState in decStates:  # iterate over h, c
                    # layers x beam*sent x dim
                    sentStates = decState.view(-1, beamSize,
                                               remainingSents,
                                               decState.size(2))[:, :, idx]
                    sentStates.data.copy_(
                        sentStates.data.index_select(
                            1, beam[b].getCurrentOrigin()))

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return Variable(view.index_select(1, activeIdx)
                                .view(*newSize), volatile=True)

            decStates = (updateActive(decStates[0]),
                         updateActive(decStates[1]))
            decOut = updateActive(decOut)
            context = updateActive(context)
            padMask = padMask.index_select(1, activeIdx)

            remainingSents = len(active)

        #  (4) package everything up
        allScores = []
        for b in range(batchSize):
            scores, ks = beam[b].sortBest()
            allScores += [scores[:conf_n_best]]

        # -> (batch_size, conf_n_best)
        p_infer = torch.stack(allScores)

        return p_infer

    def confidence(self, srcBatchWord, tgtBatchWord, confidence_method, opt):
        confidence_method_split = confidence_method.split(':')
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatchWord, tgtBatchWord)
        srcBatch, tgtBatch, indices = dataset[0]
        conf_word_score_save = {}

        #  (2) compute confidence
        conf_score_dict = {}
        if 'prb_word' in confidence_method_split:
            p_infer = self.drp_conf_once(
                srcBatch, tgtBatch, confidence_method, opt)
            if 'min' in confidence_method_split:
                # do not need masking because of min()
                conf_score_dict[confidence_method.replace(
                    ':sum', '')], __ = p_infer.min(0)  # .double().exp_()
            if 'sum' in confidence_method_split:
                p_infer_exp_masked = p_infer.clone().double().exp_()
                # masking
                for it, tgt_t in zip(p_infer_exp_masked, tgtBatch[1:].data):
                    it.masked_fill_(tgt_t.unsqueeze(
                        1).eq(onmt.Constants.PAD), 0)
                r = 1.0 / add_eps_(p_infer_exp_masked.sqrt())
                conf_word_score_save[confidence_method.replace(
                    ':sum', '').replace(':min', '')] = r.sub_(r.min())
                conf_score_dict[confidence_method.replace(
                    ':min', '')] = torch.sum(p_infer_exp_masked, 0)
        elif 'beam' in confidence_method_split:
            p_infer_stack = self.beam_conf_once(
                srcBatch, tgtBatch, confidence_method, opt.conf_n_best).double().exp_()
            if 'var' in confidence_method_split:
                conf_score = p_infer_stack.var(1)
                conf_score_dict['beam:var'] = conf_score.clone()
                if 'miu_norm' in confidence_method_split:
                    # std deviation / |miu|
                    conf_score_dict['beam:var:miu_norm'] = conf_score.sqrt().div_(
                        add_eps_(p_infer_stack.mean(1).abs_()))
        elif ('drp' in confidence_method_split) or ('noise' in confidence_method_split):
            if 'drp' in confidence_method_split:
                if 'normal' in confidence_method_split:
                    self.model.train()
                elif 'enc' in confidence_method_split:
                    self.model.encoder.train()
                elif 'enc_word' in confidence_method_split:
                    self.model.encoder.set_confidence_dropout_word(True, None)
                elif 'dec_word' in confidence_method_split:
                    self.model.decoder.set_confidence_dropout_word(True)
            elif 'noise' in confidence_method_split:
                if 'enc_word' in confidence_method_split:
                    self.model.encoder.set_confidence_noise_word(
                        get_noise_type(confidence_method_split), opt.noise_sigma, None)
                elif 'dec_word' in confidence_method_split:
                    self.model.decoder.set_confidence_noise_word(
                        get_noise_type(confidence_method_split), opt.noise_sigma)

            p_infer_list = []
            for i_infer in xrange(opt.infer_times):
                p_infer = self.drp_conf_once(
                    srcBatch, tgtBatch, confidence_method, opt)
                p_infer_list.append(p_infer)
            _p_infer_stack = torch.stack(p_infer_list)

            name_base = ':'.join(confidence_method_split[:3]) if confidence_method_split[
                0] == 'noise' else ':'.join(confidence_method_split[:2])
            # compute word-level variance
            if ('max' in confidence_method_split) or ('sum' in confidence_method_split):
                p_infer_stack = _p_infer_stack.clone().double().exp_()
                # (opt.infer_times * seq_len * batch_size) -> (seq_len * batch_size)
                conf_score_stack = p_infer_stack.var(0).squeeze_(0)
                conf_word_score_save[name_base] = conf_score_stack
                # (seq_len * batch_size) -> (batch_size)
                if 'max' in confidence_method_split:
                    conf_score_dict[name_base +
                                    ':max'], __ = torch.max(conf_score_stack, 0)
                if 'sum' in confidence_method_split:
                    conf_score_dict[name_base +
                                    ':sum'] = torch.sum(conf_score_stack, 0)
                # miu_norm
                conf_score_stack = conf_score_stack.sqrt().div_(
                    add_eps_(p_infer_stack.mean(0).abs_()))
                conf_word_score_save[name_base +
                                     ':miu_norm'] = conf_score_stack
                # (seq_len * batch_size) -> (batch_size)
                if 'max' in confidence_method_split:
                    conf_score_dict[name_base +
                                    ':miu_norm:max'], __ = torch.max(conf_score_stack, 0)
                if 'sum' in confidence_method_split:
                    conf_score_dict[name_base +
                                    ':miu_norm:sum'] = torch.sum(conf_score_stack, 0)

            # compute sequence-level variance
            # (opt.infer_times * seq_len * batch_size) -> (opt.infer_times * batch_size)
            p_infer_stack = _p_infer_stack.clone().double().sum(1)
            # the basic computation
            conf_score_dict[name_base] = _var(p_infer_stack)
            # :exp
            if 'exp' in confidence_method_split:
                p_infer_stack = p_infer_stack.double().exp_()
                conf_score = _var(p_infer_stack)
                conf_score_dict[name_base + ':exp'] = conf_score.clone()
                # :exp:miu_norm
                if 'miu_norm' in confidence_method_split:
                    # std deviation / |miu|
                    conf_score_dict[name_base + ':exp:miu_norm'] = conf_score.sqrt().div_(
                        add_eps_(p_infer_stack.mean(0).abs_()))

            for conf_score in conf_score_dict.values():
                conf_score.mul_(-1)
        elif 'ent' in confidence_method_split:
            if 'forced_dec' in confidence_method_split:
                p_infer = self.drp_conf_once(
                    srcBatch, tgtBatch, confidence_method, opt)
                conf_word_score_save[confidence_method.replace(
                    ':sum', '').replace(':max', '')] = -p_infer
                if 'max' in confidence_method_split:
                    # use .min(0) because -entropy is returned
                    conf_score_dict[confidence_method.replace(
                        ':sum', '')], __ = p_infer.min(0)
                if 'sum' in confidence_method_split:
                    conf_score_dict[confidence_method.replace(
                        ':max', '')] = p_infer.sum(0)
            elif 'seq' in confidence_method_split:
                p_infer_list = []
                for i_infer in xrange(opt.infer_times):
                    p_infer = self.ent_conf_sample_once(srcBatch)
                    p_infer_list.append(p_infer)
                conf_score_dict['ent:seq'] = torch.stack(p_infer_list).mean(0)
        else:
            raise NotImplementedError

        self.model.eval()
        self.model.encoder.set_confidence_dropout_word(False, None)
        self.model.decoder.set_confidence_dropout_word(False)
        self.model.encoder.set_confidence_noise_word(None, None, None)
        self.model.decoder.set_confidence_noise_word(None, None)

        assert len(conf_score_dict) > 0, 'len(conf_score_dict) > 0'

        r_dict = {}
        for k, conf_score in conf_score_dict.iteritems():
            # recover the order
            r_dict[k] = recover_order(conf_score.view(-1), indices)

        return r_dict, conf_word_score_save

    def confidence_each_word(self, srcBatchWord, tgtBatchWord, confidence_method, opt):
        confidence_method_split = confidence_method.split(':')
        assert ('drp' in confidence_method_split) or (
            'noise' in confidence_method_split), 'confidence_each_word() only supports drp/noise'

        #  (1) convert words to indexes
        dataset = self.buildData(srcBatchWord, tgtBatchWord)
        srcBatch, tgtBatch, indices = dataset[0]

        #  (2) compute confidence
        max_src_len = srcBatch[0].data.size(0)
        lengths_enc = srcBatch[1].data.view(-1)
        lengths_enc_cuda = lengths_enc.double().cuda()
        word_score_list = []
        for i_src in xrange(max_src_len):
            conf_score_dict = {}
            if 'drp' in confidence_method_split:
                if 'enc_word' in confidence_method_split:
                    self.model.encoder.set_confidence_dropout_word(True, i_src)
            elif 'noise' in confidence_method_split:
                if 'enc_word' in confidence_method_split:
                    self.model.encoder.set_confidence_noise_word(
                        get_noise_type(confidence_method_split), opt.noise_sigma, i_src)

            p_infer_list = []
            for i_infer in xrange(opt.infer_times):
                p_infer = self.drp_conf_once(
                    srcBatch, tgtBatch, confidence_method, opt)
                p_infer_list.append(p_infer)
            _p_infer_stack = torch.stack(p_infer_list)

            name_base = ':'.join(confidence_method_split[:3]) if confidence_method_split[
                0] == 'noise' else ':'.join(confidence_method_split[:2])
            # compute word-level variance
            p_infer_stack = _p_infer_stack.clone().double().exp_()
            # (opt.infer_times * seq_len * batch_size) -> (seq_len * batch_size)
            conf_score_stack = p_infer_stack.var(0).squeeze_(0)
            # (seq_len * batch_size) -> (batch_size)
            conf_score_dict[
                name_base + ':sum'] = torch.sum(conf_score_stack, 0).div_(lengths_enc_cuda)
            # miu_norm
            conf_score_stack = conf_score_stack.sqrt().div_(
                add_eps_(p_infer_stack.mean(0).abs_()))
            # (seq_len * batch_size) -> (batch_size)
            conf_score_dict[
                name_base + ':miu_norm:sum'] = torch.sum(conf_score_stack, 0).div_(lengths_enc_cuda)

            # compute sequence-level variance
            # (opt.infer_times * seq_len * batch_size) -> (opt.infer_times * batch_size)
            p_infer_stack = _p_infer_stack.clone().double().sum(1)
            # the basic computation
            conf_score_dict[name_base] = _var(p_infer_stack)
            # :exp
            if 'exp' in confidence_method_split:
                p_infer_stack = p_infer_stack.double().exp_()
                conf_score = _var(p_infer_stack)
                conf_score_dict[name_base + ':exp'] = conf_score.clone()
                # :exp:miu_norm
                if 'miu_norm' in confidence_method_split:
                    # std deviation / |miu|
                    conf_score_dict[name_base + ':exp:miu_norm'] = conf_score.sqrt().div_(
                        add_eps_(p_infer_stack.mean(0).abs_()))
            word_score_list.append(conf_score_dict)

        self.model.eval()
        self.model.encoder.set_confidence_dropout_word(False, None)
        self.model.decoder.set_confidence_dropout_word(False)
        self.model.encoder.set_confidence_noise_word(None, None, None)
        self.model.decoder.set_confidence_noise_word(None, None)

        assert len(conf_score_dict) > 0, 'len(conf_score_dict) > 0'

        r_dict = {}
        for method_name in word_score_list[0].keys():
            # (batch_size * src_len)
            t = torch.stack([conf_score_dict[method_name].view(-1)
                             for conf_score_dict in word_score_list], 1)
            r_dict[method_name] = recover_order(
                trim_tensor_by_length(t, lengths_enc), indices)

        return r_dict

    def confidence_bp(self, srcBatchWord, tgtBatchWord, conf_word_score, opt_train):
        """backpropagate confidence scores to input words (-> batch_size * enc_seq_len)
        Args:
            conf_word_score (Tensor): dec_seq_len * batch_size (reordered, need to be recovered by indices)
        """
        #  (0) convert words to indexes
        conf_word_score = conf_word_score.float().mul_(1e3)

        # DEBUG
        # tgtBatchWord[0][-1] = 'A'

        dataset = self.buildData(srcBatchWord, tgtBatchWord)
        srcBatch, tgtBatch, indices = dataset[0]
        #  (1) forward propagation
        encStates, context, rnnSize = self.conf_encode(srcBatch)
        batchSize = self._getBatchSize(srcBatch[0])
        decStates = encStates
        decOut = self.model.make_init_decoder_output(context)
        self.model.decoder.attn.applyMask(
            srcBatch[0].data.eq(onmt.Constants.PAD).t())
        initOutput = self.model.make_init_decoder_output(context)
        decOut, decStates, attn = self.model.decoder(
            tgtBatch[:-1], decStates, context, initOutput)
        # (dec_len * batch_size * dec_voc)
        tgt_scores = []
        generator_hook = self.model.generator[1].register_forward_hook(
            lambda _module, _input, _output: tgt_scores.append(_output.data.clone().exp_()))
        for dec_t, tgt_t in zip(decOut, tgtBatch[1:].data):
            gen_t = self.model.generator.forward(dec_t)
            tgt_t = tgt_t.unsqueeze(1)
            scores = gen_t.data.gather(1, tgt_t)
            scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
        generator_hook.remove()

        #  (3) backpropagate confidence
        cbp = {}

        def cbp_linear(W, x, b, c, method='wx'):
            """confidence bp for linear layer
            be careful about the *_() (inplace operation)
            Args:
                W: (out * in)
                x: (batch_size * in)
                b: None or (1 * out)
                c: (batch_size * out)
            """
            W_t = W.transpose(0, 1).contiguous()
            batchSize, in_size, out_size = x.size(0), x.size(1), W.size(0)
            if method == 'w2':
                r0 = W_t.pow_(2)
                r1 = W_t.sum(0).expand_as(W_t)
                # (batch_size * in * out)
                r = r0.div_(add_eps_(r1).view(
                    batchSize, 1, out_size).expand_as(r0))
            elif method in ('abs', 'wx>0'):
                # (batch_size * in * out)
                W_t_expand = W_t.view(
                    1, in_size, out_size).expand(batchSize, in_size, out_size)
                # (batch_size * in * out)
                r0 = W_t_expand.mul(
                    x.view(batchSize, in_size, 1).expand_as(W_t_expand))
                if method == 'abs':
                    r0.abs_()
                elif method == 'wx>0':
                    # only makes sense for relu networks (all positive)
                    r0.masked_fill_(r0.lt(0), 0)
                # (batch_size * out)
                r1 = r0.sum(1).unsqueeze_(1)
                # (batch_size * in * out)
                r = r0.div_(add_eps_(r1).view(
                    batchSize, 1, out_size).expand_as(r0))
            elif method == 'wx':
                # (batch_size * in * out)
                W_t_expand = W_t.view(
                    1, in_size, out_size).expand(batchSize, in_size, out_size)
                # (batch_size * in * out)
                r0 = W_t_expand.mul(
                    x.view(batchSize, in_size, 1).expand_as(W_t_expand))
                r0_abs = r0.abs()
                # (batch_size * out)
                r1 = r0.sum(1).unsqueeze_(1)
                r1_abs = r0_abs.sum(1).unsqueeze_(1)
                # (batch_size * in * out)
                r = r0.div_(add_eps_(r1).view(
                    batchSize, 1, out_size).expand_as(r0))
                r_abs = r0_abs.div_(add_eps_(r1_abs).view(
                    batchSize, 1, out_size).expand_as(r0_abs))
                r = r_abs.mul_(r.sign_())
            # (batch_size * in * out) * (batch_size * out * 1) -> (batch_size * in * 1) -> (batch_size * in)
            return r.bmm(c.view(batchSize, -1, 1)).squeeze_(2)
            # if method.find('bias') >= 0 and (b is not None):
            #   r1.add_(b.abs().expand_as(r1))

        # used for x_0 + x_1
        def cbp_direct_add_ratio(x0, x1, c, method):
            # x0/(x0+x1)*c, x1/(x0+x1)*c
            if method == 'abs':
                x0_abs, x1_abs = x0.abs(), x1.abs()
                x1_abs.add_(x0_abs)
                c0 = x0_abs.div_(add_eps_(x1_abs)).mul_(c)
                return c0, c - c0
            elif method == 'wx':
                x0_abs, x1_abs = x0.abs(), x1.abs()
                c0_abs = x0_abs.div(add_eps(x0_abs + x1_abs))
                c1_abs = 1 - c0_abs
                c0 = x0.div(add_eps(x0 + x1))
                c1 = 1 - c0
                c0 = c0_abs.mul(c0.sign())
                c1 = c1_abs.mul(c1.sign())
                c0.mul_(c)
                c1.mul_(c)
                return c0, c1

        # used for x_0 .* x_1 (x_0 >= 0, only x_1 could be negative)
        def cbp_direct_dot_ratio(x0, x1, c, method):
            # if method == 'abs':
            #   return cbp_direct_add_ratio(x0, x1, c, method)
            # elif method == 'wx':
            #   x0_abs, x1_abs = x0.abs(), x1.abs()
            #   c0_abs = x0_abs.div(add_eps(x0_abs + x1_abs))
            #   c1_abs = 1 - c0_abs
            #   c0 = c0_abs
            #   c1 = c1_abs#.mul(x1.sign())
            # c0.mul_(c)
            # c1.mul_(c)
            # return c0, c1

            x0_abs, x1_abs = x0.abs(), x1.abs()
            # x0_mask = (x0_abs >= 0) & (x0_abs < 1)
            # x1_mask = (x1_abs >= 0) & (x1_abs < 1)
            # e0 = x0_abs.clone().masked_fill_(x0_mask, 0) + \
            #     add_eps(x0_abs).pow_(-1).masked_fill_(1 - x0_mask, 0)
            # e1 = x1_abs.clone().masked_fill_(x1_mask, 0) + \
            #     add_eps(x1_abs).pow_(-1).masked_fill_(1 - x1_mask, 0)

            x0_abs.masked_fill_(x0_abs.eq(1), 1 + 1e-6)
            x1_abs.masked_fill_(x1_abs.eq(1), 1 + 1e-6)
            e0 = add_eps_(x0_abs).log_()
            e1 = add_eps_(x1_abs).log_()
            return cbp_direct_add_ratio(e0, e1, c, 'abs')

        # (3.-) y_t -> h_att (through classifier layer)
        cbp['h_att'] = []
        catt = None
        # (|dec_voc| * |h_att|)
        W_classifier, b_classifier = self.model.generator[
            0].weight.data, self.model.generator[0].bias.data
        # (batch_size * dec_voc)
        conf_pred = self.tt.FloatTensor(batchSize, W_classifier.size(0))
        for t, dec_t, tgt_t, scores_t, conf_t in zip(count(), decOut.data, tgtBatch[1:].data, tgt_scores, conf_word_score):
            # (batch_size)
            conf_t_masked = conf_t.masked_fill_(
                tgt_t.eq(onmt.Constants.PAD), 0)
            conf_pred.zero_()
            for i in xrange(batchSize):
                conf_pred[i, tgt_t[i]] = conf_t_masked[i]
                # DEBUG
                # if tgt_t[i] == self.tgt_dict.lookup('A0'):
                #   conf_pred[i, self.tgt_dict.lookup('A')] = -conf_t_masked[i]
            # conf_pred.copy_(scores_t).mul_(
            #     conf_t_masked.view(-1, 1).expand_as(conf_pred))
            cbp_att = cbp_linear(W_classifier, dec_t, b_classifier,
                                 conf_pred, self.opt.conf_bp)
            cbp['h_att'].append(cbp_att)

            # weighted sum uncertainty scores by the attention scores
            # (batch_size * src_len)
            att_score = self.model.decoder.cache_attn[t]
            r = att_score.mul(conf_t_masked.view(-1, 1).expand_as(att_score))
            if catt is None:
                catt = r
            else:
                catt.add_(r)
        conf_pred = None

        # (3.-) h_att -> c_t (weightedContext), h_dec_t (through attention layer)
        cbp['h_dec_top'] = []
        cbp['h_enc_top'] = None
        # (2d * d)
        W_attn = self.model.decoder.attn.linear_out.weight.data
        for t, conf_t in zip(count(), cbp['h_att']):
            contextCombined = torch.cat((self.model.decoder.cache_weightedContext[
                                        t], self.model.decoder.cache_output[t]), 1)
            conf_bp = cbp_linear(W_attn, contextCombined, None,
                                 conf_t, self.opt.conf_bp)
            c_t, h_dec_t = conf_bp.chunk(2, 1)
            cbp['h_dec_top'].append(h_dec_t)
            # (3.-) c_t -> h_enc_t (through attention layer)
            # context.data: (src_len * batch_size * d)
            # self.model.decoder.cache_attn[t]: (batch_size * src_len)
            # c_t: (batch_size * d)
            # -> (src_len * batch_size * d)
            # r0 = self.model.decoder.cache_attn[t].t().unsqueeze(2).expand_as(context.data).mul(context.data)
            # r1 =
            # self.model.decoder.cache_weightedContext[t].unsqueeze(0).expand_as(context.data)
            r0 = self.model.decoder.cache_attn[t].t().unsqueeze(
                2).expand_as(context.data).mul(context.data).abs_()
            r1 = r0.sum(0).expand_as(context.data)
            r = r0.div(add_eps_(r1))
            if cbp['h_enc_top'] is None:
                cbp['h_enc_top'] = c_t.unsqueeze(0).expand_as(r).mul(r)
            else:
                cbp['h_enc_top'].add_(c_t.unsqueeze(0).expand_as(r).mul(r))
        self.model.decoder.clean_cache()

        def cbp_lstm(save, num_layers, W_rnn, conf_h, conf_h_from_next, conf_c_from_next, cbp_top, seq_len=None):
            if seq_len is not None:
                save_conf_h_from_next = list(
                    map(lambda x: x.clone(), conf_h_from_next))
                save_conf_c_from_next = list(
                    map(lambda x: x.clone(), conf_c_from_next))
            cbp_input = []
            for t, conf_from_top in reversed(zip(count(), cbp_top)):
                conf_h.copy_(conf_h_from_next)
                if seq_len is not None:
                    # mask confidence
                    for i_batch in xrange(batchSize):
                        if t >= seq_len[i_batch]:
                            for i_layer in xrange(num_layers):
                                conf_h[i_layer, i_batch, :].zero_()
                                conf_c_from_next[i_layer, i_batch, :].zero_()
                        else:
                            if t == seq_len[i_batch] - 1:
                                for i_layer in xrange(num_layers):
                                    conf_h[i_layer, i_batch, :].copy_(
                                        save_conf_h_from_next[i_layer][i_batch, :])
                                    conf_c_from_next[i_layer, i_batch, :].copy_(
                                        save_conf_c_from_next[i_layer][i_batch, :])
                conf_h[num_layers - 1].add_(conf_from_top)
                for i_layer in reversed(range(num_layers)):
                    W_x, W_h1, b_x, b_h1 = map(
                        lambda x: x.data, W_rnn[i_layer])
                    # h -> o .* tanh(c)
                    cbp_o, cbp_c = cbp_direct_dot_ratio(save[i_layer][t]['o'], save[i_layer][t][
                                                        'tanh_c'], conf_h[i_layer], self.opt.conf_bp)
                    cbp_c.add_(conf_c_from_next[i_layer])
                    # c -> f .* c_(t-1) + i .* g
                    cbp_fc1, cbp_ig = cbp_direct_add_ratio(save[i_layer][t]['fc1'], save[i_layer][
                                                           t]['ig'], cbp_c, self.opt.conf_bp)
                    cbp_f, cbp_c1 = cbp_direct_dot_ratio(save[i_layer][t]['f'], save[i_layer][
                                                         t]['c1'], cbp_fc1, self.opt.conf_bp)
                    conf_c_from_next[i_layer].copy_(cbp_c1)
                    cbp_i, cbp_g = cbp_direct_dot_ratio(save[i_layer][t]['i'], save[i_layer][
                                                        t]['g'], cbp_ig, self.opt.conf_bp)
                    # i, f, g, o -> x_t, h_(t-1)
                    cbp_ifgo = torch.cat([cbp_i, cbp_f, cbp_g, cbp_o], 1)
                    W_comb = torch.cat([W_x, W_h1], 1)
                    b_comb = b_x.view(1, -1) + b_h1.view(1, -1)
                    xh1_comb = torch.cat(
                        [save[i_layer][t]['x'], save[i_layer][t]['h1']], 1)
                    cbp_x_h1 = cbp_linear(
                        W_comb, xh1_comb, b_comb, cbp_ifgo, self.opt.conf_bp)
                    if i_layer >= 1:
                        cbp_x, cbp_h1 = cbp_x_h1.chunk(2, 1)
                        conf_h[i_layer - 1].add_(cbp_x)
                    elif i_layer == 0:
                        # rnn_size != word_vec_size
                        cbp_x, cbp_h1 = cbp_x_h1[:, :opt_train['word_vec_size']
                                                 ], cbp_x_h1[:, opt_train['word_vec_size']:]
                        cbp_input.append(cbp_x.clone())
                    conf_h_from_next[i_layer].copy_(cbp_h1)
            return list(reversed(cbp_input))

        def mask_cbp_list(cbp_top, lengths, batchSize):
            for t in xrange(max(lengths)):
                for i_batch in xrange(batchSize):
                    if t >= lengths[i_batch]:
                        cbp_top[t][i_batch].zero_()

        def resize_to_batch_size(save, batchSize):
            for save_layer in save:
                for save_cell in save_layer:
                    for name in ('x', 'h1', 'c1', 'i', 'f', 'g', 'o', 'c', 'tanh_c', 'fc1', 'ig'):
                        t = save_cell[name]
                        org_size = t.size(0)
                        if org_size < batchSize:
                            t.resize_(batchSize, *(t.size()[1:]))
                            t[org_size:batchSize].zero_()
            return save

        # (3.-) h_dec_t -> dec_t, dec_t -> dec_(t-1) (through decoder LSTM layer)
        W_rnn_dec = list(map(lambda x: (x.weight_ih, x.weight_hh,
                                        x.bias_ih, x.bias_hh), self.model.decoder.rnn.layers))
        conf_h, conf_h_from_next, conf_c_from_next = [self.tt.FloatTensor(
            opt_train['layers'], batchSize, self.model.decoder.hidden_size).zero_() for __ in xrange(3)]
        # run functional rnn to save internal vectors
        lengths_dec_unorder = list(map(lambda x: len(x), tgtBatchWord))
        # length + 1 because of </s>
        lengths_dec = list(map(lambda x: lengths_dec_unorder[x] + 1, indices))
        a, b, save_dec = mod.FuncRNN('LSTM', opt_train['word_vec_size'], self.model.decoder.hidden_size, num_layers=opt_train[
            'layers'], batch_first=False, dropout=0, train=False, bidirectional=False, batch_sizes=None)(self.model.decoder.get_word_vector(tgtBatch[:-1]), W_rnn_dec, encStates)
        # import pudb ; pu.db
        mask_cbp_list(cbp['h_dec_top'], lengths_dec, batchSize)
        cbp_lstm(save_dec, opt_train['layers'], W_rnn_dec, conf_h,
                 conf_h_from_next, conf_c_from_next, cbp['h_dec_top'], None)
        # (3.-) h_dec_0 -> h_enc_T (through encoder-decoder layer)
        # use the same conf_h, conf_h_from_next, conf_c_from_next
        # (3.-) h_enc_t -> enc_t, enc_t -> enc_(t-1) (through encoder LSTM layer)
        W_rnn_enc = [(getattr(self.model.encoder.rnn, 'weight_ih_l%d' % (i,)), getattr(self.model.encoder.rnn, 'weight_hh_l%d' % (i,)),
                      getattr(self.model.encoder.rnn, 'bias_ih_l%d' % (i,)), getattr(self.model.encoder.rnn, 'bias_hh_l%d' % (i,))) for i in xrange(opt_train['layers'])]
        # run functional rnn to save internal vectors
        lengths_enc = srcBatch[1].data.view(-1).tolist()
        emb_packed = pack(
            self.model.encoder.get_word_vector(srcBatch[0]), lengths_enc)
        a, b, save_enc = mod.FuncRNN('LSTM', opt_train['word_vec_size'], self.model.encoder.hidden_size, num_layers=opt_train[
            'layers'], batch_first=False, dropout=0, train=False, bidirectional=opt_train['brnn'], batch_sizes=emb_packed.batch_sizes)(emb_packed.data, W_rnn_enc, None)
        # import pudb ; pu.db
        mask_cbp_list(cbp['h_enc_top'], lengths_enc, batchSize)
        cbp['enc_word'] = cbp_lstm(resize_to_batch_size(save_enc, batchSize), opt_train[
                                   'layers'], W_rnn_enc, conf_h, conf_h_from_next, conf_c_from_next, cbp['h_enc_top'], lengths_enc)
        # (3.-) sum(enc_t -> word_emb_t)
        # mask_cbp_list(cbp['enc_word'], lengths_enc, batchSize)
        # (seq_len * batch_size * emb_dim) -> (batch_size * seq_len)
        # cbp_src_tensor = torch.stack(cbp['enc_word']).abs_().sum(2).squeeze_(2).t()
        cbp_src_tensor = torch.stack(cbp['enc_word'])
        # cbp_src_tensor.masked_fill_(cbp_src_tensor.lt(0), 0)
        if self.opt.conf_bp_abs:
            cbp_src_tensor.abs_()
        cbp_src_tensor = cbp_src_tensor.sum(2).squeeze_(2).t()
        # recover the order
        return recover_order(trim_tensor_by_length(cbp_src_tensor, lengths_enc), indices), recover_order(trim_tensor_by_length(catt, lengths_enc), indices), recover_order(trim_tensor_by_length(conf_word_score.transpose(0, 1), lengths_dec), indices)
