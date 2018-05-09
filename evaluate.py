from __future__ import division

import os
import codecs
import onmt
import torch
import argparse
import math
import numpy as np
from path import Path
from tqdm import tqdm
import glob
from tensorboard_logger import Logger
import json
import itertools
from itertools import izip
from PTB_Tree_eval import PTB_Tree_eval
from pprint import pprint
from tabulate import tabulate
from utils.utils import is_valid_by_eval_type_ifttt, get_split, line_iter
import kenlm
# import ast
# import astor
# from lang.py.parse import tokenize_code, de_canonicalize_code

parser = argparse.ArgumentParser(description='evaluate.py')

parser.add_argument('-metric', default="tree_acc",
                    help='tree_acc: Tree-level accuracy; ifttt: F1, channel-level accuracy, function-level accuracy;')
parser.add_argument('-confidence', default="none",
                    help='''none - no confidence values;

                    prb_word:min - min(probabilities of all the decoded words);
                    prb_word:sum - sum(probabilities of all the decoded words);

                    drp:normal - enable all dropout as during training;
                    drp:enc - only dropout encoder;
                    drp:enc_word - word-level dropout (all encoder words);
                    drp:dec_word - word-level dropout (all decoder words);
                    drp:bridge - dropout the bridge between encoder and decoder;
                    drp:classifier - dropout the classifier in decoder;
                    drp:normal:max - enable all dropout as during training, but use the max variance values of decoding tokens for uncertainty scores;
                    drp:normal:sum - enable all dropout as during training, but use the sum of variance values of decoding tokens for uncertainty scores;
                    drp:enc_word:max - word-level dropout (all encoder words), but use the max variance values of decoding tokens for uncertainty scores;
                    drp:enc_word:sum - word-level dropout (all encoder words), but use the sum of variance values of decoding tokens for uncertainty scores;
                    [exp] - exp(log_score);
                    [miu_norm] - normalize std deviation by miu (mean values);

                    noise:enc_word - word-level noise injection (all encoder words);
                    noise:dec_word - word-level noise injection (all decoder words);
                    noise:bridge - inject noise for the bridge between encoder and decoder;
                    noise:classifier - inject noise for the classifier in decoder;
                    ... similar to drp ...
                    [add] - w + N(0, /sigma^2)
                    [mul] - w * N(1, /sigma^2)

                    ent:forced_dec:sum - sum(entropy values with forced decoding words);
                    ent:forced_dec:max - max(entropy values with forced decoding words);
                    ent:seq - sequence-level entropy -sum_{y}{p(y|x) log{p(y|x)}}
                    
                    beam:var - sequence-level variance for beam results;

                    lm - log(language model scores);
                    lm:norm - log(language model scores) normalized by the length of source sentence;

                    src_unk - number of unk tokens in the source sentence;
                    src_unk:norm - src_unk normalized by the length of source sentence;''')
parser.add_argument('-conf_each_word', default='none',
                    help="Compute uncertainty scores for each source word. drp:enc_word:exp:miu_norm,noise:enc_word:add:exp:miu_norm,noise:enc_word:mul:exp:miu_norm")
parser.add_argument('-conf_bp', default='none',
                    help="Confidence back-propagation: [abs]: use abs(); [wx]: keep positive and negative;")
parser.add_argument('-conf_bp_abs', action="store_true",
                    help="Use abs() when summarize the confidence scores of word vectors.")
parser.add_argument('-conf_bp_bias', action="store_true",
                    help="Consider bias term in confidence backpropagation.")
parser.add_argument('-dropout_rate', type=float, default=0.1)
parser.add_argument('-noise_sigma', type=float, default=0.05)
parser.add_argument('-infer_times', type=int, default=30,
                    help='how many times do we infer for confidence estimation')

parser.add_argument('-model_path', required=True,
                    help='Path to model file (can contain shell-style wildcards)')
parser.add_argument('-lm_path', default='/disk/scratch_ssd/lidong/lang2logic-py/ifttt/lm.arpa',
                    help='Path to language model file')
parser.add_argument('-src', required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-tgt', help='True target sequence (optional)')
parser.add_argument('-ifttt_eval_category', default='/disk/scratch_ssd/lidong/lang2logic-py/ifttt/test.url.txt',
                    help='True target sequence (optional)')
parser.add_argument('-beam_size', type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=600,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had highest attention weight.""")
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-dump_beam', action="store_true",
                    help='File to dump beam information to.')

parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")
parser.add_argument('-conf_n_best', type=int, default=10,
                    help='N-best size during confidence inference')

parser.add_argument('-gpu', type=int, default=0,
                    help="Device to run on")


_tree_template = "(ROOT (TRI (%s (%s (%s)))) (ACT (%s (%s (%s)))))"


def ifttt_f1_metric(gold, pred):
    t_gold_str = _tree_template % (gold['if_channel'], gold['if_func'], gold[
                                   'if_cmd'], gold['then_func'], gold['then_channel'], gold['then_cmd'])
    t_pred_str = _tree_template % (pred['if_channel'], pred['if_func'], pred[
                                   'if_cmd'], pred['then_func'], pred['then_channel'], pred['then_cmd'])

    t_gold, t_pred = PTB_Tree_eval(t_gold_str), PTB_Tree_eval(t_pred_str)

    prod_gold, prod_pred = t_gold.get_production_list(), t_pred.get_production_list()

    # <U>, _url_ in gold will not be predicted
    prod_gold = map(lambda x: x.replace('<U>', '<G>'), prod_gold)
    prod_gold = map(lambda x: x.replace('_url_', '_goldurl_'), prod_gold)

    c = sum([min(prod_gold.count(it), prod_pred.count(it))
             for it in set(prod_gold)])
    if c == 0 or len(prod_pred) == 0 or len(prod_gold) == 0:
        return 0.0
    else:
        p = float(c) / float(len(prod_pred))
        r = float(c) / float(len(prod_gold))
        return 2 * p * r / (p + r)


def is_tree_eq(s1, s2, vocab_dict):
    t1, t2 = onmt.Tree(vocab_dict), onmt.Tree(vocab_dict)
    t1.init_from_str(s1)
    t2.init_from_str(s2)
    return t1.is_eq(t2)


def is_py_eq(ref_code, pred_code):
    return ref_code == pred_code
    # def code2ast2token(c):
    #     return tokenize_code(astor.to_source(ast.parse(c).body[0]).strip())
    # ref_tokens = code2ast2token(ref_code)
    # pred_tokens = code2ast2token(pred_code)
    # return ref_tokens == pred_tokens


def get_epoch_from_model_path(fn_model):
    return int(Path(fn_model).name.split('_')[1])


def split_at_least(s, sp, n):
    tk_list = s.split(sp, n)
    while len(tk_list) <= n:
        tk_list += ['<none>']
    return tk_list


def decompose_ifttt_tgt(s):
    s_if, s_then = split_at_least(s.strip(), ' <then> ', 1)
    if_channel, if_func, if_cmd = split_at_least(s_if, ' ', 2)
    then_channel, then_func, then_cmd = split_at_least(s_then, ' ', 2)
    return {'if_channel': if_channel, 'if_func': if_func, 'if_cmd': if_cmd, 'then_channel': then_channel, 'then_func': then_func, 'then_cmd': then_cmd}


def evaluate_main(opt):
    # read language model
    if 'lm' in opt.confidence.split(','):
        if Path(opt.lm_path).exists():
            lm = kenlm.Model(opt.lm_path)
        else:
            print('==== LM does not exist: ' + opt.lm_path)

    for fn_model in tqdm(glob.glob(opt.model_path)):
        print(fn_model)
        opt_train = json.load(
            open(os.path.join(Path(fn_model).dirname(), 'opt.json'), 'r'))
        opt.model = fn_model
        translator = onmt.Translator(opt)
        # reset dropout rate
        translator.reset_dropout_rate(opt.dropout_rate)
        outF = codecs.open('%s.%s.sample' %
                           (opt.model, get_split(opt)), 'w', encoding='utf-8')
        tgtF = codecs.open(opt.tgt, 'r', encoding='utf-8')
        srcBatch, tgtBatch = [], []
        count = 0
        eval_info_list = []

        if opt.dump_beam:
            translator.initBeamAccum()

        for line in line_iter(codecs.open(opt.src, 'r', encoding='utf-8')):
            if line is not None:
                srcTokens = line.split()
                srcBatch += [srcTokens]
                tgtTokens = tgtF.readline().split()
                tgtBatch += [tgtTokens]

                if len(srcBatch) < opt.batch_size:
                    continue
            else:
                # at the end of file, check last batch
                if len(srcBatch) == 0:
                    break

            predBatch, predScore, goldScore, attn = translator.translate(
                srcBatch, tgtBatch)
            conf, conf_bp_src, conf_att_src, conf_tgt, conf_each_word = {}, {}, {}, {}, {}
            if opt.confidence != 'none':
                for confidence_method in set(opt.confidence.split(',')):
                    if confidence_method == 'lm':
                        conf[confidence_method] = list(
                            map(lambda x: lm.score(' '.join(x[0])), srcBatch))
                    else:
                        d, w = translator.confidence(srcBatch, list(
                            map(lambda x: x[0], predBatch)), confidence_method, opt)
                        for k, v in d.iteritems():
                            conf[k] = v
                        if (opt.conf_bp != 'none') and (len(w) > 0):
                            for k, v in w.iteritems():
                                conf_bp_src[k], conf_att_src[k], conf_tgt[k] = translator.confidence_bp(
                                    srcBatch, list(map(lambda x: x[0], predBatch)), v, opt_train)
            if opt.conf_each_word != 'none':
                for confidence_method in set(opt.conf_each_word.split(',')):
                    d = translator.confidence_each_word(srcBatch, list(
                        map(lambda x: x[0], predBatch)), confidence_method, opt)
                    for k, v in d.iteritems():
                        conf_each_word[k] = v

            for b in range(len(predBatch)):
                count += 1
                if opt.metric == 'django':
                    # post-process: copy <unk> tokens from srcBatch
                    def copy_unk(src, pred, attn_score):
                        post = []
                        for i, pred_token in enumerate(pred):
                            if pred_token == '<unk>':
                                _, ids = attn_score[i].sort(0, descending=True)
                                post.append(src[ids[0]])
                            else:
                                post.append(pred_token)
                        return post
                    outF.write(
                        " ".join(copy_unk(srcBatch[b], predBatch[b][0], attn[b][0])) + '\n')
                else:
                    outF.write(" ".join(predBatch[b][0]) + '\n')
                outF.flush()

                info = {'id': count, 'pred_score': predScore[b][0], 'pred_len': len(predBatch[b][0]), 'gold_score': goldScore[b], 'gold_len': len(
                    tgtBatch[b]), 'src_len': len(srcBatch[b]), 'src_unk': sum([0 if (translator.src_dict.lookup(w, None) is not None) else 1 for w in srcBatch[b]])}
                if opt.confidence != 'none':
                    info['confidence'] = dict(
                        [(k, v[b]) for k, v in conf.iteritems()])
                if opt.conf_bp != 'none':
                    info['conf_bp_src'] = dict([(k, v[b])
                                                for k, v in conf_bp_src.iteritems()])
                    info['conf_att_src'] = dict([(k, v[b])
                                                 for k, v in conf_att_src.iteritems()])
                    info['conf_tgt'] = dict([(k, v[b])
                                             for k, v in conf_tgt.iteritems()])
                if opt.conf_each_word != 'none':
                    info['conf_each_word'] = dict(
                        [(k, v[b]) for k, v in conf_each_word.iteritems()])
                eval_info_list.append(info)

                if opt.verbose:
                    print('')
                    # show attention score
                    print(" ".join(predBatch[b][0]))
                    for i, w in enumerate(predBatch[b][0]):
                        print(w)
                        _, ids = attn[b][0][i].sort(0, descending=True)
                        for j in ids[:5].tolist():
                            w_src = translator.src_dict.getLabel(translator.src_dict.lookup(
                                srcBatch[b][j], default=translator.src_dict.lookup(onmt.Constants.UNK_WORD)))
                            print("\t%s\t%d\t%3f" %
                                  (w_src, j, attn[b][0][i][j]))

                    srcSent = ' '.join(srcBatch[b])
                    if translator.tgt_dict.lower:
                        srcSent = srcSent.lower()
                    print('SENT %d: %s' % (count, srcSent))
                    print('PRED %d: %s' % (count, " ".join(predBatch[b][0])))
                    print("PRED SCORE: %.4f" % predScore[b][0])

                    tgtSent = ' '.join(tgtBatch[b])
                    if translator.tgt_dict.lower:
                        tgtSent = tgtSent.lower()
                    print('GOLD %d: %s ' % (count, tgtSent))
                    print("GOLD SCORE: %.4f" % goldScore[b])

                    if opt.n_best > 1:
                        print('\nBEST HYP:')
                        for n in range(opt.n_best):
                            print("[%.4f] %s" % (predScore[b][n],
                                                 " ".join(predBatch[b][n])))
                    print('')

            srcBatch, tgtBatch = [], []
        outF.close()
        tgtF.close()

        if opt.dump_beam:
            json.dump(translator.beam_accum, open('%s.%s.beam' %
                                                  (opt.model, get_split(opt)), 'w'))

        # read golden results and predictions
        with codecs.open(opt.tgt, 'r', encoding='utf-8') as f_in:
            gold_tgt_list = [l.strip() for l in f_in]
        with codecs.open('%s.%s.sample' % (opt.model, get_split(opt)), 'r', encoding='utf-8') as f_in:
            pred_tgt_list = [l.strip() for l in f_in]
        assert len(gold_tgt_list) == len(pred_tgt_list), '%d\t%d' % (
            len(gold_tgt_list), len(pred_tgt_list))

        # tree-level accuracy
        if opt.metric == 'tree_acc':
            for i, gold_tgt, pred_tgt in izip(itertools.count(), gold_tgt_list, pred_tgt_list):
                eval_info_list[i]['acc'] = 1 if is_tree_eq(
                    gold_tgt, pred_tgt, translator.tgt_dict) else 0
            m = {'acc': np.mean(list(map(lambda x: x['acc'], eval_info_list)))}
            pprint(m)
        elif opt.metric == 'django':
            for i, gold_tgt, pred_tgt in izip(itertools.count(), gold_tgt_list, pred_tgt_list):
                eval_info_list[i]['acc'] = 1 if is_py_eq(
                    gold_tgt, pred_tgt) else 0
            m = {'acc': np.mean(list(map(lambda x: x['acc'], eval_info_list)))}
            pprint(m)
        elif opt.metric == 'ifttt':
            if get_split(opt) == 'test':
                # read test category
                with codecs.open(opt.ifttt_eval_category, 'r', encoding='utf-8') as f_in:
                    eval_category_list = [
                        int(l.strip().split('\t')[1]) for l in f_in]
                assert len(gold_tgt_list) == len(eval_category_list), '%d\t%d' % (
                    len(gold_tgt_list), len(eval_category_list))
            for i, gold_tgt, pred_tgt in izip(itertools.count(), gold_tgt_list, pred_tgt_list):
                gold_decomp = decompose_ifttt_tgt(gold_tgt)
                pred_decomp = decompose_ifttt_tgt(pred_tgt)

                eval_info_list[i]['channel_acc'] = 1 if all(map(lambda k: gold_decomp[k] == pred_decomp[
                    k], ('if_channel', 'then_channel'))) else 0
                eval_info_list[i]['func_acc'] = 1 if all(map(lambda k: gold_decomp[k] == pred_decomp[
                    k], ('if_channel', 'then_channel', 'if_func', 'then_func'))) else 0
                eval_info_list[i]['f1'] = ifttt_f1_metric(
                    gold_decomp, pred_decomp)
                if get_split(opt) == 'test':
                    eval_info_list[i]['eval_category'] = eval_category_list[i]

            m = {}
            for eval_type in ('omit_non_eng', '+unintel', 'only>=3'):
                if get_split(opt) == 'test':
                    valid_eval_info_list = [
                        eval_info for eval_info in eval_info_list if is_valid_by_eval_type_ifttt(eval_info, eval_type)]
                else:
                    valid_eval_info_list = eval_info_list
                for metric_type in ('channel_acc', 'func_acc', 'f1'):
                    m['%s:%s' % (eval_type, metric_type)] = np.mean(
                        list(map(lambda x: x[metric_type], valid_eval_info_list)))
            # print the table of results
            table = []
            for eval_type in ('omit_non_eng', '+unintel', 'only>=3'):
                row = [eval_type]
                row += [m['%s:%s' % (eval_type, metric_type)]
                        for metric_type in ('channel_acc', 'func_acc', 'f1')]
                table.append(row)
            # print(tabulate(table, headers=[
            #       '', 'channel_acc', 'func_acc', 'f1']))
        elif opt.metric == 'word_f1':
            for i, gold_tgt, pred_tgt in izip(itertools.count(), gold_tgt_list, pred_tgt_list):
                tk_gold_tgt, tk_pred_tgt = gold_tgt.split(
                    ' '), pred_tgt.split(' ')
                min_len = min(len(tk_gold_tgt), len(tk_pred_tgt))
                c = sum([1 for it in xrange(min_len)
                         if tk_gold_tgt[it] == tk_pred_tgt[it]])
                if c == 0:
                    eval_info_list[i]['f1'] = 0
                else:
                    p = float(c) / float(len(tk_pred_tgt))
                    r = float(c) / float(len(tk_gold_tgt))
                    eval_info_list[i]['f1'] = 2 * p * r / (p + r)
            m = {'f1': np.mean(list(map(lambda x: x['f1'], eval_info_list)))}
            pprint(m)
        else:
            raise NotImplementedError

        with codecs.open('%s.%s.eval' % (opt.model, get_split(opt)), 'w', encoding='utf-8') as evalF:
            json.dump(eval_info_list, evalF, indent=2)


if __name__ == "__main__":
    opt = parser.parse_args()
    opt.cuda = True

    torch.manual_seed(123)
    torch.cuda.set_device(opt.gpu)
    torch.cuda.manual_seed(123)

    evaluate_main(opt)
