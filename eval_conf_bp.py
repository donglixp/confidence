from __future__ import division

import os
import sys
import codecs
import argparse
import math
from path import Path
from tqdm import tqdm
import glob
import json
import itertools
import more_itertools
import numpy as np
from scipy.stats import entropy, spearmanr, kendalltau
from tabulate import tabulate
import onmt
from utils.EvalInfo import EvalInfo
from utils.utils import get_split, safe_numpy_div, softmax, norm_to_range, list_topk
import random as rnd
from multiprocessing import Process, Queue, Pool
rnd.seed(123)

parser = argparse.ArgumentParser(description='eval_conf_bp.py')

parser.add_argument('-dataset', default="ifttt", help='dataset name;')
parser.add_argument('-metric', default="kl",
                    help='evaluation metric: kl/spearmanr/intersectionK')
parser.add_argument('-model_path', required=True,
                    help='Path to model file (can contain shell-style wildcards)')
parser.add_argument('-src', required=True,
                    help='Source sequence to decode (one line per sequence)')


# KL divergence
def compute_KL_score(_pred, _gold, k=4):
    p = softmax(norm_to_range(np.array(_pred)))
    g = softmax(norm_to_range(np.array(_gold)))
    return entropy(g, p)


def compute_spearmanr_score(_pred, _gold, k=4):
    r, pval = spearmanr(_pred, _gold)
    return r


def compute_intersection_score(_pred, _gold, k=4):
    return float(len(set(list_topk(_pred, k)) & set(list_topk(_gold, k)))) / float(min(k, len(_pred)))


# (larger -> better)
def eval_func(_in):
    conf_each_word_method, dev_eval_info_list, opt_metric = _in

    metric_k = 4
    if opt_metric == 'kl':
        compute_score_func = compute_KL_score
        larger_is_better = False
    elif opt_metric == 'spearmanr':
        compute_score_func = compute_spearmanr_score
        larger_is_better = True
    elif opt_metric.startswith('intersection'):
        compute_score_func = compute_intersection_score
        larger_is_better = True
        metric_k = int(opt_metric.split('tion')[1])

    conf_bp_method_list = sorted(dev_eval_info_list[0].conf_bp_src.keys())
    s = {}
    for conf_bp_method in conf_bp_method_list:
        s['bp-%s' % (conf_bp_method,)] = []
        s['att-%s' % (conf_bp_method,)] = []
    for eval_info in dev_eval_info_list:
        src_word_uncert = eval_info.conf_each_word[conf_each_word_method]
        for conf_bp_method in conf_bp_method_list:
            s['bp-%s' % (conf_bp_method,)].append(compute_score_func(
                eval_info.conf_bp_src[conf_bp_method], src_word_uncert, metric_k))
            s['att-%s' % (conf_bp_method,)].append(compute_score_func(
                eval_info.conf_att_src[conf_bp_method], src_word_uncert, metric_k))
    for conf_bp_method in conf_bp_method_list:
        s['bp-%s' % (conf_bp_method,)] = np.mean(s['bp-%s' %
                                                   (conf_bp_method,)])
        s['att-%s' % (conf_bp_method,)] = np.mean(s['att-%s' %
                                                    (conf_bp_method,)])
    return s


def eval_main(opt, fn_model):
    dev_eval_info_list = [EvalInfo(it) for it in json.load(
        open('%s.%s.eval' % (fn_model, get_split(opt)), 'r'))]
    # read source tokens
    with codecs.open(opt.src, 'r', encoding='utf-8') as f_in:
        for i, l in enumerate(f_in):
            dev_eval_info_list[i].src = l.strip().split(' ')
    # filter: src_len >= 2
    dev_eval_info_list = list(
        filter(lambda x: len(x.src) >= 2, dev_eval_info_list))

    # start evaluation
    r = eval_func(('noise:enc_word:mul:exp:miu_norm',
                   dev_eval_info_list, opt.metric))
    print(opt.metric)
    print('ATT: %f' % r['att-noise:enc_word:mul:miu_norm'])
    print('BP: %f' % r['bp-noise:enc_word:mul:miu_norm'])
    print('')


if __name__ == "__main__":
    opt = parser.parse_args()

    for fn_model in glob.glob(opt.model_path):
        eval_main(opt, fn_model)
