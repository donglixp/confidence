from __future__ import division

import os
import codecs
import argparse
import math
from path import Path
from tqdm import tqdm
import glob
import json
import itertools
import more_itertools


class EvalInfo(object):

    def __init__(self, it):
        super(EvalInfo, self).__init__()

        self.conf = {}
        self.compute_confidence_scores(it)

        self.conf_bp_src = it['conf_bp_src']
        self.conf_att_src = it['conf_att_src']
        self.conf_tgt = it['conf_tgt']
        self.conf_each_word = it['conf_each_word']

    def compute_confidence_scores(self, it):
        def get_score(logged_score, use_log_score):
            return logged_score if use_log_score else math.exp(logged_score)

        def zero_add_one(v):
            return 1 if v == 0 else v

        self.correct = it['f1'] if it.has_key('f1') else it['acc']
        self.conf['prb'] = get_score(it['pred_score'], False)
        self.conf['perplexity'] = get_score(
            it['pred_score'], True) / float(zero_add_one(it['pred_len']))
        self.conf['src_len'] = -it['src_len']
        self.conf['pred_len'] = -it['pred_len']
        # self.conf['src_unk'] = -it['src_unk']
        self.conf['src_unk:norm'] = - \
            float(it['src_unk']) / float(it['src_len'])
        for conf_method in it.get('confidence', {}).keys():
            if conf_method == 'lm':
                self.conf['lm:norm'] = it['confidence']['lm'] / \
                    float(it['src_len'])
            else:
                self.conf[conf_method] = it['confidence'][conf_method]
                if conf_method.endswith(':sum'):
                    self.conf[conf_method + ':avg'] = self.conf[conf_method] / \
                        float(zero_add_one(it['pred_len']))
