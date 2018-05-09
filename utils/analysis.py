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
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams.update({'text.usetex': True})
import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
from matplotlib import cm as cm
from sklearn.metrics import roc_curve, roc_auc_score, make_scorer
from feature import FeatureManager
from EvalInfo import EvalInfo
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn import preprocessing
import lightgbm as lgb
from scipy.stats import spearmanr, kendalltau, pearsonr
from utils import is_valid_by_eval_type_ifttt, spearmanr_significance_test, norm_by_max
from tabulate import tabulate
import multiprocessing
import seaborn as sb
sb.set_context("paper", font_scale=2)

parser = argparse.ArgumentParser(description='analysis.py')

parser.add_argument('-dataset', default="ifttt", help='dataset name;')
parser.add_argument('-metric', default="auc",
                    help='auc: ROC AUC; spearmanr: Spearman rank-order correlation coefficient; kendalltau: Kendall\'s tau, a correlation measure for ordinal data;')
parser.add_argument('-eval_type', default="omit_non_eng",
                    help='omit_non_eng/+unintel/only>=3')
parser.add_argument('-dev_prb', default="",
                    help='[dataset = upper] .prb file name for dev')
parser.add_argument('-test_prb', default="",
                    help='[dataset = upper] .prb file name for test')
parser.add_argument('-model_path', required=True,
                    help='Path to model file (can contain shell-style wildcards)')
parser.add_argument('-num_bin', type=int, default=30,
                    help='number of bins to plot accuracy')
parser.add_argument('-conf_model', default="none",
                    help='''none: do not train confidence model;
                    xgb: xgboost;
                    lgb: lightgbm;
                    svc: libsvm;
                    lr: libsvm;
                    ''')
parser.add_argument('-group', default="",
                    help="group features for evaluation.")
parser.add_argument('-significance_test', action="store_true",
                    help="significance test for spearmanr")
parser.add_argument('-plot_feature_importance', action="store_true",
                    help="plot feature importance")


def spearmanr_scorer(ground_truth, predictions):
    r, pval = spearmanr(predictions, ground_truth)
    return r


def convert_conf_feature_matrix(feature_extractor, ei_list):
    # map to feature vectors
    f_list = []
    for ei in ei_list:
        f_list.append(feature_extractor.map2vec(ei.conf))
    return f_list


def plot_conf_score_vs_f1(ei_list, opt):
    plt.clf()
    # print('plot_conf_threshold_vs_f1')
    # ei_list = list(filter(lambda it: math.fabs(it.conf['src_len']) >= 2, ei_list))
    x = np.array(range(30, 100 + 1, 10)) * 0.01
    ei_list = sorted(
        ei_list, key=lambda it: it.conf['conf_model'], reverse=True)
    y = [np.array(list(map(lambda it: it.correct, ei_list[:int(
        len(ei_list) * top_percent)]))).mean() for top_percent in x]
    ei_list = sorted(ei_list, key=lambda it: it.conf['prb'], reverse=True)
    y_prb = [np.array(list(map(lambda it: it.correct, ei_list[:int(
        len(ei_list) * top_percent)]))).mean() for top_percent in x]
    # print('\n'.join(['%d%% %f %f' % (a * 100, b, c)
    #                  for a, b, c in reversed(zip(x, y, y_prb))]))
    bar_width = 0.35
    fig, ax = plt.subplots()
    ax.grid(False)
    index = np.array(range(len(x))) + 1
    index_rev = np.array(list(reversed(index)))
    plt.bar(index_rev, y_prb, bar_width,
            alpha=0.95, label=r'\textsc{Posterior}')
    plt.bar(index_rev + bar_width, y,
            bar_width, alpha=0.95, label=r'\textsc{Conf}')

    plt.xlim([0.8, 8.5])
    if opt.dataset == 'ifttt':
        plt.ylim([0.45, 0.72])
        yticks_list = np.array(range(5, 7 + 1)) * 0.1
        plt.ylabel('F1 Score')
    elif opt.dataset == 'django':
        plt.ylim([0.4, 1.0])
        yticks_list = np.array(range(5, 10 + 1)) * 0.1
        plt.ylabel('Accuracy')
    plt.xlabel('Proportion of Examples')
    plt.xticks(index + bar_width / 2,
               reversed(map(lambda it: r'$' + str(int(x[it - 1] * 100)) + r'\%$', index)))
    plt.yticks(yticks_list, map(lambda it: r'$' + '%.1f' %
                                (it,) + r'$', yticks_list))

    leg = plt.legend(loc=2, fancybox=True, ncol=1)
    leg.get_frame().set_alpha(0.9)

    # plt.show()
    plt.savefig('analysis/conf_threshold_vs_f1.pdf', bbox_inches='tight')
    plt.clf()


def get_xgb_feature_importance(m):
    trees = m.get_dump('', with_stats=True)

    importance_type = 'gain='
    fmap = {}
    gmap = {}
    for tree in trees:
        for line in tree.split('\n'):
            # look for the opening square bracket
            arr = line.split('[')
            # if no opening bracket (leaf node), ignore this line
            if len(arr) == 1:
                continue

            # look for the closing bracket, extract only info within that bracket
            fid = arr[1].split(']')

            # extract gain or cover from string after closing bracket
            g = float(fid[1].split(importance_type)[1].split(',')[0])

            # extract feature name from string before closing bracket
            fid = fid[0].split('<')[0]

            if fid not in fmap:
                # if the feature hasn't been seen yet
                fmap[fid] = 1
                gmap[fid] = g
            else:
                fmap[fid] += 1
                gmap[fid] += g

    # calculate average value (gain/cover) for each feature
    r_dict = {}
    for fid in gmap:
        r_dict[fid] = (gmap[fid], fmap[fid])

    return r_dict


def plot_feature_importance(m, feature_extractor, opt):
    print('plot_feature_importance')
    matplotlib.rcParams.update({'text.usetex': False})
    # xgb.plot_importance(m)

    importance_dict = {}
    # for k, v in m.get_booster().get_score(importance_type='gain').iteritems():
    for k, v in get_xgb_feature_importance(m.get_booster()).iteritems():
        idx = int(k.strip('f'))
        if feature_extractor.idx2feat.has_key(idx):
            importance_dict[feature_extractor.idx2feat[idx]] = v

    label_dict = {'drp:': 'Dout', 'noise:': 'Noise', 'prb': 'PR', 'perplexity': 'PPL',
                  'lm': 'LM', 'src_unk': '\\#UNK', 'beam:': 'Var', 'ent:': 'Ent'}

    # group the features, and average the values
    group_dict = {}
    for k, v in importance_dict.iteritems():
        g_name = None
        for t in label_dict.keys():
            if k.startswith(t):
                g_name = t
                break
        if g_name is not None:
            if group_dict.has_key(g_name):
                group_dict[g_name].append(v)
            else:
                group_dict[g_name] = [v]
    print(group_dict)

    def _agg_score(l):
        # return np.max(np.array(v))
        # return np.sum(np.array(v))
        z1, z2 = zip(*v)
        return np.sum(np.array(z1)) / float(np.sum(np.array(z2)))

    org_labels, labels, values = zip(*list(sorted([(k, label_dict[k], _agg_score(v))
                                                   for k, v in group_dict.iteritems()], key=lambda x: x[-1])))

    # plot the bar chart
    _, ax = plt.subplots(1, 1)

    ylocs = np.arange(len(values))
    ax.barh(ylocs, values, align='center')

    # for x, y in zip(values, ylocs):
    #     ax.text(x + 1, y, x, va='center')

    ax.set_yticks(ylocs)
    ax.set_yticklabels(labels)

    xlim = (0, max(values) * 1.1)
    ax.set_xlim(xlim)

    ylim = (-1, len(values))
    ax.set_ylim(ylim)

    ax.set_xlabel('Metric Importance')
    # ax.set_ylabel('Metric')
    ax.grid(False)

    plt.savefig('analysis/metric_importance-%s.pdf' %
                (opt.dataset,), bbox_inches='tight')
    plt.clf()
    matplotlib.rcParams.update({'text.usetex': True})

    # dump to a latex table
    print(org_labels)
    v_dict = dict(zip(org_labels, list(norm_by_max(np.array(values)))))
    tbl_list = []
    for k in ['drp:', 'noise:', 'prb', 'perplexity', 'lm', 'src_unk', 'beam:', 'ent:']:
        tbl_list.append(' & %.2f' % (v_dict[k],))
    with open('analysis/metric_importance-%s.txt' % (opt.dataset,), 'w') as f_out:
        f_out.write('\n'.join(tbl_list))


def confidence_main(opt, fn_model, group_name):
    dev_eval_info_list = [EvalInfo(it) for it in json.load(
        open(fn_model + '.dev.eval', 'r'))]
    if opt.dataset == 'ifttt':
        test_eval_info_list = [EvalInfo(it) for it in json.load(
            open(fn_model + '.test.eval', 'r')) if is_valid_by_eval_type_ifttt(it, opt.eval_type)]
    else:
        test_eval_info_list = [EvalInfo(it) for it in json.load(
            open(fn_model + '.test.eval', 'r'))]

    if opt.conf_model != 'none':
        feature_extractor = FeatureManager()
        if group_name is not None:
            feature_extractor.set_valid_uncertainty_type(group_name)

        X_test = np.array(convert_conf_feature_matrix(
            feature_extractor, test_eval_info_list))
        y_test = np.array([it.correct for it in test_eval_info_list])

        X = np.array(convert_conf_feature_matrix(
            feature_extractor, dev_eval_info_list))
        y = np.array([it.correct for it in dev_eval_info_list])
        X, y = shuffle(X, y, random_state=123)

        def expand_with_small_noise(t, n):
            t_new = np.random.normal(0, 1e-6, (t.shape[0], 6))
            t_new[:, 0:t.shape[1]] = t
            return t_new

        if X_test.shape[1] < 6:
            X_test = expand_with_small_noise(X_test, 6)
            X = expand_with_small_noise(X, 6)

        # train model
        print('Training confidence model...')
        if opt.conf_model == 'lr':
            # more features, more estimators
            if (group_name is None) or (group_name == 'all'):
                _n_estimators = 50
            else:
                _n_estimators = 20
            param_grid = {'max_depth': [3, 4, 5], 'min_child_weight': [
                1, 3, 5], 'gamma': [0], 'n_estimators': [_n_estimators], 'learning_rate': [0.1], 'subsample': [0.8], 'colsample_bytree': [0.8], 'reg_alpha': [1e-1, 1e-2], 'reg_lambda': [1e-1, 1e-2]}
            param_fit = {'eval_metric': 'auc', 'verbose': False,
                         'early_stopping_rounds': 1}
            model_builder = xgb.XGBRegressor(
                random_state=123, booster='gbtree', objective='reg:logistic')

            conf_model = GridSearchCV(model_builder, param_grid,
                                      # fit_params=param_fit,
                                      cv=8, verbose=False, n_jobs=multiprocessing.cpu_count() - 1, scoring=make_scorer(spearmanr_scorer, greater_is_better=True))
        else:
            raise NotImplementedError

        conf_model.fit(X, y)
        # print('Best score:', conf_model.best_score_)
        # print('Best param:', conf_model.best_params_)

        if opt.plot_feature_importance:
            plot_feature_importance(
                conf_model.best_estimator_, feature_extractor, opt)

        if opt.conf_model == 'lr':
            y_pred = conf_model.predict(X_test)
        else:
            y_pred = conf_model.predict_proba(X_test)[:, 1]
        for i, it in enumerate(test_eval_info_list):
            it.conf['conf_model'] = y_pred[i]

    metric_result_list = []
    save_name_list = test_eval_info_list[0].conf.keys()
    for save_name in save_name_list:
        scores = np.array([it.conf[save_name]
                            for it in test_eval_info_list])
        if opt.metric == 'spearmanr':
            r, pval = spearmanr(scores, y_test)
        elif opt.metric == 'kendalltau':
            r, pval = kendalltau(scores, y_test)
        else:
            raise NotImplementedError
        metric_result_list.append((save_name, r))
    if opt.conf_model != 'none':
        plot_conf_score_vs_f1(test_eval_info_list, opt)
        if opt.significance_test:
            scores_conf = np.array([it.conf['conf_model']
                                    for it in test_eval_info_list])
            scores_prb = np.array([it.conf['prb']
                                    for it in test_eval_info_list])
            print('significance_test: %f' % (
                spearmanr_significance_test(scores_conf, scores_prb, y_test),))

    return metric_result_list, np.array([it.conf.get('conf_model', 0) for it in test_eval_info_list]), np.array([it.correct for it in test_eval_info_list])


def summurize_result(r_list):
    r_dict = {}
    for metric_result_list in r_list:
        for save_name, v in metric_result_list:
            if r_dict.has_key(save_name):
                r_dict[save_name].append(v)
            else:
                r_dict[save_name] = [v]
    metric_result_list = [(k, np.mean(np.array(v)))
                          for k, v in r_dict.iteritems()]
    # filter NaN
    metric_result_list = list(
        filter(lambda x: not math.isnan(x[1]), metric_result_list))
    # sort according to auc, and print
    metric_result_list.sort(key=lambda x: x[1], reverse=True)
    return metric_result_list


def wrap_multiple_col(s, n):
    if n <= 1:
        return s
    else:
        s_list = s.split('\n')
        group_list = [list(itertools.chain(s_list[:2], it))
                      for it in more_itertools.divide(n, s_list[2:])]
        max_len = max([len(it) for it in group_list])
        for it in group_list:
            for i in xrange(max_len - len(it)):
                it.append('')
        return '\n'.join(['\t\t'.join(it) for it in zip(*group_list)])


if __name__ == "__main__":
    opt = parser.parse_args()

    if len(opt.group) > 0:
        summary_metric_result_list, summary_s_list = [], []
        significance_dict = {}
        for group_name in opt.group.split(','):
            r_list, s_list = [], []
            for fn_model in glob.glob(opt.model_path):
                metric_result_list, scores, y_test = confidence_main(
                    opt, fn_model, group_name)
                significance_dict[group_name] = (scores, y_test)
                r_list.append(metric_result_list)
                s_list.append(scores)
            summary_metric_result_list.append((group_name, list(
                filter(lambda x: x[0] == 'conf_model', summurize_result(r_list)))[0][1]))
            summary_s_list.append((group_name, np.array(s_list).mean(axis=0)))
        if opt.significance_test and ('all' in opt.group.split(',')):
            for group_name in opt.group.split(','):
                if group_name != 'all':
                    assert(np.array_equal(
                        significance_dict['all'][1], significance_dict[group_name][1]))
                    print('%s - %s: %f' % ('all', group_name, spearmanr_significance_test(
                        significance_dict['all'][0], significance_dict[group_name][0], significance_dict['all'][1])))
    else:
        r_list = []
        for fn_model in glob.glob(opt.model_path):
            r_list.append(confidence_main(opt, fn_model, None)[0])
        summary_metric_result_list = list(filter(lambda x: x[0] == 'conf_model', summurize_result(r_list)))
    print(wrap_multiple_col(tabulate([(save_name, auc) for save_name, auc in summary_metric_result_list], headers=[
          'Method', opt.metric], floatfmt=".5f"), min(int(len(summary_metric_result_list) / 15.0), 2)))
