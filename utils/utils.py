from collections import OrderedDict
from scipy.stats import spearmanr, kendalltau
import numpy as np


def list_topk(l, k):
    return list(reversed(np.array(l).argsort()))[:k]


def is_valid_by_eval_type_ifttt(eval_info, eval_type):
    if (eval_type in ('omit_non_eng', '+unintel')) and (eval_info['eval_category'] == 0):
        return False
    if (eval_type == '+unintel') and (eval_info['eval_category'] == 1):
        return False
    if (eval_type == 'only>=3') and (eval_info['eval_category'] != 2):
        return False
    return True


def add_eps(x, inplace=False):
    if inplace:
        return x.masked_fill_(x.eq(0), 1e-6)
    else:
        return x.clone().masked_fill_(x.eq(0), 1e-6)

    x_sign = x.sign()
    x_sign.masked_fill_(x_sign.eq(0), 1)
    x_sign.mul_(0.3)
    if inplace:
        return x.add_(x_sign)
    else:
        return x + x_sign


def add_eps_(x):
    return add_eps(x, True)


def num_nan(t):
    return (t != t).sum()


def get_split(opt):
    if opt.src.find('train') >= 0:
        return 'train'
    elif opt.src.find('dev') >= 0:
        return 'dev'
    elif opt.src.find('test') >= 0:
        return 'test'
    elif opt.src.find('1k') >= 0:
        return '1k'
    else:
        print(opt.src)
        raise NotImplementedError


def line_iter(f):
    for line in f:
        yield line
    yield None


def safe_numpy_div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[c == np.inf] = 0
        c = np.nan_to_num(c)
    return c


def norm_to_range(m):
    return safe_numpy_div(m - m.min(), m.ptp())


def norm_by_sum(m):
    return safe_numpy_div(m, m.sum())


def norm_by_max(m):
    return safe_numpy_div(m, m.max())


def trim_tensor_by_length(t, len_list):
    return [t[i].tolist()[:len_list[i]] for i in xrange(t.size(0))]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def spearmanr_significance_test(scores_conf, scores_prb, y_test):
    better_list = []
    for i_sig in xrange(10000):
        sp_mask = np.random.choice(
            [True, False], size=(len(y_test),), p=[0.7, 0.3])
        v_conf, __ = spearmanr(
            scores_conf[sp_mask], y_test[sp_mask])
        v_prb, __ = spearmanr(scores_prb[sp_mask], y_test[sp_mask])
        if v_conf > v_prb:
            better_list.append(1)
        else:
            better_list.append(0)
    return np.mean(better_list)
