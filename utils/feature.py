

class FeatureManager(object):

    def __init__(self):
        super(FeatureManager, self).__init__()
        self.feat2idx = {}
        self.idx2feat = {}
        self.start_idx = 0
        self.block_set = set()
        self.block_option_set = set()
        self.valid_uncertainty_type = None

    def set_valid_uncertainty_type(self, t):
        self.valid_uncertainty_type = t
        if t == 'all':
            self.valid_prefix_set = set(('',))
        elif t == 'model':
            self.valid_prefix_set = set(
                ('drp:', 'noise:', 'prb', 'perplexity'))
        elif t == 'data':
            self.valid_prefix_set = set(('lm', 'src_unk', 'src_len'))
        elif t == 'input':
            self.valid_prefix_set = set(('beam:', 'ent:'))
        elif t == 'nomodel':
            self.valid_prefix_set = set(
                ('lm', 'src_unk', 'src_len', 'beam:', 'ent:'))
        elif t == 'nodata':
            self.valid_prefix_set = set(
                ('drp:', 'noise:', 'prb', 'perplexity', 'beam:', 'ent:'))
        elif t == 'noinput':
            self.valid_prefix_set = set(
                ('drp:', 'noise:', 'prb', 'perplexity', 'lm', 'src_unk', 'src_len'))
        elif t in ('token', 'seq'):
            self.token_valid_suffix_set = set((':max', ':min', ':sum', ':avg'))
        elif t in ('drp:', 'noise:', 'prb', 'perplexity', 'lm', 'src_unk', 'src_len', 'beam:', 'ent:'):
            self.valid_prefix_set = set((t,))
        else:
            raise NotImplementedError

    def is_idx_blocked(self, idx):
        return self.is_feat_blocked(self.idx2feat.get(idx, ''))

    def is_feat_blocked(self, f):
        option_list = f.split(':')
        for option in option_list:
            if option in self.block_option_set:
                return True

        if (f in self.block_set):
            return True

        if (self.valid_uncertainty_type in ('model', 'data', 'input')) or (self.valid_uncertainty_type in ('nomodel', 'nodata', 'noinput')) or (self.valid_uncertainty_type in ('drp:', 'noise:', 'prb', 'perplexity', 'lm', 'src_unk', 'src_len', 'beam:', 'ent:')):
            is_valid = False
            for valid_prefix in self.valid_prefix_set:
                if f.startswith(valid_prefix):
                    is_valid = True
                    break
            if not is_valid:
                return True
        elif self.valid_uncertainty_type in ('token', 'seq'):
            is_token = False
            for token_valid_suffix in self.token_valid_suffix_set:
                if f.endswith(token_valid_suffix):
                    is_token = True
                    break
            if not ((is_token and (self.valid_uncertainty_type == 'token')) or (not(is_token) and (self.valid_uncertainty_type == 'seq'))):
                return True

        return False

    def add_feature_name(self, k):
        if (not self.feat2idx.has_key(k)) and (not self.is_feat_blocked(k)):
            idx = len(self.feat2idx) + self.start_idx
            self.feat2idx[k] = idx
            self.idx2feat[idx] = k

    def map2vec(self, ft_dict):
        f = []
        for k, v in ft_dict.iteritems():
            if (not self.feat2idx.has_key(k)) and (not self.is_feat_blocked(k)):
                self.add_feature_name(k)
        for i in xrange(self.start_idx, len(self.feat2idx) + self.start_idx):
            if not self.is_idx_blocked(i):
                f.append(ft_dict.get(self.idx2feat[i], 0))
        return f
