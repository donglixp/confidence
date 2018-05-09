
class Tree(object):

  def __init__(self, vocab_dict):
    self.parent = None
    self.child = []
    self.vocab_dict = vocab_dict

  def idx2sym(self, tk):
    return self.vocab_dict.getLabel(tk)

  def init_from_list(self, r_list, i_left=None, i_right=None):
    if i_left is None:
      i_left = 0
    if i_right is None:
      i_right = len(r_list)
    level = 0
    left = -1
    for i in xrange(i_left, i_right):
      if r_list[i] == self.vocab_dict.lookup('('):
        if level == 0:
          left = i
        level = level + 1
      elif r_list[i] == self.vocab_dict.lookup(')'):
        level = level - 1
        if level == 0:
          c = Tree(self.vocab_dict)
          c.init_from_list(r_list, left + 1, i)
          self.add_child(c)
      elif level == 0:
        self.add_child(r_list[i])

  def init_from_str(self, s):
    idx_list = self.vocab_dict.convertToIdx_list(filter(lambda x: len(
        x) > 0, s.replace('(', ' ( ').replace(')', ' ) ').strip().split(' ')), '<u>')
    if len(idx_list) > 0:
      if (idx_list[0] == self.vocab_dict.lookup('(')) and (idx_list[-1] == self.vocab_dict.lookup(')')):
        self.init_from_list(idx_list, 1, len(idx_list) - 1)
      else:
        self.init_from_list(idx_list)

  def add_child(self, c):
    if isinstance(c, Tree):
      c.parent = self
    self.child.append(c)

  def __str__(self):
    return ' '.join(map(lambda x: self.vocab_dict.getLabel(x), self.to_list()))

  def to_list(self):
    s_list = []
    for c in self.child:
      if isinstance(c, Tree):
        s_list.append(self.vocab_dict.lookup('('))
        s_list.extend(c.to_list())
        s_list.append(self.vocab_dict.lookup(')'))
      else:
        s_list.append(c)
    return s_list

  def child_to_str(self, c):
    if isinstance(c, Tree):
      return str(c)
    else:
      return self.vocab_dict.getLabel(c)

  def normalize(self):
    for c in self.child:
      if isinstance(c, Tree):
        c.normalize()
    if (len(self.child) > 1) and (self.child[0] in (self.vocab_dict.lookup('and'), self.vocab_dict.lookup('or'))):
      new_child = [self.child[0], ]
      new_child.extend(
          sorted(self.child[1:], key=lambda x: self.child_to_str(x)))
      self.child = new_child

  def is_eq(self, t):
    return self.to_list() == t.to_list()

if __name__ == "__main__":
  import sys
  import os
  sys.path.append(os.path.join(os.path.dirname(
      os.path.realpath(__file__)), '../onmt/'))
  from Dict import Dict
  d = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4,
       'and': 5, '(': 6, ')': 7, '<u>': -1, 'or': 8}
  vocab_dict = Dict(d.keys())
  t = Tree(vocab_dict)
  s = "(a (and e d b) (or c d a))"
  print(s)
  t.init_from_str(s)
  print(t)
  print(t.to_list())
  t.normalize()
  print(t)
