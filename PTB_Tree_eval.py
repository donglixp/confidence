# -*- coding: utf-8 -*-
import sys
import re


def quota_normalize(s):
  if s.startswith("{'lat'"):
    return "_{geo}"
  elif s.startswith("{'name'"):
    return "_{name}"
  return s


class PTB_Tree_eval:
  '''Tree for PTB format

  >>> tree = PTB_Tree_eval()
  >>> tree.set_by_text("(ROOT (NP (NNP Newspaper)))")
  >>> print tree
  (ROOT (NP (NNP Newspaper)))
  >>> tree = PTB_Tree_eval()
  >>> tree.set_by_text("(ROOT (S (NP-SBJ (NNP Ms.) (NNP Haag) ) (VP (VBZ plays) (NP (NNP Elianti) )) (. .) ))")
  >>> print tree
  (ROOT (S (NP-SBJ (NNP Ms.) (NNP Haag)) (VP (VBZ plays) (NP (NNP Elianti))) (. .)))
  >>> print tree.word_yield()
  Ms. Haag plays Elianti .
  >>> tree = PTB_Tree_eval()
  >>> tree.set_by_text("(ROOT (NFP ...))")
  >>> print tree
  (ROOT (NFP ...))
  >>> tree.word_yield()
  '...'
  '''
# Convert text from the PTB to a tree.  For example:
# ( (S (NP-SBJ (NNP Ms.) (NNP Haag) ) (VP (VBZ plays) (NP (NNP Elianti) )) (.
# .) ))
# This is a compressed form of:
# ( (S
#     (NP-SBJ (NNP Ms.) (NNP Haag))
#     (VP (VBZ plays)
#       (NP (NNP Elianti)))
#     (.  .)))

  def __init__(self, text=None):
    self.subtrees = []
    self.text = None
    if text != None:
      self.set_by_text(text)

  def set_by_text(self, text, pos=0):
    depth = 0
    no_quot_flag = True
    for i in xrange(pos + 1, len(text)):
      char = text[i]
      if char == '\"':
        no_quot_flag = not no_quot_flag
      elif no_quot_flag:
        # update the depth
        if char == '(':
          depth += 1
          if depth == 1:
            subtree = PTB_Tree_eval()
            subtree.set_by_text(text, i)
            self.subtrees.append(subtree)
            if self.text is None:
              self.text = text[pos + 1:i].strip()
        elif char == ')':
          depth -= 1
          if len(self.subtrees) == 0:
            if depth >= 0:
              print 'ERR: depth >= 0'
            self.text = text[pos + 1:i]
      # we've reached the end of the scope for this bracket
      if depth < 0:
        break

  def get_production_list(self, depth=0):
    r_list = []
    if len(self.subtrees) > 0:
      # add production of this level
      if depth > 0:
        prod = ['(', self.text]
        for subtree in self.subtrees:
          prod.append('(%s)' % (subtree.text,))
        prod.append(')')
        r_list.append(' '.join(prod))
      # travel to subtree
      for subtree in self.subtrees:
        r_list.extend(subtree.get_production_list(depth + 1))
    return r_list
