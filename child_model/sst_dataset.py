# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import io
import sys
import csv
import json

import numpy as np
import math
import random
import torch
import pickle


# Taken from Jonathan K's Berkeley parser with minor modification
class PTBTree:
  WORD_TO_WORD_MAPPING = {
      "{": "-LCB-",
      "}": "-RCB-"
  }

  def __init__(self):
    self.subtrees = []
    self.word = None
    self.label = ""
    self.parent = None
    self.span = (-1, -1)
    self.word_vector = None  # HOS, store dx1 RNN word vector
    self.prediction = None  # HOS, store Kx1 prediction vector

  def is_leaf(self):
    return len(self.subtrees) == 0

  def set_by_text(self, text, pos=0, left=0):
    depth = 0
    right = left
    for i in range(pos + 1, len(text)):
      char = text[i]
      # update the depth
      if char == "(":
        depth += 1
        if depth == 1:
          subtree = PTBTree()
          subtree.parent = self
          subtree.set_by_text(text, i, right)
          right = subtree.span[1]
          self.span = (left, right)
          self.subtrees.append(subtree)
      elif char == ")":
        depth -= 1
        if len(self.subtrees) == 0:
          pos = i
          for j in range(i, 0, -1):
            if text[j] == " ":
              pos = j
              break
          self.word = text[pos + 1:i]
          self.span = (left, left + 1)

      # we've reached the end of the category that is the root of this subtree
      if depth == 0 and char == " " and self.label == "":
        self.label = text[pos + 1:i]
      # we've reached the end of the scope for this bracket
      if depth < 0:
        break

    # Fix some issues with variation in output, and one error in the treebank
    # for a word with a punctuation POS
    self.standardise_node()

  def standardise_node(self):
    if self.word in self.WORD_TO_WORD_MAPPING:
      self.word = self.WORD_TO_WORD_MAPPING[self.word]

  def __repr__(self, single_line=True, depth=0):
    ans = ""
    if not single_line and depth > 0:
      ans = "\n" + depth * "\t"
    ans += "(" + self.label
    if self.word is not None:
      ans += " " + self.word
    for subtree in self.subtrees:
      if single_line:
        ans += " "
      ans += subtree.__repr__(single_line, depth + 1)
    ans += ")"
    return ans

def read_tree(source):
  cur_text = []
  depth = 0
  while True:
    line = source.readline()
    # Check if we are out of input
    if line == "":
      return None
    # strip whitespace and only use if this contains something
    line = line.strip()
    if line == "":
      continue
    cur_text.append(line)
    # Update depth
    for char in line:
      if char == "(":
        depth += 1
      elif char == ")":
        depth -= 1
    # At depth 0 we have a complete tree
    if depth == 0:
      tree = PTBTree()
      tree.set_by_text(" ".join(cur_text))
      return tree
  return None

def read_trees(source, max_sents=-1):
  with open(source) as fp:
    trees = []
    while True:
      tree = read_tree(fp)
      if tree is None:
        break
      trees.append(tree)
      if len(trees) >= max_sents > 0:
        break
    return trees

def sst_load_trees(filename):
  trees = read_trees(filename)
  return trees

def sst_get_id_input(content, word_id_dict, max_input_length):
    words = content.split(' ')
    sentence = [word_id_dict["<pad>"]] * max_input_length
    mask = [0] * max_input_length
    for i, word in enumerate(words[:max_input_length]):
        sentence[i] = word_id_dict.get(word, word_id_dict["<unknown>"])
        mask[i] = 1
    return sentence, mask

def sst_get_phrases(trees, sample_ratio=1.0, is_binary=False, only_sentence=False):
  all_phrases = []
  for tree in trees:
    if only_sentence == True:
      sentence = get_sentence_by_tree(tree)
      label = int(tree.label)
      pair = (sentence, label)
      all_phrases.append(pair)
    else:
      phrases = get_phrases_by_tree(tree)
      sentence = get_sentence_by_tree(tree)
      pair = (sentence, int(tree.label))
      all_phrases.append(pair)
      all_phrases.extend(phrases)
  np.random.shuffle(all_phrases)
  result_phrases = []
  for pair in all_phrases:
    if is_binary:
      phrase = pair[0]
      label = pair[1]
      if label <= 1:
        pair = (phrase, 0)
      elif label >= 3:
        pair = (phrase, 1)
      else:
        continue
    if sample_ratio == 1.0:
      result_phrases.append(pair)
    else:
      rand_portion = np.random.random()
      if rand_portion < sample_ratio:
        result_phrases.append(pair)
  return result_phrases

def get_phrases_by_tree(tree):
  phrases = []
  if tree == None:
    return phrases
  if tree.is_leaf():
    pair = (tree.word, int(tree.label))
    phrases.append(pair)
    return phrases
  left_child_phrases = get_phrases_by_tree(tree.subtrees[0])
  right_child_phrases = get_phrases_by_tree(tree.subtrees[1])
  phrases.extend(left_child_phrases)
  phrases.extend(right_child_phrases)
  sentence = get_sentence_by_tree(tree)
  pair = (sentence, int(tree.label))
  phrases.append(pair)
  return phrases

def get_sentence_by_tree(tree):
  sentence = ""
  if tree == None:
    return sentence
  if tree.is_leaf():
    return tree.word
  left_sentence = get_sentence_by_tree(tree.subtrees[0])
  right_sentence = get_sentence_by_tree(tree.subtrees[1])
  sentence = left_sentence + " " + right_sentence
  return sentence.strip()

def get_word_id_dict(word_num_dict, word_id_dict, min_count):
  z = [k for k in sorted(word_num_dict.keys())]
  for word in z:
    count = word_num_dict[word]
    if count >= min_count:
      index = len(word_id_dict)
      if word not in word_id_dict:
        word_id_dict[word] = index
  return word_id_dict

def load_word_num_dict(phrases, word_num_dict):
  for (sentence, label) in phrases:
    words = sentence.split(' ')
    for cur_word in words:
      word = cur_word.strip()
      if word not in word_num_dict:
        word_num_dict[word] = 1
      else:
        word_num_dict[word] += 1
  return word_num_dict

def init_trainable_embedding(embedding, word_id_dict, word_embed_model, unknown_word_embed, embed_dim=300):
  embed_dim = unknown_word_embed.shape[0]
  embedding[0] = np.zeros(embed_dim)
  embedding[1] = unknown_word_embed
  for word in sorted(word_id_dict.keys()):
    idx = word_id_dict[word]
    if idx == 0 or idx == 1:
      continue
    if word in word_embed_model:
      embedding[idx] = word_embed_model[word]
    else:
      embedding[idx] = np.random.rand(embed_dim) / 2.0 - 0.25
  return embedding

def sst_get_trainable_data(phrases, word_id_dict, word_embed_model,
                           split_label, max_input_length, is_binary):
  sent_ids, labels, mask = [], [], []

  for (phrase, label) in phrases:
    if len(phrase.split(' ')) < 1:
      continue
    phrase_input, mask_input = sst_get_id_input(phrase, word_id_dict, max_input_length)
    sent_ids.append(phrase_input)
    mask.append(mask_input)
    labels.append(int(label))
  labels = np.array(labels, dtype=np.int64)
  if split_label == 1:
    split_label_str = "train"
  elif split_label == 2:
    split_label_str = "test"
  else:
    split_label_str = "valid"
  sent_ids = np.reshape(sent_ids, [-1, max_input_length]).astype(np.int64)
  mask = np.reshape(mask, [-1, max_input_length]).astype(np.int64)
  print(split_label_str, sent_ids.shape, labels.shape, mask.shape)
  return sent_ids, labels, mask

def load_glove_model(filename, embed_dim):
  if os.path.exists(filename + ".cache"):
    print("found cache. loading...")
    with open(filename + ".cache", "rb") as fp:
      return pickle.load(fp)

  embedding_dict = {}
  with open(filename) as f:
    for line in f:
      vocab_word, vec = line.strip().split(' ', 1)
      embed_vector = list(map(float, vec.split()))
      embedding_dict[vocab_word] = embed_vector

  with open(filename + ".cache", "wb") as fp:
    pickle.dump(embedding_dict, fp)
  return embedding_dict

def load_embedding(embedding_model, embedding_path, embed_dim=300):
  word_embed_model = {}

  if embedding_model == "glove" or embedding_model == "all":
    word_embed_model["glove"] = load_glove_model(embedding_path, embed_dim)

  unknown_word_embed = np.random.rand(embed_dim)
  unknown_word_embed = (unknown_word_embed - 0.5) / 2.0
  return word_embed_model, unknown_word_embed

def read_data_sst(data_path, max_input_length, embedding_model, min_count,
                  train_ratio, valid_ratio, embedding_path,
                  is_binary=False, embed_dim=300):
  word_id_dict = {}
  word_num_dict = {}
  sent_ids, labels, mask = {}, {}, {}

  print("-" * 80)
  print("Reading SST data")

  train_file_name = os.path.join(data_path, 'train.txt')
  valid_file_name = os.path.join(data_path, 'dev.txt')
  test_file_name = os.path.join(data_path, 'test.txt')

  train_trees = sst_load_trees(train_file_name)
  train_phrases = sst_get_phrases(train_trees, train_ratio, is_binary, False)
  print("finish load train_phrases")
  valid_trees = sst_load_trees(valid_file_name)
  valid_phrases = sst_get_phrases(valid_trees, valid_ratio, is_binary, False)
  train_phrases = train_phrases + valid_phrases
  valid_phrases = None
  test_trees = sst_load_trees(test_file_name)
  test_phrases = sst_get_phrases(test_trees, valid_ratio, is_binary, True)
  print("finish load test_phrases")

  word_id_dict["<pad>"] = 0
  word_id_dict["<unknown>"] = 1
  load_word_num_dict(train_phrases, word_num_dict)
  print("finish load train words: {0}".format(len(word_num_dict)))
  load_word_num_dict(test_phrases, word_num_dict)
  print("finish load test words: {0}".format(len(word_num_dict)))
  word_id_dict = get_word_id_dict(word_num_dict, word_id_dict, min_count)
  print("after trim words: {0}".format(len(word_id_dict)))

  word_embed_model, unknown_word_embed = load_embedding(embedding_model, embedding_path, embed_dim)
  embedding = {}
  for model_name in word_embed_model:
    embedding[model_name] = np.random.random([len(word_id_dict), embed_dim]).astype(np.float32) / 2.0 - 0.25
    embedding[model_name] = init_trainable_embedding(embedding[model_name], word_id_dict,
                            word_embed_model[model_name], unknown_word_embed, embed_dim)

  embedding["none"] = np.random.random([len(word_id_dict), embed_dim]).astype(np.float32) / 2.0 - 0.25
  embedding["none"][0] = np.zeros([embed_dim])

  print("finish initialize word embedding")

  sent_ids["train"], labels["train"], mask["train"] = sst_get_trainable_data(
      train_phrases, word_id_dict, word_embed_model, 1, max_input_length, is_binary)
  sent_ids["test"], labels["test"], mask["test"] = sst_get_trainable_data(
      test_phrases, word_id_dict, word_embed_model, 2, max_input_length, is_binary)

  return sent_ids, mask, labels, embedding

class SSTDataset(torch.utils.data.Dataset):
  def __init__(self, sents, mask, labels):
    self.sents = sents
    self.labels = labels
    self.mask = mask

  def __getitem__(self, index):
    return self.sents[index], self.mask[index], self.labels[index]

  def __len__(self):
    return len(self.sents)
