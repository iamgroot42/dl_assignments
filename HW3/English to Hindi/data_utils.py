# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for  tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib

from indicnlp.tokenize import indic_tokenize
from nltk.tokenize.moses import MosesTokenizer

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

# URLs for WMT data.
IITB_TRAIN_DIR = "./data/IITB.en-hi"
IITB_DEV_DIR = "./data/dev"


def tokenize_hindi(sentence):
  return indic_tokenize.trivial_tokenize(sentence.decode('utf-8'))


def tokenize_english(sentence):
  tknzr = MosesTokenizer()
  return tknzr.tokenize(sentence.decode('utf-8'))


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        line = tf.compat.as_bytes(line)
        tokens = tokenizer(line)
        for w in tokens:
          word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  words = tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                            tokenizer, normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_iitb_data(data_dir, en_vocabulary_size, hi_vocabulary_size):
  """Get IITB data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    en_vocabulary_size: size of the English vocabulary to create and use.
    hi_vocabulary_size: size of the Hindi vocabulary to create and use.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.

  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for English training data-set,
      (2) path to the token-ids for Hindi training data-set,
      (3) path to the token-ids for English development data-set,
      (4) path to the token-ids for Hindi development data-set,
      (5) path to the English vocabulary file,
      (6) path to the Hindi vocabulary file.
  """
  # Get iitb data to the specified directory.
  en_tokenizer = tokenize_english
  hi_tokenizer = tokenize_hindi
  from_train_path = IITB_TRAIN_DIR + ".en"
  to_train_path = IITB_TRAIN_DIR + ".hi"
  from_dev_path = IITB_DEV_DIR + ".en"
  to_dev_path = IITB_DEV_DIR + ".hi"
  return prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path, en_vocabulary_size,
                      hi_vocabulary_size, en_tokenizer, hi_tokenizer)


def prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path, from_vocabulary_size,
                 to_vocabulary_size, en_tokenizer, hi_tokenizer):

  # Create vocabularies of the appropriate sizes.
  to_vocab_path = os.path.join(data_dir, "vocab%d.to" % to_vocabulary_size)
  from_vocab_path = os.path.join(data_dir, "vocab%d.from" % from_vocabulary_size)
  create_vocabulary(to_vocab_path, to_train_path , to_vocabulary_size, hi_tokenizer)
  create_vocabulary(from_vocab_path, from_train_path , from_vocabulary_size, en_tokenizer)

  # Create token ids for the training data.
  to_train_ids_path = to_train_path + (".ids%d" % to_vocabulary_size)
  from_train_ids_path = from_train_path + (".ids%d" % from_vocabulary_size)
  data_to_token_ids(to_train_path, to_train_ids_path, to_vocab_path, hi_tokenizer)
  data_to_token_ids(from_train_path, from_train_ids_path, from_vocab_path, en_tokenizer)

  # Create token ids for the development data.
  to_dev_ids_path = to_dev_path + (".ids%d" % to_vocabulary_size)
  from_dev_ids_path = from_dev_path + (".ids%d" % from_vocabulary_size)
  data_to_token_ids(to_dev_path, to_dev_ids_path, to_vocab_path, hi_tokenizer)
  data_to_token_ids(from_dev_path, from_dev_ids_path, from_vocab_path, en_tokenizer)

  return (from_train_ids_path, to_train_ids_path,
          from_dev_ids_path, to_dev_ids_path,
          from_vocab_path, to_vocab_path)