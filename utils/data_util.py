# coding=utf-8
"""util for data processing"""
import sys
import codecs
import logging
import numpy as np


def is_number(str):
    try:
        float(str)
        return True
    except ValueError:
        pass
    return str.isnumeric()


def sentence_gen(files):
    """Generator that yield each sentence in a line.
    Parameters
    ----------
        files: list
            data file list
    """
    if not isinstance(files, list):
        files = [files]
    for filename in files:
        with codecs.open(filename, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip().lower()
                if len(line):
                    yield line


# batch preparation of a given sequence pair for training
def prepare_train_pair_batch(source_seq, targets_list, source_maxlen=sys.maxsize, target_maxlen=sys.maxsize, dtype='int32'):
    source_tokens, source_lengths = prepare_train_batch(source_seq, source_maxlen)
    if len(source_lengths) < 1:
        return tuple([None] * 4)

    data_list = list()
    for target_seq in zip(*targets_list):
        target_tokens, target_lengths = prepare_train_batch(target_seq, target_maxlen)
        if len(target_lengths) < 1 or len(target_tokens) != len(source_tokens):
            logging.error("prepare train batch, source token length:%d, target token length:%d", len(source_tokens), len(target_tokens))
            return tuple([None] * 4)
        data_list.append((target_tokens, target_lengths))
    target_tokens = np.concatenate([target for target, _ in data_list], axis=1)
    target_lengths = np.vstack([target_lengths for _, target_lengths in data_list])
    target_lengths = np.transpose(target_lengths)
    return source_tokens, source_lengths, target_tokens, target_lengths


# batch preparation of a given sequence for embedding or decoder
def prepare_train_batch(seqs, maxlen=None, dtype='int32'):
    # seqs_x, seqs_y: a list of sentences
    seqs = list(map(lambda x: x[:maxlen], seqs))
    lengths = [len(s) for s in seqs]

    if len(lengths) < 1:
        return None, None

    batch_size = len(seqs)

    lengths = np.array(lengths)

    maxlen = np.max(lengths)

    x = np.zeros((batch_size, maxlen)).astype(dtype)

    for idx, s_x in enumerate(seqs):
        x[idx, :lengths[idx]] = s_x
    return x, lengths
