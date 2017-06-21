# coding: utf-8
# pylint: disable=invalid-name
"""constant value"""
bos_word = '<s>'  # begin of sentence
eos_word = '</s>'  # end of sentence
unk_word = '<unk>'  # unknown word
pad_word = '<pad>'  # pad word
special_words = {pad_word: 0, unk_word: 1, bos_word: 2, eos_word: 3}  # special word: index
