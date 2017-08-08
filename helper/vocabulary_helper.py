import os
import sys
import logging
import re
from collections import Counter
from itertools import product
from tqdm import tqdm
from utils.config_decouple import config
from utils.serialize import save_obj_json, load_json_object
from utils.data_util import is_number
from exception.resource_exception import ResourceNotFoundError


class VocabularyHelper(object):

    def __init__(self, vocabulary_data_dir=None, vocabulary_name='vocab_newest',
                 words_freq_counter_name="words_freq_counter"):
        self.vocabulary_data_dir = vocabulary_data_dir or config('vocabulary_dir')
        self.words_freq_counter_path = os.path.join(self.vocabulary_data_dir, words_freq_counter_name)
        self.vocab_path = os.path.join(self.vocabulary_data_dir, vocabulary_name)

    def build_word_counter(self, corpus_files, parser):
        word_counter = Counter()
        for text_list in tqdm(parser.data_text_list_generator(corpus_files)):
            for item in text_list:
                word_counter.update(item.split())
        logging.info("Saving word counter, size: %d", len(word_counter))
        save_obj_json(word_counter, self.words_freq_counter_path, True)

    @property
    def trigram_word_generator(self):
        string = "abcdefghijklmnopqrstuvwxyz1234567890#.'-,"
        trigram = product(string, repeat=3)
        for ele in trigram:
            yield ''.join(ele)

    def build_vocabulary_from_counter(self, vocabulary_size=sys.maxsize, special_words=dict()):
        """load vocabulary from pickle
        """
        if os.path.isfile(self.words_freq_counter_path):
            words_freq_counter = Counter(load_json_object(self.words_freq_counter_path))
            # If top_words is None, then list all element counts.
            words_freq_list = words_freq_counter.most_common(vocabulary_size)

            words_num = len(words_freq_list)
            special_words_num = len(special_words)

            if words_num <= special_words_num and special_words_num > 0:
                raise ValueError("the size of total words must be larger than the size of special_words")

            if special_words_num > 0 and vocabulary_size <= special_words_num:
                raise ValueError("the value of most_common_words_num must be larger "
                                 "than the size of special_words")

            vocab = dict()
            vocab.update(special_words)
            for word, _ in words_freq_list:
                # 27 = len('honorificabilitudinitatibus')
                if len(str(word)) < 4 or is_number(word) or re.search('[a-zA-Z]+', word) is None or not word.isprintable() or len(str(word)) > 27:
                    continue
                if vocabulary_size <= len(vocab):
                    break
                if word not in vocab:
                    vocab[word] = len(vocab)

            for word in self.trigram_word_generator:
                if word not in vocab:
                    vocab[word] = len(vocab)
            logging.info("Saving vocabulary, size: %d", len(vocab))
            save_obj_json(vocab, self.vocab_path, True)
        else:
            raise ResourceNotFoundError(
                "Failed to load vocabulary resource, please check words_freq_counter %s" % self.words_freq_counter_path)
        return vocab

    def load_vocabulary(self, vocab_path=None):
        vocab_path = vocab_path if vocab_path else self.vocab_path
        vocab = load_json_object(vocab_path) if os.path.isfile(vocab_path) else dict()
        logging.info("Vocabulary size is %d, vocabulary path: %s", len(vocab), vocab_path)
        return vocab
