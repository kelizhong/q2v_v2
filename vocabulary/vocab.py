# coding=utf-8
"""Generate vocabulary"""
import os
import logbook as logging
import sys
from itertools import product
from vocabulary.ventilator import VentilatorProcess
from vocabulary.worker import WorkerProcess
from vocabulary.collector import CollectorProcess
from utils.data_util import sentence_gen

from utils.data_util import basic_tokenizer
from utils.pickle_util import save_obj_pickle, load_pickle_object
from collections import Counter
from exception.resource_exception import ResourceNotFoundError


class VocabularyBase(object):
    def __init__(self, vocabulary_data_dir, top_words, special_words,
                 words_freq_counter_name="words_freq_counter"):
        self.vocabulary_data_dir = vocabulary_data_dir
        self.top_words = top_words
        self.special_words = special_words
        self.words_freq_counter_path = os.path.join(self.vocabulary_data_dir, words_freq_counter_name)
        self.vocab_path = os.path.join(self.vocabulary_data_dir, "vocab_%d" % top_words)

    @property
    def special_words_size(self):
        return len(self.special_words)

    def build_vocabulary_from_pickle(self):
        """load vocabulary from pickle
        Parameters
        ----------
            path: str
                corpus path
            top_words: int
                the max words num in the vocabulary
            special_words: dict
             special_words like <unk>, <s>, </s>
        Returns
        -------
            vocabulary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            9322221111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111-=

        """

        if os.path.isfile(self.vocab_path):
            vocab = load_pickle_object(self.vocab_path)
        elif os.path.isfile(self.words_freq_counter_path):
            words_freq_counter = load_pickle_object(self.words_freq_counter_path)
            # If top_words is None, then list all element counts.
            words_freq_list = words_freq_counter.most_common(self.top_words)

            words_num = len(words_freq_list)
            special_words_num = self.special_words_size

            if words_num <= special_words_num and special_words_num > 0:
                raise ValueError("the size of total words must be larger than the size of special_words")

            if special_words_num > 0 and self.top_words <= special_words_num:
                raise ValueError("the value of most_common_words_num must be larger "
                                 "than the size of special_words")

            vocab = dict()
            vocab.update(self.special_words)
            for word, _ in words_freq_list:
                if 0 < self.top_words <= len(vocab):
                    break
                if word not in self.special_words:
                    vocab[word] = len(vocab)
            save_obj_pickle(vocab, self.vocab_path, True)
        else:
            raise ResourceNotFoundError(
                "Failed to load vocabulary resource, please check vocabulary file %s or words_freq_counter %s" % (
                    self.vocab_path, self.words_freq_counter_path))
        return vocab


class VocabularyFromLocalFile(VocabularyBase):
    def __init__(self, vocabulary_data_dir, top_words=40000, special_words=dict(),
                 words_freq_counter_name="words_freq_counter"):
        super(VocabularyFromLocalFile, self).__init__(vocabulary_data_dir, top_words, special_words,
                                                      words_freq_counter_name)

    def build_words_frequency_counter(self, raw_data_path, tokenizer=None):
        """
        Create vocabulary file (if it does not exist yet) from data file.
        Data file should have one sentence per line.
        Each sentence will be tokenized.
        Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
        We write it to vocabulary_path in a one-token-per-line format, so that later
        token in the first line gets id=0, second line gets id=1, and so on.
        Args:
          vocabulary_path: path where the vocabulary will be created.
          data_path: data file that will be used to create vocabulary.
          max_vocabulary_size: limit on the size of the created vocabulary.
        """
        if not os.path.isfile(self.words_freq_counter_path):
            print("Building words frequency counter %s from data %s" % (self.words_freq_counter_path, raw_data_path))

            def _word_generator():
                with open(raw_data_path, 'r+') as f:
                    for num, line in enumerate(f):
                        if num % 100000 == 0:
                            print("  processing line %d" % num)
                        try:
                            tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                        except Exception, e:
                            print("Tokenize failure: " + line)
                            continue
                        for word in tokens:
                            yield word

            counter = Counter(_word_generator())
            save_obj_pickle(counter, self.words_freq_counter_path, True)
            print('Vocabulary file created')

    def build_vocabulary_from_pickle(self, raw_data_path=None):
        # Load vocabulary

        if raw_data_path:
            self.build_words_frequency_counter(raw_data_path)
        logging.info("Loading vocabulary")
        vocab = super(VocabularyFromLocalFile, self).build_vocabulary_from_pickle()
        logging.info("Vocabulary size: %d" % len(vocab))
        return vocab


class VocabFromZMQ(VocabularyBase):
    """
    Create vocabulary file (if it does not exist yet) from data file.
    Data file should have one sentence per line.
    Each sentence will be tokenized.
    Vocabulary contains the most-frequent tokens up to top_words.
    We write it to vocab_file in pickle format.
    Parameters
    ----------
        corpus_files: list
            corpus files list that will be used to create vocabulary
        vocab_save_path: str
            vocab file name where the vocabulary will be created
        sentence_gen: generator
            generator which produce the sentence in corpus data
        top_words: int
            limit on the size of the created vocabulary
        workers_num: int
            numbers of workers to parse the sentence
        ip: str
            the ip address string without the port to pass to ``Socket.bind()``
        ventilator_port: int
            port for ventilator process socket
        collector_port: int
            port for collector process socket
        overwrite: bool
            whether to overwrite the existed vocabulary
    """

    def __init__(self, vocabulary_data_dir, special_words=dict(), sentence_gen=sentence_gen, workers_num=1,
                 top_words=100000,
                 ip='127.0.0.1', ventilator_port='5555', collector_port='5556',
                 overwrite=True, words_freq_counter_name="words_freq_counter"):
        super(VocabFromZMQ, self).__init__(vocabulary_data_dir, top_words, special_words,
                                                      words_freq_counter_name)
        self.sentence_gen = sentence_gen
        self.workers_num = workers_num
        self.top_words = top_words
        self.ip = ip
        self.ventilator_port = ventilator_port
        self.collector_port = collector_port
        self.overwrite = overwrite

    def build_words_frequency_counter(self, corpus_files):
        process_pool = []
        v = VentilatorProcess(corpus_files, self.ip, self.ventilator_port, sentence_gen=self.sentence_gen)
        v.start()
        process_pool.append(v)
        for i in xrange(self.workers_num):
            w = WorkerProcess(self.ip, self.ventilator_port, self.collector_port, name='WorkerProcess_{}'.format(i))
            w.start()
            process_pool.append(w)
        c = CollectorProcess(self.ip, self.collector_port)
        counter = Counter(c.collect())
        self._terminate_process(process_pool)
        logging.info("Finish counting. {} unique words, a total of {} words in all files."
                     , len(counter), sum(counter.values()))

        logging.info("store vocabulary with most_common_words file, vocabulary size: {}", len(counter))
        save_obj_pickle(counter, self.words_freq_counter_path, self.overwrite)

    def _terminate_process(self, pool):
        for p in pool:
            p.terminate()
            logging.info('terminated process {}', p.name)


class VocabularyFromCustomStringTrigram(VocabularyBase):
    def __init__(self, vocabulary_data_dir, top_words=sys.maxsize, special_words=dict(),
                 words_freq_counter_name="words_freq_counter"):
        super(VocabularyFromCustomStringTrigram, self).__init__(vocabulary_data_dir, top_words, special_words,
                                                      words_freq_counter_name)

    def build_words_frequency_counter(self, string=None):
        """
        Create vocabulary file (if it does not exist yet) from data file.
        Data file should have one sentence per line.
        Each sentence will be tokenized.
        Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
        We write it to vocabulary_path in a one-token-per-line format, so that later
        token in the first line gets id=0, second line gets id=1, and so on.
        Args:
          vocabulary_path: path where the vocabulary will be created.
          data_path: data file that will be used to create vocabulary.
          max_vocabulary_size: limit on the size of the created vocabulary.
        """
        string = string if string else 'abcdefghijklmnopqrstuvwxyz1234567890#.&\\'
        if not os.path.isfile(self.words_freq_counter_path):
            print("Building words frequency counter %s from custom string %s" % (self.words_freq_counter_path, string))

            counter = Counter(product(string, repeat=3))
            save_obj_pickle(counter, self.words_freq_counter_path, True)
            print('Vocabulary file created')

    def build_vocabulary_from_pickle(self, string=None):
        # Load vocabulary

        if string:
            self.build_words_frequency_counter(string)
        logging.info("Loading vocabulary")
        vocab = super(VocabularyFromCustomStringTrigram, self).build_vocabulary_from_pickle()
        logging.info("Vocabulary size: %d" % len(vocab))
        return vocab