# coding=utf-8
"""Generate vocabulary"""
import os
import logbook as logging
import sys
import codecs
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
    """
    Parameters
    ----------
        vocabulary_data_dir: str
            Path where the vocabulary will be created for vocabulary from AKSIS corpus data or custom string
        top_words: int
            limit on the size of the created vocabulary
        special_words:  dict
            special vocabulary symbols(begin of sentence, end of sentence, unknown word) - we always put them at the start.
        words_freq_counter_name: str
            word frequency counter name
    """

    def __init__(self, vocabulary_data_dir, top_words, special_words, vocabulary_name='vocab',
                 words_freq_counter_name="words_freq_counter"):
        self.vocabulary_data_dir = vocabulary_data_dir
        self.top_words = top_words
        self.special_words = special_words
        self.words_freq_counter_path = os.path.join(self.vocabulary_data_dir, words_freq_counter_name)
        self.vocab_path = os.path.join(self.vocabulary_data_dir, "%s_%d" % (vocabulary_name, top_words))

    @property
    def special_words_size(self):
        return len(self.special_words)

    @property
    def vocabulary_size(self):
        if os.path.isfile(self.vocab_path):
            vocab = load_pickle_object(self.vocab_path)
            return len(vocab)
        else:
            logging.warn("Vocabulary {} has not been built", self.vocab_path)
            return 0

    def build_vocabulary_from_words_list(self, words_list_file=None):
        if os.path.isfile(self.vocab_path) and words_list_file is None:
            vocab = load_pickle_object(self.vocab_path)
        else:
            vocab = dict()
            with codecs.open(words_list_file, encoding='utf-8', errors='ignore') as f:
                for word in f:
                    word = word.strip().lower()
                    vocab[word] = len(vocab)
                    if len(vocab) > self.top_words:
                        break
            save_obj_pickle(vocab, self.vocab_path, True)
        return vocab

    def build_vocabulary_from_pickle(self):
        """load vocabulary from pickle
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


class VocabularyFromWordList(VocabularyBase):
    def __init__(self, vocabulary_data_dir, top_words=40000, special_words=dict(), vocabulary_name='wordslist'):
        super(VocabularyFromWordList, self).__init__(vocabulary_data_dir, top_words, special_words, vocabulary_name=vocabulary_name)


class VocabularyFromLocalFile(VocabularyBase):
    """create vocabulary from local file"""

    def __init__(self, vocabulary_data_dir, top_words=40000, special_words=dict(), vocabulary_name='localvocab',
                 words_freq_counter_name="words_freq_counter"):
        super(VocabularyFromLocalFile, self).__init__(vocabulary_data_dir, top_words, special_words, vocabulary_name=vocabulary_name,
                                                      words_freq_counter_name=words_freq_counter_name)

    def build_words_frequency_counter(self, raw_data_path, tokenizer=None):
        """
        build word frequency counter(if it does not exist yet) from raw data file.
        Data file should have one sentence per line.
        Each sentence will be tokenized.
        Parameters
        ----------
          raw_data_path: raw data path for corpus that will be used to create word frequency counter.
          tokenizer: tokenizer to tokenize the sentence in raw data
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
                        except Exception as e:
                            print("Tokenize failure: " + line)
                            continue
                        for word in tokens:
                            yield word

            counter = Counter(_word_generator())
            save_obj_pickle(counter, self.words_freq_counter_path, True)
            print('Vocabulary file created')
        return self

    def build_vocabulary_from_pickle(self, raw_data_path=None):
        # Load vocabulary

        if raw_data_path:
            self.build_words_frequency_counter(raw_data_path)
        logging.info("Loading vocabulary")
        vocab = super(VocabularyFromLocalFile, self).build_vocabulary_from_pickle()
        logging.info("Vocabulary size: %d" % len(vocab))
        return vocab


class VocabFromZMQ(VocabularyBase):
    """build vocabulary using zmq to parse and generate data
    Parameters
    ----------
        vocabulary_data_dir: str
            vocab file name where the vocabulary and word counter will be created
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
        words_freq_counter_name: str
            word frequency counter name
        special_words:  dict
            special vocabulary symbols(begin of sentence, end of sentence, unknown word) - we always put them at the start.
    """

    def __init__(self, vocabulary_data_dir, special_words=dict(), sentence_gen=sentence_gen, workers_num=1,
                 top_words=100000,
                 ip='127.0.0.1', ventilator_port=5555, collector_port=5556, vocabulary_name='zmqvocab',
                 overwrite=True, words_freq_counter_name="words_freq_counter"):
        super(VocabFromZMQ, self).__init__(vocabulary_data_dir, top_words, special_words, vocabulary_name=vocabulary_name,
                                           words_freq_counter_name=words_freq_counter_name)
        self.sentence_gen = sentence_gen
        self.workers_num = workers_num
        self.top_words = top_words
        self.ip = ip
        self.ventilator_port = ventilator_port
        self.collector_port = collector_port
        self.overwrite = overwrite

    def build_words_frequency_counter(self, corpus_files):
        """
        build word frequency counter(if it does not exist yet) from corpus_files.
        Data file should have one sentence per line.
        Each sentence will be tokenized in worker process.
        We write it to `words_freq_counter_name` in pickle format.
        Parameters
        ----------
            corpus_files: list
                corpus files list that will be used to create word frequency counter
        """
        process_pool = []
        v = VentilatorProcess(corpus_files, self.ip, self.ventilator_port, sentence_gen=self.sentence_gen)
        v.start()
        process_pool.append(v)
        for i in range(self.workers_num):
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
        return self

    def _terminate_process(self, pool):
        for p in pool:
            p.terminate()
            logging.info('terminated process {}', p.name)


class VocabularyFromCustomStringTrigram(VocabularyBase):
    """build vocabulary from custom string with trigram parser
    Parameters
    ----------
        vocabulary_data_dir: str
            vocab file name where the vocabulary and word counter will be created
        top_words: int
            limit on the size of the created vocabulary
        words_freq_counter_name: str
            word frequency counter name
        special_words:  dict
            special vocabulary symbols(begin of sentence, end of sentence, unknown word) - we always put them at the start.
    """

    def __init__(self, vocabulary_data_dir, top_words=sys.maxsize, special_words=dict(), vocabulary_name='stringvocab',
                 words_freq_counter_name="words_freq_counter"):
        super(VocabularyFromCustomStringTrigram, self).__init__(vocabulary_data_dir, top_words, special_words, vocabulary_name=vocabulary_name,
                                                                words_freq_counter_name=words_freq_counter_name)

    def build_words_frequency_counter(self, string=None):
        string = string if string else 'abcdefghijklmnopqrstuvwxyz1234567890#.&\\'
        if not os.path.isfile(self.words_freq_counter_path):
            print("Building words frequency counter %s from custom string %s" % (self.words_freq_counter_path, string))

            counter = Counter(product(string, repeat=3))
            save_obj_pickle(counter, self.words_freq_counter_path, True)
            print('Vocabulary file created')
        return self

    def build_vocabulary_from_pickle(self, string=None):
        # Load vocabulary

        self.build_words_frequency_counter(string)
        logging.info("Loading vocabulary")
        vocab = super(VocabularyFromCustomStringTrigram, self).build_vocabulary_from_pickle()
        logging.info("Vocabulary size: %d" % len(vocab))
        return vocab
