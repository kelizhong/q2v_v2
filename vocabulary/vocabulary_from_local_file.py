import os
import logbook as logging
from utils.data_util import basic_tokenizer
from utils.pickle_util import save_obj_pickle, load_pickle_object
from collections import Counter


class VocabularyBase(object):
    def __init__(self, vocabulary_data_dir, top_words=40000, special_words=dict(), words_freq_counter_name="words_freq_counter"):
        self.vocabulary_data_dir = vocabulary_data_dir
        self.top_words = top_words
        self.special_words = special_words
        self.words_freq_counter_path = os.path.join(self.vocabulary_data_dir, words_freq_counter_name)
        self.vocab_path = os.path.join(self.vocabulary_data_dir, "vocab_%d".format(top_words))

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
            vocabulary
        """

        words_freq_counter = load_pickle_object(self.words_freq_counter_path)
        words_freq_list = words_freq_counter.most_common(self.top_words)

        words_num = len(words_freq_list)
        special_words_num = self.special_words_size

        if words_num <= special_words_num:
            raise ValueError("the size of total words must be larger than the size of special_words")

        if self.top_words <= special_words_num:
            raise ValueError("the value of most_common_words_num must be larger "
                             "than the size of special_words")

        vocab = dict()
        vocab.update(self.special_words)
        for word, _ in words_freq_list:
            if len(vocab) >= self.top_words:
                break
            if word not in self.special_words:
                vocab[word] = len(vocab)
        return vocab


class VocabularyFromLocalFile(VocabularyBase):
    def __init__(self, vocabulary_data_dir, top_words=40000, special_words=dict(), words_freq_counter_name="words_freq_counter"):
        super(VocabularyFromLocalFile, self).__init__(vocabulary_data_dir, top_words, special_words, words_freq_counter_name)

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