from vocabulary.vocabulary_from_local_file import VocabularyFromLocalFile
from utils.data_util import query_title_score_generator_from_aksis_data, sentence_to_padding_tokens


class AksisDataStream(object):
    def __init__(self, vocabulary_data_dir, top_words, special_words, max_seq_length, batch_size, raw_data_path=None, stop_freq=-1):
        self.max_seq_length = max_seq_length
        self.stop_freq = stop_freq
        self.batch_size = batch_size
        self.raw_data_path = raw_data_path
        self.vocabulary = self._init_vocabulary(vocabulary_data_dir, top_words, special_words, raw_data_path)
        self._sources, self._source_lens, self._targets, self._target_lens, self._labels = [], [], [], [], []

    def __iter__(self):
        return self

    def next(self):
        """return the data from collector"""
        batch_data = self.generate_batch_data()
        return batch_data

    def generate_batch_data(self):
        for num, (source, target) in enumerate(query_title_score_generator_from_aksis_data(self.raw_data_path)):
            if num % 1000 == 0:
                print("  reading data line %d" % num)

            if num > 50000:
                break
            source_len, source_tokens = sentence_to_padding_tokens(source, self.vocabulary, self.max_seq_length)
            target_len, target_tokens = sentence_to_padding_tokens(target, self.vocabulary, self.max_seq_length)
            self.insert_data_object(source_tokens, source_len, target_tokens, target_len, 1)
            if self.data_object_length == self.batch_size:
                yield self.data_object
                self.clear_data_object()

    def _init_vocabulary(self, vocabulary_data_dir, top_words=40000, special_words=dict(), raw_data_path=None):
        # Load vocabulary
        vocab = VocabularyFromLocalFile(vocabulary_data_dir, top_words, special_words).build_vocabulary_from_pickle(
            raw_data_path)
        return vocab

    @property
    def data_object_length(self):
        return len(self._sources)

    def insert_data_object(self, source_tokens, source_len, target_tokens, target_len, label_id):
        self._sources.append(source_tokens)
        self._source_lens.append(source_len)
        self._targets.append(target_tokens)
        self._target_lens.append(target_len)
        self._labels.append(label_id)

    def clear_data_object(self):
        del self._sources[:]
        del self._source_lens[:]
        del self._targets[:]
        del self._target_lens[:]
        del self._labels[:]

    @property
    def data_object(self):
        return self._sources, self._source_lens, self._targets, self._target_lens, self._labels

