import abc
import six
from utils.data_util import trigram_encoding


@six.add_metaclass(abc.ABCMeta)
class BatchDataHandler(object):
    """handler to parse and generate data
    Parameters
    ----------
        vocabulary: vocabulary object
            vocabulary from AKSIS corpus data or custom string
        batch_size: int
            Batch size for each data batch
    """

    def __init__(self, vocabulary, batch_size):
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self._sources, self._source_tokens, self._targets, self._target_tokens, self._labels = [], [], [], [], []

    @property
    def data_object_length(self):
        return len(self._source_tokens)

    @abc.abstractmethod
    def parse_and_insert_data_object(self, source, target, label=1):
        """parse data using trigram parser, insert it to data_object to generate batch data"""
        raise NotImplementedError

    def insert_data_object(self, source, source_tokens, target, target_tokens, label_id):
        if self.data_object_length == self.batch_size:
            self.clear_data_object()
        if len(source_tokens) > 0:
            self._sources.append(source)
            self._source_tokens.append(source_tokens)
            self._targets.append(target)
            self._target_tokens.append(target_tokens)
            self._labels.append(label_id)
        return self.data_object

    def clear_data_object(self):
        del self._sources[:]
        del self._source_tokens[:]
        del self._targets[:]
        del self._target_tokens[:]
        del self._labels[:]

    @property
    def data_object(self):
        return self._sources, self._source_tokens, self._targets, self._target_tokens, self._labels


class BatchDataTrigramHandler(BatchDataHandler):
    """handler to parse with trigram parser and generate data
    Parameters
    ----------
        vocabulary: vocabulary object
            vocabulary from AKSIS corpus data or custom string
        batch_size: int
            Batch size for each data batch
        min_words: int
            ignore the source wit length less than `min_words`
    """

    def __init__(self, vocabulary, batch_size, min_words=2):
        super().__init__(vocabulary, batch_size)
        self.min_words = min_words

    def parse_and_insert_data_object(self, source, target, label=1):
        """parse data using trigram parser, insert it to data_object to generate batch data"""
        if source and len(source.split()) >= self.min_words and len(target.split()) >= self.min_words:
            # discard source with length less than `min_words`
            source_tokens, source = trigram_encoding(source, self.vocabulary)
            target_tokens, target = trigram_encoding(target, self.vocabulary)
            data_object = self.insert_data_object(source, source_tokens, target, target_tokens, label)
        else:
            data_object = self.data_object
        return data_object
