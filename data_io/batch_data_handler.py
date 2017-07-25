# coding=utf-8
import abc
import six
import logging
from utils.data_util import data_encoding


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

    def __init__(self, vocabulary, batch_size, enable_target):
        self.logger = logging.getLogger("data")
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.enable_target = enable_target
        self._sources, self._sources_token, self._targets_list = [], [], []
        self._labels = []

    @property
    def data_object_length(self):
        return len(self._sources_token)

    @abc.abstractmethod
    def parse_and_insert_data_object(self, source, target_sample_items, label):
        """parse data using trigram parser, insert it to data_object to generate batch data"""
        raise NotImplementedError

    def insert_data_object(self, source, source_tokens, targets_list, label):
        if self.data_object_length == self.batch_size:
            self.clear_data_object()
        if self.enable_target:
            self._labels.append(label)
            self._targets_list.append(targets_list)

        if source and len(source_tokens) > 0:
            self._sources.append(source)
            self._sources_token.append(source_tokens)
        return self.data_object

    def clear_data_object(self):
        del self._sources[:]
        del self._sources_token[:]
        del self._targets_list[:]
        del self._labels[:]

    @property
    def data_object(self):

        if self.enable_target:
            data = self._sources, self._sources_token, self._targets_list, self._labels
        else:
            data = self._sources, self._sources_token
        return data


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

    def __init__(self, vocabulary, batch_size, min_words=2, enable_target=True):
        super().__init__(vocabulary, batch_size, enable_target)
        self.min_words = min_words
        self.enable_target = enable_target

    def parse_and_insert_data_object(self, source, target_sample_items, label=None):
        """parse data using trigram parser, insert it to data_object to generate batch data"""
        _items_list = list()
        if self.enable_target:
            for item in target_sample_items:
                if item and len(item.split()) >= self.min_words:
                    _item_token, _item = data_encoding(item, self.vocabulary)
                    _items_list.append(_item_token)
                else:
                    logging.error("Failed to parse %s", source)
                    return self.data_object
        if source and len(source.split()) >= self.min_words:
            # discard source with length less than `min_words`
            _source_token, _source = data_encoding(source, self.vocabulary)
            data_object = self.insert_data_object(_source, _source_token, _items_list, label)
        else:
            data_object = self.data_object
        return data_object
