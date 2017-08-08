# coding=utf-8
import abc
import six
import logging


@six.add_metaclass(abc.ABCMeta)
class BatchDataHandler(object):
    """handler to parse and generate data
    Parameters
    ----------
        batch_size: int
            Batch size for each data batch
        enable_target: bool
            True for train mode, False for encode mode

    """

    def __init__(self, batch_size, enable_target):
        self.logger = logging.getLogger("data")
        self.batch_size = batch_size
        self.enable_target = enable_target
        self._sources, self._sources_token, self._targets_list = [], [], []
        self._labels = []

    @property
    def data_object_length(self):
        """return data object length

        data object length equal to length of source token

        Returns
        -------
        int
            sources token length
        """
        assert not self.enable_target or all([len(self._sources_token) == len(self._targets_list),
                                              len(self._sources_token) == len(
                                                  self._labels)]), "source shape:%d, target shape:%d, label shape:%d" % (
            len(self._sources_token), len(self._targets_list), len(self._sources_token))
        return len(self._sources_token)

    @abc.abstractmethod
    def parse_and_insert_data_object(self, source, target_sample_items, label):
        """parse data using trigram parser, insert it to data_object to generate batch data"""
        raise NotImplementedError

    def insert_data_object(self, source, source_tokens, targets_list, label):
        if self.data_object_length == self.batch_size:
            self.clear_data_object()
        if self.enable_target:
            if all([label is not None, targets_list is not None, len(label) > 0, len(targets_list) > 0]):
                self._labels.append(label)
                self._targets_list.append(targets_list)
            else:
                logging.error(len(targets_list))
                logging.error("label/target_list is None or empty. Source:%s", source)
                return self.data_object

        if source_tokens and len(source_tokens) > 0:
            self._sources.append(source)
            self._sources_token.append(source_tokens)
        elif self.enable_target:
            self._labels.pop()
            self._targets_list.pop()
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
        batch_size: int
            Batch size for each data batch
        tokens_fn: function
            funtion to convert word to token/id
        min_words: int
            ignore the source wit length less than `min_words`
        enable_target: bool
            True for train mode, False for encode mode
    """

    def __init__(self, batch_size, tokens_fn, min_words=2, enable_target=True):
        super().__init__(batch_size, enable_target)
        self.min_words = min_words
        self.enable_target = enable_target
        self.tokens_fn = tokens_fn

    def parse_and_insert_data_object(self, source, target_sample_items, label=None):
        """parse data using trigram parser, insert it to data_object to generate batch data"""
        _items_list = list()
        if self.enable_target:
            for item in target_sample_items:
                if item and len(item.split()) >= self.min_words:
                    _item_token, _item = self.tokens_fn(item)
                    _items_list.append(_item_token)
                else:
                    logging.error("Failed to parse %s", source)
                    return self.data_object
        if source and len(source.split()) >= self.min_words:
            # discard source with length less than `min_words`
            _source_token, _source = self.tokens_fn(source)
            data_object = self.insert_data_object(_source, _source_token, _items_list, label)
        else:
            data_object = self.data_object
        return data_object
