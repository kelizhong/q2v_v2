import abc
import six
import logbook as logging
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

    def __init__(self, vocabulary, batch_size, enable_target):
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.enable_target = enable_target
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
        if self.enable_target:
            if target and len(target_tokens) > 0:
                self._targets.append(target)
                self._target_tokens.append(target_tokens)
                self._labels.append(label_id)
            else:
                logging.error("BatchDataHandler insert object: allow target, but the target is None or length is 0")
                self.validate_data_object_size()
                return self.data_object
        if source and len(source_tokens) > 0:
            self._sources.append(source)
            self._source_tokens.append(source_tokens)
        elif self.enable_target and target and len(target_tokens) > 0:
            self._targets.pop()
            self._target_tokens.pop()
            self._labels.pop()

        if self.enable_target:
            self.validate_data_object_size()

        return self.data_object

    def validate_data_object_size(self):
        if len(self._sources) != len(self._targets) or len(self._source_tokens) != len(self._target_tokens) \
                or len(self._source_tokens) != len(self._labels):
            raise ValueError("Data object must be equal in their "
                             "batch_size, sources_size:%d, source_tokens_size:%d, targets_size:%d, targets_tokens_size:%d, labels_size:%d" % (
                             len(self._sources), len(self._source_tokens), len(self._targets), len(self._target_tokens),
                             len(self._labels)))

    def clear_data_object(self):
        del self._sources[:]
        del self._source_tokens[:]
        del self._targets[:]
        del self._target_tokens[:]
        del self._labels[:]

    @property
    def data_object(self):
        if self.enable_target:
            data = self._sources, self._source_tokens, self._targets, self._target_tokens, self._labels
        else:
            data = self._sources, self._source_tokens
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

    def parse_and_insert_data_object(self, source, target, label=1):
        """parse data using trigram parser, insert it to data_object to generate batch data"""
        _target_tokens, _target = None, None
        if self.enable_target:
            if target and len(target.split()) >= self.min_words:
                _target_tokens, _target = trigram_encoding(target, self.vocabulary)
            else:
                return self.data_object
        if source and len(source.split()) >= self.min_words:
            # discard source with length less than `min_words`
            _source_tokens, _source = trigram_encoding(source, self.vocabulary)
            data_object = self.insert_data_object(_source, _source_tokens, _target, _target_tokens, label)
        else:
            data_object = self.data_object
        return data_object
