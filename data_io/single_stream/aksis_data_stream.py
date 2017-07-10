from vocabulary.vocab import VocabularyFromCustomStringTrigram
from utils.data_util import query_title_score_generator_from_aksis_data
from ..batch_data_handler import BatchDataTrigramHandler
from utils.decorator_util import memoized
from config.config import special_words


class AksisDataStream(object):
    def __init__(self, vocabulary_data_dir, top_words, batch_size, raw_data_path=None, stop_freq=-1):
        self.stop_freq = stop_freq
        self.batch_size = batch_size
        self.raw_data_path = raw_data_path
        self.vocabulary_data_dir = vocabulary_data_dir
        self.top_words = top_words
        self.batch_data = BatchDataTrigramHandler(self.vocabulary, batch_size)

    def generate_batch_data(self):
        for num, (source, target) in enumerate(query_title_score_generator_from_aksis_data(self.raw_data_path)):
            if num % 1000 == 0:
                print("  reading data line %d" % num)

            self.batch_data.parse_and_insert_data_object(source, target)
            if self.batch_data.data_object_length == self.batch_size:
                yield self.batch_data.data_object

    @property
    @memoized
    def vocabulary(self):
        """load vocabulary"""
        vocab = VocabularyFromCustomStringTrigram(self.vocabulary_data_dir, special_words=special_words,
                                                  top_words=self.top_words).build_vocabulary_from_pickle()
        return vocab

