from vocabulary.vocab import VocabularyFromWordList
from utils.data_util import siamese_data_generator
from ..batch_data_handler import BatchDataTrigramHandler
from utils.decorator_util import memoized


class AksisDataStream(object):
    def __init__(self, vocabulary_data_dir, top_words, batch_size, raw_data_path=None, words_list_file=None):
        self.batch_size = batch_size
        self.raw_data_path = raw_data_path
        self.vocabulary_data_dir = vocabulary_data_dir
        self.top_words = top_words
        self.words_list_file = words_list_file
        self.batch_data = BatchDataTrigramHandler(self.vocabulary, batch_size, min_words=-1)

    def generate_batch_data(self):
        for num, (source, target, label) in enumerate(siamese_data_generator(self.raw_data_path)):
            if num % 1000 == 0:
                print("  reading data line %d" % num)
            self.batch_data.parse_and_insert_data_object(source, target, label)
            if self.batch_data.data_object_length == self.batch_size:
                yield self.batch_data.data_object

    @property
    @memoized
    def vocabulary(self):
        """load vocabulary"""
        vocab = VocabularyFromWordList(self.vocabulary_data_dir, top_words=self.top_words).build_vocabulary_from_words_list(self.words_list_file)
        return vocab
