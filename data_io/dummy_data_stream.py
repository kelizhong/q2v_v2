from data_io.batch_data_handler import BatchDataTrigramHandler
from helper.data_parser import QueryPairParser


class DummyDataStream(object):
    """This class only for test purpose
    """

    def __init__(self, batch_size, raw_data_path):
        """Data stream for dummy
        Parameters
        ----------
        batch_size : int
            data batch size
        raw_data_path : str
            raw data path for training
        """
        self.batch_size = batch_size
        self.raw_data_path = raw_data_path
        self.batch_data = BatchDataTrigramHandler(batch_size, tokens_fn=None, min_words=-1)

    def generate_batch_data(self):
        """Generator for dummy batch data

        Yields
        ------
        data object
            Including source tokens, target toekns, labels
        """
        parser = QueryPairParser()
        for num, (source_tokens, target_list, label_list) in enumerate(
                parser.siamese_sequences_to_tokens_generator(self.raw_data_path)):
            self.batch_data.insert_data_object(None, source_tokens, target_list, label_list)
            if self.batch_data.data_object_length == self.batch_size:
                yield self.batch_data.data_object
