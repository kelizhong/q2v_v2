import tensorflow as tf

from config import special_words
from data_io.single_stream.aksis_data_stream import BatchDataHandler
from helper.model_helper import create_model
from vocabulary.vocabulary_from_local_file import VocabularyFromLocalFile


class Inference(object):
    def __init__(self, vocabulary_data_dir, top_words, max_seq_length, batch_size):
        self.sess = tf.Session()
        self.model = self._init_model()

        self.batch_data = self._init_batch_data_handler(vocabulary_data_dir, max_seq_length, batch_size, top_words)
        print(self.model)

    def _init_batch_data_handler(self, vocabulary_data_dir, max_seq_length, batch_size, top_words):
        # Load vocabulary
        vocab = VocabularyFromLocalFile(vocabulary_data_dir, top_words, special_words).build_vocabulary_from_pickle()
        batch_data = BatchDataHandler(vocab, max_seq_length, batch_size)
        return batch_data

    def _init_model(self):
        model = create_model(self.sess, True)
        model.batch_size = 1
        return model

    def encode(self, source):
        self.batch_data.parse_and_insert_data_object(source, "apple apple", 1)
        sources, source_lens, targets, target_lens, labels = self.batch_data.data_object
        step_loss, vec = self.model.step(self.sess, sources, source_lens, targets, target_lens, labels)
        return vec


def main(_):
    i = Inference('./data/vocabulary', 50000, 55, 128)
    print(i.encode("apple"))

if __name__ == "__main__":
    tf.app.run()

