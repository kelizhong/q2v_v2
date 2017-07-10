import os
import tensorflow as tf
import sys
from data_io.batch_data_handler import BatchDataTrigramHandler
from helper.model_helper import create_model
from utils.decorator_util import memoized
from vocabulary.vocab import VocabularyFromCustomStringTrigram
from config.config import FLAGS
from utils.math_util import cos_distance
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector


class Inference(object):
    def __init__(self):
        self.sess = tf.Session()
        self.model = self._init_model()
        self.batch_data = BatchDataTrigramHandler(self.vocabulary, source_maxlen=FLAGS.source_max_seq_length, target_maxlen=FLAGS.target_max_seq_length, batch_size=sys.maxsize)
        self.batch_size = 2

    def _init_model(self):
        model = create_model(self.sess, True)
        model.batch_size = 1
        return model

    def encode(self, inputs):
        self.batch_data.clear_data_object()
        sources = None
        source_tokens = []
        self.batch_data.clear_data_object()
        for each in inputs:
            sources, source_tokens, source_lens, targets, target_tokens, target_lens, labels = self.batch_data.parse_and_insert_data_object(each, None)
        step_loss, vec = self.model.step(self.sess, source_tokens, source_lens, target_tokens, target_lens, labels)
        return sources, vec

    def batch_encode(self, file):
        model_path = os.path.join(FLAGS.model_dir, "embedding")
        writer = tf.summary.FileWriter(model_path, self.sess.graph)
        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'item_embedding'
        embed.metadata_path = os.path.join(model_path, 'metadata.csv')
        projector.visualize_embeddings(writer, config)
        sources = set()
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                sources.add(line)
        metadata_path = embed.metadata_path

        sources_list = []
        embedding_list = []
        count = 0
        # the left over elements that would be truncated by zip
        for each in zip(*[iter(sources)]*self.batch_size):
            batch_sources, result = self.encode(each)
            if result is not None:
                sources_list.extend(batch_sources)
                embedding_list.append(result)
                count += self.batch_size
                print("Finished: %d" % count)

        with open(metadata_path, 'w+') as item_file:
            item_file.write('id\tchar\n')
            for i, each in enumerate(sources_list):
                item_file.write('{}\t{}\n'.format(i, each))
            print('metadata file created')

        concat = np.concatenate(embedding_list, axis=0)
        item_size, unit_size = concat.shape
        item_embedding = tf.get_variable(embed.tensor_name, [item_size, unit_size])
        assign_op = item_embedding.assign(concat)
        self.sess.run(assign_op)
        saver = tf.train.Saver([item_embedding])
        saver.save(self.sess, model_path, global_step=self.model.global_step)


    @property
    @memoized
    def vocabulary(self):
        """load vocabulary"""
        vocab = VocabularyFromCustomStringTrigram(FLAGS.vocabulary_data_dir).build_vocabulary_from_pickle()
        return vocab

def main(_):
    i = Inference()
    i.batch_encode('source')
    #  print(cos_distance(i.encode(["nike shoe men"]), i.encode("apple shoe men")))
    # print(i.encode(["nike shoe men"]))


if __name__ == "__main__":
    tf.app.run()

