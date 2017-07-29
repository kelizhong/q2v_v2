import os
import sys
from collections import OrderedDict
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm
from data_io.batch_data_handler import BatchDataTrigramHandler
from helper.model_helper import create_model
from utils.decorator_util import memoized
from config.config import FLAGS
from utils.math_util import cos_distance
from utils.data_util import prepare_train_batch
from helper.tokenizer_helper import TokenizerHelper
from helper.tokens_helper import TokensHelper
from config.config import unk_token, _PUNC, _NUM
from helper.vocabulary_helper import VocabularyHelper
import heapq


class Inference(object):
    """inference model to encode the sequence to vector"""

    def __init__(self, config=None, batch_size=65536, min_words=-1, max_sequences=sys.maxsize):
        """
        Parameters
        ----------
        batch_size: define batch size in embedding process
        min_words: ignore the sequence with length < min_words
        max_sequences: tensorflow can not support the variable with memory > 2GB, define `max_sequences` to meet the tf requirement
        """
        self.config = config
        self.sess = tf.Session()
        self.model = self._init_model()
        tokenizer = TokenizerHelper(unk_token=unk_token, num_word=_NUM, punc_word=_PUNC)
        self.tokens_helper = TokensHelper(tokenize_fn=tokenizer.tokenize, vocabulary=self.vocabulary, unk_token=unk_token)
        self.batch_data = BatchDataTrigramHandler(tokens_fn=self.tokens_helper.tokens,
                                                  batch_size=sys.maxsize, min_words=min_words, enable_target=False)
        self.batch_size = batch_size
        self.max_sequences = max_sequences

    def _init_model(self):
        """initialize the q2v model"""

        self.config['max_vocabulary_size'] = len(self.vocabulary)
        model = create_model(self.sess, config=self.config, mode='encode')

        return model

    def encode(self, inputs):
        sources = None
        source_tokens = []
        self.batch_data.clear_data_object()
        for each in inputs:
            sources, source_tokens = self.batch_data.parse_and_insert_data_object(each, None)
        source_tokens, source_lens = prepare_train_batch(source_tokens)
        vectors = None
        if source_tokens is not None and len(source_tokens) > 0:
            vectors = self.model.encode(self.sess, source_tokens, source_lens)

        return sources, vectors

    def visualize(self, file, tensor_name="item_embedding", proj_name="q2v"):
        """
        encode the sequences in file to vector and store in model to visualize them in tensorboard
        Parameters
        ----------
        file: file contains sequences
        tensor_name: embedding tensorf name
        proj_name: project name

        """
        model_path = os.path.join(FLAGS.model_dir, 'visualize')
        writer = tf.summary.FileWriter(model_path, self.sess.graph)
        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = tensor_name
        meta_path = os.path.join(model_path, "%s.tsv" % proj_name)
        embed.metadata_path = meta_path
        projector.visualize_embeddings(writer, config)
        sources = set(map(str.strip, open(file)))
        metadata_path = embed.metadata_path

        sources_list = []
        embedding_list = []
        # the left over elements that would be truncated by zip
        with tqdm(total=len(sources)) as pbar:
            for count, each in enumerate(zip(*[iter(sources)] * self.batch_size)):
                batch_sources, result = self.encode(each)
                if result is not None:
                    sources_list.extend(batch_sources)
                    embedding_list.append(result)
                pbar.update(self.batch_size)
                if count * self.batch_size > self.max_sequences:
                    print("Have reached to the max sequence %d" % self.max_sequences)
                    break

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
        saver.save(self.sess, os.path.join(model_path, "%s.embs" % proj_name))

    @property
    @memoized
    def vocabulary(self):
        """load vocabulary"""
        vocab = VocabularyHelper().load_vocabulary()
        return vocab

    def nearest(self, item, file, n):
        item_v = self.encode([item])[1][0]
        sources = set(map(str.strip, open(file)))
        h = []
        with tqdm(total=len(sources)) as pbar:
            for count, each in enumerate(zip(*[iter(sources)] * self.batch_size)):
                batch_sources, result = self.encode(each)
                pbar.update(self.batch_size)
                for source, v in zip(batch_sources, result):
                    similarity = cos_distance(item_v, v)
                    if len(h) < n:
                        heapq.heappush(h, (similarity, source))
                    else:
                        # Equivalent to a push, then a pop, but faster
                        heapq.heappushpop(h, (similarity, source))
        for i in h:
            print(i)


def main(_):
    config = OrderedDict(sorted(FLAGS.__flags.items()))
    i = Inference(batch_size=10, config=config)
    # i.visualize('names')
    i.nearest("zone diet cookbook", '/Users/keliz/PycharmProjects/q2v_v4/data/rawdata/query_sample_inference', 10)


if __name__ == "__main__":
    tf.app.run()
