import os
import sys
import logging.config

import heapq
import pprint
from collections import OrderedDict
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm
import h5py
import yaml

from utils.config_decouple import config
from data_io.batch_data_handler import BatchDataTrigramHandler
from helper.model_helper import create_model
from utils.decorator_util import memoized
from config.config import FLAGS
from utils.math_util import cos_distance
from utils.data_util import prepare_train_batch
from helper.tokenizer_helper import TextBlobTokenizerHelper
from helper.tokens_helper import TokensHelper
from helper.vocabulary_helper import VocabularyHelper
from utils.file_util import ensure_dir_exists


class Inference(object):
    """inference model to encode the sequence to vector"""

    def __init__(self, tf_config=None, batch_size=65536, min_words=-1, max_sequences=sys.maxsize):
        """
        Parameters
        ----------
        batch_size: define batch size in embedding process
        min_words: ignore the sequence with length < min_words
        max_sequences: tensorflow can not support the variable with memory > 2GB, define `max_sequences` to meet the tf requirement
        """
        self.tf_config = tf_config
        self.sess = tf.Session()
        self.model = self._init_model()
        tokenizer = TextBlobTokenizerHelper()
        self.tokens_helper = TokensHelper(tokenize_fn=tokenizer.tokenize, vocabulary=self.vocabulary,
                                          unk_token=config('_unk_', section='vocabulary_symbol'))
        self.batch_data = BatchDataTrigramHandler(tokens_fn=self.tokens_helper.tokens,
                                                  batch_size=sys.maxsize, min_words=min_words, enable_target=False)
        self.batch_size = batch_size
        self.max_sequences = max_sequences

    def _init_model(self):
        """initialize the q2v model"""

        self.tf_config['max_vocabulary_size'] = len(self.vocabulary)
        model = create_model(self.sess, config=self.tf_config, mode='encode',
                             model_dir=self.tf_config['model_dir'])

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
        model_path = os.path.join(config('project_dir'), 'data', 'visualize')
        writer = tf.summary.FileWriter(model_path, self.sess.graph)
        proj_config = projector.ProjectorConfig()
        embed = proj_config.embeddings.add()
        embed.tensor_name = tensor_name
        meta_path = os.path.join(model_path, "%s.tsv" % proj_name)
        embed.metadata_path = meta_path
        projector.visualize_embeddings(writer, proj_config)
        sources = set(map(str.strip, open(file)))
        metadata_path = embed.metadata_path

        sources_list = []
        embedding_list = []
        h5f = h5py.File('%s.h5' % proj_name, 'w')
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
        h5f.create_dataset(proj_name, data=concat)
        item_size, unit_size = concat.shape
        item_embedding = tf.get_variable(embed.tensor_name, [item_size, unit_size])
        assign_op = item_embedding.assign(concat)
        self.sess.run(assign_op)
        saver = tf.train.Saver([item_embedding])
        saver.save(self.sess, os.path.join(model_path, "%s.embs" % proj_name))

    def vectorization(self, file, proj_name="q2v"):
        """
        encode the sequences in file to vector and store in model to visualize them in tensorboard
        Parameters
        ----------
        file: file contains sequences
        tensor_name: embedding tensorf name
        proj_name: project name

        """
        sources = set(map(str.strip, open(file)))

        sources_list = []
        embedding_list = []
        vector_path = os.path.join(config('project_dir'), 'data', 'vectorization')
        ensure_dir_exists(vector_path)
        h5_path = os.path.join(vector_path, "%s.h5" % proj_name)
        h5f = h5py.File(h5_path, 'w')
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

        meta_path = os.path.join(vector_path, "%s.tsv" % proj_name)
        with open(meta_path, 'w+') as item_file:
            item_file.write('id\tchar\n')
            for i, each in enumerate(sources_list):
                item_file.write('{}\t{}\n'.format(i, each))
            print('metadata file created')
        concat = np.concatenate(embedding_list, axis=0)
        h5f.create_dataset(proj_name, data=concat)

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
                if count % 10 == 0:
                    pprint.pprint(heapq.nlargest(len(h), h))


def setup_logger():
    logging_config_path = config('logging_config_path')
    with open(logging_config_path) as f:
        dictcfg = yaml.load(f)
        logging.config.dictConfig(dictcfg)


def main(_):
    setup_logger()
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    tf_config = OrderedDict(sorted(FLAGS.__flags.items()))
    i = Inference(batch_size=1024, tf_config=tf_config)
    # i.visualize('./data/rawdata/query_sample_inference')
    # i.nearest("women grey nike shoes", './data/rawdata/query_inference', 10)
    # i.nearest("Microsoft Windows 10 Home USB Flash Drive", './data/rawdata/query_inference', 50)
    i.vectorization('./data/rawdata/query_inference')


if __name__ == "__main__":
    tf.app.run()
