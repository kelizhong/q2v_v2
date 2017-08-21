# coding=utf-8
"""
Change the hardcoded host urls below with your own hosts.
Run like this:
pc-01$ CUDA_VISIBLE_DEVICES='' python train.py --job_name="ps" --task_index=0 --data_stream_port=5558 --gpu=0 --ps_hosts='localhost:2221' --worker_hosts='localhost:2222,localhost:2223,localhost:2224'
pc-02$ CUDA_VISIBLE_DEVICES=1 python train.py --job_name="worker" --task_index=0 --data_stream_port=5558 --gpu=1 --ps_hosts='localhost:2221' --worker_hosts='localhost:2222,localhost:2223,localhost:2224'
pc-03$ CUDA_VISIBLE_DEVICES=2 python train.py --job_name="worker" --task_index=1 --data_stream_port=5558 --gpu=2 --ps_hosts='localhost:2221' --worker_hosts='localhost:2222,localhost:2223,localhost:2224'
pc-04$ CUDA_VISIBLE_DEVICES=3 python train.py --job_name="worker" --task_index=2 --data_stream_port=5558 --gpu=3 --ps_hosts='localhost:2221' --worker_hosts='localhost:2222,localhost:2223,localhost:2224'

# single machine with zmq stream:
CUDA_VISIBLE_DEVICES=0 python train.py --gpu 0 --data_stream_port 5558

# single machine with local file stream:
CUDA_VISIBLE_DEVICES=0 python train.py --gpu 0


"""

from __future__ import print_function

import glob
import logging.config
import math
import os
import time
from collections import OrderedDict
from collections import defaultdict

import numpy as np
import tensorflow as tf
import yaml

from utils.config_decouple import config
from config.config import FLAGS
from data_io.dummy_data_stream import DummyDataStream
from helper.data_record_helper import DataRecordHelper
from helper.model_helper import create_model, export_model
from helper.vocabulary_helper import VocabularyHelper
from utils.data_util import prepare_train_pair_batch
from utils.decorator_util import memoized

logger = logging.getLogger("model")


class Trainer(object):

    def __init__(self, tf_config):
        self.job_name = tf_config.get('job_name')
        self.ps_hosts = tf_config.get('ps_hosts', '').split(",")
        self.worker_hosts = tf_config.get('worker_hosts').split(",")
        self.task_index = tf_config.get('task_index')
        self.gpu = tf_config.get('gpu')
        self.label_size = tf_config.get('label_size')
        self.model_dir = tf_config.get('model_dir')
        self.is_sync = tf_config.get('is_sync')
        self.raw_data_path = tf_config.get('raw_data_path')
        self.display_freq = tf_config.get('display_freq')
        self.batch_size = tf_config.get('batch_size')
        self.max_vocabulary_size = self.vocabulary_size
        self.tfrecord_train_file = tf_config['tfrecord_train_file']
        self.model_export_path = tf_config['model_export_path']
        # add max vocabulary size to config
        tf_config['max_vocabulary_size'] = self.vocabulary_size
        self.dummy_model_dir = tf_config['dummy_model_dir']
        self.dummy_model_name = tf_config['dummy_model_name']
        self.model_name = tf_config['model_name']
        self.tf_config = tf_config

    @property
    @memoized
    def vocabulary_size(self):
        vocab = VocabularyHelper().load_vocabulary()
        vocabulary_size = len(vocab)
        return vocabulary_size

    @property
    @memoized
    def master(self):
        if self.job_name == "single":
            master = ""
        else:
            master = self.server.target
        return master

    @property
    @memoized
    def server(self):
        server = tf.train.Server(self.cluster, job_name=self.job_name, task_index=self.task_index)
        return server

    @property
    @memoized
    def cluster(self):
        """represents the set of processes that participate in a distributed TensorFlow computation"""
        assert self.job_name != 'single', "Not support cluster for single machine training"
        assert len(self.ps_hosts) > 0, "No parameter server are found"
        assert len(self.worker_hosts) > 0, "No worker_hosts are found"
        cluster = tf.train.ClusterSpec({"ps": self.ps_hosts, "worker": self.worker_hosts})
        return cluster

    @property
    @memoized
    def core_str(self):
        core_str = "cpu:0" if (self.gpu is None or self.gpu == "") else "gpu:%d" % int(self.gpu)
        return core_str

    @property
    @memoized
    def device(self):
        if self.job_name == "worker":
            # TODO What may happen sometimes is that a single variable is huge,
            # in which case you would need to break variable into smaller pieces first manually
            # or using PartitionedVariable
            ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(
                len(self.ps_hosts), tf.contrib.training.byte_size_load_fn)
            device = tf.train.replica_device_setter(cluster=self.cluster,
                                                    ps_strategy=ps_strategy,
                                                    worker_device='job:worker/task:%d/%s' % (
                                                        self.task_index, self.core_str),
                                                    ps_device='job:ps/task:%d/%s' % (self.task_index, self.core_str))
        else:
            device = "/" + self.core_str

        return device

    def _log_variable_info(self):
        for attribute, value in self.tf_config.items():
            logger.info("%s : %s", attribute, value)
        tensor_memory = defaultdict(int)
        for item in tf.global_variables():
            logger.info("variable:%s, device:%s", item.name, item.device)
        # TODO int32 and float32, dtype?
        for item in tf.trainable_variables():
            tensor_memory[item.device] += int(np.prod(item.shape))
        for key, value in tensor_memory.items():
            logger.info("device: %s, memory:%s", key, value)

    def build_model(self, model_dir):
        with tf.Session(target=self.master,
                            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            model = create_model(sess, config=self.tf_config, model_dir=model_dir)
        return model

    def export_model(self, model_dir):
        with tf.Session(target=self.master,
                            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            model = export_model(sess, config=self.tf_config, model_export_path=self.model_export_path, model_dir=model_dir)
        return model

    def dummy_train(self):
        stream = DummyDataStream(raw_data_path=self.raw_data_path, batch_size=self.batch_size)
        with tf.Session() as sess:
            model = self.build_model(self.dummy_model_dir)
            init = tf.global_variables_initializer()
            sess.run(init)
            step, step_time, loss = 0.0, 0.0, 0.0
            words_done, sents_done = 0.0, 0.0
            for _, _sources_token, _targets_list, _labels in stream.generate_batch_data():
                start_time = time.time()
                source_tokens, source_lengths, target_tokens, target_lengths = prepare_train_pair_batch(_sources_token,
                                                                                                        _targets_list)
                step_loss = model.train(sess, src_inputs=source_tokens,
                                        src_inputs_length=source_lengths,
                                        tgt_inputs=target_tokens, tgt_inputs_length=target_lengths,
                                        labels=_labels)

                if len(source_tokens) != len(target_tokens) or len(source_tokens) != len(target_lengths) or len(
                        target_tokens) != len(target_lengths):
                    raise ValueError("Shape error")
                loss += step_loss
                step += 1
                avg_loss = loss / (step + 1)
                step_time = time.time() - start_time
                words_done += (float(np.sum(source_lengths) + np.sum(target_lengths)))
                sents_done += float(source_tokens.shape[0] * (1 + self.label_size))  # batch_size
                # Increase the epoch index of the model
                global_step = model.global_epoch_step_op.eval()
                if global_step % self.display_freq == 0:
                    avg_perplexity = math.exp(float(avg_loss)) if avg_loss < 300 else float("inf")
                    words_per_sec = words_done / step_time / self.display_freq
                    sents_per_sec = sents_done / step_time / self.display_freq
                    logger.info(
                        "global step %d, learning rate %.8f, step-time:%.2f, step-loss:%.8f, avg-loss:%.8f, perplexity:%.4f, %.4f sents/s, %.4f words/s" %
                        (global_step, model.learning_rate.eval(), step_time, step_loss, avg_loss,
                         avg_perplexity,
                         sents_per_sec, words_per_sec))
                    # set zero timer and loss.
                    words_done, sents_done = 0.0, 0.0
                    model.saver.save(sess, os.path.join(self.dummy_model_dir, self.dummy_model_name), global_step=global_step)

    def train(self):
        if self.job_name == "ps":
            self.server.join()
        else:
            self._log_variable_info()
            file_list = glob.glob(self.tfrecord_train_file)
            record = DataRecordHelper()
            source_batch_data, source_batch_length, targets_batch_data, targets_batch_length, label_batch = record.get_padded_batch(
                file_list, batch_size=self.batch_size, label_size=self.label_size)
            model = self.build_model(self.model_dir)
            init_op = tf.global_variables_initializer()
            sv = tf.train.Supervisor(is_chief=(self.task_index == 0),
                                     logdir=self.model_dir,
                                     init_op=init_op,
                                     saver=model.saver,
                                     global_step=model.global_step,
                                     save_model_secs=180,
                                     checkpoint_basename=self.model_name)
            gpu_options = tf.GPUOptions(allow_growth=True, allocator_type="BFC")
            session_config = tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False,
                                            gpu_options=gpu_options,
                                            intra_op_parallelism_threads=16)
            with sv.prepare_or_wait_for_session(master=self.master, config=session_config) as sess:

                if self.task_index == 0 and self.is_sync and self.job_name == "worker":
                    sv.start_queue_runners(sess, [model.chief_queue_runner])
                    sess.run(model.init_token_op)

                coord = sv.coord
                try:
                    step, step_time, loss = 0.0, 0.0, 0.0
                    words_done, sents_done = 0.0, 0.0

                    # Supervisor: http://blog.csdn.net/lenbow/article/details/52218551
                    while not coord.should_stop():
                        start_time = time.time()
                        # Get a batch from training parallel data
                        _source_batch_data, _source_batch_length, _targets_batch_data, _targets_batch_length, _label_batch = sess.run(
                            [source_batch_data, source_batch_length, targets_batch_data, targets_batch_length,
                             label_batch])
                        step_loss = model.train(sess, src_inputs=_source_batch_data,
                                                src_inputs_length=_source_batch_length,
                                                tgt_inputs=_targets_batch_data, tgt_inputs_length=_targets_batch_length,
                                                labels=_label_batch)
                        loss += step_loss
                        step += 1
                        avg_loss = loss / (step + 1)
                        step_time = time.time() - start_time
                        words_done += (float(np.sum(_source_batch_length) + np.sum(_targets_batch_length)))
                        sents_done += float(_source_batch_data.shape[0] * (1 + self.label_size))  # batch_size
                        # Increase the epoch index of the model
                        global_step = model.global_epoch_step_op.eval()
                        if global_step % self.display_freq == 0:
                            avg_perplexity = math.exp(float(avg_loss)) if avg_loss < 300 else float("inf")
                            words_per_sec = words_done / step_time / self.display_freq
                            sents_per_sec = sents_done / step_time / self.display_freq
                            logger.info(
                                "global step %d, learning rate %.8f, step-time:%.2f, step-loss:%.8f, avg-loss:%.8f, perplexity:%.4f, %.4f sents/s, %.4f words/s" %
                                (global_step, model.learning_rate.eval(), step_time, step_loss, avg_loss, avg_perplexity, sents_per_sec, words_per_sec))
                            # set zero timer and loss.
                            words_done, sents_done = 0.0, 0.0
                except tf.errors.OutOfRangeError:
                    logger.info('Finished training.')
                finally:
                    coord.request_stop()
            sv.stop()


def setup_logger():
    logging_config_path = config('logging_config_path')
    with open(logging_config_path) as f:
        dictcfg = yaml.load(f)
        logging.config.dictConfig(dictcfg)


def main(_):
    setup_logger()
    if FLAGS.debug:
        # https://github.com/tensorflow/tensorflow/commit/ec1403e7dc2b919531e527d36d28659f60621c9e
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu if FLAGS.gpu else ''
    tf_config = OrderedDict(sorted(FLAGS.__flags.items()))
    trainer = Trainer(tf_config=tf_config)
    if FLAGS.use_dummy:
        trainer.dummy_train()
    elif FLAGS.export_model:
        trainer.export_model(tf_config['model_dir'])
    else:
        trainer.train()


if __name__ == "__main__":
    tf.app.run()
