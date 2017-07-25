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

import time

import logging.config
import tensorflow as tf
import os
import numpy as np
import math
import yaml
from collections import defaultdict
from config.config import FLAGS, logging_config_path
from data_io.distribute_stream.aksis_data_receiver import AksisDataReceiver
from data_io.single_stream.aksis_data_stream import AksisDataStream
from helper.model_helper import create_model
from utils.decorator_util import memoized
from utils.data_util import prepare_train_pair_batch
from collections import OrderedDict


class Trainer(object):
    def __init__(self, config):
        self.job_name = config.get('job_name')
        self.ps_hosts = config.get('ps_hosts', '').split(",")
        self.worker_hosts = config.get('worker_hosts').split(",")
        self.task_index = config.get('task_index')
        self.gpu = config.get('gpu')
        self.model_dir = os.path.join(config.get('model_dir'), config.get('model_name'))
        self.is_sync = config.get('is_sync')
        self.raw_data_path = config.get('raw_data_path')
        self.display_freq = config.get('display_freq')
        self.batch_size = config.get('batch_size')
        self.source_maxlen = config.get('source_maxlen')
        self.target_maxlen = config.get('target_maxlen')
        self.max_vocabulary_size = config.get('max_vocabulary_size')
        self.vocabulary_data_dir = config.get('vocabulary_data_dir')
        self.data_stream_port = config.get('data_stream_port')
        self.words_list_path = config['words_list_path']
        self.logger = logging.getLogger("model")

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

    @property
    def data_zmq_stream(self):
        if self.data_stream_port is None:
            raise Exception("port is not defined for zmq stream")
        data_stream = AksisDataReceiver(port=self.data_stream_port)
        return data_stream

    @property
    def data_local_stream(self):
        data_stream = AksisDataStream(self.vocabulary_data_dir, top_words=self.max_vocabulary_size,
                                      batch_size=self.batch_size, words_list_file=self.words_list_path,
                                      raw_data_path=self.raw_data_path).generate_batch_data()
        return data_stream

    @property
    def data_stream(self):
        if self.data_stream_port:
            stream = self.data_zmq_stream
        else:
            stream = self.data_local_stream
        return stream

    def _log_variable_info(self):
        tensor_memory = defaultdict(int)
        for item in tf.global_variables():
            self.logger.info("variable:%s, device:%s", item.name, item.device)
        # TODO int32 and float32, dtype?
        for item in tf.trainable_variables():
            tensor_memory[item.device] += int(np.prod(item.shape))
        for key, value in tensor_memory.items():
            self.logger.info("device: %s, memory:%s", key, value)

    def train(self):
        if self.job_name == "ps":
            self.server.join()
        else:
            with tf.device(self.device):
                with tf.Session(target=self.master,
                                config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
                    model = create_model(sess)
                self._log_variable_info()
                summary_op = tf.summary.merge_all()
                init_op = tf.global_variables_initializer()
            sv = tf.train.Supervisor(is_chief=(self.task_index == 0),
                                     logdir=self.model_dir,
                                     init_op=init_op,
                                     summary_op=summary_op,
                                     saver=model.saver,
                                     global_step=model.global_step,
                                     save_model_secs=180)
            gpu_options = tf.GPUOptions(allow_growth=True, allocator_type="BFC")
            session_config = tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False,
                                            gpu_options=gpu_options,
                                            intra_op_parallelism_threads=16)
            with sv.prepare_or_wait_for_session(master=self.master, config=session_config) as sess:

                if self.task_index == 0 and self.is_sync and self.job_name == "worker":
                    sv.start_queue_runners(sess, [model.chief_queue_runner])
                    sess.run(model.init_token_op)

                step_time, loss = 0.0, 0.0
                words_done, sents_done = 0, 0
                for _ in range(1300):
                    data_stream = self.data_stream
                    for step, (_, source_tokens, _, target_tokens, labels) in enumerate(data_stream):

                        start_time = time.time()
                        source_tokens, source_lens, target_tokens, target_lens = prepare_train_pair_batch(source_tokens, target_tokens, source_maxlen=self.source_maxlen, target_maxlen=self.target_maxlen)
                        # Get a batch from training parallel data
                        if source_tokens is None or target_tokens is None or len(source_tokens) == 0 or len(target_tokens) == 0:
                            self.logger.warning('No samples under source_max_seq_length %d or target_max_seq_length %d',
                                         self.source_maxlen, self.target_maxlen)
                            continue

                        # Execute a single training step
                        step_loss = model.train(sess, src_inputs=source_tokens, src_inputs_length=source_lens,
                                                tgt_inputs=target_tokens, tgt_inputs_length=target_lens, labels=labels)
                        step_time = time.time() - start_time
                        loss += step_loss
                        words_done += float(np.sum(source_lens + target_lens))
                        sents_done += float(source_tokens.shape[0])  # batch_size
                        avg_loss = loss/(step+1)
                        # Increase the epoch index of the model
                        model.global_epoch_step_op.eval()
                        if model.global_step.eval() % self.display_freq == 0:
                            avg_perplexity = math.exp(float(avg_loss)) if avg_loss < 300 else float("inf")
                            words_per_sec = words_done / step_time / self.display_freq
                            sents_per_sec = sents_done / step_time / self.display_freq
                            self.logger.info(
                                "global step %d, learning rate %.8f, step-time:%.2f, step-loss:%.8f, avg-loss:%.8f, perplexity:%.4f, %.4f sents/s, %.4f words/s" %
                                (model.global_step.eval(), model.learning_rate.eval(), step_time, step_loss, avg_loss, avg_perplexity,
                                 sents_per_sec, words_per_sec))
                            # set zero timer and loss.
                            words_done, sents_done = 0.0, 0.0

            sv.stop()


def setup_logger():
    with open(logging_config_path) as f:
        dictcfg = yaml.load(f)
        logging.config.dictConfig(dictcfg)


def main(_):
    setup_logger()
    if FLAGS.debug:
        # https://github.com/tensorflow/tensorflow/commit/ec1403e7dc2b919531e527d36d28659f60621c9e
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu if FLAGS.gpu else ''
    config = OrderedDict(sorted(FLAGS.__flags.items()))
    trainer = Trainer(config=config)
    trainer.train()


if __name__ == "__main__":
    tf.app.run()
