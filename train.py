# coding=utf-8
"""
Change the hardcoded host urls below with your own hosts.
Run like this:
pc-01$ python example.py --job_name="ps" --task_index=0
pc-02$ python example.py --job_name="worker" --task_index=0
pc-03$ python example.py --job_name="worker" --task_index=1
pc-04$ python example.py --job_name="worker" --task_index=2
"""

from __future__ import print_function

import time

import logbook as logging
import tensorflow as tf

from common.constant import special_words
from config.config import FLAGS
from data_io.distribute_stream.aksis_data_receiver import AksisDataReceiver
from data_io.single_stream.aksis_data_stream import AksisDataStream
from helper.model_helper import create_model
from utils.decorator_util import memoized
from utils.log_util import setup_logger


class Trainer(object):
    def __init__(self, job_name, ps_hosts, worker_hosts, task_index, gpu, model_dir, is_sync, raw_data_path, batch_size,
                 steps_per_checkpoint,
                 source_max_seq_length=None, target_max_seq_length=None, special_words=None, top_words=None, vocabulary_data_dir=None, port=None):
        self.job_name = job_name
        self.ps_hosts = ps_hosts.split(",")
        self.worker_hosts = worker_hosts.split(",")
        self.task_index = task_index
        self.gpu = gpu
        self.model_dir = model_dir
        self.is_sync = is_sync
        self.raw_data_path = raw_data_path
        self.steps_per_checkpoint = steps_per_checkpoint
        self.batch_size = batch_size
        self.source_max_seq_length = source_max_seq_length
        self.target_max_seq_length = target_max_seq_length
        self.special_words = special_words
        self.top_words = top_words
        self.vocabulary_data_dir = vocabulary_data_dir
        self.port = port

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
            device = tf.train.replica_device_setter(cluster=self.cluster,
                                                    worker_device='job:worker/task:%d/%s' % (
                                                        self.task_index, self.core_str),
                                                    ps_device='job:ps/task:%d/%s' % (self.task_index, self.core_str))
        else:
            device = "/" + self.core_str

        return device

    @property
    def data_zmq_stream(self):
        if self.port is None:
            raise Exception("port is not defined for zmq stream")
        data_stream = AksisDataReceiver(port=self.port)
        return data_stream

    @property
    def data_local_stream(self):
        data_stream = AksisDataStream(self.vocabulary_data_dir, top_words=self.top_words,
                                      special_words=self.special_words, source_max_seq_length=self.source_max_seq_length, target_max_seq_length=self.target_max_seq_length,
                                      batch_size=self.batch_size,
                                      raw_data_path=self.raw_data_path).generate_batch_data()
        return data_stream

    @property
    def data_stream(self):
        if self.port:
            stream = self.data_zmq_stream
        else:
            stream = self.data_local_stream
        return stream

    def train(self):
        with tf.device(self.device):
            with tf.Session(target=self.master,
                            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                model = create_model(sess, False)
            for item in tf.global_variables():
                logging.info("variable:{}, device:{}", item.name, item.device)
            init_op = tf.global_variables_initializer()
        sv = tf.train.Supervisor(is_chief=(self.task_index == 0),
                                 logdir=self.model_dir,
                                 init_op=init_op,
                                 summary_op=None,
                                 saver=model.saver,
                                 global_step=model.global_step,
                                 save_model_secs=60)
        gpu_options = tf.GPUOptions(allow_growth=True)
        session_config = tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=True,
                                        gpu_options=gpu_options,
                                        intra_op_parallelism_threads=16)
        with sv.prepare_or_wait_for_session(master=self.master, config=session_config) as sess:
            # 如果是同步模式
            if self.task_index == 0 and self.is_sync:
                sv.start_queue_runners(sess, [model.chief_queue_runner])
                sess.run(model.init_token_op)

            step_time, loss = 0.0, 0.0
            data_stream = self.data_stream
            for current_step, (sources, source_lens, targets, target_lens, labels) in enumerate(data_stream):
                start_time = time.time()

                step_loss, _ = model.step(sess, sources, source_lens, targets, target_lens, labels)
                step_time += (time.time() - start_time) / self.steps_per_checkpoint
                loss += step_loss / self.steps_per_checkpoint

                # Once in a while, print statistics, and run evals.
                if current_step % self.steps_per_checkpoint == 0:
                    logging.info("global step %d, learning rate %.4f step-time:%.2f step-loss:%.4f loss:%.4f" %
                                 (model.global_step.eval(), model.learning_rate, step_time, step_loss, loss))
                    # set zero timer and loss.
                    step_time, loss = 0.0, 0.0

        sv.stop()


def main(_):
    setup_logger(FLAGS.log_file_name)
    trainer = Trainer(special_words=special_words, raw_data_path=FLAGS.raw_data_path,
                      vocabulary_data_dir=FLAGS.vocabulary_data_dir,
                      port=FLAGS.data_stream_port, top_words=FLAGS.max_vocabulary_size, source_max_seq_length=FLAGS.source_max_seq_length, target_max_seq_length=FLAGS.target_max_seq_length,
                      job_name=FLAGS.job_name,
                      ps_hosts=FLAGS.ps_hosts, worker_hosts=FLAGS.worker_hosts, task_index=FLAGS.task_index,
                      gpu=FLAGS.gpu,
                      model_dir=FLAGS.model_dir, is_sync=FLAGS.is_sync, batch_size=FLAGS.batch_size,
                      steps_per_checkpoint=FLAGS.steps_per_checkpoint)
    trainer.train()


if __name__ == "__main__":
    tf.app.run()
