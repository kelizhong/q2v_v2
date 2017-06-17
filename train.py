# coding=utf-8
from __future__ import print_function
import os
import time
import sys
import codecs
import math
import tensorflow as tf
from data_util import create_vocabulary, sentence_to_padding_tokens, initialize_vocabulary
from model import Q2VModel

# cluster specification
parameter_servers = ["pc-01:2222"]
workers = ["pc-02:2222",
           "pc-03:2222",
           "pc-04:2222"]
cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})

tf.app.flags.DEFINE_float("learning_rate", 0.3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training(positive pair count based).")
tf.app.flags.DEFINE_integer("embedding_size", 50, "Size of word embedding vector.")
tf.app.flags.DEFINE_integer("encoding_size", 80,
                            "Size of sequence encoding vector. Same number of nodes for each model layer.")
tf.app.flags.DEFINE_integer("src_cell_size", 96, "LSTM cell size in source RNN model.")
tf.app.flags.DEFINE_integer("tgt_cell_size", 96,
                            "LSTM cell size in target RNN model. Same number of nodes for each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("max_vocabulary_size", 50000, "Sequence vocabulary size in the mapping task.")

tf.app.flags.DEFINE_integer("max_seq_length", 55, "max number of words in each source or target sequence.")
tf.app.flags.DEFINE_integer("max_epoch", 8, "max epoc number for training procedure.")
tf.app.flags.DEFINE_integer("predict_nbest", 20, "max top N for evaluation prediction.")

tf.app.flags.DEFINE_string("data_dir", 'data', "Data directory")
tf.app.flags.DEFINE_string("train_data_file", 'data/rawdata/TrainPairs', "Train Data file")
tf.app.flags.DEFINE_string("model_dir", 'models', "Trained model directory.")
tf.app.flags.DEFINE_string("export_dir", 'exports', "Trained model directory.")
tf.app.flags.DEFINE_string("device", "gpu:0",
                           "Default to use GPU:0. Softplacement used, if no GPU found, further default to cpu:0.")

tf.app.flags.DEFINE_integer("steps_per_checkpoint", 10,
                            "How many training steps to do per checkpoint.")

tf.app.flags.DEFINE_boolean("embeddingMode", False,
                            "Set to True to generate embedding vectors file for entries in targetIDs file.")

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "0.0.0.0:2221",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "0.0.0.0:2222",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "single", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 0, "")

FLAGS = tf.app.flags.FLAGS


def create_model(session, forward_only):
    """Create SSE model and initialize or load parameters in session."""
    model = Q2VModel(FLAGS.max_seq_length, FLAGS.max_vocabulary_size, FLAGS.embedding_size, FLAGS.encoding_size,
                     FLAGS.num_layers, FLAGS.src_cell_size, FLAGS.tgt_cell_size,
                     FLAGS.batch_size, FLAGS.learning_rate, FLAGS.max_gradient_norm)

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if forward_only:
            print('Error!!!Could not load model from specified folder: %s' % FLAGS.model_dir)
            exit(-1)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())
    return model


def read_train_data(train_data_path, vocabulary):
    """Read data from source and target files.

    Args:
      train_data_path: targetID, encoded source tokenIDs.
      vocabulary: targetID and encoded target seqence tokenIDs.

    Returns:
      data_set: a list of positive (sourceTokenIDs, targetTokenIDs) pairs read from the provided data files.
      batchsize_per_epoc: how many batched needed to go through out one epoch.
    """
    data_set = []
    counter = 0
    for line in codecs.open(train_data_path, "r", 'utf-8'):
        source, target, _ = line.strip().split('\t')
        counter += 1
        if counter % 1000 == 0:
            print("  reading data line %d" % counter)
            sys.stdout.flush()

        if counter > 50000:
            break
        source_len, source_tokens = sentence_to_padding_tokens(source, vocabulary, FLAGS.max_seq_length)
        target_len, target_tokens = sentence_to_padding_tokens(target, vocabulary, FLAGS.max_seq_length)

        data_set.append([source_tokens, source_len, target_tokens, target_len, 1])

    return data_set, int(math.floor(counter / FLAGS.batch_size))


def train_1(checkpoint_dir, gpu=""):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    task_index = FLAGS.task_index
    job_name = FLAGS.job_name
    if job_name == "single":
        master = ""
    else:
        cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
        server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=task_index)
        master = server.target
    issync = FLAGS.issync
    if FLAGS.job_name == "ps":
        server.join()
    else:
        # Device setting
        core_str = "cpu:0" if (gpu is None or gpu == "") else "gpu:%d" % int(gpu)
        if job_name == "worker":
            device = tf.train.replica_device_setter(cluster=cluster,
                                                    worker_device='job:worker/task:%d/%s' % (task_index, core_str),
                                                    ps_device='job:ps/task:%d/%s' % (task_index, core_str))
        else:
            device = "/" + core_str

        with tf.device(device):
            with tf.Session() as sess:
                print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.embedding_size))
                model = create_model(sess, False)
            for each in tf.global_variables():
                print(each.name, each.device)
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir=checkpoint_dir,
                                 init_op=init_op,
                                 summary_op=None,
                                 saver=saver,
                                 global_step=model.global_step,
                                 save_model_secs=60)
        gpu_options = tf.GPUOptions(allow_growth=True)
        session_config = tf.ConfigProto(allow_soft_placement=True,
                                        # log_device_placement=True,
                                        gpu_options=gpu_options,
                                        intra_op_parallelism_threads=16)
        with sv.prepare_or_wait_for_session(master=master, config=session_config) as sess:
            # 如果是同步模式
            if FLAGS.task_index == 0 and issync == 1:
                sv.start_queue_runners(sess, [model.chief_queue_runner])
                sess.run(model.init_token_op)

            vocab_path = os.path.join(FLAGS.data_dir, "vocab")
            create_vocabulary(vocab_path, FLAGS.train_data_file, FLAGS.max_vocabulary_size)
            print("Loading vocabulary")
            vocab, _ = initialize_vocabulary(vocab_path)
            print("Vocabulary size: %d" % len(vocab))
            train_set, steps = read_train_data(FLAGS.train_data_file, vocab)
            step_time, loss = 0.0, 0.0
            current_step = 0
            for epoch in range(FLAGS.max_epoch):
                epoch_start_time = time.time()
                for step in range(steps):  # basic drop out here
                    start_time = time.time()
                    sources, targets, sources_len, targets_len, labels = [], [], [], [], []
                    for idx in xrange(FLAGS.batch_size):
                        source_tokens, source_len, target_tokens, target_len, label = train_set[
                            step * FLAGS.batch_size + idx]
                        sources.append(source_tokens)
                        sources_len.append(source_len)
                        targets.append(target_tokens)
                        targets_len.append(target_len)
                        labels.append(label)
                    source_partitions = model.generate_partition(FLAGS.batch_size, sources_len)
                    target_partitions = model.generate_partition(FLAGS.batch_size, targets_len)
                    d = model.get_train_feed_dict(sources, sources_len, targets, targets_len, labels, source_partitions,
                                                  target_partitions)
                    ops = [model.train, model.loss]
                    _, step_loss = sess.run(ops, feed_dict=d)
                    step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                    loss += step_loss / FLAGS.steps_per_checkpoint
                    current_step += 1

                    # Once in a while, we save checkpoint, print statistics, and run evals.
                    if current_step % FLAGS.steps_per_checkpoint == 0:
                        print("global epoc: %.3f, global step %d, learning rate %.4f step-time:%.2f loss:%.4f " %
                              (float(model.global_step.eval()) / float(steps), model.global_step.eval(),
                               model.learning_rate,
                               step_time, step_loss))
                        # Save checkpoint and zero timer and loss.
                        step_time, loss = 0.0, 0.0

                        sys.stdout.flush()

                # give out epoc statistics
                epoch_train_time = time.time() - epoch_start_time
                print('epoch# %d  took %f hours' % (epoch, epoch_train_time / (60.0 * 60)))

        sv.stop()


def train():
    vocab_path = os.path.join(FLAGS.data_dir, "vocab")
    create_vocabulary(vocab_path, FLAGS.train_data_file, FLAGS.max_vocabulary_size)
    print("Loading vocabulary")
    vocab, _ = initialize_vocabulary(vocab_path)
    print("Vocabulary size: %d" % len(vocab))
    train_set, steps = read_train_data(FLAGS.train_data_file, vocab)
    cfg = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    with tf.device('/' + FLAGS.device), tf.Session(config=cfg) as sess:
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.embedding_size))
        model = create_model(sess, False)

        # setup tensorboard logging
        # sw = tf.train.SummaryWriter(FLAGS.model_dir, sess.graph, flush_secs=120)
        # summary_op = model.add_summaries()

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        for epoch in range(FLAGS.max_epoch):
            epoch_start_time = time.time()
            for step in range(steps):  # basic drop out here
                start_time = time.time()
                sources, targets, sources_len, targets_len, labels = [], [], [], [], []
                for idx in xrange(FLAGS.batch_size):
                    source_tokens, source_len, target_tokens, target_len, label = train_set[
                        step * FLAGS.batch_size + idx]
                    sources.append(source_tokens)
                    sources_len.append(source_len)
                    targets.append(target_tokens)
                    targets_len.append(target_len)
                    labels.append(label)
                source_partitions = model.generate_partition(FLAGS.batch_size, sources_len)
                target_partitions = model.generate_partition(FLAGS.batch_size, targets_len)
                d = model.get_train_feed_dict(sources, sources_len, targets, targets_len, labels, source_partitions,
                                              target_partitions)
                ops = [model.train, model.loss]
                _, step_loss = sess.run(ops, feed_dict=d)
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                current_step += 1

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % FLAGS.steps_per_checkpoint == 0:
                    print("global epoc: %.3f, global step %d, learning rate %.4f step-time:%.2f loss:%.4f " %
                          (float(model.global_step.eval()) / float(steps), model.global_step.eval(),
                           model.learning_rate,
                           step_time, step_loss))
                    # Save checkpoint and zero timer and loss.
                    checkpoint_path = os.path.join(FLAGS.model_dir, "q2v.ckpt")
                    model.save(sess, checkpoint_path, global_step=model.global_step)
                    step_time, loss = 0.0, 0.0

                    sys.stdout.flush()

            # give out epoc statistics
            epoch_train_time = time.time() - epoch_start_time
            print('epoch# %d  took %f hours' % (epoch, epoch_train_time / (60.0 * 60)))
            # Save checkpoint at end of each epoch
            checkpoint_path = os.path.join(FLAGS.model_dir, "q2v.ckpt")
            model.save(sess, checkpoint_path + '-epoch-%d' % epoch)


def main(_):
    #train()
    train_1("models")


if __name__ == "__main__":
    tf.app.run()
