# coding=utf-8
import tensorflow as tf
import numpy as np


class Q2VModel(object):
    def __init__(self, max_seq_length, vocab_size, word_embed_size, seq_embed_size, num_layers, src_cell_size,
                 tgt_cell_size, batch_size, learning_rate, max_gradient_norm, worker_hosts, issync=0, use_lstm=True):
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.word_embed_size = word_embed_size
        self.seq_embed_size = seq_embed_size
        self.num_layers = num_layers
        self.src_cell_size = src_cell_size
        self.tgt_cell_size = tgt_cell_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.source_partitions = tf.placeholder(tf.int32, [None])
        self.target_partitions = tf.placeholder(tf.int32, [None])
        self.issync = issync
        self.worker_hosts = worker_hosts
        self.use_lstm = use_lstm

        # List of operations to be called after each training step, see
        # _add_post_train_ops
        self._post_train_ops = []

        self.graph()

    def graph(self):
        # placeholder for input data
        self._src_input_data = tf.placeholder(tf.int32, [None, self.max_seq_length], name='source_sequence')
        self._tgt_input_data = tf.placeholder(tf.int32, [None, self.max_seq_length], name='target_sequence')
        self._labels = tf.placeholder(tf.float32, [None], name='labels')
        self._src_lens = tf.placeholder(tf.int32, [None], name='source_seq_lengths')
        self._tgt_lens = tf.placeholder(tf.int32, [None], name='target_seq_lengths')

        # create word embedding vectors
        self.word_embedding = tf.get_variable('word_embedding', [self.vocab_size, self.word_embed_size],
                                              initializer=tf.random_uniform_initializer(-0.25, 0.25))

        # transform input tensors from tokenID to word embedding
        self.src_input_distributed = tf.nn.embedding_lookup(self.word_embedding, self._src_input_data,
                                                            name='dist_source')
        self.tgt_input_distributed = tf.nn.embedding_lookup(self.word_embedding, self._tgt_input_data,
                                                            name='dist_target')

        self._def_network()
        self._def_loss()
        self._def_optimize()
        #self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

    def _def_network(self):
        # Build shared encoder
        with tf.variable_scope('shared_encoder'):
            # TODO: need play with forgetGate and peeholes here
            # for tf1.1, tf.contrib.rnn.LSTMCell
            if self.use_lstm:
                # src_single_cell = tf.nn.rnn_cell.LSTMCell(self.src_cell_size, forget_bias=1.0, use_peepholes=False)
                src_single_cell = tf.contrib.rnn.LSTMCell(self.src_cell_size, forget_bias=1.0, use_peepholes=False)
            else:
                # src_single_cell = tf.nn.rnn_cell.GRUCell(self.src_cell_size)
                src_single_cell = tf.contrib.rnn.GRUCell(self.src_cell_size)

            src_cell = src_single_cell
            if self.num_layers > 1:
                # src_cell = tf.nn.rnn_cell.MultiRNNCell([src_single_cell] * self.num_layers)
                src_cell = tf.contrib.rnn.MultiRNNCell([src_single_cell] * self.num_layers)

            # compute source sequence related tensors
            src_output, _ = tf.nn.dynamic_rnn(src_cell, self.src_input_distributed, sequence_length=self._src_lens,
                                              dtype=tf.float32)
            src_last_output = self._last_output(src_output, src_cell.output_size, self.source_partitions)
            self.src_M = tf.get_variable('src_M', shape=[self.src_cell_size, self.seq_embed_size],
                                         initializer=tf.truncated_normal_initializer())
            # self.src_b = tf.get_variable('src_b', shape=[self.seq_embed_size])
            self.src_seq_embedding = tf.matmul(src_last_output, self.src_M)  # + self.src_b

            # declare tgt_M tensor before reuse them
            self.tgt_M = tf.get_variable('tgt_M', shape=[self.src_cell_size, self.seq_embed_size],
                                         initializer=tf.truncated_normal_initializer())
            # self.tgt_b = tf.get_variable('tgt_b', shape=[self.seq_embed_size])

        with tf.variable_scope('shared_encoder', reuse=True):
            # compute target sequence related tensors by reusing shared_encoder model
            tgt_output, _ = tf.nn.dynamic_rnn(src_cell, self.tgt_input_distributed, sequence_length=self._tgt_lens,
                                              dtype=tf.float32)
            tgt_last_output = self._last_output(tgt_output, src_cell.output_size, self.target_partitions)

            self.tgt_seq_embedding = tf.matmul(tgt_last_output, self.tgt_M)  # + self.tgt_b

    @staticmethod
    def _last_output_legacy(output, length):
        b_size = tf.shape(output)[0]
        max_len = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        idx = tf.range(0, b_size) * max_len + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, idx)
        # output = tf.transpose(output, [1, 0, 2])
        # last = tf.gather(output, int(output.get_shape()[0]) - 1)
        return relevant

    @staticmethod
    def _last_output(output, output_size, partitions):
        outputs = tf.reshape(tf.stack(output), [-1, output_size])

        num_partitions = 2

        res_out = tf.dynamic_partition(outputs, partitions, num_partitions)

        return res_out[1]

    @staticmethod
    def contrastive_loss(y, d, batch_size):
        # tmp = y * tf.square(d)
        tmp = tf.multiply(y, tf.square(d))
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2

    def _def_loss(self):

        self.distance = tf.sqrt(
            tf.reduce_sum(tf.square(tf.subtract(self.src_seq_embedding, self.tgt_seq_embedding)), 1, keep_dims=True))

        self.distance = tf.div(self.distance,
                               tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.src_seq_embedding), 1, keep_dims=True)),
                                      tf.sqrt(tf.reduce_sum(tf.square(self.tgt_seq_embedding), 1, keep_dims=True))))

        self.distance = tf.reshape(self.distance, [-1], name="distance")

        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self._labels, self.distance, self.batch_size)
            # compute src / tgt similarity

    def get_train_feed_dict(self, sources, sources_len, targets, targets_len, labels, source_partitions, target_partitions):
        """
        Returns a batch feed dict for given srcSquence and tgtSequences.

        """
        d = dict()
        d[self._src_input_data] = np.array(sources, dtype=np.int32)
        d[self._src_lens] = np.array(sources_len, dtype=np.int32)
        d[self._tgt_input_data] = np.array(targets, dtype=np.int32)
        d[self._tgt_lens] = np.array(targets_len, dtype=np.int32)
        d[self._labels] = np.array(labels, dtype=np.int64)
        d[self.source_partitions] = np.array(source_partitions, dtype=np.int32)
        d[self.target_partitions] = np.array(target_partitions, dtype=np.int32)
        return d

    def _def_optimize(self):
        """
        Builds graph to minimize loss function.
        """

        # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)

        grads_and_vars = optimizer.compute_gradients(self.loss)
        if self.issync == 1:
            rep_op = tf.train.SyncReplicasOptimizer(optimizer,
                                                    replicas_to_aggregate=len(
                                                        self.worker_hosts),
                                                    total_num_replicas=len(
                                                        self.worker_hosts),
                                                    use_locking=True)
            self.train = rep_op.apply_gradients(grads_and_vars,
                                                   global_step=self.global_step)
            self.init_token_op = rep_op.get_init_tokens_op()
            self.chief_queue_runner = rep_op.get_chief_queue_runner()
        else:
            self.train = optimizer.apply_gradients(grads_and_vars,
                                                      global_step=self.global_step)

        self._add_post_train_ops()

    def _add_post_train_ops(self):
        """
        Replaces the self.train operation with an operation group, consisting of
        the training operation itself and the operations listed in
        self.post_train_ops.

        Called by _def_optimize().

        """
        with tf.control_dependencies([self.train]):
            self.train = tf.group(self.train, *self._post_train_ops)

    def save(self, session, path, global_step=None):
        """ Saves variables to given path """
        return self.saver.save(session, path, global_step)

    def generate_partition(self, batch_size, seqlen):
        partitions = [0] * (batch_size * self.max_seq_length)
        step = 0
        for each in seqlen:
            idx = each + self.max_seq_length * step
            partitions[idx - 1] = 1
            step += 1
        return partitions
