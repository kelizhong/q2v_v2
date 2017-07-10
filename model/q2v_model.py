# coding=utf-8
import math

import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper

import logbook as logging
from external.cocob_optimizer import COCOB


class Q2VModel(object):
    def __init__(self, config, mode='train'):

        assert mode.lower() in ['train', 'decode', 'encode']

        self.config = config
        self.mode = mode.lower()

        self.dtype = tf.float16 if config['use_fp16'] else tf.float32
        self.num_layers = config['num_layers']
        self.cell_type = config['cell_type']
        self.hidden_units = config['hidden_units']

        self.use_dropout = config['use_dropout']
        self.keep_prob = 1.0 - config['dropout_rate']
        self.use_residual = config['use_residual']

        self.max_vocabulary_size = config['max_vocabulary_size']
        self.embedding_size = config['embedding_size']

        self.learning_rate = config['learning_rate']
        self.optimizer = config['optimizer']
        self.max_gradient_norm = config['max_gradient_norm']

        self.is_sync = config['is_sync']
        self.worker_hosts = config['worker_hosts']

        # List of operations to be called after each training step, see
        # _add_post_train_ops
        self._post_train_ops = []

        self.build_model()
        self.saver = tf.train.Saver(tf.global_variables())

    def build_model(self):
        logging.info("building model..")
        self.init_placeholders()
        self.build_source_encoder()
        if self.mode == 'train':
            self.build_target_encoder()
            self.init_loss()
            self.init_optimizer()

    def init_placeholders(self):

        self.keep_prob_placeholder = tf.placeholder(self.dtype, shape=[], name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        # TODO use MutableHashTable to store word->id mapping in checkpoint
        # source_inputs: [batch_size, max_time_steps]
        self.src_inputs = tf.placeholder(dtype=tf.int32,
                                             shape=(None, None), name='source_inputs')

        # source_inputs_length: [batch_size]
        self.src_inputs_length = tf.placeholder(
            dtype=tf.int32, shape=(None,), name='source_inputs_length')

        self.src_partitions = tf.placeholder(tf.int32, [None], name='source_partitions')

        # get dynamic batch_size
        self.batch_size = tf.to_float(tf.shape(self.src_inputs)[0])

        if self.mode == 'train':
            # target_inputs: [batch_size, max_time_steps]
            self.tgt_inputs = tf.placeholder(
                dtype=tf.int32, shape=(None, None), name='target_inputs')
            # decoder_inputs_length: [batch_size]
            self.tgt_inputs_length = tf.placeholder(
                dtype=tf.int32, shape=(None,), name='target_inputs_length')

            self.tgt_partitions = tf.placeholder(tf.int32, [None], name='target_partitions')

            self.labels = tf.placeholder(self.dtype, [None], name='labels')

    def build_single_cell(self):
        cell_type = LSTMCell
        if self.cell_type.lower() == 'gru':
            cell_type = GRUCell
        cell = cell_type(self.hidden_units)

        if self.use_dropout:
            cell = DropoutWrapper(cell, dtype=self.dtype,
                                  output_keep_prob=self.keep_prob_placeholder, )
        if self.use_residual:
            cell = ResidualWrapper(cell)

        return cell

    # Building encoder cell
    def build_encoder_cell(self):

        return MultiRNNCell([self.build_single_cell() for _ in range(self.num_layers)])

    def build_source_encoder(self):
        logging.info("building source encoder..")
        with tf.variable_scope('shared_encoder'):
            # Building encoder_cell
            self.src_cell = self.build_encoder_cell()

            # Initialize encoder_embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)

            self.src_embeddings = tf.get_variable(name='embedding',
                                                      shape=[self.max_vocabulary_size, self.embedding_size],
                                                      initializer=initializer, dtype=self.dtype)

            # Embedded_inputs: [batch_size, time_step, embedding_size]
            self.src_inputs_embedded = tf.nn.embedding_lookup(
                params=self.src_embeddings, ids=self.src_inputs)

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(self.hidden_units, dtype=self.dtype, name='input_projection')

            # Embedded inputs having gone through input projection layer
            self.src_inputs_embedded = input_layer(self.src_inputs_embedded)

            # Encode input sequences into context vectors:
            # encoder_outputs: [batch_size, max_time_step, cell_output_size]
            # encoder_state: [batch_size, cell_output_size]
            self.src_outputs, self.src_encoder_last_state = tf.nn.dynamic_rnn(
                cell=self.src_cell, inputs=self.src_inputs_embedded,
                sequence_length=self.src_inputs_length, dtype=self.dtype,
                time_major=False)

            self.src_last_output = self._last_output(self.src_outputs, self.src_cell.output_size, self.src_partitions)

    def build_target_encoder(self):
        logging.info("building target encoder..")
        with tf.variable_scope('shared_encoder', reuse=True):
            # Building encoder_cell
            self.tgt_cell = self.build_encoder_cell()

            # Initialize encoder_embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)

            self.tgt_embeddings = tf.get_variable(name='embedding',
                                                      shape=[self.max_vocabulary_size, self.embedding_size],
                                                      initializer=initializer, dtype=self.dtype)

            # Embedded_inputs: [batch_size, time_step, embedding_size]
            self.tgt_inputs_embedded = tf.nn.embedding_lookup(
                params=self.tgt_embeddings, ids=self.tgt_inputs)

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(self.hidden_units, dtype=self.dtype, name='input_projection')

            # Embedded inputs having gone through input projection layer
            self.tgt_inputs_embedded = input_layer(self.tgt_inputs_embedded)

            # Encode input sequences into context vectors:
            # encoder_outputs: [batch_size, max_time_step, cell_output_size]
            # encoder_state: [batch_size, cell_output_size]
            self.tgt_outputs, self.tgt_encoder_last_state = tf.nn.dynamic_rnn(
                cell=self.tgt_cell, inputs=self.tgt_inputs_embedded,
                sequence_length=self.tgt_inputs_length, dtype=self.dtype,
                time_major=False)

            self.tgt_last_output = self._last_output(self.tgt_outputs, self.tgt_cell.output_size, self.tgt_partitions)

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

    def init_loss(self):

        self.distance = tf.sqrt(
            tf.reduce_sum(tf.square(tf.subtract(self.src_last_output, self.tgt_last_output)), 1, keep_dims=True))

        self.distance = tf.div(self.distance,
                               tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.src_last_output), 1, keep_dims=True)),
                                      tf.sqrt(tf.reduce_sum(tf.square(self.tgt_last_output), 1, keep_dims=True))))

        self.distance = tf.reshape(self.distance, [-1], name="distance")

        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self.labels, self.distance, self.batch_size)
            # compute src / tgt similarity

    def check_feeds(self, src_inputs, src_inputs_length, src_partitions, tgt_inputs, tgt_inputs_length, tgt_partitions, labels):
        """
        Args:
          src_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          src_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          tgt_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          tgt_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
        Returns:
          A feed for the model that consists of encoder_inputs, encoder_inputs_length,
          decoder_inputs, decoder_inputs_length
        """

        input_batch_size = src_inputs.shape[0]
        if input_batch_size != src_inputs_length.shape[0]:
            raise ValueError("Encoder inputs and their lengths must be equal in their "
                             "batch_size, %d != %d" % (input_batch_size, src_inputs_length.shape[0]))

        if self.mode == 'train':
            target_batch_size = tgt_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError("Encoder inputs and Decoder inputs must be equal in their "
                                 "batch_size, %d != %d" % (input_batch_size, target_batch_size))
            if target_batch_size != tgt_inputs_length.shape[0]:
                raise ValueError("Decoder targets and their lengths must be equal in their "
                                 "batch_size, %d != %d" % (target_batch_size, tgt_inputs_length.shape[0]))
        input_feed = dict()

        input_feed[self.src_inputs.name] = src_inputs
        input_feed[self.src_inputs_length.name] = src_inputs_length
        input_feed[self.src_partitions.name] = src_partitions

        if self.mode == 'train':

            input_feed[self.tgt_inputs.name] = tgt_inputs
            input_feed[self.tgt_inputs_length.name] = tgt_inputs_length
            input_feed[self.labels.name] = labels
            input_feed[self.tgt_partitions.name] = tgt_partitions

        return input_feed

    def train(self, sess, src_inputs, src_inputs_length, tgt_inputs, tgt_inputs_length, labels):
        # Check if the model is 'training' mode
        if self.mode.lower() != 'train':
            raise ValueError("train step can only be operated in train mode")

        src_inputs_max_seq_length = src_inputs.shape[1]
        src_partitions = self._generate_partition(src_inputs_length, src_inputs_max_seq_length)

        tgt_inputs_max_seq_length = tgt_inputs.shape[1]
        tgt_partitions = self._generate_partition(tgt_inputs_length, tgt_inputs_max_seq_length)

        input_feed = self.check_feeds(src_inputs, src_inputs_length, src_partitions, tgt_inputs, tgt_inputs_length, tgt_partitions, labels)

        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = self.keep_prob

        output_feed = [self.updates,  # Update Op that does optimization
                       self.loss]  # Loss for this batch

        outputs = sess.run(output_feed, input_feed)
        return outputs[1]  # loss

    def init_optimizer(self):
        """
        Builds graph to minimize loss function.
        """
        print("setting optimizer..")
        # Gradients and SGD update operation for training the model
        trainable_params = tf.trainable_variables()
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'cocob':
            self.opt = COCOB()
        else:
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        # Compute gradients of loss w.r.t. all trainable variables
        gradients = tf.gradients(self.loss, trainable_params)

        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

        # Update the model
        self.updates = self.opt.apply_gradients(
            zip(gradients, trainable_params), global_step=self.global_step)

        grads_and_vars = self.opt.compute_gradients(self.loss)
        if self.is_sync and self.worker_hosts:
            rep_op = tf.train.SyncReplicasOptimizer(self.opt,
                                                    replicas_to_aggregate=len(
                                                        self.worker_hosts),
                                                    total_num_replicas=len(
                                                        self.worker_hosts),
                                                    use_locking=True)
            self.updates = rep_op.apply_gradients(grads_and_vars,
                                                global_step=self.global_step)
            self.init_token_op = rep_op.get_init_tokens_op()
            self.chief_queue_runner = rep_op.get_chief_queue_runner()
        else:
            self.updates = self.opt.apply_gradients(grads_and_vars,
                                                   global_step=self.global_step)

        self._add_post_train_ops()

    def _add_post_train_ops(self):
        """
        Replaces the self.train operation with an operation group, consisting of
        the training operation itself and the operations listed in
        self.post_train_ops.

        Called by _def_optimize().

        """
        with tf.control_dependencies([self.updates]):
            self.updates = tf.group(self.updates, *self._post_train_ops)


    @staticmethod
    def _generate_partition(seqlen, max_seq_length):
        batch_size = len(seqlen)
        partitions = [0] * (batch_size * max_seq_length)
        step = 0
        for each in seqlen:
            idx = each + max_seq_length * step
            partitions[idx - 1] = 1
            step += 1
        return partitions

    def encode(self, sess, src_inputs, src_inputs_length):

        assert self.mode == 'encode', "encode function only support encode mode"
        max_seq_length = src_inputs.shape[1]
        src_partitions = self._generate_partition(src_inputs_length, max_seq_length)
        input_feed = self.check_feeds(src_inputs, src_inputs_length, src_partitions,
                                      tgt_inputs=None, tgt_inputs_length=None, tgt_partitions= None, labels=None)

        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = 1.0
        input_feed[self.src_partitions.name] = src_partitions

        output_feed = [self.src_last_output]
        outputs = sess.run(output_feed, input_feed)

        return outputs[0] # encode: [batch_size, cell.output_size]

