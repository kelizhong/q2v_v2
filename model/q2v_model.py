# coding=utf-8
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper

import logbook as logging
from external.cocob_optimizer import COCOB


class Q2VModel(object):
    def __init__(self, config, mode='train'):

        assert mode.lower() in ['train', 'decode', 'encode']

        self.config = config
        self.mode = mode.lower()

        self.job_name = config['job_name']
        self.worker_hosts_size = len(config['worker_hosts'].split(",")) if config['worker_hosts'] else 0

        self.bidirectional = config['bidirectional']
        self.dtype = tf.float16 if config['use_fp16'] else tf.float32
        self.num_layers = config['num_layers']
        self.cell_type = config['cell_type']
        self.hidden_units = config['hidden_units']

        self.use_dropout = config['use_dropout']
        self.keep_prob = 1.0 - config['dropout_rate']
        self.use_residual = config['use_residual']

        self.max_vocabulary_size = config['max_vocabulary_size']
        self.embedding_size = config['embedding_size']

        self.optimizer = config['optimizer']
        self.max_gradient_norm = config['max_gradient_norm']

        self.is_sync = config['is_sync']
        self.worker_hosts = config['worker_hosts']

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        self.learning_rate = tf.maximum(
            config['min_learning_rate'],  # min_lr_rate.
            tf.train.exponential_decay(config['learning_rate'], self.global_step, config['decay_steps'],
                                       config['lr_decay_factor']))

        # List of operations to be called after each training step, see
        # _add_post_train_ops
        self._post_train_ops = []

        self.build_model()
        self.saver = tf.train.Saver(tf.global_variables())

    def build_model(self):
        logging.info("building model..")
        self.init_placeholders()
        self._init_embedding_layer()
        self.build_source_encoder()
        if self.mode == 'train':
            self.build_target_encoder()
            self.init_loss()
            self.init_optimizer()

    @staticmethod
    def dense_batch_relu(x, phase, scope):
        with tf.variable_scope(scope):
            h1 = tf.contrib.layers.fully_connected(x, 100,
                                                   activation_fn=None,
                                                   scope='dense')
            h2 = tf.contrib.layers.batch_norm(h1,
                                              center=True, scale=True,
                                              is_training=phase,
                                              scope='bn')
            return tf.nn.relu(h2, 'relu')

    def init_placeholders(self):

        self.keep_prob_placeholder = tf.placeholder(self.dtype, shape=[], name='keep_prob')

        # TODO use MutableHashTable to store word->id mapping in checkpoint
        # source_inputs: [batch_size, max_time_steps]
        self.src_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=(None, None), name='source_inputs')

        # source_inputs_length: [batch_size]
        self.src_inputs_length = tf.placeholder(
            dtype=tf.int32, shape=(None,), name='source_inputs_length')

        # get dynamic batch_size
        self.batch_size = tf.to_float(tf.shape(self.src_inputs)[0])

        if self.mode == 'train':
            # target_inputs: [batch_size, max_time_steps]
            self.tgt_inputs = tf.placeholder(
                dtype=tf.int32, shape=(None, None), name='target_inputs')
            # decoder_inputs_length: [batch_size]
            self.tgt_inputs_length = tf.placeholder(
                dtype=tf.int32, shape=(None,), name='target_inputs_length')

            self.labels = tf.placeholder(tf.int32, [None], name='labels')

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

    def _init_embedding_layer(self):
        # create word embedding vectors
        self.encoder_embeddings = tf.get_variable(name='embedding',
                                              shape=[self.max_vocabulary_size, self.embedding_size],
                                              initializer=tf.random_uniform_initializer(-1.0, 1.0), dtype=self.dtype)

    def build_source_encoder(self):
        logging.info("building source encoder..")
        with tf.variable_scope('shared_encoder', dtype=self.dtype) as scope:

            # Embedded_inputs: [batch_size, time_step, embedding_size]
            src_inputs_embedded = tf.nn.embedding_lookup(
                params=self.encoder_embeddings, ids=self.src_inputs)

            # src_inputs_embedded = self.dense_batch_relu(src_inputs_embedded, True, scope=scope)

            if self.use_residual:
                # Input projection layer to feed embedded inputs to the cell
                # ** Essential when use_residual=True to match input/output dims
                input_layer = Dense(self.hidden_units, dtype=self.dtype, name='input_projection')

                # Embedded inputs having gone through input projection layer
                src_inputs_embedded = input_layer(src_inputs_embedded)

            if self.bidirectional:
                logging.info("building bidirectional encoder..")
                self.fw_encoder_cell = self.build_encoder_cell()
                self.bw_encoder_cell = self.build_encoder_cell()
                self.src_outputs, self.src_last_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.fw_encoder_cell, cell_bw=self.bw_encoder_cell,
                    inputs=src_inputs_embedded,
                    sequence_length=self.src_inputs_length,
                    time_major=False,
                    dtype=self.dtype,
                    parallel_iterations=16,
                    scope=scope)

                fw_state, bw_state = self.src_last_state
                self.src_last_state = []
                for f, b in zip(fw_state, bw_state):
                    if isinstance(f, LSTMStateTuple):
                        self.src_last_state.append(
                            LSTMStateTuple(tf.concat([f.c, b.c], axis=1), tf.concat([f.h, b.h], axis=1)))
                    else:
                        self.src_last_state.append(tf.concat([f, b], 1))
                self.src_outputs = tf.concat([self.src_outputs[0], self.src_outputs[1]], axis=2)

            else:
                logging.info("building encoder..")
                # Building encoder_cell
                self.encoder_cell = self.build_encoder_cell()
                # Encode input sequences into context vectors:
                # encoder_outputs: [batch_size, max_time_step, cell_output_size]
                # encoder_state: [batch_size, cell_output_size]
                self.src_outputs, self.src_encoder_last_state = tf.nn.dynamic_rnn(
                    cell=self.encoder_cell, inputs=src_inputs_embedded,
                    sequence_length=self.src_inputs_length, dtype=self.dtype,
                    time_major=False)
            # [batch_size, hidden unit]
            self.src_last_output = self.extract_last_output(self.src_outputs, self.src_inputs_length-1)

    def build_target_encoder(self):
        logging.info("building target encoder..")
        with tf.variable_scope('shared_encoder', dtype=self.dtype, reuse=True) as scope:

            # Embedded_inputs: [batch_size, time_step, embedding_size]
            tgt_inputs_embedded = tf.nn.embedding_lookup(
                params=self.encoder_embeddings, ids=self.tgt_inputs)
            # tgt_inputs_embedded = self.dense_batch_relu(tgt_inputs_embedded, True, scope=scope)

            if self.use_residual:
                # Input projection layer to feed embedded inputs to the cell
                # ** Essential when use_residual=True to match input/output dims
                input_layer = Dense(self.hidden_units, dtype=self.dtype, name='input_projection')

                # Embedded inputs having gone through input projection layer
                tgt_inputs_embedded = input_layer(tgt_inputs_embedded)

            if self.bidirectional:
                logging.info("building bidirectional encoder..")
                self.tgt_outputs, self.tgt_last_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.fw_encoder_cell, cell_bw=self.bw_encoder_cell,
                    inputs=tgt_inputs_embedded,
                    sequence_length=self.tgt_inputs_length,
                    time_major=False,
                    dtype=self.dtype,
                    parallel_iterations=16,
                    scope=scope)

                fw_state, bw_state = self.tgt_last_state
                self.tgt_last_state = []
                for f, b in zip(fw_state, bw_state):
                    if isinstance(f, LSTMStateTuple):
                        self.tgt_last_state.append(
                            LSTMStateTuple(tf.concat([f.c, b.c], axis=1), tf.concat([f.h, b.h], axis=1)))
                    else:
                        self.tgt_last_state.append(tf.concat([f, b], 1))
                self.tgt_outputs = tf.concat([self.tgt_outputs[0], self.tgt_outputs[1]], axis=2)

            else:
                logging.info("building encoder..")
                # Building encoder_cell
                # Encode input sequences into context vectors:
                # encoder_outputs: [batch_size, max_time_step, cell_output_size]
                # encoder_state: [batch_size, cell_output_size]
                self.tgt_outputs, self.tgt_encoder_last_state = tf.nn.dynamic_rnn(
                    cell=self.encoder_cell, inputs=tgt_inputs_embedded,
                    sequence_length=self.tgt_inputs_length, dtype=self.dtype,
                    time_major=False)

            self.tgt_last_output = self.extract_last_output(self.tgt_outputs, self.tgt_inputs_length-1)

    @staticmethod
    def extract_last_output(output, ind):
        """
        Get specified elements along the first axis of tensor.
        :param data: Tensorflow tensor that will be subsetted.
        :param ind: Indices to take (one for each element along axis 0 of data).
        :return: Subsetted tensor.
        """

        batch_range = tf.range(tf.shape(output)[0])
        indices = tf.stack([batch_range, ind], axis=1)
        res = tf.gather_nd(output, indices)

        return res

    def cos_similarity_loss(self):
        labels = tf.cast(self.labels, self.dtype)
        src_last_output_normalize = tf.nn.l2_normalize(self.src_last_output, dim=1)
        tgt_last_output_normalize = tf.nn.l2_normalize(self.tgt_last_output, dim=1)
        distance = tf.losses.cosine_distance(src_last_output_normalize, tgt_last_output_normalize, dim=1)
        loss = tf.add(tf.multiply(labels, tf.maximum(1 - distance, 0)), tf.multiply((1 - labels), tf.maximum(1 + distance, 0)))
        loss_mean = tf.reduce_mean(loss)
        return loss_mean

    def cos_loss(self):
        labels = tf.cast(self.labels, self.dtype)
        # src_last_output_normalize = tf.nn.l2_normalize(self.src_last_output, dim=1)
        # tgt_last_output_normalize = tf.nn.l2_normalize(self.tgt_last_output, dim=1)
        distance = self.compute_euclidean_distance(self.src_last_output, self.tgt_last_output)
        loss_mean = tf.reduce_mean(distance)
        return loss_mean

    @staticmethod
    def compute_euclidean_distance(x, y):
        """
        Computes the euclidean distance between two tensors
        """

        with tf.name_scope('euclidean_distance') as scope:
            distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), 1))
        return distance

    @staticmethod
    def compute_cosine_distance(x, y):
        x_normalize = tf.nn.l2_normalize(x, dim=1)
        y_normalize = tf.nn.l2_normalize(y, dim=1)
        distance = tf.losses.cosine_distance(x_normalize, y_normalize, dim=1)
        return distance

    @staticmethod
    def compute_manhattan_distance(x, y):
        tf.subtract(x, y)
        distance = tf.reduce_sum(tf.abs(tf.subtract(x, y)), reduction_indices=1)
        return distance

    def contrastive_loss(self):

        labels = tf.cast(self.labels, self.dtype)
        margin = tf.constant(5.)
        # Euclidean distance between x1,x2
        distance = self.compute_euclidean_distance(self.src_last_output, self.tgt_last_output)

        # distance = tf.div(distance,
        #                 tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.src_last_output), 1, keep_dims=True)),
        #                        tf.sqrt(tf.reduce_sum(tf.square(self.tgt_last_output), 1, keep_dims=True))))

        loss = tf.add(tf.multiply(labels, tf.square(distance)), tf.multiply((1 - labels), tf.square(tf.nn.relu(margin - distance))))
        loss_mean = tf.reduce_mean(loss)
        return loss_mean

    def l1_loss(self):
        # Calculate L1 Distance
        labels = tf.cast(self.labels, self.dtype)
        distance = tf.reduce_sum(tf.abs(tf.add(self.src_last_output, tf.negative(self.tgt_last_output))), reduction_indices=1)
        # loss = tf.nn.sigmoid(distance)
        # MSE error
        loss = tf.square(labels - tf.exp(tf.negative(distance)))
        loss_mean = tf.reduce_mean(loss)
        return loss_mean

    def init_loss(self):

        with tf.name_scope("loss"):
            # self.loss = self.contrastive_loss_distance()
            # self.loss = self.contrastive_loss()
            # self.loss = self.cos_similarity_loss()
            # self.loss = self.dot_product_loss()
            # self.loss = self.cos_loss()
            self.loss = self.l1_loss()

    def check_feeds(self, src_inputs, src_inputs_length, tgt_inputs, tgt_inputs_length, labels):
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

        if self.mode == 'train':
            input_feed[self.tgt_inputs.name] = tgt_inputs
            input_feed[self.tgt_inputs_length.name] = tgt_inputs_length
            input_feed[self.labels.name] = labels

        return input_feed

    def train(self, sess, src_inputs, src_inputs_length, tgt_inputs, tgt_inputs_length, labels):
        # Check if the model is 'training' mode
        if self.mode.lower() != 'train':
            raise ValueError("train step can only be operated in train mode")

        input_feed = self.check_feeds(src_inputs, src_inputs_length, tgt_inputs, tgt_inputs_length, labels)

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
        logging.info("setting optimizer..")
        # Gradients and SGD update operation for training the model
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower == 'adagrad':
            self.opt = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'cocob':
            self.opt = COCOB()
        else:
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        # Compute gradients of loss w.r.t. all trainable variables

        if self.job_name == "worker" and self.is_sync:
            self.opt = tf.train.SyncReplicasOptimizer(self.opt,
                                                      replicas_to_aggregate=self.worker_hosts_size,
                                                      total_num_replicas=self.worker_hosts_size,
                                                      use_locking=True
                                                      )
            grads_and_vars = self.opt.compute_gradients(loss=self.loss)
            gradients, variables = zip(*grads_and_vars)
        else:
            gradients = tf.gradients(self.loss, tf.trainable_variables(), aggregation_method=2)
            variables = tf.trainable_variables()

        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.updates = self.opt.apply_gradients(zip(clipped_gradients, variables), self.global_step)

        if self.job_name == "worker":
            self.init_token_op = self.opt.get_init_tokens_op()
            self.chief_queue_runner = self.opt.get_chief_queue_runner()

        # self._add_post_train_ops()

    def _add_post_train_ops(self):
        """
        Replaces the self.train operation with an operation group, consisting of
        the training operation itself and the operations listed in
        self.post_train_ops.

        Called by _def_optimize().

        """
        with tf.control_dependencies([self.updates]):
            self.updates = tf.group(self.updates, *self._post_train_ops)

    def encode(self, sess, src_inputs, src_inputs_length):

        assert self.mode == 'encode', "encode function only support encode mode"
        input_feed = self.check_feeds(src_inputs, src_inputs_length,
                                      tgt_inputs=None, tgt_inputs_length=None, labels=None)

        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = 1.0

        output_feed = [self.src_last_output]
        outputs = sess.run(output_feed, input_feed)

        return outputs[0]  # encode: [batch_size, cell.output_size]
