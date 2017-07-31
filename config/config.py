# coding=utf-8
import os
import tensorflow as tf


project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
rawdata_dir = os.path.join(project_dir, 'data', 'rawdata')
traindata_dir = os.path.join(project_dir, 'data', 'traindata')
vocabulary_dir = os.path.join(project_dir, 'data', 'vocabulary')
logging_config_path = os.path.join(project_dir, 'config', 'logging.yaml')

# Extra vocabulary symbols
_PAD = '_pad_'
_UNK = '_unk_'
_NUM = '_num_'
_PUNC = '_punc_'

extra_tokens = [_PAD, _UNK, _NUM, _PUNC]

pad_token = extra_tokens.index(_PAD)  # pad_token = 0
unk_token = extra_tokens.index(_UNK)	 # unknown_token = 1
num_token = extra_tokens.index(_NUM)	 # number_token = 2
punc_token = extra_tokens.index(_PUNC)	 # punc_token = 3

special_words = {_PAD: pad_token, _UNK: unk_token, _NUM: num_token, _PUNC: punc_token}


# Run time variables
tf.app.flags.DEFINE_string("model_dir", os.path.join(project_dir, 'data/models'), "Trained model directory.")
tf.app.flags.DEFINE_integer("display_freq", 1, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string("gpu", None, "specify the gpu to use")
tf.app.flags.DEFINE_string("tfrecord_train_file", os.path.join(traindata_dir, 'train.tfrecords'), "tfrecord train file")
tf.app.flags.DEFINE_integer("data_stream_port", None, "port for data zmq stream")
tf.app.flags.DEFINE_boolean('debug', False, 'Enable debug')
tf.app.flags.DEFINE_integer("label_size", 4, "How many target samples in one example")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training(positive pair count based).")
# Network parameters
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("embedding_size", 64, "Size of word embedding vector.")
tf.app.flags.DEFINE_boolean('bidirectional', False, 'Enable bidirectional encoder')
tf.app.flags.DEFINE_integer('hidden_units', 64, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_boolean('use_fp16', False, 'Use half precision float16 instead of float32 as dtype')
tf.app.flags.DEFINE_string('cell_type', 'gru', 'RNN cell for encoder and decoder, default: lstm')
tf.app.flags.DEFINE_float('dropout_rate', 0.3, 'Dropout probability for input/output/state units (0.0: no dropout)')
tf.app.flags.DEFINE_boolean('use_dropout', False, 'Use dropout in each rnn cell')
tf.app.flags.DEFINE_boolean('use_residual', False, 'Use residual connection between layers')
tf.app.flags.DEFINE_string('optimizer', 'cocob', 'Optimizer for training: (adadelta, adam, rmsprop, cocob, adagrad)')
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_string('model_name', 'q2v', 'File name used for model checkpoints')
tf.app.flags.DEFINE_string("model_export_path", os.path.join(project_dir, 'data', 'export'), "model export path")

# For distributed
# cluster specification
tf.app.flags.DEFINE_string("ps_hosts", "0.0.0.0:2221",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "0.0.0.0:2222",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "single", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_boolean("is_sync", True, "whether to synchronize, aggregate gradients")

# Training parameters
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'Learning rate')
tf.app.flags.DEFINE_float('min_learning_rate', 0.0002, 'minimum Learning rate')
tf.app.flags.DEFINE_integer('decay_steps', 10000, 'how many steps to update the learning rate.')
tf.app.flags.DEFINE_float("lr_decay_factor", 0.9, "Learning rate decays by this much.")

# dummy train
tf.app.flags.DEFINE_boolean('use_dummy', False, 'dummy train for test and debug')
tf.app.flags.DEFINE_string("raw_data_path", os.path.join(project_dir, 'data/rawdata', 'dummy_train_data'), "port for data zmq stream")

FLAGS = tf.app.flags.FLAGS