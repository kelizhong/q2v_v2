# coding=utf-8
import os
import tensorflow as tf

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
logging_config_path = os.path.join(project_dir, 'config', 'logging.yaml')

# Extra vocabulary symbols
_GO = '_GO_'
_EOS = '_EOS_' # also function as PAD
_UNK = '_UNK_'

extra_tokens = [_GO, _EOS, _UNK]

start_token = extra_tokens.index(_GO)  # start_token = 0
end_token = extra_tokens.index(_EOS)	 # end_token = 1
unk_token = extra_tokens.index(_UNK)	 # unknown_token = 2

special_words = {_GO: start_token, _EOS: end_token, _UNK: unk_token}

vocabulary_size = 132922 + len(special_words)

# Run time variables
tf.app.flags.DEFINE_string("model_dir", os.path.join(project_dir, 'data/models'), "Trained model directory.")
tf.app.flags.DEFINE_integer("display_freq", 1,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string("gpu", None, "specify the gpu to use")
tf.app.flags.DEFINE_string("log_file_name", os.path.join(project_dir, 'data/logs', 'q2v.log'), "Log data file name")
tf.app.flags.DEFINE_integer("data_stream_port", None, "port for data zmq stream")
tf.app.flags.DEFINE_boolean('debug', False, 'Enable debug')

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
tf.app.flags.DEFINE_integer('max_vocabulary_size', vocabulary_size, 'Source vocabulary size')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop, cocob, adagrad)')
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("source_maxlen", 60, "max number of words in each source sequence.")
tf.app.flags.DEFINE_integer("target_maxlen", 60, "max number of words in each target sequence.")
tf.app.flags.DEFINE_string('model_name', 'q2v', 'File name used for model checkpoints')

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
tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')
tf.app.flags.DEFINE_float('min_learning_rate', 0.0002, 'minimum Learning rate')
tf.app.flags.DEFINE_integer('decay_steps', 1000, 'how many steps to update the learning rate.')
tf.app.flags.DEFINE_float("lr_decay_factor", 0.9, "Learning rate decays by this much.")

# local data stream
tf.app.flags.DEFINE_string("raw_data_path", os.path.join(project_dir, 'data/rawdata', 'person_name_labels'), "port for data zmq stream")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "Batch size to use during training(positive pair count based).")
tf.app.flags.DEFINE_string("vocabulary_data_dir", os.path.join(project_dir, 'data/vocabulary'), "port for data zmq stream")
tf.app.flags.DEFINE_string("words_list_path", os.path.join(project_dir, 'data/vocabulary/words_list'), "custom words list to build vocabulary")

FLAGS = tf.app.flags.FLAGS
ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")