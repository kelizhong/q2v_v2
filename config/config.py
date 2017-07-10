import os
import tensorflow as tf

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

# Extra vocabulary symbols
_GO = '_GO'
EOS = '_EOS' # also function as PAD
UNK = '_UNK'

extra_tokens = [_GO, EOS, UNK]

start_token = extra_tokens.index(_GO)	# start_token = 0
end_token = extra_tokens.index(EOS)	# end_token = 1
unk_token = extra_tokens.index(UNK)

special_words = {_GO: start_token, EOS: end_token, UNK: unk_token}
# Run time variables

tf.app.flags.DEFINE_float("learning_rate", 0.3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "Batch size to use during training(positive pair count based).")
tf.app.flags.DEFINE_integer("embedding_size", 128, "Size of word embedding vector.")
tf.app.flags.DEFINE_integer("encoding_size", 80,
                            "Size of sequence encoding vector. Same number of nodes for each model layer.")
tf.app.flags.DEFINE_integer("src_cell_size", 96, "LSTM cell size in source RNN model.")
tf.app.flags.DEFINE_integer("tgt_cell_size", 96,
                            "LSTM cell size in target RNN model. Same number of nodes for each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")

tf.app.flags.DEFINE_integer("source_max_seq_length", 30, "max number of words in each source or target sequence.")
tf.app.flags.DEFINE_integer("target_max_seq_length", 100, "max number of words in each source or target sequence.")
tf.app.flags.DEFINE_integer("max_epoch", 8, "max epoc number for training procedure.")
tf.app.flags.DEFINE_integer("predict_nbest", 20, "max top N for evaluation prediction.")

tf.app.flags.DEFINE_string("data_dir", 'data', "Data directory")
tf.app.flags.DEFINE_string("train_data_file", 'data/rawdata/TrainPairs', "Train Data file")
tf.app.flags.DEFINE_string("model_dir", os.path.join(project_dir, 'data/models'), "Trained model directory.")
tf.app.flags.DEFINE_string("export_dir", 'exports', "Trained model directory.")
tf.app.flags.DEFINE_string("device", "gpu:0",
                           "Default to use GPU:0. Softplacement used, if no GPU found, further default to cpu:0.")

tf.app.flags.DEFINE_integer("display_freq", 1,
                            "How many training steps to do per checkpoint.")

tf.app.flags.DEFINE_string("gpu", None, "specify the gpu to use")

tf.app.flags.DEFINE_string("log_file_name", os.path.join(project_dir, 'data/logs', 'q2v.log'), "Log data file name")
tf.app.flags.DEFINE_integer("data_stream_port", None, "port for data zmq stream")
tf.app.flags.DEFINE_string("raw_data_path", os.path.join(project_dir, 'data/rawdata', 'test.add'), "port for data zmq stream")
tf.app.flags.DEFINE_string("vocabulary_data_dir", os.path.join(project_dir, 'data/vocabulary'), "port for data zmq stream")


tf.app.flags.DEFINE_integer('hidden_units', 256, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_boolean('use_fp16', False, 'Use half precision float16 instead of float32 as dtype')
tf.app.flags.DEFINE_string('cell_type', 'lstm', 'RNN cell for encoder and decoder, default: lstm')
tf.app.flags.DEFINE_float('dropout_rate', 0.3, 'Dropout probability for input/output/state units (0.0: no dropout)')
tf.app.flags.DEFINE_boolean('use_dropout', True, 'Use dropout in each rnn cell')
tf.app.flags.DEFINE_boolean('use_residual', True, 'Use residual connection between layers')
tf.app.flags.DEFINE_integer('max_vocabulary_size', 64005, 'Source vocabulary size')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop, cocob)')
tf.app.flags.DEFINE_integer("source_maxlen", 30, "max number of words in each source sequence.")
tf.app.flags.DEFINE_integer("target_maxlen", 100, "max number of words in each target sequence.")

# For distributed
# cluster specification
tf.app.flags.DEFINE_string("ps_hosts", "0.0.0.0:2221",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "0.0.0.0:2222",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "single", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_boolean("is_sync", False, "")

FLAGS = tf.app.flags.FLAGS
ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")