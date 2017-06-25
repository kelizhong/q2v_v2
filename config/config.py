import os
import tensorflow as tf

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

"""constant value"""
bos_word = '<s>'  # begin of sentence
eos_word = '</s>'  # end of sentence
unk_word = '<unk>'  # unknown word
pad_word = '<pad>'  # pad word

# Special vocabulary symbols - we always put them at the start.
special_words = {pad_word: 0, unk_word: 1, bos_word: 2, eos_word: 3}
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
tf.app.flags.DEFINE_integer("num_layers", 4, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("max_vocabulary_size", 64001, "Sequence vocabulary size in the mapping task.")

tf.app.flags.DEFINE_integer("source_max_seq_length", 50, "max number of words in each source or target sequence.")
tf.app.flags.DEFINE_integer("target_max_seq_length", 150, "max number of words in each source or target sequence.")
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

tf.app.flags.DEFINE_string("gpu", None, "specify the gpu to use")

tf.app.flags.DEFINE_string("log_file_name", os.path.join(project_dir, 'data/logs', 'q2v.log'), "Log data file name")
tf.app.flags.DEFINE_integer("data_stream_port", 5558, "port for data zmq stream")
tf.app.flags.DEFINE_string("raw_data_path", './data/rawdata/test.add', "port for data zmq stream")
tf.app.flags.DEFINE_string("vocabulary_data_dir", './data/vocabulary', "port for data zmq stream")

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