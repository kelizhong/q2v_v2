# coding=utf-8
import tensorflow as tf
from utils.config_decouple import config

tf_train = config(section='tf_train')
# Run time variables
tf.app.flags.DEFINE_string("model_dir", tf_train['model_dir'], "Trained model directory.")
tf.app.flags.DEFINE_integer("display_freq", tf_train['display_freq'], "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string("gpu", tf_train['gpu'], "specify the gpu to use")
tf.app.flags.DEFINE_string("tfrecord_train_file", tf_train['tfrecord_train_file'], "tfrecord train file")
tf.app.flags.DEFINE_boolean('debug', tf_train['debug'], 'Enable debug')
tf.app.flags.DEFINE_integer("label_size", tf_train['label_size'], "How many target samples in one example")
tf.app.flags.DEFINE_integer("batch_size", tf_train['batch_size'], "Batch size to use during training(positive pair count based).")
# Network parameters
tf.app.flags.DEFINE_integer("num_layers", tf_train['num_layers'], "Number of layers in the model.")
tf.app.flags.DEFINE_integer("embedding_size", tf_train['embedding_size'], "Size of word embedding vector.")
tf.app.flags.DEFINE_boolean('bidirectional', tf_train['bidirectional'], 'Enable bidirectional encoder')
tf.app.flags.DEFINE_integer('hidden_units', tf_train['hidden_units'], 'Number of hidden units in each layer')
tf.app.flags.DEFINE_boolean('use_fp16', tf_train['use_fp16'], 'Use half precision float16 instead of float32 as dtype')
tf.app.flags.DEFINE_string('cell_type', tf_train['cell_type'], 'RNN cell for encoder and decoder, default: lstm')
tf.app.flags.DEFINE_float('dropout_rate', tf_train['dropout_rate'], 'Dropout probability for input/output/state units (0.0: no dropout)')
tf.app.flags.DEFINE_boolean('use_dropout', tf_train['use_dropout'], 'Use dropout in each rnn cell')
tf.app.flags.DEFINE_boolean('use_residual', tf_train['use_residual'], 'Use residual connection between layers')
tf.app.flags.DEFINE_string('optimizer', tf_train['optimizer'], 'Optimizer for training: (adadelta, adam, rmsprop, cocob, adagrad)')
tf.app.flags.DEFINE_float("max_gradient_norm", tf_train['max_gradient_norm'], "Clip gradients to this norm.")
tf.app.flags.DEFINE_string('model_name', tf_train['model_name'], 'File name used for model checkpoints')
tf.app.flags.DEFINE_string("model_export_path", tf_train['model_export_path'], "model export path")
tf.app.flags.DEFINE_integer('seq_embed_size', tf_train['seq_embed_size'], 'Sequence output size')
tf.app.flags.DEFINE_float('regularized_lambda', tf_train['regularized_lambda'], 'lambda for loss regularized item')


# For distributed
# cluster specification
tf.app.flags.DEFINE_string("ps_hosts", tf_train['ps_hosts'],
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", tf_train['worker_hosts'],
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", tf_train['job_name'], "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", tf_train['task_index'], "Index of task within the job")
tf.app.flags.DEFINE_boolean("is_sync", tf_train['is_sync'], "whether to synchronize, aggregate gradients")

# Training parameters
tf.app.flags.DEFINE_float('learning_rate', tf_train['learning_rate'], 'Learning rate')
tf.app.flags.DEFINE_float('min_learning_rate', tf_train['min_learning_rate'], 'minimum Learning rate')
tf.app.flags.DEFINE_integer('decay_steps', tf_train['decay_steps'], 'how many steps to update the learning rate.')
tf.app.flags.DEFINE_float("lr_decay_factor", tf_train['lr_decay_factor'], "Learning rate decays by this much.")

# dummy train
tf.app.flags.DEFINE_boolean('use_dummy', tf_train['use_dummy'], 'dummy train for test and debug')
tf.app.flags.DEFINE_string("raw_data_path", tf_train['raw_data_path'], "port for data zmq stream")
tf.app.flags.DEFINE_string("dummy_model_dir", tf_train['dummy_model_dir'], "port for data zmq stream")
tf.app.flags.DEFINE_string("dummy_model_name", tf_train['dummy_model_name'], "port for data zmq stream")

tf.app.flags.DEFINE_boolean('export_model', tf_train['export_model'], 'export model')

FLAGS = tf.app.flags.FLAGS
