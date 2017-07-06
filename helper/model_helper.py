import logbook as logging
import tensorflow as tf

from config.config import FLAGS
from model.q2v_model import Q2VModel


def create_model(session, forward_only):
    """Create query2vec model and initialize or load parameters in session."""
    logging.info("Creating {} layers of {} units.", FLAGS.num_layers, FLAGS.embedding_size)
    model = Q2VModel(FLAGS.source_max_seq_length, FLAGS.target_max_seq_length, FLAGS.max_vocabulary_size, FLAGS.src_cell_size, FLAGS.encoding_size,
                     FLAGS.num_layers, FLAGS.src_cell_size, FLAGS.tgt_cell_size,
                     FLAGS.batch_size, FLAGS.learning_rate, FLAGS.max_gradient_norm, is_sync=FLAGS.is_sync)

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        logging.info("Reading model parameters from {}", ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if forward_only:
            logging.info('Error!!!Could not load model from specified folder: {}', FLAGS.model_dir)
            exit(-1)
        else:
            logging.info("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())
    return model
