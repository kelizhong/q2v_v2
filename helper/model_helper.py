import os
import logging
import tensorflow as tf

from config.config import FLAGS
from model.q2v_model import Q2VModel
from collections import OrderedDict


def create_model(session, mode='train', model_name='q2v'):
    """Create query2vec model and initialize or load parameters in session."""
    logging.info("Creating %s layers of %s units.", FLAGS.num_layers, FLAGS.hidden_units)
    config = OrderedDict(sorted(FLAGS.__flags.items()))
    model = Q2VModel(config, mode)

    ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.model_dir, model_name))
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logging.info("Reloading model parameters from %s", ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)

    else:
        assert mode == 'train', "Can not find existed model, please double check your model path"
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    return model
