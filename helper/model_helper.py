import os
import logging
import shutil
import tensorflow as tf

from config.config import FLAGS
from model.q2v_model import Q2VModel


def create_model(session, config, model_dir, mode='train'):
    """Create query2vec model and initialize or load parameters in session."""
    logging.info("Creating %s layers of %s units.", config['num_layers'], config['hidden_units'])

    model = Q2VModel(config, mode)

    ckpt = tf.train.get_checkpoint_state(os.path.join(model_dir))
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logging.info("Reloading model parameters from %s", ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        assert mode == 'train', "Can not find existed model, please double check your model path"
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    return model


def export_model(sess, config, export_dir, model_dir):
    model = create_model(sess, config, mode='encode', model_dir=model_dir)
    output_path = os.path.join(
        tf.compat.as_bytes(export_dir),
        tf.compat.as_bytes(str(model.global_step.eval())))
    if os.path.exists(output_path):
        logging.info("Removing duplicate: %s" % output_path)
        shutil.rmtree(output_path)

    model.export(sess, output_path)

