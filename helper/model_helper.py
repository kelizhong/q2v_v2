import os
import logging
import shutil
import tensorflow as tf

from model.q2v_model import Q2VModel


def create_model(session, config, model_dir=None, mode='train'):
    """Create query2vec model and initialize or load parameters in session.


    Parameters
    ----------
    session : {session}
        tensoflow session
    config : {dict}
        configuration dictionary
    model_dir : {str}
        path for model file
    mode : {str}, optional
        mode: train, encode. (the default is 'train')

    Returns
    -------
    [Object]
        Tensorflow model object
    """
    logging.info("Creating %s layers of %s units.", config['num_layers'], config['hidden_units'])

    model = Q2VModel(config, mode)
    model_dir = model_dir or config['model_dir']
    ckpt = tf.train.get_checkpoint_state(os.path.join(model_dir))
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logging.info("Reloading model parameters from %s", ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        assert mode == 'train', "Can not find existed model, please double check your model path"
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    return model


def export_model(sess, config, model_export_path=None, model_dir=None):
    """Export model for tensorflow serving

    Parameters
    ----------
    sess : {Session}
        Tensorflow session
    config : {dict}
        configuration dictionary
    model_dir : {str}
        path for model file
    model_export_path : {str}, optional
        path for exported model (the default is None)
    """
    model_export_path = model_export_path or config['model_export_path']
    model_dir = model_dir or config['model_dir']

    model = create_model(sess, config, mode='encode', model_dir=model_dir)
    output_path = os.path.join(
        tf.compat.as_bytes(model_export_path),
        tf.compat.as_bytes(str(model.global_step.eval())))
    if os.path.exists(output_path):
        logging.info("Removing duplicate: %s" % output_path)
        shutil.rmtree(output_path)
    # export model
    model.export(sess, output_path)
