# coding=utf-8
"""model util.e.g.load mode, save model"""
import os
import logbook as logging
import mxnet as mx
from utils.file_util import ensure_dir_exists


def load_model(model_prefix, rank=0, load_epoch=None):
    """load existed model
    Parameters
    ----------
        model_prefix: str
            the prefix for the parameters file
        rank: int
            the rank of worker node
        load_epoch: int
            Epoch number of model we would like to load.
        Returns
        -------
        symbol : Symbol
            The symbol configuration of computation network.
        arg_params : dict of str to NDArray
            Model parameter, dict of name to NDArray of net's weights.
        aux_params : dict of str to NDArray
            Model parameter, dict of name to NDArray of net's auxiliary states.

    Notes
    -----
    - if load_epoch is None or value < 0, will return None, None, None
    - symbol will be loaded from ``prefix-symbol.json``.
    - parameters will be loaded from ``prefix-epoch.params``.
    """
    if load_epoch is None or load_epoch < 0:
        return None, None, None
    assert model_prefix is not None
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, load_epoch)
    logging.info('Loaded query2vec %s_%04d.params' % (model_prefix, load_epoch))
    return sym, arg_params, aux_params


def save_model_callback(model_prefix, rank=0, period=1):
    """Callback to checkpoint the model to prefix every period.

    Parameters
        ----------
        model_prefix : str
            The file prefix to checkpoint to
        rank: int
            the rank of worker node
        period : int
            How many epochs to wait before checkpointing. Default is 1.

    Returns
    -------
        callback : function
            The callback function that can be passed as iter_end_callback to fit.
    Notes
    -----
    - model_prefix is not a directory name
    """
    if model_prefix is None:
        return None
    ensure_dir_exists(model_prefix, is_dir=False)
    return mx.callback.do_checkpoint(model_prefix if rank == 0 else "%s-%d" % (
        model_prefix, rank), period)
