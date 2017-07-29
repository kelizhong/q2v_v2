# coding=utf-8
import logging
import pickle
import os
import json
from utils.file_util import ensure_dir_exists


def save_obj_pickle(obj, filename, overwrite=False):
    """save the object into pickle
    Parameters
    ----------
        obj: python object, e.g. dict, list, set
            object we would like to save into pickle format.
        filename: str:
            pickle file save name
        overwrite: bool
            whether overwrite the existed file
    """
    ensure_dir_exists(filename, is_dir=False)
    if os.path.isfile(filename) and not overwrite:
        logging.warning("Not saving %s, already exists.", filename)
    else:
        if os.path.isfile(filename):
            logging.info("Overwriting %s.", filename)
        else:
            logging.info("Saving to %s.", filename)
        with open(filename, 'w', encoding="utf8") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle_object(path):
    """
    Load object from pickle file

    Parameters
    ----------
        path: the pickle path

    Returns
    -------
        object
    """
    with open(path, 'r', encoding="utf8") as f:
        obj = pickle.load(f)
    return obj


def save_obj_json(obj, filename, overwrite=False):
    """save the object into pickle
    Parameters
    ----------
        obj: python object, e.g. dict, list, set
            object we would like to save into pickle format.
        filename: str:
            pickle file save name
        overwrite: bool
            whether overwrite the existed file
    """
    ensure_dir_exists(filename, is_dir=False)
    if os.path.isfile(filename) and not overwrite:
        logging.warning("Not saving %s, already exists.", filename)
    else:
        if os.path.isfile(filename):
            logging.info("Overwriting %s.", filename)
        else:
            logging.info("Saving to %s.", filename)
        with open(filename, 'w', encoding="utf8") as f:
            json.dump(obj, f)


def load_json_object(path):
    """
    Load object from json file

    Parameters
    ----------
        path: the json path

    Returns
    -------
        object
    """
    with open(path, 'r', encoding="utf8") as f:
        obj = json.load(f)
    return obj