#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the TensorDetect model."""
import json
import logging
import os
import sys
# sys.path.insert(1, 'include')
import include.tensorvision.train as train
import include.tensorvision.utils as utils


# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070

# import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('name', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('project', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('hypes', 'hypes/kittiBox.json',
                    'File storing model parameters.')

flags.DEFINE_boolean('save', True,
                     ('Whether to save the run. In case --nosave (default) '
                      'output will be saved to the folder TV_DIR_RUNS/debug, '
                      'hence it will get overwritten by further runs.'))


def main(_):
    utils.set_gpus_to_use()

    try:
        import include.tensorvision.train
    except ImportError:
        logging.error("Could not import the submodules.")
        logging.error("Please execute:"
                      "'git submodule update --init --recursive'")
        exit(1)

    with open(tf.app.flags.FLAGS.hypes, 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)
    # utils.load_plugins()

    if 'TV_DIR_RUNS' in os.environ:
        os.environ['TV_DIR_RUNS'] = os.path.join(os.environ['TV_DIR_RUNS'], 'KittiBox')
    utils.set_dirs(hypes, tf.app.flags.FLAGS.hypes)
    # utils._add_paths_to_sys(hypes)
    logging.info("Initialize training folder")
    # ????????????????????????????????????????????????
    train.initialize_training_folder(hypes)
    # train.maybe_download_and_extract(hypes)
    logging.info("Start training")
    # ????????????,????????????
    train.do_training(hypes)


if __name__ == '__main__':
    tf.compat.v1.app.run()
