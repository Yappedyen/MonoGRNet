#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core functions of TV."""

import logging

import os
import numpy as np
import tensorflow as tf

# import utils as utils

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    'summary', True, 'Whether or not to save summaries to tensorboard.')


def load_weights(checkpoint_dir, sess, saver):
    """
    Load the weights of a model stored in saver.

    Parameters
    ----------
    checkpoint_dir : str
        The directory of checkpoints.
    sess : tf.Session
        A Session to use to restore the parameters.
    saver : tf.train.Saver

    Returns
    -----------
    int
        training step of checkpoint
    """
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        logging.info(ckpt.model_checkpoint_path)
        file = os.path.basename(ckpt.model_checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_dir, file)
        saver.restore(sess, checkpoint_path)
        return int(file.split('-')[1])


def build_training_graph(hypes, queue, modules):
    """
    Build the tensorflow graph out of the model files.

    Parameters
    ----------
    hypes : dict
        Hyper_parameters
    queue: tf.queue
        Data Queue
    modules : tuple
        The modules load in utils.

    Returns
    -------
    tuple
        (q, train_op, loss, eval_lists) where
        q is a dict with keys 'train' and 'val' which includes queues,
        train_op is a tensorflow op,
        loss is a float,
        eval_lists is a dict with keys 'train' and 'val'
    """

    data_input = modules['input']
    encoder = modules['arch']
    objective = modules['objective']
    optimizer = modules['solver']

    learning_rate = tf.compat.v1.placeholder(tf.float32)

    # Add Input Producers to the Graph
    with tf.name_scope("Inputs"):
        # kitti_input.py中的inputs函数
        image, labels = data_input.inputs(hypes, queue, phase='train')

    # backbone 已装载权重
    # Run inference on the encoder network
    # 全连接层的输出，logits就是一个向量，下一步将被投给 softmax 的向量
    # 返回的是early和deep层特征,已经装载了预训练的权重
    logits = encoder.inference(hypes, image, train=True)
    
    # Build decoder on top of the logits
    decoded_logits = objective.decoder(hypes, logits, labels, train=True)

    # Add to the Graph the Ops for loss calculation.
    with tf.name_scope("Loss"):
        losses = objective.loss(hypes, decoded_logits, labels)

    # Add to the Graph the Ops that calculate and apply gradients.
    with tf.name_scope("Optimizer"):
        # 不可训练，初始化为0
        global_step = tf.Variable(0, trainable=False)
        # Build training operation
        train_op = optimizer.training(hypes, losses, global_step, learning_rate)

    with tf.name_scope("Evaluation"):

        # Add the Op to compare the logits to the labels during evaluation.
        eval_list = objective.evaluation(hypes, image, labels, decoded_logits, losses, global_step)

        summary_op = tf.compat.v1.summary.merge_all()

    graph = {
        'losses': losses,
        'eval_list': eval_list,
        'summary_op': summary_op,
        'train_op': train_op,
        'global_step': global_step,
        'learning_rate': learning_rate,
        'decoded_logits': decoded_logits
    }

    return graph


def build_inference_graph(hypes, modules, image, calib_pl, xy_scale_pl):
    """Run one evaluation against the full epoch of data.

    Parameters
    ----------
    hypes : dict
        Hyper——parameters
    modules : tuple
        the modules load in utils
    image : placeholder
    calib_pl:
        calib
    xy_scale_pl:
        xy_scale
    --------
    Returns
    -------
        graph_ops

    """
    with tf.name_scope("Validation"):
        logits = modules['arch'].inference(hypes, image, train=False)
        labels = (0, 0, 0, calib_pl, 0, xy_scale_pl)
        decoded_logits = modules['objective'].decoder(hypes, logits, labels, train=False)

    return decoded_logits


def start_tv_session(hypes):
    """
    Run one evaluation against the full epoch of data.

    Parameters
    ----------
    hypes : dict
        Hyper_parameters

    Returns
    -------
    tuple
        (sess, saver, summary_op, summary_writer, threads)
    """
    # Build the summary operation based on the TF collection of Summaries.
    if FLAGS.summary:
        tf.contrib.layers.summarize_collection(tf.GraphKeys.WEIGHTS)
        tf.contrib.layers.summarize_collection(tf.GraphKeys.BIASES)
        summary_op = tf.summary.merge_all()
    else:
        summary_op = None

    # Create a saver for writing training checkpoints.
    # if 'keep_checkpoint_every_n_hours' in hypes['solver']:
    #     kc = hypes['solver']['keep_checkpoint_every_n_hours']
    # else:
    #     kc = 10000.0

    # 保存和加载模型实例化一个 tf.train.Saver
    # 调用一次保存操作会创建后3个数据文件并创建一个检查点（checkpoint）文件，简单理解就是权重等参数被保存到 .ckpt.data 文件中，以字典的形式；
    # 图和元数据被保存到 .ckpt.meta 文件中，可以被 tf.train.import_meta_graph 加载到当前默认的图。
    #  max_to_keep 参数表示要保留的最近检查点文件的最大数量，创建新文件时，将删除旧文件，默认为 5
    saver = tf.compat.v1.train.Saver(max_to_keep=int(hypes['logging']['max_to_keep']))
    # 主线程
    sess = tf.get_default_session()

    # Run the Op to initialize the variables.
    if 'init_function' in hypes:
        _initalize_variables = hypes['init_function']
        _initalize_variables(hypes)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    # Start the queue runners.
    # 创建一个线程管理器（协调器）对象。用来管理之后在Session中启动的所有线程
    coord = tf.train.Coordinator()
    # 启动入队线程，由多个或单个线程，按照设定规则，把文件读入Filename Queue中。
    # 只有调用 tf.train.start_queue_runners 之后，才会真正把tensor推入内存序列中，供计算单元调用，否则会由于内存序列为空，数据流图会处于一直等待状态
    # 启动执行文件名队列填充的线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(hypes['dirs']['output_dir'], graph=sess.graph)

    tv_session = {
        'sess': sess,
        'saver': saver,
        'summary_op': summary_op,
        'writer': summary_writer,
        'coord': coord,
        'threads': threads
    }

    return tv_session
