import matplotlib
import itertools
import os
import random
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import imageio
import tensorflow as tf
from include.utils.data_utils import (annotation_to_h5)
from include.utils.annolist import AnnotationLib as AnnoLib
import threading
from collections import namedtuple
import logging
import json


# nametuple 继承于tuple
# 定义一个nametuple类型的fake_anno_object，包含rects属性
fake_anno = namedtuple('fake_anno_object', ['rects'])
matplotlib.use('Agg')


def _noise(image):
    if np.random.uniform() < 0.5:
        return image
    scale = np.random.uniform(0, 32)
    noise = np.random.normal(-scale, scale, size=image.shape)
    image_new = np.clip(image.astype(np.float32) + 
                        noise, 0, 255).astype(np.uint8)
    return image_new


# 图像增强
def _enhance(image):
    if np.random.uniform() < 0.5:
        return image
    # 实现array到image的转换
    image_obj = Image.fromarray(image)
    # PIL image转换成array
    # img = np.asarray(image_obj)
    # 颜色增强
    image_obj = ImageEnhance.Color(image_obj).enhance(np.random.uniform(0.5, 1.5))
    # 亮度增强
    image_obj = ImageEnhance.Brightness(image_obj).enhance(np.random.uniform(0.7, 1.3))
    # 对比度增强
    image_obj = ImageEnhance.Contrast(image_obj).enhance(np.random.uniform(0.7, 1.3))
    # img = np.array(image_obj)
    img = np.asarray(image_obj)
    return img


def _projection(point, calib):
    point_r = np.reshape(point, (3, ))
    point_exp = np.reshape([point_r[0], point_r[1], point_r[2], 1], (4, 1))
    point_proj = np.dot(calib, point_exp)
    point_proj = point_proj[:2] / point_proj[2]
    return np.reshape(point_proj, (2, ))


def _vis(im_obj, anno, index):
    plt.figure(figsize=(12, 4))
    # 截取函数,限定到范围0-255
    plt.imshow(np.clip(im_obj, 0, 255).astype(np.int32))

    for r in anno.rects:
        if r.classID == -1:
            continue
        plt.plot([r.x1, r.x2, r.x2, r.x1, r.x1], 
                 [r.y1, r.y1, r.y2, r.y2, r.y1])
        bottom_proj = _projection([r.x_3d, r.y_3d, r.z_3d], r.calib)
        plt.scatter(bottom_proj[0], bottom_proj[1])

    plt.show()
    plt.savefig('/home/yappedyen/{}'.format(index))
    plt.close()
    return


# 图像做一个起伏
def _jitter(im_obj, anno, jitter_pixel=24):
    im = np.array(im_obj)
    trans = np.random.normal(scale=jitter_pixel, size=(2, ))
    height_jitter, width_jitter = np.clip(trans,
                                          a_min = -jitter_pixel * 2,
                                          a_max = +jitter_pixel * 2).astype(np.int32)
    image_jitter = np.zeros(shape=np.shape(im), dtype=np.uint8)
    image_means = im.mean(axis=(0, 1), keepdims=True).astype(np.uint8)
    image_jitter += image_means
    height, width, channels = np.shape(im)
    left_new = max(0, width_jitter)
    left_ori = max(0, -width_jitter)
    right_new = min(width + width_jitter, width)
    right_ori = min(width - width_jitter, width)
    top_new = max(0, height_jitter)
    top_ori = max(0, -height_jitter)
    bottom_new = min(height + height_jitter, height)
    bottom_ori = min(height - height_jitter, height)
    image_jitter[top_new:bottom_new, left_new:right_new] = im[top_ori:bottom_ori, left_ori:right_ori]
    new_rects = []

    for r in anno.rects:
        focal_length = r.calib.reshape(3, 4)[0, 0]
        r.x_3d += r.z_3d * width_jitter / focal_length
        r.y_3d += r.z_3d * height_jitter / focal_length
        r.x1 = max(r.x1 + width_jitter, 0)
        r.x2 = min(r.x2 + width_jitter, width)
        r.y1 = max(r.y1 + height_jitter, 0)
        r.y2 = min(r.y2 + height_jitter, height)
        if r.x1 < r.x2 and r.y1 < r.y2:
            new_rects.append(r)
    anno.rects = new_rects
    return image_jitter, anno


def _flip(im_obj, anno):
    if np.random.uniform() < 0.5:
        return im_obj, anno
    # 左右方向上翻转每行的元素，列保持不变，但是列的显示顺序变了。
    im_obj = np.fliplr(im_obj)
    height, width, channels = np.shape(im_obj)
    for r in anno.rects:
        calib = r.calib.reshape((3, 4))
        focal_length = calib[0, 0]
        ppoint_x = calib[0, 2]
        trans_x = calib[0, 3]
        delta_x = (r.z_3d*(width-1-2*ppoint_x) - 2*trans_x)/focal_length - 2*r.x_3d
        r.x_3d += delta_x
        r.x1, r.x2 = (width-1-r.x2, width-1-r.x1)
        r.alpha = np.pi - r.alpha if r.alpha > 0 else -np.pi - r.alpha 
    return im_obj, anno


def read_kitti_anno(label_file, calib_file, detect_truck):
    """ Reads a kitti annotation file.

    Args:
    label_file: Path to file

    Returns:
      Lists of rectangles: Cars and don't care area.
    """
    """
    [['Car', '0.00', '0', '1.89', '561.93', '186.85', '698.62',  '273.77', 
    '1.48', '1.51', '4.35', '0.55', '1.80', '14.99', '1.92'], 
    ['Car', '0.00', '2', '1.65', '805.83', '179.99', '856.66', '218.93', 
    '1.47', '1.68', '3.88', '9.09', '1.78', '29.59', '1.94']]
    """
    labels = [line.rstrip().split(' ') for line in open(label_file)]
    # label_file = '/home/l4v/MonoGRNet/data/KittiBox/training/label_2/007474.txt'
    label_file_split = label_file.rstrip().split('/')
    index = label_file_split[-1].split('.')[0]
    # import pdb
    # pdb.set_trace()
    calibs = [line.rstrip().split(' ') for line in open(calib_file)]
    assert calibs[2][0] == 'P2:'
    """
    [[7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01]
    [0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01]
    [0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03]]
    """
    calib = np.reshape(calibs[2][1:], (3, 4)).astype(np.float32)
    # calib_pinv calib伪逆
    calib_pinv = np.linalg.pinv(calib)
    rect_list = []
    for label in labels:
        # 只检测 Car/Van/Truck
        if not (label[0] == 'Car' or label[0] == 'Van' or
                label[0] == 'Truck' or label[0] == 'DontCare'):
            continue
        notruck = not detect_truck
        if notruck and label[0] == 'Truck':  # truck不显示
            continue
        if label[0] == 'DontCare':
            class_id = -1
        else:
            class_id = 1
        object_rect = AnnoLib.AnnoRect(
            x1=float(label[4]), y1=float(label[5]),
            x2=float(label[6]), y2=float(label[7]),
            height=float(label[8]), width=float(label[9]),
            length=float(label[10]), x=float(label[11]), 
            y=float(label[12]), z=float(label[13]), 
            alpha=float(label[14]), calib=calib, 
            calib_pinv=calib_pinv)
        assert object_rect.x1 < object_rect.x2
        assert object_rect.y1 < object_rect.y2
        object_rect.classID = class_id

        # The ground truth rotation (in camera coordinates) is converted to the local coordinates.
        view_angle = np.arctan2(object_rect.z_3d, object_rect.x_3d)
        object_rect.alpha += view_angle - np.pi * 0.5

        rect_list.append(object_rect)
    return rect_list


# 改变bbox的大小
def _rescale_boxes(current_shape, anno, target_height, target_width):
    x_scale = target_width / float(current_shape[1])
    y_scale = target_height / float(current_shape[0])
    z_3d_scale = ((x_scale**2 + y_scale**2)*0.5)**0.5
    for r in anno.rects:
        assert r.x1 < r.x2
        r.x1 *= x_scale
        r.x2 *= x_scale
        assert r.x1 < r.x2
        r.y1 *= y_scale
        r.y2 *= y_scale
        r.xy_scale = np.array([x_scale, y_scale], dtype=np.float32)
    return anno


# 生成 mask
def _generate_mask(hypes, ignore_rects):

    width = hypes["image_width"]
    height = hypes["image_height"]
    grid_width = hypes["grid_width"]
    grid_height = hypes["grid_height"]

    mask = np.ones([grid_height, grid_width])

    if not hypes['use_mask']:
        return mask

    for rect in ignore_rects:
        left = int((rect.x1+2)/width*grid_width)
        right = int((rect.x2-2)/width*grid_width)
        top = int((rect.y1+2)/height*grid_height)
        bottom = int((rect.y2-2)/height*grid_height)
        for x in range(left, right+1):
            for y in range(top, bottom+1):
                mask[y, x] = 0

    return mask


def _load_kitti_txt(kitti_txt, hypes, jitter=False, random_shuffel=True):
    """
    Take the txt file and net configuration and create a generator
    that outputs a jittered version of a random image from the annolist
    that is mean corrected.
    Args:
        kitti_txt: path_to_txt
        hypes: hypes
        jitter: Image preprocessing
        random_shuffel: Random sorting of images
    """
    # /home/l4v/MonoGRNet/data/KittiBox
    base_path = os.path.realpath(os.path.dirname(kitti_txt))
    # ['training/image_2/00445.png training/label_2/00445.txt',...,'...']
    files = [line.rstrip() for line in open(kitti_txt)]

    if hypes['data']['truncate_data']:
        files = files[:10]
        random.seed(0)
    # 无穷迭代器
    # 将数据不断读取到队列
    for epoch in itertools.count():
        if random_shuffel:
            # 将序列的所有元素随机排序
            random.shuffle(files)
        for file in files:
            image_file, gt_image_file = file.split(" ")  # image_address, label_address
            image_file_split = image_file.split('/')
            index = image_file_split[-1].split('.')[0]  # 007474 image_index
            # /home/l4v/MonoGRNet/data/KittiBox/training/calib/007474.txt
            calib_file = os.path.join(base_path, image_file_split[0], 'calib', index + '.txt')

            #  assert false  igger exception
            assert os.path.exists(calib_file), \
                "File does not exist: %s" % calib_file

            # /home/l4v/MonoGRNet/data/KittiBox/training/image_2/007474.png
            image_file = os.path.join(base_path, image_file)
            assert os.path.exists(image_file), \
                "File does not exist: %s" % image_file
            # /home/l4v/MonoGRNet/data/KittiBox/training/label_2/007474.txt
            gt_image_file = os.path.join(base_path, gt_image_file)
            assert os.path.exists(gt_image_file), \
                "File does not exist: %s" % gt_image_file
            # rect_list = [x1,y1,...]  读取label和calib
            # 每一张图片注释的矩形框列表
            rect_list = read_kitti_anno(gt_image_file, calib_file,
                                        detect_truck=hypes['detect_truck'])

            anno = AnnoLib.Annotation()
            anno.rects = rect_list

            im = imageio.imread(image_file)
            if im.shape[2] == 4:
                im = im[:, :, :3]

            if jitter:
                # 图片翻转
                im, anno = _flip(im, anno)
                # 图片做一个起伏
                im, anno = _jitter(im, anno)
                # 图片增强并加噪
                im = _noise(_enhance(im))

            # _vis(im, anno, index)

            # 变换图像尺寸
            anno = _rescale_boxes(im.shape, anno, hypes["image_height"], hypes["image_width"])
            im = Image.fromarray(im).resize(size=(hypes["image_width"], hypes["image_height"]))

            # 只关注classid=1的注释
            pos_list = [rect for rect in anno.rects if rect.classID == 1]
            # 存入nametuple
            pos_anno = fake_anno(pos_list)
            # boxes: [1, grid_height*grid_width, 11, max_len, 1]
            # for each cell, this array contains the ground truth boxes around it (within focus area,
            # defined by center distance)
            # confs: [1, grid_height*grid_width, 1, max_len, 1]
            # record the valid boxes, since max_len is greater than the number of ground truth boxes
            boxes, confs, calib, calib_pinv, xy_scale = annotation_to_h5(hypes, pos_anno, hypes["grid_width"],
                                                                         hypes["grid_height"], hypes["rnn_len"])
            # masks are zero in "Don't care" area
            mask_list = [rect for rect in anno.rects if rect.classID == -1]
            mask = _generate_mask(hypes, mask_list)

            boxes = boxes.reshape([hypes["grid_height"], hypes["grid_width"], 11])
            confs = confs.reshape(hypes["grid_height"], hypes["grid_width"])
            calib = calib.reshape(hypes["grid_height"], hypes["grid_width"], 3, 4)
            xy_scale = xy_scale.reshape(hypes["grid_height"], hypes["grid_width"], 2)
            calib_pinv = calib_pinv.reshape(hypes['grid_height'], hypes['grid_width'], 4, 3)
            # yield 生成器类似于return，但是迭代一次遇到yield时就返回yield后面(右边)的值。
            # 重点是：下一次迭代时，从上一次迭代遇到的yield后面的代码(下一行)开始执行。
            # 第一次调用时必须先next()或send(None)，否则会报错。
            # 通过重复调用next()方法，直到捕获一个异常。或者通过迭代（for）从上次位置开始
            yield {"image": im, "boxes": boxes, "confs": confs, "calib": calib, "calib_pinv": calib_pinv,
                   "xy_scale": xy_scale, "rects": pos_list, "mask": mask}


def _make_sparse(n, d):
    v = np.zeros((d,), dtype=np.float32)
    v[n] = 1.
    return v


def create_queues(hypes, phase):
    """Create Queues."""

    dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
    grid_size = hypes['grid_width'] * hypes['grid_height']
    shapes = ([hypes['image_height'], hypes['image_width'], 3],  # image
              [hypes['grid_height'], hypes['grid_width']],       # confs
              [hypes['grid_height'], hypes['grid_width'], 11],  # boxes的11 个参数
              [hypes['grid_height'], hypes['grid_width']],      # mask
              [hypes['grid_height'], hypes['grid_width'], 3, 4],  # calib
              [hypes['grid_height'], hypes['grid_width'], 4, 3],  # calib_pinv
              [hypes['grid_height'], hypes['grid_width'], 2])     # x_scale y_scale 缩放比例
    capacity = 32
    # 创建一个队列存放数据
    q = tf.queue.FIFOQueue(capacity=capacity, dtypes=dtypes, shapes=shapes)
    return q


def _processe_image(hypes, image):
    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    augment_level = hypes['augment_level']
    if augment_level > 0:
        image = tf.image.random_brightness(image, max_delta=30)
        image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    if augment_level > 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.6)
        image = tf.image.random_hue(image, max_delta=0.15)
    # 返回最小值
    image = tf.minimum(image, 255.0)
    image = tf.maximum(image, 0)

    return image


def start_enqueuing_threads(hypes, q, phase, sess):
    """Start enqueuing threads."""

    # Creating Placeholder for the Queue
    # 占位符
    x_in = tf.placeholder(tf.float32)
    confs_in = tf.placeholder(tf.float32)
    boxes_in = tf.placeholder(tf.float32)
    mask_in = tf.placeholder(tf.float32)
    calib_in = tf.placeholder(tf.float32)
    calib_pinv_in = tf.placeholder(tf.float32)
    xy_scale_in = tf.placeholder(tf.float32)

    # Creating Enqueue OP
    # 入列元素
    enqueue_op = q.enqueue((x_in, confs_in, boxes_in, mask_in, calib_in, calib_pinv_in, xy_scale_in))

    def make_feed(data):
        return {x_in: data['image'],
                confs_in: data['confs'],
                boxes_in: data['boxes'],
                mask_in: data['mask'], 
                calib_in: data['calib'], 
                calib_pinv_in: data['calib_pinv'], 
                xy_scale_in: data['xy_scale']}

    def thread_loop(sess, enqueue_op, gen):
        # 参数 feed_dict给使用placeholder创建出来的tensor赋值, 也允许调用者替换图中张量的值
        # 通过迭代来给队列喂数据
        for d in gen:
            sess.run(enqueue_op, feed_dict=make_feed(d))

    data_file = hypes["data"]['%s_file' % phase]  # train_file
    data_dir = hypes['dirs']['data_dir']
    data_file = os.path.join(data_dir, data_file)  # ../data/KittiBox/train.txt

    # 返回的是生成器
    gen = _load_kitti_txt(data_file, hypes, jitter={'train': hypes['solver']['use_jitter'], 'val': False}[phase])
    # 第一次调用gen用next(),yield的数据返回给data
    data = next(gen)
    sess.run(enqueue_op, feed_dict=make_feed(data))
    # 多线程
    # 这个子线程用来装载数据
    t = threading.Thread(target=thread_loop, args=(sess, enqueue_op, gen))
    # daemon守护线程，只有这个线程时，线程结束
    t.daemon = True
    # 线程启动
    t.start()


def inputs(hypes, q, phase):

    if phase == 'val':
        image, confidences, boxes, mask, calib, calib_pinv, xy_scale = q.dequeue()
        image = tf.expand_dims(image, 0)
        confidences = tf.expand_dims(confidences, 0)
        boxes = tf.expand_dims(boxes, 0)
        mask = tf.expand_dims(mask, 0)
        calib = tf.expand_dims(calib, 0)
        calib_pinv = tf.expand_dims(calib_pinv, 0)
        xy_scale = tf.expand_dims(xy_scale, 0)
        return image, (confidences, boxes, mask, calib, calib_pinv, xy_scale)
    elif phase == 'train':
        # dequeue_many(n)将n个元素连接到一起移出队列
        # image=Tensor(shape=(8,384,1248,3))
        # confidences=Tensor(shape=(8,12,39))
        # boxes=Tensor(shape=(8,12,39,11))
        # mask=Tensor(shape=(8,12,39))
        # calib=Tensor(shape=(8,12,39,3,4))
        # calib_pinv=Tensor(shape=(8,12,39,4,3))
        # xy_scale=Tensor(shape=(8,12,39,2))
        image, confidences, boxes, mask, calib, calib_pinv, xy_scale = q.dequeue_many(hypes['batch_size'])
        image = _processe_image(hypes, image)
        return image, (confidences, boxes, mask, calib, calib_pinv, xy_scale)
    else:
        assert("Bad phase: {}".format(phase))


if __name__ == '__main__':
    with open("../hypes/kittiBox.json", 'r') as f:
        logging.info("f: %s", f)
        hype = json.load(f)

    data_file1 = hype["data"]['%s_file' % 'train']  # train_file
    data_dir1 = hype['dirs']['data_dir']
    data_file1 = os.path.join(data_dir1, data_file1)

    gen1 = _load_kitti_txt(data_file1, hype, jitter=False)
    data1 = next(gen1)