import os
import logging
from math import ceil
import pickle
import numpy as np
import tensorflow as tf
import sys

# defaultencoding = 'utf-8'
# if sys.getdefaultencoding() != defaultencoding:
#     reload(sys)
#     sys.setdefaultencoding(defaultencoding)

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

VGG_MEAN = [103.939, 116.779, 123.68]


class FCN8VGG:

    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = sys.modules[self.__class__.__module__].__file__
            # print path
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print path
            path = os.path.join(path, "model_2D.pkl")
            vgg16_npy_path = path
            logging.info("Load npy file from '%s'.", vgg16_npy_path)
        if not os.path.isfile(vgg16_npy_path):
            logging.error(("File {} not found".format(vgg16_npy_path)))
            sys.exit(1)
        # read binary file
        with open(vgg16_npy_path, 'rb') as file:
            self.data_dict = pickle.load(file, encoding='latin1')  # encoding='latin1'
            file.close()
        self.wd = 1e-5
        print("pkl file loaded")

    def build(self, rgb, train=False, num_classes=20, random_init_fc8=False, debug=False):
        """
        Build the VGG model using loaded weights

        Parameters
        ----------
        rgb: image batch tensor
            Image in rgb shap. Scaled to Intervall [0, 255]
        train: bool
            Whether to build train or inference graph
        num_classes: int
            How many classes should be predicted (by fc8)
        random_init_fc8 : bool
            Whether to initialize fc8 layer randomly.
            Finetuning is required in this case.
        debug: bool
            Whether to print additional Debug Information.
        """

        # Convert RGB to BGR
        self.train = train
        # 命名空间Processing
        with tf.name_scope('Processing'):
            # red:Tensor(shape=(batch_size,384,1248,1))
            # green:Tensor(shape=(batch_size,384,1248,1))
            # blue:Tensor(shape=(batch_size,384,1248,1))
            # split()切割tensor,tf.split(value,num_or_size_splits,axis=0) value是要切的Tensor，num_or_size_splits切成小张量数量
            # axis代表切割哪个维度
            # 在第三个维度上把rgb切割成3个tensor,分别为r,g,b
            red, green, blue = tf.split(rgb, 3, 3)
            # assert red.get_shape().as_list()[1:] == [224, 224, 1]
            # assert green.get_shape().as_list()[1:] == [224, 224, 1]
            # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            # 拼接tensor
            # tensor(shape=(batch_size,384,1248,3))
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], 3)

            if debug:
                bgr = tf.print(bgr, [tf.shape(bgr)],
                               message='Shape of input image: ',
                               summarize=4, first_n=1)
        # conv1_1/filter (3,3,3,64)
        self.conv1_1 = self._conv_layer(bgr, "conv1_1")
        # conv1_2/filter (3,3,64,64)
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
        # pool1 (batch_size,192,624,64)
        self.pool1 = self._max_pool(self.conv1_2, 'pool1', debug)
        # conv2_1/filter (3,3,64,128)
        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
        # conv2_2/filter (3,3,128,128)
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
        # pool2 (batch_size,96,312,128)
        self.pool2 = self._max_pool(self.conv2_2, 'pool2', debug)
        # conv3_1/filter (3,3,128,256)
        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
        # conv3_2/filter (3,3,256,256)
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
        # conv3_3/filter (3,3,256,256)
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
        # pool3 (batch_size,48,156,256)
        self.pool3 = self._max_pool(self.conv3_3, 'pool3', debug)
        # conv4_1/filter (3,3,256,512)
        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
        # conv4_2/filter (3,3,512,512)
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
        # conv4_3/filter (3,3,512,512)
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
        # pool4 (batch_size,24,78,512)
        self.pool4 = self._max_pool(self.conv4_3, 'pool4', debug)

        # conv4_3 feature_map(batch_size,48,156,512) 经3次pool所得
        # 从conv4_3的特征图上做多任务 depth、location、corner
        # conv4_depth(batch_size,48,156,128)
        # conv4_location(batch_size,48,156,128)
        # conv4_corners(batch_size,48,156,128)
        self.conv4_depth, self.conv4_location, self.conv4_corner = self._image_encoder(
            self.conv4_3, 128, (6, 12, 18), 'conv4_encoder')

        # conv5_1/filter (3,3,512,512)
        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
        # conv5_2/filter (3,3,512,512)
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
        # conv5_3/filter (3,3,512,512)
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
        # pool5 (batch_size,12,39,512)
        self.pool5 = self._max_pool(self.conv5_3, 'pool5', debug)
        # 从pool5 的特征图上做多任务 depth、location、corner
        # pool5_depth(batch_size,12,39,128)
        # pool5_location(batch_size,12,39,128)
        # pool5_corners(batch_size,12,39,128)
        self.pool5_depth, self.pool5_location, self.pool5_corner = self._image_encoder(
            self.pool5, 128, (2, 4, 8), 'pool5_encoder')

        self.fc8 = None

    def _inverted_residual(self, bottom, num_blocks, out_channels, name, branch):
        batch, height, width, in_channels = bottom.get_shape().as_list()
        x = self._conv_layer_new(bottom, name + '_start', 3, 1, (in_channels, 64), branch, False, True)
        for loop in range(num_blocks):
            inputs = x
            up = self._conv_layer_new(inputs, name + '_conv_{}_up'.format(loop+1), 3, 1, (64, 256), branch, True, True)
            down = self._conv_layer_new(up, name + '_conv_{}_down'.format(loop+1), 3, 1, (256, 64), branch, True, True)
            x = inputs + down
        outputs = self._conv_layer_new(x, name + '_end', 3, 1, (64, out_channels), branch, True, True)
        return outputs

    def _image_encoder(self, bottom, out_channels, rates, name):
        # conv4_3 bottom:(batch_size,48,156,512)
        # batch:batch_size height:48 weight:156 in_channels=512
        # pool5 (batch_size,12,39,512)
        batch, height, width, in_channels = bottom.get_shape().as_list()
        # depth_branch
        # conv4_encoder_full_neck/filter (3,3,512,256) full_neck = (batch_size,48,156,256)
        # pool5_encoder_full_neck/filter (3,3,512,256) full_neck = (batch_size,12,39,256)
        full_neck = self._conv_layer_new(bottom, name + '_full_neck', 3, 1, (in_channels, 256), 'depth')
        # conv4_encoder_full/filter (3,3,256,128) full = (batch_size,48,156,128)
        # pool5_encoder_full/filter (3,3,256,128) full = (batch_size,12,39,128)
        full = self._conv_layer_new(full_neck, name + '_full', 3, 1, (256, 128), 'depth')
        # 按照不同的步长做卷积得到atrous conv4_3_step(6,12,18)  pool5_step (2,4,8)
        # conv4_encoder_atrous_1/filter (3,3,256,128) atrous_1=(batch_size,48,156,128)
        # pool5_encoder_atrous_1/filter (3,3,256,128) atrous_1=(batch_size,12,39,128)
        atrous_1 = self._atrous_conv_layer_new(full_neck, name + '_atrous_1', 3, rates[0], (256, 128), 'depth')
        # conv4_encoder_atrous_2/filter (3,3,256,128) atrous_2=(batch_size,48,156,128)
        # pool5_encoder_atrous_2/filter (3,3,256,128) atrous_2=(batch_size,12,39,128)
        atrous_2 = self._atrous_conv_layer_new(full_neck, name + '_atrous_2', 3, rates[1], (256, 128), 'depth')
        # conv4_encoder_atrous_3/filter (3,3,256,128) atrous_3=(batch_size,48,156,128)
        # pool5_encoder_atrous_3/filter (3,3,256,128) atrous_3=(batch_size,12,39,128)
        atrous_3 = self._atrous_conv_layer_new(full_neck, name + '_atrous_3', 3, rates[2], (256, 128), 'depth')
        # 拼接tensor，full、atrous_1、atrous_2、atrous_3
        # conv4_3 neck (batch_size,48,156,512)
        # pool5 neck (batch_size,12,39,512)
        neck = tf.concat([full, atrous_1, atrous_2, atrous_3], axis=3)
        # conv4_depth_head (batch_size,48,156,128)
        # pool5_depth_head (batch_size,12,39,128)
        depth_head = self._conv_layer_new(neck, name + '_conv_neck', 1, 1, (512, out_channels), 'depth')
        # location_branch
        # conv4_location_neck (batch_size,48,156,256)
        # pool5_location_neck (batch_size,12,39,256)
        location_neck = self._conv_layer_new(bottom, name + '_location_neck', 3, 1, (in_channels, 256), 'location')
        # conv4_location_head (batch_size,48,156,128)
        # pool5_location_head (batch_size,12,39,128)
        location_head = self._conv_layer_new(location_neck, name + '_location_head', 3, 1, (256, out_channels),
                                             'location')
        # corners_branch
        # conv4_corners_neck (batch_size,48,156,256)
        # pool5_corners_neck (batch_size,12,39,256)
        corners_neck = self._conv_layer_new(bottom, name + '_corners_neck', 3, 1, (in_channels, 256), 'corners')
        # conv4_corners_head (batch_size,48,156,128)
        # pool5_corners_head (batch_size,12,39,128)
        corners_head = self._conv_layer_new(corners_neck, name + '_corners_head', 3, 1, (256, out_channels), 'corners')
        return depth_head, location_head, corners_head

    def _max_pool(self, bottom, name, debug):
        # ksize 池化窗口大小
        pool = tf.nn.max_pool2d(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

        if debug:
            pool = tf.print(pool, [tf.shape(pool)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return pool

    def _conv_layer(self, bottom, name, channels=None, new_weights=False, use_relu=True):
        # shape_filt = (3, 3, channels[0], channels[1]) if not channels == None else None
        shape_filt = None if channels is None else (3, 3, channels[0], channels[1])
        shape_bias = None if channels is None else channels[1]
        with tf.compat.v1.variable_scope(name) as scope:
            # 卷积核 conv1_1/filter (3,3,3,64)
            filt = self.get_conv_filter(name, 'backbone', shape_filt, new_weights)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            # 每一个conv加上一个bias,有64个filter,每一个一个bias
            conv_biases = self.get_bias(name, None, 'backbone', shape_bias, new_weights)
            bias = tf.nn.bias_add(conv, conv_biases)
            # 激活函数relu
            relu = tf.nn.relu(bias) if use_relu else bias
            # Add summary to Tensorboard
            _activation_summary(relu)
            return relu

    def _conv_layer_new(self, bottom, name, ksize, stride, channels, branch, use_relu=True, use_norm=True):
        # shape_filter (3,3,512,256) # shape_filter (3,3,256,128)
        shape_filt = (ksize, ksize, channels[0], channels[1])
        # shape_bias = 256  # shape_bias = 128
        shape_bias = channels[1]
        with tf.compat.v1.variable_scope(name) as scope:
            filt = self.get_conv_filter(name, branch, shape_filt, True)
            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding='SAME')
            # conv (batch_size,48,156,256) # conv (batch_size,48,156,128)
            conv = self._group_norm(conv) if use_norm else conv
            conv_biases = self.get_bias(name, None, branch, shape_bias, True)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias) if use_relu else bias
            _activation_summary(relu)
            return relu

    def _atrous_conv_layer(self, bottom, name, channels=None, new_weights=False, use_relu=True, use_norm=True):
        shape_filt = (3, 3, channels[0], channels[1]) if not channels == None else None
        shape_bias = channels[1] if not channels == None else None
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name, shape_filt, new_weights)
            conv = tf.nn.atrous_conv2d(bottom, filt, rate=2, padding='SAME')
            conv_biases = self.get_bias(name, None, shape_bias, new_weights)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias) if use_relu else bias
            _activation_summary(relu)
            return relu

    def _atrous_conv_layer_new(self, bottom, name, ksize, rate, channels, branch, use_relu=True, use_norm=True):
        shape_filt = (ksize, ksize, channels[0], channels[1])
        shape_bias = channels[1]
        with tf.compat.v1.variable_scope(name) as scope:
            filt = self.get_conv_filter(name, branch, shape_filt, True)
            conv = tf.nn.atrous_conv2d(bottom, filt, rate=rate, padding='SAME')
            conv = self._group_norm(conv) if use_norm else conv
            conv_biases = self.get_bias(name, None, branch, shape_bias, True)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias) if use_relu else bias
            _activation_summary(relu)
            return relu

    def _group_norm(self, x, G=16, eps=1e-5, scope='group_norm'):
        with tf.compat.v1.variable_scope(scope):
            N, H, W, C = x.get_shape().as_list()
            G = min(G, C)
            # x=(batch_size,48,156,16,16)
            x = tf.reshape(x, [N, H, W, G, C // G])
            # 计算1，2，4维度上的均值和方差
            mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
            x = (x - mean) / tf.sqrt(var + eps)
            gamma = tf.compat.v1.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
            x = tf.reshape(x, [N, H, W, C]) * gamma
            tf.compat.v1.add_to_collection('trainable', gamma)
            return x

    def _fc_layer(self, bottom, name, num_classes=None,
                  relu=True, debug=False):
        with tf.compat.v1.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()

            if name == 'fc6':
                filt = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])
            elif name == 'score_fr':
                name = 'fc8'  # Name of score_fr layer in VGG Model
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 1000],
                                                  num_classes=num_classes)
            else:
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])

            self._add_wd_and_summary(filt, self.wd, "fc_wlosses")

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name, num_classes=num_classes)
            bias = tf.nn.bias_add(conv, conv_biases)

            if relu:
                bias = tf.nn.relu(bias)
            _activation_summary(bias)

            if debug:
                bias = tf.Print(bias, [tf.shape(bias)],
                                message='Shape of %s' % name,
                                summarize=4, first_n=1)
            return bias

    def _score_layer(self, bottom, name, num_classes):
        with tf.compat.v1.variable_scope(name) as scope:
            # get number of input channels
            in_features = bottom.get_shape()[3].value
            shape = [1, 1, in_features, num_classes]
            # He initialization Sheme
            if name == "score_fr":
                num_input = in_features
                stddev = (2 / num_input)**0.5
            elif name == "score_pool4":
                stddev = 0.001
            elif name == "score_pool3":
                stddev = 0.0001
            # Apply convolution
            w_decay = self.wd

            weights = self._variable_with_weight_decay(shape, stddev, w_decay,
                                                       decoder=True)
            conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
            # Apply bias
            conv_biases = self._bias_variable([num_classes], constant=0.0)
            bias = tf.nn.bias_add(conv, conv_biases)

            _activation_summary(bias)

            return bias

    def _upscore_layer(self, bottom, shape,
                       num_classes, name, debug,
                       ksize=4, stride=2):
        strides = [1, stride, stride, 1]
        with tf.compat.v1.variable_scope(name):
            in_features = bottom.get_shape()[3].value

            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(bottom)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]
            output_shape = tf.stack(new_shape)

            logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]

            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input)**0.5

            weights = self.get_deconv_filter(f_shape)
            self._add_wd_and_summary(weights, self.wd, "fc_wlosses")
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)

        _activation_summary(deconv)
        return deconv

    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        heigh = f_shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        var = tf.compat.v1.get_variable(name="up_filter", initializer=init, shape=weights.shape)
        return var

    def get_conv_filter(self, name, branch, shape=None, new_weights=False):
        # 用预训练的参数的参数初始化backbone中的卷积核
        if name in self.data_dict:
            init = tf.constant_initializer(value=self.data_dict[name][0], dtype=tf.float32)
            shape = self.data_dict[name][0].shape
        else:
            init = tf.contrib.layers.xavier_initializer()
            assert shape is not None
        print('Layer name: %s' % name)
        print('Layer filter shape: %s' % str(shape))
        var = tf.compat.v1.get_variable(name="filter", initializer=init, shape=shape)
        if new_weights:
            tf.compat.v1.add_to_collection('trainable', var)
            tf.compat.v1.add_to_collection(branch, var)
        weights_key = 'new_weights' if new_weights else tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
        # if not tf.compat.v1.get_variable_scope().reuse:
        # 利用L2范数来计算张量的误差值，但是没有开方并且只取L2范数的值的一半  l2_loss(t) = sum(t**2)/2
        weight_decay = tf.nn.l2_loss(var) * self.wd
        tf.compat.v1.add_to_collection(weights_key, weight_decay)
        tf.compat.v1.add_to_collection(branch + '_decay', weight_decay)
        _variable_summaries(var)
        return var

    def get_bias(self, name, num_classes=None, branch='backbone', shape=None, new_weights=False):
        if name in self.data_dict:
            bias_wights = self.data_dict[name][1]
            shape = self.data_dict[name][1].shape
            init = tf.constant_initializer(value=bias_wights, dtype=tf.float32)
        else:
            init = tf.contrib.layers.xavier_initializer()
            assert shape is not None
        var = tf.compat.v1.get_variable(name="biases", initializer=init, shape=shape)
        if new_weights:
            tf.compat.v1.add_to_collection('trainable', var)
            tf.compat.v1.add_to_collection(branch, var)
        _variable_summaries(var)
        return var

    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.compat.v1.get_variable(name="weights", initializer=init, shape=shape)
        if not tf.compat.v1.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                       name='weight_loss')
            tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)
        _variable_summaries(var)
        return var

    def _bias_reshape(self, bweight, num_orig, num_new):
        """ Build bias weights for filter produces with `_summary_reshape`

        """
        n_averaged_elements = num_orig//num_new
        avg_bweight = np.zeros(num_new)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
        return avg_bweight

    def _summary_reshape(self, fweight, shape, num_new):
        """ Produce weights for a reduced fully-connected layer.

        FC8 of VGG produces 1000 classes. Most semantic segmentation
        task require much less classes. This reshapes the original weights
        to be used in a fully-convolutional layer which produces num_new
        classes. To archive this the average (mean) of n adjanced classes is
        taken.

        Consider reordering fweight, to perserve semantic meaning of the
        weights.

        Args:
          fweight: original weights
          shape: shape of the desired fully-convolutional layer
          num_new: number of new classes


        Returns:
          Filter weights for `num_new` classes.
        """
        num_orig = shape[3]
        shape[3] = num_new
        assert(num_new < num_orig)
        n_averaged_elements = num_orig//num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, :, :, avg_idx] = np.mean(
                fweight[:, :, :, start_idx:end_idx], axis=3)
        return avg_fweight

    def _variable_with_weight_decay(self, shape, stddev, wd, decoder=False):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.

        Args:

          self: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """

        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.compat.v1.get_variable('weights', shape=shape, initializer=initializer)

        collection_name = tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
        if wd and (not tf.compat.v1.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.compat.v1.add_to_collection(collection_name, weight_decay)
        _variable_summaries(var)
        return var

    def _add_wd_and_summary(self, var, wd, collection_name=None):
        if collection_name is None:
            collection_name = tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
        if wd and (not tf.compat.v1.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.compat.v1.add_to_collection(collection_name, weight_decay)
        _variable_summaries(var)
        return var

    def _bias_variable(self, shape, constant=0.0):
        initializer = tf.constant_initializer(constant)
        var = tf.compat.v1.get_variable(name='biases', shape=shape,
                              initializer=initializer)
        _variable_summaries(var)
        return var

    def get_fc_weight_reshape(self, name, shape, num_classes=None):
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)
        weights = self.data_dict[name][0]
        weights = weights.reshape(shape)
        if num_classes is not None:
            weights = self._summary_reshape(weights, shape,
                                            num_new=num_classes)
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        var = tf.compat.v1.get_variable(name="weights", initializer=init, shape=shape)
        return var


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.compat.v1.summary.histogram(tensor_name + '/activations', x)
    tf.compat.v1.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.compat.v1.get_variable_scope().reuse:
        name = var.op.name
        logging.info("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            # 计算tensor均值
            mean = tf.reduce_mean(var)
            # 用来显示标量信息
            tf.compat.v1.summary.scalar(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.compat.v1.summary.scalar(name + '/sttdev', stddev)
            tf.compat.v1.summary.scalar(name + '/max', tf.reduce_max(var))
            tf.compat.v1.summary.scalar(name + '/min', tf.reduce_min(var))
            # 用来显示直方图信息
            tf.compat.v1.summary.histogram(name, var)
