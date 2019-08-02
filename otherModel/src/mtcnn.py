"""Main script. Contain model definition and training code."""

# MIT License
#
# Copyright (c) 2017 Baoming Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os

import tensorflow as tf
import numpy as np


def layer(op):

    def layer_decorated(self, *args, **kwargs):

        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        layer_output = op(self, layer_input, *args, **kwargs)
        tf.add_to_collection('feature_map', layer_output)
        self.layers[name] = layer_output
        self.feed(layer_output)
        return self

    return layer_decorated


class NetWork(object):

    def __init__(self, inputs, trainable=True,
                 weight_decay_coeff=4e-3, mode='train'):

        self.inputs = inputs
        self.terminals = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.mode = mode
        self.out_put = []
        self.weight_decay_coeff = weight_decay_coeff

        if self.mode == 'train':
            self.tasks = [inp[0] for inp in inputs]
            self.weight_decay = {}
            self.setup_training_graph()
        else:
            self.setup()

    def setup_training_graph(self):

        for index, task in enumerate(self.tasks):
            self.weight_decay[task] = []
            reuse_bool = False
            if index is not 0:
                reuse_bool = True
            self.setup(task=task, reuse=reuse_bool)

    def setup(self, task='data'):

        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, prefix, ignore_missing=False):

        data_dict = np.load(data_path, encoding='latin1').item()
        for op_name in data_dict:
            with tf.variable_scope(prefix + op_name, reuse=True):
                for param_name, data in data_dict[op_name].items():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):

        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):

        return self.terminals[-1]

    def get_all_output(self):

        return self.out_put

    def get_weight_decay(self):

        assert self.mode == 'train'
        return self.weight_decay

    def get_unique_name(self, prefix):

        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):

        return tf.get_variable(
            name,
            shape,
            trainable=self.trainable,
            initializer=tf.truncated_normal_initializer(
                stddev=1e-4))

    def validate_padding(self, padding):

        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, inp, k_h, k_w, c_o, s_h, s_w, name,
             task=None, relu=True, padding='SAME',
             group=1, biased=True, wd=None):

        self.validate_padding(padding)
        c_i = int(inp.get_shape()[-1])
        assert c_i % group == 0
        assert c_o % group == 0

        def convolve(i, k): return tf.nn.conv2d(
            i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var(
                'weights', shape=[
                    k_h, k_w, c_i / group, c_o])
            if group == 1:
                output = convolve(inp, kernel)
            else:
                input_groups = tf.split(inp, group, 3)
                kernel_groups = tf.split(kernel, group, 3)
                output_groups = [convolve(i, k) for i, k in
                                 zip(input_groups, kernel_groups)]
                output = tf.concat(output_groups, 3)
            if (wd is not None) and (self.mode == 'train'):
                self.weight_decay[task].append(
                    tf.multiply(tf.nn.l2_loss(kernel), wd))
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def prelu(self, inp, name):

        with tf.variable_scope(name):
            i = int(inp.get_shape()[-1])
            alpha = self.make_var('alpha', shape=(i,))
            return tf.nn.relu(inp) + tf.multiply(alpha, -tf.nn.relu(-inp))

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name,
                 padding='SAME'):

        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def fc(self, inp, num_out, name, task=None, relu=True, wd=None):

        with tf.variable_scope(name):
            input_shape = inp.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= int(d)
                feed_in = tf.reshape(inp, [-1, dim])
            else:
                feed_in, dim = (inp, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            if (wd is not None) and (self.mode == 'train'):
                self.weight_decay[task] \
                    .append(tf.multiply(tf.nn.l2_loss(weights), wd))
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            return op(feed_in, weights, biases, name=name)

    @layer
    def softmax(self, target, name=None):

        with tf.variable_scope(name):
            return tf.nn.softmax(target, name=name)


class PNet(NetWork):

    def setup(self, task='data', reuse=False):

        with tf.variable_scope('pnet', reuse=reuse):
            (
                self.feed(task) .conv(
                    3,
                    3,
                    15,
                    1,
                    1,
                    padding='VALID',
                    relu=False,
                    name='conv1') .prelu(
                    name='PReLU1') .max_pool(
                    2,
                    2,
                    2,
                    2,
                    name='pool1') .conv(
                    3,
                    3,
                    16,
                    1,
                    1,
                    padding='VALID',
                    relu=False,
                    name='conv2') .prelu(
                        name='PReLU2') .conv(
                            3,
                            3,
                            32,
                            1,
                            1,
                            task=task,
                            padding='VALID',
                            relu=False,
                            name='conv3',
                            wd=self.weight_decay_coeff) .prelu(
                                name='PReLU3'))

        if self.mode == 'train':
            if task == 'cls':
                (self.feed('PReLU3')
                     .conv(1, 1, 2, 1, 1, task=task, relu=False,
                           name='pnet/conv4-1', wd=self.weight_decay_coeff))
            elif task == 'bbx':
                (self.feed('PReLU3')
                     .conv(1, 1, 4, 1, 1, task=task, relu=False,
                           name='pnet/conv4-2', wd=self.weight_decay_coeff))
            elif task == 'pts':
                (self.feed('PReLU3')
                     .conv(1, 1, 10, 1, 1, task=task, relu=False,
                           name='pnet/conv4-3', wd=self.weight_decay_coeff))
            self.out_put.append(self.get_output())
        else:
            (self.feed('PReLU3')
                 .conv(1, 1, 2, 1, 1, relu=False, name='pnet/conv4-1')
                 .softmax(name='softmax'))
            self.out_put.append(self.get_output())
            (self.feed('PReLU3')
                 .conv(1, 1, 4, 1, 1, relu=False, name='pnet/conv4-2'))
            self.out_put.append(self.get_output())


class RNet(NetWork):

    def setup(self, task='data', reuse=False):

        with tf.variable_scope('rnet', reuse=reuse):
            (
                self.feed(task) .conv(
                    3,
                    3,
                    28,
                    1,
                    1,
                    padding='VALID',
                    relu=False,
                    name='conv1') .prelu(
                    name='prelu1') .max_pool(
                    3,
                    3,
                    2,
                    2,
                    name='pool1') .conv(
                    3,
                    3,
                    48,
                    1,
                    1,
                    padding='VALID',
                    relu=False,
                    name='conv2') .prelu(
                        name='prelu2') .max_pool(
                            3,
                            3,
                            2,
                            2,
                            padding='VALID',
                            name='pool2') .conv(
                                2,
                                2,
                                64,
                                1,
                                1,
                                padding='VALID',
                                task=task,
                                relu=False,
                                name='conv3',
                                wd=self.weight_decay_coeff) .prelu(
                                    name='prelu3') .fc(
                                        128,
                                        task=task,
                                        relu=False,
                                        name='conv4',
                                        wd=self.weight_decay_coeff) .prelu(
                                            name='prelu4'))

        if self.mode == 'train':
            if task == 'cls':
                (self.feed('prelu4')
                     .fc(2, task=task, relu=False,
                         name='rnet/conv5-1', wd=self.weight_decay_coeff))
            elif task == 'bbx':
                (self.feed('prelu4')
                     .fc(4, task=task, relu=False,
                         name='rnet/conv5-2', wd=self.weight_decay_coeff))
            elif task == 'pts':
                (self.feed('prelu4')
                     .fc(10, task=task, relu=False,
                         name='rnet/conv5-3', wd=self.weight_decay_coeff))
            self.out_put.append(self.get_output())
        else:
            (self.feed('prelu4')
                 .fc(2, relu=False, name='rnet/conv5-1')
                 .softmax(name='softmax'))
            self.out_put.append(self.get_output())
            (self.feed('prelu4')
                 .fc(4, relu=False, name='rnet/conv5-2'))
            self.out_put.append(self.get_output())


class ONet(NetWork):

    def setup(self, task='data', reuse=False):

        with tf.variable_scope('onet', reuse=reuse):
            (
                self.feed(task) .conv(
                    3,
                    3,
                    32,
                    1,
                    1,
                    padding='VALID',
                    relu=False,
                    name='conv1') .prelu(
                    name='prelu1') .max_pool(
                    3,
                    3,
                    2,
                    2,
                    name='pool1') .conv(
                    3,
                    3,
                    64,
                    1,
                    1,
                    padding='VALID',
                    relu=False,
                    name='conv2') .prelu(
                        name='prelu2') .max_pool(
                            3,
                            3,
                            2,
                            2,
                            padding='VALID',
                            name='pool2') .conv(
                                3,
                                3,
                                64,
                                1,
                                1,
                                padding='VALID',
                                relu=False,
                                name='conv3') .prelu(
                                    name='prelu3') .max_pool(
                                        2,
                                        2,
                                        2,
                                        2,
                                        name='pool3') .conv(
                                            2,
                                            2,
                                            128,
                                            1,
                                            1,
                                            padding='VALID',
                                            relu=False,
                                            name='conv4') .prelu(
                                                name='prelu4') .fc(
                                                    256,
                                                    relu=False,
                                                    name='conv5') .prelu(
                                                        name='prelu5'))

        if self.mode == 'train':
            if task == 'cls':
                (self.feed('prelu5')
                     .fc(2, task=task, relu=False,
                         name='onet/conv6-1', wd=self.weight_decay_coeff))
            elif task == 'bbx':
                (self.feed('prelu5')
                     .fc(4, task=task, relu=False,
                         name='onet/conv6-2', wd=self.weight_decay_coeff))
            elif task == 'pts':
                (self.feed('prelu5')
                     .fc(10, task=task, relu=False,
                         name='onet/conv6-3', wd=self.weight_decay_coeff))
            self.out_put.append(self.get_output())
        else:
            (self.feed('prelu5')
                 .fc(2, relu=False, name='onet/conv6-1')
                 .softmax(name='softmax'))
            self.out_put.append(self.get_output())
            (self.feed('prelu5')
                 .fc(4, relu=False, name='onet/conv6-2'))
            self.out_put.append(self.get_output())
            (self.feed('prelu5')
                 .fc(10, relu=False, name='onet/conv6-3'))
            self.out_put.append(self.get_output())


def read_and_decode(filename_queue, label_type, shape):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32)

    image = (image - 127.5) * (1. / 128.0)
    image.set_shape([shape * shape * 3])
    image = tf.reshape(image, [shape, shape, 3])
    label = tf.decode_raw(features['label_raw'], tf.float32)

    if label_type == 'cls':
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        label.set_shape([2])
    elif label_type == 'bbx':
        label.set_shape([4])
    elif label_type == 'pts':
        label.set_shape([10])

    return image, label


def inputs(filename, batch_size, num_epochs, label_type, shape):

    with tf.device('/cpu:0'):
        if not num_epochs:
            num_epochs = None

        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(
                filename, num_epochs=num_epochs)

        image, label = read_and_decode(filename_queue, label_type, shape)

        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            min_after_dequeue=1000)

        return images, sparse_labels


def train_net(Net, training_data, base_lr, loss_weight,
              train_mode, num_epochs=[1, None, None],
              batch_size=64, weight_decay=4e-3,
              load_model=False, load_filename=None,
              save_model=False, save_filename=None,
              num_iter_to_save=10000,
              gpu_memory_fraction=1):

    images = []
    labels = []
    tasks = ['cls', 'bbx', 'pts']
    shape = 12
    if Net.__name__ == 'RNet':
        shape = 24
    elif Net.__name__ == 'ONet':
        shape = 48
    for index in range(train_mode):
        image, label = inputs(filename=[training_data[index]],
                              batch_size=batch_size,
                              num_epochs=num_epochs[index],
                              label_type=tasks[index],
                              shape=shape)
        images.append(image)
        labels.append(label)
    while len(images) is not 3:
        images.append(tf.placeholder(tf.float32, [None, shape, shape, 3]))
        labels.append(tf.placeholder(tf.float32))
    net = Net((('cls', images[0]), ('bbx', images[1]), ('pts', images[2])),
              weight_decay_coeff=weight_decay)

    print('all trainable variables:')
    all_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES)
    for var in all_vars:
        print(var)

    print('all local variable:')
    local_variables = tf.local_variables()
    for l_v in local_variables:
        print(l_v.name)

    prefix = str(all_vars[0].name[0:5])
    out_put = net.get_all_output()
    cls_output = tf.reshape(out_put[0], [-1, 2])
    bbx_output = tf.reshape(out_put[1], [-1, 4])
    pts_output = tf.reshape(out_put[2], [-1, 10])

    # cls loss
    softmax_loss = loss_weight[0] * \
        tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels[0],
                                                logits=cls_output))
    weight_losses_cls = net.get_weight_decay()['cls']
    losses_cls = softmax_loss + tf.add_n(weight_losses_cls)

    # bbx loss
    square_bbx_loss = loss_weight[1] * \
        tf.reduce_mean(tf.squared_difference(bbx_output, labels[1]))
    weight_losses_bbx = net.get_weight_decay()['bbx']
    losses_bbx = square_bbx_loss + tf.add_n(weight_losses_bbx)

    # pts loss
    square_pts_loss = loss_weight[2] * \
        tf.reduce_mean(tf.squared_difference(pts_output, labels[2]))
    weight_losses_pts = net.get_weight_decay()['pts']
    losses_pts = square_pts_loss + tf.add_n(weight_losses_pts)

    global_step_cls = tf.Variable(1, name='global_step_cls', trainable=False)
    global_step_bbx = tf.Variable(1, name='global_step_bbx', trainable=False)
    global_step_pts = tf.Variable(1, name='global_step_pts', trainable=False)

    train_cls = tf.train.AdamOptimizer(learning_rate=base_lr) \
                        .minimize(losses_cls, global_step=global_step_cls)
    train_bbx = tf.train.AdamOptimizer(learning_rate=base_lr) \
                        .minimize(losses_bbx, global_step=global_step_bbx)
    train_pts = tf.train.AdamOptimizer(learning_rate=base_lr) \
                        .minimize(losses_pts, global_step=global_step_pts)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    config.gpu_options.allow_growth = True

    loss_agg_cls = [0]
    loss_agg_bbx = [0]
    loss_agg_pts = [0]
    step_value = [1, 1, 1]

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        saver = tf.train.Saver(max_to_keep=200000)
        if load_model:
            saver.restore(sess, load_filename)
        else:
            net.load(load_filename, sess, prefix)
        if save_model:
            save_dir = os.path.split(save_filename)[0]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                choic = np.random.randint(0, train_mode)
                if choic == 0:
                    _, loss_value_cls, step_value[0] = sess.run(
                        [train_cls, softmax_loss, global_step_cls])
                    loss_agg_cls.append(loss_value_cls)
                elif choic == 1:
                    _, loss_value_bbx, step_value[1] = sess.run(
                        [train_bbx, square_bbx_loss, global_step_bbx])
                    loss_agg_bbx.append(loss_value_bbx)
                else:
                    _, loss_value_pts, step_value[2] = sess.run(
                        [train_pts, square_pts_loss, global_step_pts])
                    loss_agg_pts.append(loss_value_pts)

                if sum(step_value) % (100 * train_mode) == 0:
                    agg_cls = sum(loss_agg_cls) / len(loss_agg_cls)
                    agg_bbx = sum(loss_agg_bbx) / len(loss_agg_bbx)
                    agg_pts = sum(loss_agg_pts) / len(loss_agg_pts)
                    print(
                        'Step %d for cls: loss = %.5f' %
                        (step_value[0], agg_cls), end='. ')
                    print(
                        'Step %d for bbx: loss = %.5f' %
                        (step_value[1], agg_bbx), end='. ')
                    print(
                        'Step %d for pts: loss = %.5f' %
                        (step_value[2], agg_pts))
                    print('number of epoch')
                    print(num_epochs[0])
                    loss_agg_cls = [0]
                    loss_agg_bbx = [0]
                    loss_agg_pts = [0]

                if save_model and (step_value[0] % num_iter_to_save == 0):
                    saver.save(sess, save_filename, global_step=step_value[0])

        except tf.errors.OutOfRangeError:
            print(
                'Done training for %d epochs, %d steps.' %
                (num_epochs[0], step_value[0]))
        finally:
            coord.request_stop()

        coord.join(threads)
