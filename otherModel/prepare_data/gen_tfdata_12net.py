"""Generate tfrecords file for pnet, which has input size of 12*12*3."""

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

import sys
import random

import cv2
import tensorflow as tf
import numpy as np
import numpy.random as npr

from tools import view_bar, bytes_feature

sys.path.append('../')


def main():

    size = 12
    net = 'native_'+str(size)

    with open('%s/pos_%s.txt' % (net, size), 'r') as f:
        pos = f.readlines()
    with open('%s/neg_%s.txt' % (net, size), 'r') as f:
        neg = f.readlines()
    with open('%s/part_%s.txt' % (net, size), 'r') as f:
        part = f.readlines()

    print('\n'+'pos')
    filename_cls = 'pnet_data_for_cls.tfrecords'
    print('Writing')
    examples = []
    writer = tf.python_io.TFRecordWriter(filename_cls)
    cur_ = 0
    sum_ = len(pos)
    for line in pos:
        view_bar(cur_, sum_)
        cur_ += 1
        words = line.split()
        image_file_name = words[0]+'.jpg'
        im = cv2.imread(image_file_name)
        h, w, ch = im.shape
        if h != 12 or w != 12:
            im = cv2.resize(im, (12, 12))
        im = im.astype('uint8')
        label = np.array([0, 1], dtype='float32')
        label_raw = label.tostring()
        image_raw = im.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': bytes_feature(label_raw),
            'image_raw': bytes_feature(image_raw)}))
        examples.append(example)

    print('\n'+'neg')
    cur_ = 0
    neg_keep = npr.choice(len(neg), size=1000000, replace=False)
    sum_ = len(neg_keep)
    for i in neg_keep:
        line = neg[i]
        view_bar(cur_, sum_)
        cur_ += 1
        words = line.split()
        image_file_name = words[0]+'.jpg'
        im = cv2.imread(image_file_name)
        h, w, ch = im.shape
        if h != 12 or w != 12:
            im = cv2.resize(im, (12, 12))
        im = im.astype('uint8')
        label = np.array([1, 0], dtype='float32')
        label_raw = label.tostring()
        image_raw = im.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': bytes_feature(label_raw),
            'image_raw': bytes_feature(image_raw)}))
        examples.append(example)
    print(len(examples))
    random.shuffle(examples)
    for example in examples:
        writer.write(example.SerializeToString())
    writer.close()

    print('\n'+'pos')
    cur_ = 0
    filename_roi = 'pnet_data_for_bbx.tfrecords'
    print('Writing')
    sum_ = len(pos)
    examples = []
    writer = tf.python_io.TFRecordWriter(filename_roi)
    for line in pos:
        view_bar(cur_, sum_)
        cur_ += 1
        words = line.split()
        image_file_name = words[0]+'.jpg'
        im = cv2.imread(image_file_name)
        h, w, ch = im.shape
        if h != 12 or w != 12:
            im = cv2.resize(im, (12, 12))
        im = im.astype('uint8')
        label = np.array([float(words[2]), float(words[3]),
                          float(words[4]), float(words[5])],
                         dtype='float32')
        label_raw = label.tostring()
        image_raw = im.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': bytes_feature(label_raw),
            'image_raw': bytes_feature(image_raw)}))
        examples.append(example)

    print('\n'+'part')
    cur_ = 0
    part_keep = npr.choice(len(part), size=300000, replace=False)
    sum_ = len(part_keep)
    for i in part_keep:
        view_bar(cur_, sum_)
        line = part[i]
        cur_ += 1
        words = line.split()
        image_file_name = words[0]+'.jpg'
        im = cv2.imread(image_file_name)
        h, w, ch = im.shape
        if h != 12 or w != 12:
            im = cv2.resize(im, (12, 12))
        im = im.astype('uint8')
        label = np.array([float(words[2]), float(words[3]),
                          float(words[4]), float(words[5])],
                         dtype='float32')
        label_raw = label.tostring()
        image_raw = im.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': bytes_feature(label_raw),
            'image_raw': bytes_feature(image_raw)}))
        examples.append(example)
    print(len(examples))
    random.shuffle(examples)
    for example in examples:
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    main()
