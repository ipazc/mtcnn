"""Use the trained pnet model to generate training data for rnet."""

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
import os

import argparse
import tensorflow as tf
import cv2
import numpy as np

from tools import detect_face_12net, IoU, view_bar
from src.mtcnn import PNet

sys.path.append('../')


def main(args):

    image_size = 24
    save_dir = str(image_size)
    anno_file = 'wider_face_train.txt'
    im_dir = 'WIDER_train/images/'

    neg_save_dir = save_dir+'/negative'
    pos_save_dir = save_dir+'/positive'
    part_save_dir = save_dir+'/part'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)

    f1 = open(save_dir+'/pos_24.txt', 'w')
    f2 = open(save_dir+'/neg_24.txt', 'w')
    f3 = open(save_dir+'/part_24.txt', 'w')
    threshold = 0.6
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print('%d pics in total' % num)

    p_idx = 0  # positive
    n_idx = 0  # negative
    d_idx = 0  # dont care
    image_idx = 0
    with tf.device('/gpu:0'):
        minsize = 20
        factor = 0.709
        model_file = args.pnet_model
        with tf.Graph().as_default():
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.per_process_gpu_memory_fraction = 0.5
            with tf.Session(config=config) as sess:
                image = tf.placeholder(tf.float32, [None, None, None, 3])
                pnet = PNet({'data': image}, mode='test')
                out_tensor = pnet.get_all_output()
                init_op = tf.global_variables_initializer()
                sess.run(init_op)
                saver = tf.train.Saver()
                saver.restore(sess, model_file)

                def pnet_fun(img): return sess.run(
                    out_tensor, feed_dict={image: img})

                for annotation in annotations:
                    annotation = annotation.strip().split(' ')
                    bbox = list(map(float, annotation[1:]))
                    gts = np.array(bbox, dtype=np.float32).reshape(-1, 4)
                    img_path = im_dir + annotation[0] + '.jpg'
                    img = cv2.imread(img_path)
                    rectangles = detect_face_12net(img, minsize, pnet_fun,
                                                   threshold, factor)
                    image_idx += 1

                    view_bar(image_idx, num)
                    for box in rectangles:
                        lis = box.astype(np.int32)
                        mask = lis < 0
                        lis[mask] = 0
                        x_left, y_top, x_right, y_bottom, _ = lis
                        crop_w = x_right - x_left + 1
                        crop_h = y_bottom - y_top + 1
                        # ignore box that is too small or beyond image border
                        if crop_w < image_size or crop_h < image_size:
                            continue

                        Iou = IoU(box, gts)
                        cropped_im = img[y_top: y_bottom+1, x_left: x_right+1]
                        resized_im = cv2.resize(cropped_im,
                                                (image_size, image_size),
                                                interpolation=cv2.INTER_LINEAR)

                        # save negative images and write label
                        if np.max(Iou) < 0.3:
                            # Iou with all gts must below 0.3
                            save_file = os.path.join(neg_save_dir,
                                                     '%s.jpg' % n_idx)
                            f2.write('%s/negative/%s' %
                                     (save_dir, n_idx) + ' 0\n')
                            cv2.imwrite(save_file, resized_im)
                            n_idx += 1
                        else:
                            # find gt_box with the highest iou
                            idx = np.argmax(Iou)
                            assigned_gt = gts[idx]
                            x1, y1, x2, y2 = assigned_gt

                            # compute bbox reg label
                            offset_x1 = (x1 - x_left) / float(crop_w)
                            offset_y1 = (y1 - y_top) / float(crop_h)
                            offset_x2 = (x2 - x_right) / float(crop_w)
                            offset_y2 = (y2 - y_bottom) / float(crop_h)

                            if np.max(Iou) >= 0.65:
                                save_file = os.path.join(pos_save_dir,
                                                         '%s.jpg' % p_idx)
                                f1.write('%s/positive/%s' % (save_dir, p_idx) +
                                         ' 1 %.2f %.2f %.2f %.2f\n' %
                                         (offset_x1, offset_y1,
                                          offset_x2, offset_y2))
                                cv2.imwrite(save_file, resized_im)
                                p_idx += 1

                            elif np.max(Iou) >= 0.4:
                                save_file = os.path.join(part_save_dir,
                                                         '%s.jpg' % d_idx)
                                f3.write('%s/part/%s' % (save_dir, d_idx) +
                                         ' -1 %.2f %.2f %.2f %.2f\n' %
                                         (offset_x1, offset_y1,
                                          offset_x2, offset_y2))
                                cv2.imwrite(save_file, resized_im)
                                d_idx += 1

    f1.close()
    f2.close()
    f3.close()


def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('--pnet_model', type=str,
                        help='The path of pnet model to generate hard example',
                        default='../save_model/seperate_net/pnet/pnet-3000000')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
