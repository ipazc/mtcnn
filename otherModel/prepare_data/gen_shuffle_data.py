"""Use random crop to generate training data for three models."""

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

import numpy as np
import cv2
import numpy.random as npr
sys.path.append('../')

from tools import IoU



def main(args):

    net = args.input_size
    anno_file = 'wider_face_train.txt'
    im_dir = 'WIDER_train/images'

    save_dir = 'native_' + str(net)
    pos_save_dir = save_dir + '/positive'
    part_save_dir = save_dir + '/part'
    neg_save_dir = save_dir + '/negative'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)

    f1 = open(os.path.join(save_dir, 'pos_' + str(net) + '.txt'), 'w')
    f2 = open(os.path.join(save_dir, 'neg_' + str(net) + '.txt'), 'w')
    f3 = open(os.path.join(save_dir, 'part_' + str(net) + '.txt'), 'w')
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print('%d pics in total' % num)
    p_idx = 0  # positive
    n_idx = 0  # negative
    d_idx = 0  # dont care
    idx = 0
    box_idx = 0
    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_path = annotation[0]
        bbox = list(map(float, annotation[1:]))
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
        idx += 1
        if idx % 10000 == 0:
            print(idx, 'images done')

        height, width, channel = img.shape

        neg_num = 0
        while neg_num < 50:
            size = npr.randint(40, min(width, height) / 2)
            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])

            Iou = IoU(crop_box, boxes)

            cropped_im = img[ny: ny + size, nx: nx + size, :]
            resized_im = cv2.resize(cropped_im, (net, net),
                                    interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                save_file = os.path.join(neg_save_dir, '%s.jpg' % n_idx)
                f2.write(save_dir + '/negative/%s' % n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            print('%s images done, pos: %s part: %s neg: %s' %
                  (idx, p_idx, d_idx, n_idx))

        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue

            for i in range(20):
                size = npr.randint(int(min(w, h) * 0.8),
                                   np.ceil(1.25 * max(w, h)))

                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)

                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                cropped_im = img[ny1: ny2, nx1: nx2, :]
                resized_im = cv2.resize(cropped_im, (net, net),
                                        interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= 0.65:
                    save_file = os.path.join(pos_save_dir, '%s.jpg' % p_idx)
                    f1.write(save_dir + '/positive/%s' % p_idx +
                             ' 1 %.2f %.2f %.2f %.2f\n' %
                             (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif IoU(crop_box, box_) >= 0.4:
                    save_file = os.path.join(part_save_dir, '%s.jpg' % d_idx)
                    f3.write(save_dir + '/part/%s' % d_idx +
                             ' -1 %.2f %.2f %.2f %.2f\n' %
                             (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
            box_idx += 1
            print('%s images done, pos: %s part: %s neg: %s' %
                  (idx, p_idx, d_idx, n_idx))

    f1.close()
    f2.close()
    f3.close()


def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('input_size', type=int,
                        help='The input size for specific net')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
