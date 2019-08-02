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
import sys
import re

import numpy as np
import tensorflow as tf
import cv2


def view_bar(num, total):

    rate = float(num) / total
    rate_num = int(rate * 100) + 1
    r = '\r[%s%s]%d%%' % ("#" * rate_num, " " * (100 - rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()


def int64_feature(value):

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_model_filenames(model_dir):

    files = os.listdir(model_dir)
    pnet = [s for s in files if 'pnet' in s and
                                os.path.isdir(os.path.join(model_dir, s))]
    rnet = [s for s in files if 'rnet' in s and
                                os.path.isdir(os.path.join(model_dir, s))]
    onet = [s for s in files if 'onet' in s and
                                os.path.isdir(os.path.join(model_dir, s))]
    if pnet and rnet and onet:
        if len(pnet) == 1 and len(rnet) == 1 and len(onet) == 1:
            _, pnet_data = get_meta_data(os.path.join(model_dir, pnet[0]))
            _, rnet_data = get_meta_data(os.path.join(model_dir, rnet[0]))
            _, onet_data = get_meta_data(os.path.join(model_dir, onet[0]))
            return (pnet_data, rnet_data, onet_data)
        else:
            raise ValueError('There should not be more '
                             'than one dir for each model')
    else:
        return get_meta_data(model_dir)


def get_meta_data(model_dir):

    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model '
                         'directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than '
                         'one meta file in the model directory (%s)'
                         % model_dir)
    meta_file = meta_files[0]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^[A-Za-z]+-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                data_file = step_str.groups()[0]
    return (os.path.join(model_dir, meta_file),
            os.path.join(model_dir, data_file))


def detect_face(img, minsize, pnet, rnet, onet, threshold, factor):
    print("points out of the net %f")

    factor_count = 0
    total_boxes = np.empty((0, 9))
    points = []
    h = img.shape[0]
    w = img.shape[1]
    minl = np.amin([h, w])
    m = 12.0 / minsize
    minl = minl * m
    # creat scale pyramid
    scales = []
    while minl >= 12:
        scales += [m * np.power(factor, factor_count)]
        minl = minl * factor
        factor_count += 1

    # first stage    second stage the P neural net

    for j in range(len(scales)):
        scale = scales[j]
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        im_data = imresample(img, (hs, ws))
        im_data = (im_data - 127.5) * (1. / 128.0)
        img_x = np.expand_dims(im_data, 0)
        out = pnet(img_x)
        out0 = out[0]
        out1 = out[1]
        boxes, _ = generateBoundingBox(out0[0, :, :, 1].copy(),
                                       out1[0, :, :, :].copy(),
                                       scale,
                                       threshold[0])

        # inter-scale nms
        pick = nms(boxes.copy(), 0.5, 'Union')
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4,
                                              total_boxes[:, 4]]))
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(
            total_boxes.copy(), w, h)

    numbox = total_boxes.shape[0]

    # second stage the r neural net
    if numbox > 0:
        tempimg = np.zeros((24, 24, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k],
                :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
            if (tmp.shape[0] > 0 and tmp.shape[1] > 0 or
                    tmp.shape[0] == 0 and tmp.shape[1] == 0):
                tempimg[:, :, :, k] = imresample(tmp, (24, 24))
            else:
                return np.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 0, 1, 2))
        out = rnet(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        score = out0[1, :]
        ipass = np.where(score > threshold[1])
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(),
                                 np.expand_dims(score[ipass].copy(), 1)])
        mv = out1[:, ipass[0]]
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
            total_boxes = rerec(total_boxes.copy())

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # third stage
        total_boxes = np.fix(total_boxes).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(
            total_boxes.copy(), w, h)
        tempimg = np.zeros((48, 48, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k],
                :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
            if (tmp.shape[0] > 0 and tmp.shape[1] > 0 or
                    tmp.shape[0] == 0 and tmp.shape[1] == 0):
                tempimg[:, :, :, k] = imresample(tmp, (48, 48))
            else:
                return np.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 0, 1, 2))
        out = onet(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        out2 = np.transpose(out[2])
        score = out0[1, :]
        points = out2
        print("points out of the net %f",points)
        ipass = np.where(score > threshold[2])
        points = points[:, ipass[0]]
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(),
                                 np.expand_dims(score[ipass].copy(), 1)])
        mv = out1[:, ipass[0]]

        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1
        points[0:10:2, :] = np.tile(w, (5, 1)) * \
            (points[0:10:2, :] + 1) / 2 + \
            np.tile(total_boxes[:, 0], (5, 1)) - 1
        points[1:11:2, :] = np.tile(h, (5, 1)) * \
            (points[1:11:2, :] + 1) / 2 + \
            np.tile(total_boxes[:, 1], (5, 1)) - 1
        if total_boxes.shape[0] > 0:
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
            pick = nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick, :]
            points = points[:, pick]

    return total_boxes, points


def detect_face_12net(img, minsize, pnet, threshold, factor):

    factor_count = 0
    total_boxes = np.empty((0, 9))
    h = img.shape[0]
    w = img.shape[1]
    minl = np.amin([h, w])
    m = 12.0 / minsize
    minl = minl * m
    # creat scale pyramid
    scales = []
    while minl >= 12:
        scales += [m * np.power(factor, factor_count)]
        minl = minl * factor
        factor_count += 1

    # first stage
    for j in range(len(scales)):
        scale = scales[j]
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        im_data = imresample(img, (hs, ws))
        im_data = (im_data - 127.5) * (1. / 128.0)
        img_x = np.expand_dims(im_data, 0)
        out = pnet(img_x)
        out0 = out[0]
        out1 = out[1]
        boxes, _ = generateBoundingBox(out0[0, :, :, 1].copy(),
                                       out1[0, :, :, :].copy(),
                                       scale,
                                       threshold)

        # inter-scale nms
        pick = nms(boxes.copy(), 0.5, 'Union')
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4,
                                              total_boxes[:, 4]]))
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
    return total_boxes


def detect_face_24net(img, minsize, pnet, rnet, threshold, factor):

    factor_count = 0
    total_boxes = np.empty((0, 9))
    h = img.shape[0]
    w = img.shape[1]
    minl = np.amin([h, w])
    m = 12.0 / minsize
    minl = minl * m
    # creat scale pyramid
    scales = []
    while minl >= 12:
        scales += [m * np.power(factor, factor_count)]
        minl = minl * factor
        factor_count += 1

    # first stage
    for j in range(len(scales)):
        scale = scales[j]
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        im_data = imresample(img, (hs, ws))
        im_data = (im_data - 127.5) * 0.0078125
        img_x = np.expand_dims(im_data, 0)
        out = pnet(img_x)
        out0 = out[0]
        out1 = out[1]
        boxes, _ = generateBoundingBox(out0[0, :, :, 1].copy(),
                                       out1[0, :, :, :].copy(),
                                       scale,
                                       threshold[0])

        # inter-scale nms
        pick = nms(boxes.copy(), 0.5, 'Union')
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4,
                                              total_boxes[:, 4]]))
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(
            total_boxes.copy(), w, h)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage
        tempimg = np.zeros((24, 24, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k],
                :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
            if (tmp.shape[0] > 0 and tmp.shape[1] > 0 or
                    tmp.shape[0] == 0 and tmp.shape[1] == 0):
                tempimg[:, :, :, k] = imresample(tmp, (24, 24))
            else:
                return np.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 0, 1, 2))
        out = rnet(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        score = out0[1, :]
        ipass = np.where(score > threshold[1])
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(),
                                 np.expand_dims(score[ipass].copy(), 1)])
        mv = out1[:, ipass[0]]
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.5, 'Union')
            total_boxes = total_boxes[pick, :]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
    return total_boxes


def nms(boxes, threshold, method):

    if boxes.size == 0:
        return np.empty((0, 3))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    s_sort = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while s_sort.size > 0:
        i = s_sort[-1]
        pick[counter] = i
        counter += 1
        idx = s_sort[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        s_sort = s_sort[np.where(o <= threshold)]
    pick = pick[0:counter]
    return pick


def bbreg(boundingbox, reg):

    if reg.shape[1] == 1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
    return boundingbox


def generateBoundingBox(imap, reg, scale, t):

    stride = 2
    cellsize = 12

    imap = np.transpose(imap)
    dx1 = np.transpose(reg[:, :, 0])
    dy1 = np.transpose(reg[:, :, 1])
    dx2 = np.transpose(reg[:, :, 2])
    dy2 = np.transpose(reg[:, :, 3])
    y, x = np.where(imap >= t)
    if y.shape[0] == 1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    score = imap[(y, x)]
    reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)],
                                  dx2[(y, x)], dy2[(y, x)]]))
    if reg.size == 0:
        reg = np.empty((0, 3))
    bb = np.transpose(np.vstack([y, x]))
    q1 = np.fix((stride * bb + 1) / scale)
    q2 = np.fix((stride * bb + cellsize - 1 + 1) / scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])
    return boundingbox, reg


def pad(total_boxes, w, h):

    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:, 0].copy().astype(np.int32)
    y = total_boxes[:, 1].copy().astype(np.int32)
    ex = total_boxes[:, 2].copy().astype(np.int32)
    ey = total_boxes[:, 3].copy().astype(np.int32)

    tmp = np.where(ex > w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
    ex[tmp] = w

    tmp = np.where(ey > h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
    ey[tmp] = h

    tmp = np.where(x < 1)
    dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
    x[tmp] = 1

    tmp = np.where(y < 1)
    dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
    y[tmp] = 1

    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph


def rerec(bboxA):

    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    size = np.maximum(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - size * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - size * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.transpose(np.tile(size, (2, 1)))
    return bboxA


def imresample(img, sz):

    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA)
    return im_data


def IoU(box, boxes):

    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


def convert_to_square(bbox):

    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox
