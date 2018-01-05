MTCNN
#####

.. image:: https://badge.fury.io/py/mtcnn.svg
    :target: https://badge.fury.io/py/mtcnn

Implementation of the MTCNN face detector for TensorFlow. It is written from scratch, using as a reference the implementation of
MTCNN from David Sandberg (`FaceNet's MTCNN <https://github.com/davidsandberg/facenet/tree/master/src/align>`_). It is based on the paper *Zhang et al. (2016)* [ZHANG2016]_.

.. image:: result.jpg


INSTALLATION
############

Currently it is only supported python3 onwards. It can be installed with pip:

.. code:: bash

    $ pip3 install mtcnn


USAGE
#####

The following example illustrates the ease of use of this package:


.. code:: python

    >>> from mtcnn.mtcnn import MTCNN
    >>> import cv2
    >>>
    >>> img = cv2.imread("ivan.jpg")
    >>> detector = MTCNN()
    >>> print(detector.detect_faces(img))
    [{'box': [277, 90, 48, 63], 'keypoints': {'nose': (303, 131), 'mouth_right': (313, 141), 'right_eye': (314, 114), 'left_eye': (291, 117), 'mouth_left': (296, 143)}, 'confidence': 0.99851983785629272}]

The detector returns a list of JSON objects. Each JSON object contains two main keys: 'box' and 'keypoints'.
The bounding box is formatted as [x, y, width, height] under the key 'box'.
The keypoints are formatted into a JSON object with the keys 'left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right'. Each keypoint is identified by a pixel position (x, y).

A good example of usage can be found in the file "`example.py`_." located in the root of this repository.


MODEL
#####

The model is adapted from the Facenet's MTCNN implementation, merged in a single file located inside the folder 'data' relative
to the module's path. It can be overriden by injecting it into the MTCNN() constructor during instantiation.

The model must be a numpy-model containing the 3 main keys "pnet", "rnet" and "onet", having each of them the weights of each of the layers of the network.

.. _example.py: example.py


REFERENCE
=========

.. [ZHANG2016] Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499–1503.