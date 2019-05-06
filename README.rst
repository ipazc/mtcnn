MTCNN
#####

.. image:: https://badge.fury.io/py/mtcnn.svg
    :target: https://badge.fury.io/py/mtcnn
.. image:: https://travis-ci.org/ipazc/mtcnn.svg?branch=master
    :target: https://travis-ci.org/ipazc/mtcnn


Implementation of the MTCNN face detector for TensorFlow in Python3.4+. It is written from scratch, using as a reference the implementation of
MTCNN from David Sandberg (`FaceNet's MTCNN <https://github.com/davidsandberg/facenet/tree/master/src/align>`_) in Facenet. It is based on the paper *Zhang, K et al. (2016)* [ZHANG2016]_.

.. image:: https://github.com/ipazc/mtcnn/raw/master/result.jpg


INSTALLATION
############

Currently it is only supported Python3.4 onwards. It can be installed through pip:

.. code:: bash

    $ pip3 install mtcnn

This implementation requires OpenCV>=4.1 and Tensorflow>=1.12.1 installed in the system, with bindings for Python3.

They can be installed through pip (if pip version >= 9.0.1):


.. code:: bash

    $ pip3 install tensorflow==1.12.1 opencv-contrib-python==4.1.0

or compiled directly from sources (`OpenCV4 <https://github.com/opencv/opencv/archive/4.1.0.zip>`_, `Tensorflow <https://www.tensorflow.org/install/install_sources>`_).

Note that a tensorflow-gpu version can be used instead if a GPU device is available on the system, which will speedup the results. It can be installed with pip:

.. code:: bash

    $ pip3 install tensorflow-gpu\>=1.12.0

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

The detector returns a list of JSON objects. Each JSON object contains three main keys: 'box', 'confidence' and 'keypoints':

- The bounding box is formatted as [x, y, width, height] under the key 'box'.
- The confidence is the probability for a bounding box to be matching a face.
- The keypoints are formatted into a JSON object with the keys 'left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right'. Each keypoint is identified by a pixel position (x, y).

Another good example of usage can be found in the file "`example.py`_." located in the root of this repository.

BENCHMARK
=========

The following tables shows the benchmark of this mtcnn implementation running on an `Intel i7-3612QM CPU @ 2.10GHz <https://www.cpubenchmark.net/cpu.php?cpu=Intel+Core+i7-3612QM+%40+2.10GHz>`_, with a **CPU-based** Tensorflow 1.4.1.

 - Pictures containing a single frontal face:

+------------+--------------+---------------+-----+
| Image size | Total pixels | Process time  | FPS |
+============+==============+===============+=====+
| 460x259    | 119,140      | 0.118 seconds | 8.5 |
+------------+--------------+---------------+-----+
| 561x561    | 314,721      | 0.227 seconds | 4.5 |
+------------+--------------+---------------+-----+
| 667x1000   | 667,000      | 0.456 seconds | 2.2 |
+------------+--------------+---------------+-----+
| 1920x1200  | 2,304,000    | 1.093 seconds | 0.9 |
+------------+--------------+---------------+-----+
| 4799x3599  | 17,271,601   | 8.798 seconds | 0.1 |
+------------+--------------+---------------+-----+

 - Pictures containing 10 frontal faces:

+------------+--------------+---------------+-----+
| Image size | Total pixels | Process time  | FPS |
+============+==============+===============+=====+
| 474x224    | 106,176      | 0.185 seconds | 5.4 |
+------------+--------------+---------------+-----+
| 736x348    | 256,128      | 0.290 seconds | 3.4 |
+------------+--------------+---------------+-----+
| 2100x994   | 2,087,400    | 1.286 seconds | 0.7 |
+------------+--------------+---------------+-----+

MODEL
#####

By default the MTCNN bundles a face detection weights model.

The model is adapted from the Facenet's MTCNN implementation, merged in a single file located inside the folder 'data' relative
to the module's path. It can be overriden by injecting it into the MTCNN() constructor during instantiation.

The model must be numpy-based containing the 3 main keys "pnet", "rnet" and "onet", having each of them the weights of each of the layers of the network.

For more reference about the network definition, take a close look at the paper from *Zhang et al. (2016)* [ZHANG2016]_.

LICENSE
#######

`MIT License`_.


REFERENCE
=========

.. [ZHANG2016] Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499–1503.

.. _example.py: example.py
.. _MIT license: LICENSE

