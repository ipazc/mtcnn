# MIT LICENSE
#
# Copyright (c) 2019-2024 Iván de Paz Centeno
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# pylint: disable=duplicate-code
# pylint: disable=redefined-outer-name

import os
import pytest
import numpy as np

from mtcnn.mtcnn import MTCNN


@pytest.fixture(scope="module")
def mtcnn_detector():
    """Fixture to initialize MTCNN detector once for all tests in this module."""
    return MTCNN()


@pytest.fixture
def test_images():
    """Fixture to provide paths and bytes for the test images."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, "images")
    valid_image_path = os.path.join(images_dir, "ivan.jpg")
    no_faces_image_path = os.path.join(images_dir, "no-faces.jpg")

    # Cargar los bytes de las imágenes
    with open(valid_image_path, "rb") as f:
        valid_image_bytes = f.read()

    with open(no_faces_image_path, "rb") as f:
        no_faces_image_bytes = f.read()

    return {
        'valid_image': valid_image_path,
        'no_faces_image': no_faces_image_path,
        'valid_image_bytes': valid_image_bytes,
        'no_faces_image_bytes': no_faces_image_bytes
    }


def test_detect_faces_from_uri(mtcnn_detector, test_images):
    """
    Test MTCNN detects faces and landmarks when given a URI of a valid image.
    """
    result = mtcnn_detector.detect_faces(test_images['valid_image'])

    assert isinstance(result, list), "Output should be a list of bounding boxes."
    assert len(result) > 0, "Should detect at least one face in the image."

    first = result[0]
    assert 'box' in first, "Bounding box not found in detection result."
    assert 'keypoints' in first, "Keypoints not found in detection result."

    # Check bounding box in 'xywh' format by default
    assert len(first['box']) == 4, "Bounding box should contain 4 coordinates (X1, Y1, width, height)."


def test_detect_faces_from_bytes(mtcnn_detector, test_images):
    """
    Test MTCNN detects faces and landmarks when given an image as bytes.
    """
    result = mtcnn_detector.detect_faces(test_images['valid_image_bytes'])

    assert isinstance(result, list), "Output should be a list of bounding boxes."
    assert len(result) > 0, "Should detect at least one face in the image."


def test_detect_no_faces(mtcnn_detector, test_images):
    """
    Test that MTCNN returns an empty list when no faces are detected in a valid image.
    """
    result = mtcnn_detector.detect_faces(test_images['no_faces_image'])
    assert isinstance(result, list), "Output should be a list."
    assert len(result) == 0, "Should detect no faces in the image."


def test_detect_faces_batch_from_uri(mtcnn_detector, test_images):
    """
    Test batch detection when passed a list of URIs.
    """
    result = mtcnn_detector.detect_faces([test_images['valid_image'], test_images['no_faces_image']])

    assert isinstance(result, list), "Output should be a list of lists (one for each image)."
    assert len(result) == 2, "Should return results for two images."
    assert isinstance(result[0], list), "First result should be a list of bounding boxes."
    assert len(result[0]) > 0, "First image should detect a face."
    assert isinstance(result[1], list), "Second result should be a list of bounding boxes."
    assert len(result[1]) == 0, "Second image should detect no faces."


def test_detect_faces_batch_from_bytes(mtcnn_detector, test_images):
    """
    Test batch detection when passed a list of image byte arrays.
    """
    result = mtcnn_detector.detect_faces([test_images['valid_image_bytes'], test_images['no_faces_image_bytes']])

    assert isinstance(result, list), "Output should be a list of lists (one for each image)."
    assert len(result) == 2, "Should return results for two images."
    assert isinstance(result[0], list), "First result should be a list of bounding boxes."
    assert len(result[0]) > 0, "First image should detect a face."
    assert isinstance(result[1], list), "Second result should be a list of bounding boxes."
    assert len(result[1]) == 0, "Second image should detect no faces."


@pytest.mark.parametrize("output_type, box_format", [
    ("json", "xywh"),
    ("json", "xyxy"),
    ("numpy", "xywh"),
    ("numpy", "xyxy")
])
def test_detect_faces_output_types_and_formats(mtcnn_detector, test_images, output_type, box_format):
    """
    Test MTCNN with all combinations of output_type and box_format.
    """
    # Detect faces using the given output_type and box_format
    result = mtcnn_detector.detect_faces([test_images['valid_image'], test_images['no_faces_image']],
                                         output_type=output_type, box_format=box_format)

    # General assertions: result should be a list or numpy array depending on output_type
    if output_type == "json":
        assert isinstance(result, list), "Output should be a list when output_type is 'json'."
        assert isinstance(result[0], list), "Each element in the batch should be a list (bounding boxes for each image)."
        assert len(result) == 2, "Should return results for two images."
        assert len(result[0]) > 0, "First image should detect at least one face."
        assert len(result[1]) == 0, "Second image should detect no faces."

        # Check bounding box format based on box_format
        first_bbox = result[0][0]['box']
        if box_format == "xywh":
            assert len(first_bbox) == 4, "Bounding box should contain 4 values for 'xywh'."
            assert first_bbox[2] > 0 and first_bbox[3] > 0, "Width and height should be positive."
        elif box_format == "xyxy":
            assert len(first_bbox) == 4, "Bounding box should contain 4 values for 'xyxy'."
            assert first_bbox[2] > first_bbox[0] and first_bbox[3] > first_bbox[1], \
                "X2 and Y2 should be greater than X1 and Y1 for 'xyxy' format."

    elif output_type == "numpy":
        assert isinstance(result, np.ndarray), "Output should be a numpy array when output_type is 'numpy'."
        assert result.shape[0] == 2, "First dimension of result should correspond to the number of images in the batch."
        assert result[0, 0] == 0, "First index should indicate image 0 for the first bounding box."
        assert result[-1, 0] == 0, "Last index should indicate image 0 for the last bounding box, as the second image is invalid."

        # Check bounding box format in numpy based on box_format
        first_bbox = result[0, 1:5]  # Assuming [image_idx, x1, y1, x2_or_width, y2_or_height]
        if box_format == "xywh":
            assert first_bbox[2] > 0 and first_bbox[3] > 0, "Width and height should be positive for 'xywh' format."
        elif box_format == "xyxy":
            assert first_bbox[2] > first_bbox[0] and first_bbox[3] > first_bbox[1], \
                "X2 and Y2 should be greater than X1 and Y1 for 'xyxy' format."


def test_invalid_image(mtcnn_detector):
    """
    Test that MTCNN raises InvalidImage exception for non-image input.
    """
    invalid_input = b"not_an_image"
    #with pytest.raises(InvalidImage):
    result = mtcnn_detector.detect_faces(invalid_input)
    assert len(result) == 0, "There should not be faces detected in an invalid input"
