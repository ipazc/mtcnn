# MIT License
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

import numpy as np


def adjust_landmarks(face_landmarks, bboxes_batch):
    """
    Adjusts face landmark coordinates to align them with the corresponding bounding boxes.
    
    The face landmarks are scaled and shifted based on the width and height of the bounding boxes
    to ensure that they are correctly aligned with the face locations.

    Args:
        face_landmarks (np.ndarray or tf.Tensor): An array of shape (n, 10) where each row represents 
                                                  the normalized coordinates of 5 facial landmarks 
                                                  (x1, x2, x3, x4, x5, y1, y2, y3, y4, y5). These 
                                                  coordinates are normalized between 0 and 1.
        bboxes_batch (np.ndarray): An array of bounding boxes of shape (n, m), where each row contains
                                   the bounding box coordinates [x1, y1, x2, y2] and possibly additional
                                   columns (e.g., image id, confidence score).

    Returns:
        np.ndarray: The adjusted face landmark coordinates, now scaled and positioned relative to the 
                    corresponding bounding boxes.
    """
    # Convert face_landmarks to a NumPy array and make a copy
    face_landmarks = face_landmarks.numpy().copy()

    # Compute the width and height of each bounding box
    w = bboxes_batch[:, 3:4] - bboxes_batch[:, 1:2] + 1  # Width
    h = bboxes_batch[:, 4:5] - bboxes_batch[:, 2:3] + 1  # Height

    # Adjust the x-coordinates of the landmarks
    face_landmarks[:, 0:5] = w * face_landmarks[:, 0:5] + bboxes_batch[:, 1:2] - 1
    # Adjust the y-coordinates of the landmarks
    face_landmarks[:, 5:10] = h * face_landmarks[:, 5:10] + bboxes_batch[:, 2:3] - 1

    return face_landmarks


def parse_landmarks(landmarks):
    """
    Parses facial landmarks from different input formats (dict or np.ndarray) into a standardized format.
    
    The landmarks can be provided as a dictionary or an ndarray. If a dictionary is used, it should contain
    a 'keypoints' field. If an ndarray is used, it should contain either 10 or 16 values depending on the 
    number of keypoints and format.

    Args:
        landmarks (dict or np.ndarray): Facial landmarks, either as a dictionary with key 'keypoints' or 
                                        as a numpy array of shape (10,) or (16,).
    
    Returns:
        dict: A dictionary containing the facial landmarks with keys: 'nose', 'mouth_right', 'right_eye',
              'left_eye', 'mouth_left'. Each key corresponds to the (x, y) coordinates of that keypoint.
    """
    if isinstance(landmarks, dict):
        if 'keypoints' in landmarks:
            landmarks = landmarks['keypoints']  # Extract 'keypoints' from dict

    if isinstance(landmarks, np.ndarray):
        offset = 0 if landmarks.shape[0] == 10 else 6  # Handle different landmark formats
        landmarks = landmarks.round().astype(int)  # Round coordinates and convert to integers
        landmarks = {
            "nose": [landmarks[offset+2], landmarks[offset+7]],
            "mouth_right": [landmarks[offset+4], landmarks[offset+9]],
            "right_eye": [landmarks[offset+1], landmarks[offset+6]],
            "left_eye": [landmarks[offset+0], landmarks[offset+5]],
            "mouth_left": [landmarks[offset+3], landmarks[offset+8]]
        }

    return landmarks
