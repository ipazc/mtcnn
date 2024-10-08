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

# pylint: disable=duplicate-code

import numpy as np

from mtcnn.network.onet import ONet

from mtcnn.utils.tensorflow import load_weights
from mtcnn.utils.images import extract_patches
from mtcnn.utils.bboxes import replace_confidence, adjust_bboxes, pick_matches, smart_nms_from_bboxes
from mtcnn.utils.landmarks import adjust_landmarks

from mtcnn.stages.base import StageBase


class StageONet(StageBase):
    """
    Stage for running the Output Network (ONet) of the MTCNN model. This stage refines the bounding box proposals
    generated by the RNet stage, adjusts the bounding boxes, predicts facial landmarks, and filters the results 
    using ONet's output.

    Args:
        stage_name (str): Name of the stage. Defaults to "Stage ONET".
        stage_id (int): Unique identifier for the stage. Defaults to 3.
        weights (str): Path to the weights file to load the model. Defaults to "onet.lz4".
    """

    def __init__(self, stage_name="Stage ONET", stage_id=3, weights="onet.lz4"):
        """
        Initializes the StageONet by loading the ONet model and setting the specified weights.

        Args:
            stage_name (str, optional): The name of the stage. Default is "Stage ONET".
            stage_id (int, optional): The ID for the stage. Default is 3.
            weights (str, optional): The file path to the weights for the ONet model. Default is "onet.lz4".
        """
        model = ONet()
        model.build()  # Building the ONet model
        model.set_weights(load_weights(weights))  # Load pre-trained weights

        super().__init__(stage_name=stage_name, stage_id=stage_id, model=model)

    def __call__(self, images_normalized, bboxes_batch, threshold_onet=0.8, nms_onet=0.7, **kwargs):
        """
        Runs the ONet stage on the input images and bounding boxes, refining the proposals generated by the RNet stage
        and adding facial landmarks prediction.

        Args:
            images_normalized (tf.Tensor): A tensor of normalized images with shape (batch_size, width, height, 3).
            bboxes_batch (np.ndarray): An array of bounding boxes produced by the RNet stage, each row representing 
                                       [image_id, x1, y1, x2, y2, confidence, landmark_x1, landmark_y1, ...].
            threshold_onet (float, optional): The confidence threshold for keeping bounding boxes after ONet refinement. Default is 0.8.
            nms_onet (float, optional): The IoU threshold for Non-Maximum Suppression after ONet refinement. Default is 0.7.
            **kwargs: Additional arguments passed to the function.

        Returns:
            np.ndarray: A numpy array of refined bounding boxes and landmarks after ONet processing, ready for the final stage.
        """
        # 1. Extract patches for each bounding box from the normalized images.
        # These patches are resized to the expected input size for ONet (48x48).
        patches = extract_patches(images_normalized, bboxes_batch, expected_size=(48, 48))

        # 2. Pass the extracted patches through ONet to get bounding box offsets, facial landmarks, and confidence scores.
        bboxes_offsets, face_landmarks, scores = self._model(patches)

        # 3. Adjust the landmarks to match the bounding box coordinates relative to the original image.
        face_landmarks = adjust_landmarks(face_landmarks, bboxes_batch)

        # 4. Replace the confidence of the bounding boxes with the ones provided by ONet.
        bboxes_batch = replace_confidence(bboxes_batch, scores)

        # 5. Adjust the bounding boxes using the offsets predicted by ONet (refinement of the proposals).
        bboxes_batch = adjust_bboxes(bboxes_batch, bboxes_offsets)

        # 6. Combine the facial landmarks with the bounding boxes batch tensor.
        bboxes_batch = np.concatenate([bboxes_batch, face_landmarks], axis=-1)

        # 7. Filter out bounding boxes based on the new confidence scores and the threshold set for ONet.
        bboxes_batch = pick_matches(bboxes_batch, scores_column=5, score_threshold=threshold_onet)

        # 8. Apply Non-Maximum Suppression (NMS) to remove overlapping boxes based on the refined boxes, scores, and landmarks.
        bboxes_batch = smart_nms_from_bboxes(bboxes_batch, threshold=nms_onet, method="min", initial_sort=True)

        return bboxes_batch
