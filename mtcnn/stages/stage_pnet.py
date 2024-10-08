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

from mtcnn.network.pnet import PNet

from mtcnn.utils.tensorflow import load_weights
from mtcnn.utils.images import build_scale_pyramid, apply_scales
from mtcnn.utils.bboxes import generate_bounding_box, upscale_bboxes, smart_nms_from_bboxes, resize_to_square

from mtcnn.stages.base import StageBase


class StagePNet(StageBase):
    """
    Stage for running the Proposal Network (PNet) of the MTCNN model. This stage takes images, builds 
    image pyramids for different scales, and runs PNet to generate bounding box proposals, applying 
    Non-Maximum Suppression (NMS) to filter overlapping boxes.

    Args:
        stage_name (str): Name of the stage. Defaults to "Stage PNET".
        stage_id (int): Unique identifier for the stage. Defaults to 1.
        weights (str): Path to the weights file to load the model. Defaults to "pnet.lz4".
    """

    def __init__(self, stage_name="Stage PNET", stage_id=1, weights="pnet.lz4"):
        """
        Initializes the StagePNet by loading the PNet model and setting the specified weights.
        
        Args:
            stage_name (str, optional): The name of the stage. Default is "Stage PNET".
            stage_id (int, optional): The ID for the stage. Default is 1.
            weights (str, optional): The file path to the weights for the PNet model. Default is "pnet.lz4".
        """
        model = PNet()
        model.build()  # Building the model (no need to specify input shape if default is provided)
        model.set_weights(load_weights(weights))  # Load pre-trained weights

        super().__init__(stage_name=stage_name, stage_id=stage_id, model=model)

    def __call__(self, images_normalized, images_oshapes, min_face_size=20, min_size=12, scale_factor=0.709,
                 threshold_pnet=0.6, nms_pnet1=0.5, nms_pnet2=0.7, **kwargs):
        """
        Runs the PNet stage on a batch of images to generate bounding box proposals.

        Args:
            images_normalized (tf.Tensor): A tensor of normalized images with shape (batch_size, width, height, 3). 
                                           These images are padded to match the size of the largest image in the batch.
            images_oshapes (tf.Tensor): A tensor containing the original shapes of the images, with shape (batch_size, 3).
            min_face_size (int, optional): The minimum size of a face to detect. Default is 20.
            min_size (int, optional): The minimum size to start the image pyramid. Default is 12.
            scale_factor (float, optional): The scaling factor for the image pyramid. Default is 0.709.
            threshold_pnet (float, optional): The confidence threshold for proposals from PNet. Default is 0.6.
            nms_pnet1 (float, optional): The IoU threshold for the first round of NMS per scale. Default is 0.5.
            nms_pnet2 (float, optional): The IoU threshold for the second round of NMS across all scales. Default is 0.7.
            **kwargs: Additional arguments passed to the function.

        Returns:
            np.ndarray: A numpy array of bounding boxes after NMS and resizing to square, ready for the next stage.
        """
        # 1. Build the pyramid scale for every image based on the size and scale factor
        scales_groups = [build_scale_pyramid(shape[1], shape[0], min_face_size=min_face_size, scale_factor=scale_factor)
                         for shape in images_oshapes]

        # 2. Apply the scales to normalized images
        scales_result, scales_index = apply_scales(images_normalized, scales_groups)
        batch_size = images_normalized.shape[0]

        # 3. Get proposals bounding boxes and confidence from the model (PNet)
        pnet_result = [self._model(s) for s in scales_result]

        # 4. Generate bounding boxes per scale group
        bboxes_proposals = [generate_bounding_box(result[0], result[1], threshold_pnet) for result in pnet_result]
        bboxes_batch_upscaled = [upscale_bboxes(bbox, np.asarray([scale] * batch_size)) for bbox, scale in zip(bboxes_proposals, scales_index)]

        # 5. Apply Non-Maximum Suppression (NMS) per scale group
        bboxes_nms = [smart_nms_from_bboxes(b, threshold=nms_pnet1, method="union", initial_sort=False) for b in bboxes_batch_upscaled]

        # 6. Concatenate and apply NMS again across all scales
        bboxes_batch = np.concatenate(bboxes_nms, axis=0) if len(bboxes_nms) > 0 else np.empty((0, 6))
        bboxes_batch = smart_nms_from_bboxes(bboxes_batch, threshold=nms_pnet2, method="union", initial_sort=True)

        # 7. Resize bounding boxes to square format
        bboxes_batch = resize_to_square(bboxes_batch)

        return bboxes_batch
