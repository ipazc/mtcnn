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

import tensorflow as tf
import numpy as np

from mtcnn.stages import StagePNet, StageRNet, StageONet

from mtcnn.utils.images import load_images_batch, standarize_batch
from mtcnn.utils.bboxes import fix_bboxes_offsets, limit_bboxes, to_json


COMMON_STAGES = {
    "face_detection_only": [StagePNet, StageRNet],
    "face_and_landmarks_detection": [StagePNet, StageRNet, StageONet],
}


class MTCNN:
    """
    MTCNN class for detecting faces and landmarks through configurable stages.
    This structure allows skipping certain stages to optimize performance based on the user's needs.

    Args:
        stages (str or list, optional): Defines the pipeline stages. It can be a string to choose from predefined
                                        configurations or a list of stage classes or instances.
                                        Options: "face_detection_only", "face_and_landmarks_detection".
                                        Default is "face_and_landmarks_detection".
        device (str, optional): The device where the model will be run. Can be "CPU:0", "GPU:0", "GPU:1", ...
                                Default is "CPU:0".
                                
    """

    def __init__(self, stages="face_and_landmarks_detection", device="CPU:0"):
        if isinstance(stages, str):
            if stages not in COMMON_STAGES:
                raise ValueError(f"Invalid stages option: {stages}. Must be one of {list(COMMON_STAGES.keys())}.")
            stages = COMMON_STAGES[stages]

        # Instantiate stages if necessary (can pass already instantiated stages too)
        self._stages = [stage() if isinstance(stage, type) else stage for stage in stages]
        self._device = device

    @property
    def device(self):
        """Returns the device where the algorithm is executed"""
        return self._device

    @property
    def stages(self):
        """Returns the list of pipeline stages."""
        return self._stages

    def get_stage(self, stage_id=None, stage_name=None):
        """
        Retrieves a stage by its ID or name.

        Args:
            stage_id (int, optional): The ID of the stage.
            stage_name (str, optional): The name of the stage.

        Returns:
            The matching stage if found, otherwise None.
        """
        for stage in self._stages:
            if stage.id == stage_id or stage.name == stage_name:
                return stage

        return None

    def predict(self, image, fit_to_image=True, limit_boundaries_landmarks=False, box_format="xywh", output_type="json", postprocess=True,
                **kwargs):
        """
        Alias for detect_faces().
        """
        return self.detect_faces(image, fit_to_image=fit_to_image, limit_boundaries_landmarks=limit_boundaries_landmarks,
                                 box_format=box_format, output_type=output_type, postprocess=postprocess, **kwargs)

    def detect_faces(self, image, fit_to_image=True, limit_boundaries_landmarks=False, box_format="xywh", output_type="json",
                     postprocess=True, batch_stack_justification="center", **kwargs):
        """
        Runs face detection on a single image or batch of images through the configured stages.

        Args:
            image (str, bytes, np.ndarray or tf.Tensor or list): The input image or batch of images.
                                                                It can be a file path, a tensor, or raw bytes.
            fit_to_image (bool, optional): Whether to fit bounding boxes and landmarks within image boundaries. Default is True.
            limit_boundaries_landmarks (bool, optional): Whether to ensure landmarks stay within image boundaries. Default is False.            
            box_format (str, optional): The format of the bounding box. Can be "xywh" for [X1, Y1, width, height] or "xyxy" for [X1, Y1, X2, Y2]. 
                                        Default is "xywh".
            output_type (str, optional): The output format. Can be "json" for dictionary output or "numpy" for numpy array output. Default is "json".
            postprocess (bool, optional): Flag to enable postprocessing. The postprocessing includes functionality affected by `fit_to_image`, 
                                          `limit_boundaries_landmarks` and removing padding effects caused by batching images with different shapes.
            batch_stack_justification (str, optional): The justification of the smaller images w.r.t. the largest images when 
                                                       stacking in batch processing, which requires padding smaller images to the size of the 
                                                       biggest one. 
            **kwargs: Additional parameters passed to the stages. The following parameters are used:

                - **StagePNet**:
                    - min_face_size (int, optional): The minimum size of a face to detect. Default is 20.
                    - min_size (int, optional): The minimum size to start the image pyramid. Default is 12.
                    - scale_factor (float, optional): The scaling factor for the image pyramid. Default is 0.709.
                    - threshold_pnet (float, optional): The confidence threshold for proposals from PNet. Default is 0.6.
                    - nms_pnet1 (float, optional): The IoU threshold for the first round of NMS per scale. Default is 0.5.
                    - nms_pnet2 (float, optional): The IoU threshold for the second round of NMS across all scales. Default is 0.7.
                    
                - **StageRNet**:
                    - threshold_rnet (float, optional): Confidence threshold for RNet proposals. Default is 0.7.
                    - nms_rnet (float, optional): IoU threshold for Non-Maximum Suppression in RNet. Default is 0.7.

                - **StageONet**:
                    - threshold_onet (float, optional): Confidence threshold for ONet proposals. Default is 0.8.
                    - nms_onet (float, optional): IoU threshold for Non-Maximum Suppression in ONet. Default is 0.7.
            
        Returns:
            list or list of lists: A list of detected faces (in case a single image) or a list of lists of detected faces 
                                   (one per image in the batch). If the stages are `face_and_landmarks_detection`, 
                                   the output will have the detected faces and landmarks in JSON format. 
                                   In case of `face_detection_only`, only the bounding boxes will be provided in 
                                   JSON format.
        """
        return_tensor = output_type == "numpy"
        as_width_height = box_format == "xywh"

        is_batch = isinstance(image, list)
        images = image if is_batch else [image]

        with tf.device(self._device):
            # Load the images into memory and normalize them into a single tensor
            try:
                images_raw = load_images_batch(images)
                images_normalized, images_oshapes, pad_param = standarize_batch(images_raw,
                                                                                justification=batch_stack_justification,
                                                                                normalize=True)

                bboxes_batch = None

                # Process images through each stage (PNet, RNet, ONet)
                for stage in self.stages:
                    bboxes_batch = stage(bboxes_batch=bboxes_batch, images_normalized=images_normalized, images_oshapes=images_oshapes, **kwargs)

            except tf.errors.InvalidArgumentError:  # No faces found
                bboxes_batch = np.empty((0, 16))
                pad_param = None

            if postprocess and pad_param is not None:
                # Adjust bounding boxes and landmarks to account for padding offsets
                bboxes_batch = fix_bboxes_offsets(bboxes_batch, pad_param)

                # Optionally, limit the bounding boxes and landmarks to stay within image boundaries
                if fit_to_image:
                    bboxes_batch = limit_bboxes(bboxes_batch, images_shapes=images_oshapes, limit_landmarks=limit_boundaries_landmarks)

            # Convert bounding boxes and landmarks to JSON format if required
            if return_tensor:
                result = bboxes_batch

                if as_width_height:
                    result[:, 3] = result[:, 3] - result[:, 1]
                    result[:, 4] = result[:, 4] - result[:, 2]

            else:
                result = to_json(bboxes_batch,
                                 images_count=len(images),
                                 output_as_width_height=as_width_height,
                                 input_as_width_height=False)
                result = result[0] if (not is_batch and len(result) > 0) else result

        return result
