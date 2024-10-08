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

from mtcnn.utils.landmarks import parse_landmarks


def generate_bounding_box(bbox_reg, bbox_class, threshold_face, strides=2, cell_size=12):
    """
    Generates bounding boxes for detected objects (e.g., faces) based on the class and regression outputs of a model,
    supporting batch input.
    
    Args:
        bbox_reg (tf.Tensor): Bounding box regression predictions with shape (batch_size, height, width, 4).
                              This contains adjustments to apply to the initial bounding box positions for each image in the batch.
        bbox_class (tf.Tensor): Class predictions (e.g., face/non-face) of shape (batch_size, height, width, 2),
                                where the second channel corresponds to the probability of a face being present.
        threshold_face (float): A threshold between 0 and 1 that determines if a detection is considered a face or not.
                                Bounding boxes are only generated for detections with probabilities greater than this value.
        strides (int, optional): The step size (in pixels) used to slide the detection window over the image. Default is 2.
        cell_size (int, optional): The size of the sliding window (in pixels) used to detect faces. Default is 12.
    
    Returns:
        np.ndarray: An array of bounding boxes for the entire batch, where each box is represented as 
                    [batch_index, x1, y1, x2, y2, confidence].
                    The `batch_index` indicates which image in the batch the bounding box belongs to.
    """
    bbox_reg = bbox_reg.numpy()
    bbox_class = bbox_class.numpy()

    # Create a mask for detected faces based on the threshold for face probability
    confidence_score = bbox_class[:,:,:,1]

    # Find the indices where the detection mask is true (i.e., face detected)
    index_bboxes = np.stack(np.where(confidence_score > threshold_face)) # batch_size, y, x
    filtered_bbox_reg = np.transpose(bbox_reg[index_bboxes[0], index_bboxes[1], index_bboxes[2]], (1,0))

    # Extract the regression values
    reg_x1, reg_y1, reg_x2, reg_y2 = filtered_bbox_reg

    # Convert strides and cell size into arrays for easy broadcasting
    strides = np.asarray([[1], [strides], [strides]])
    cellsize = [np.asarray([[0], [1], [1]]), np.asarray([[0], [cell_size], [cell_size]])]

    # Calculate the top-left and bottom-right corners of the bounding boxes
    bbox_up_left = index_bboxes * strides + cellsize[0]
    bbox_bottom_right = index_bboxes * strides + cellsize[1]

    # Calculate width and height for the bounding boxes
    reg_w = bbox_bottom_right[2] - bbox_up_left[2]  # width of bounding box
    reg_h = bbox_bottom_right[1] - bbox_up_left[1]  # height of bounding box

    # Apply the regression to adjust the bounding box coordinates
    x1 = bbox_up_left[2] + reg_x1 * reg_w  # Adjusted x1
    y1 = bbox_up_left[1] + reg_y1 * reg_h  # Adjusted y1
    x2 = bbox_bottom_right[2] + reg_x2 * reg_w  # Adjusted x2
    y2 = bbox_bottom_right[1] + reg_y2 * reg_h  # Adjusted y2

    # Concatenate the bounding box coordinates and detection information, keeping batch index
    bboxes_result = np.stack([
        index_bboxes[0], x1, y1, x2, y2, confidence_score[index_bboxes[0], index_bboxes[1], index_bboxes[2]]
    ], axis=0).T

    # Sort bounding boxes by score in descending order
    bboxes_result = sort_by_scores(bboxes_result, scores=bboxes_result[:, -1], ascending=False)

    return bboxes_result


def upscale_bboxes(bboxes_result, scales):
    """
    Upscales bounding boxes to their original size based on the scaling factors applied during image resizing,
    supporting batch input.
    
    Args:
        bboxes_result (np.ndarray): Array of bounding boxes, where each box is represented as 
                                    [batch_index, x1, y1, x2, y2, confidence, reg_x1, reg_y1, reg_x2, reg_y2].
        scales (np.ndarray): Array of scaling factors used during image resizing, typically one scale per image or detection.
                             The shape of `scales` should be (batch_size,), where each entry corresponds to the scale applied to an 
                             image in the batch.
    
    Returns:
        np.ndarray: The input bounding boxes, but with the coordinates scaled back to the original image dimensions, 
                    adjusted for each image in the batch according to its respective scale.
    """

    # Broadcast the scales to match the shape of the bounding boxes, ensuring the correct scale is applied to each batch entry
    scales_bcast = np.expand_dims(scales[bboxes_result[:,0].astype(int)], axis=-1)

    # Scale the bounding box coordinates (x1, y1, x2, y2) back to the original image size
    bboxes_result[:,1:5] = bboxes_result[:,1:5] / scales_bcast

    return bboxes_result


def iou(bboxes, method="union"):
    """
    Computes the Intersection over Union (IoU) for a set of bounding boxes based on the specified method ("union" or "min").

    Args:
        bboxes (list or np.ndarray): List or array of bounding boxes, where each bounding box is represented as
                                     [row1, col1, row2, col2] (coordinates of the top-left and bottom-right corners).
        method (str, optional): Method to compute the IoU. Options are:
                                - "union": Computes IoU based on the union of the bounding boxes.
                                - "min": Computes IoU based on the minimum area of the bounding boxes.
                                Default is "union".

    Returns:
        np.ndarray: A matrix of shape (N, N) where each element [i, j] represents the IoU between the i-th and j-th bounding box.
                    The matrix is symmetric, with diagonal elements equal to 1 (IoU of a box with itself).
    """

    # Convert the list of bounding boxes to a NumPy array
    bboxes = np.stack(bboxes, axis=0)

    # Calculate the area of each bounding box
    area_bboxes = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    # Expand dimensions to compute pairwise IoU (N x N matrix)
    bboxes_a = np.expand_dims(bboxes, axis=0)
    bboxes_b = np.expand_dims(bboxes, axis=1)

    # Calculate the intersection coordinates
    row_inter_top = np.maximum(bboxes_a[:, :, 0], bboxes_b[:, :, 0])
    col_inter_left = np.maximum(bboxes_a[:, :, 1], bboxes_b[:, :, 1])
    row_inter_bottom = np.minimum(bboxes_a[:, :, 2], bboxes_b[:, :, 2])
    col_inter_right = np.minimum(bboxes_a[:, :, 3], bboxes_b[:, :, 3])

    # Calculate the intersection area
    height_inter = np.maximum(0, row_inter_bottom - row_inter_top)
    width_inter = np.maximum(0, col_inter_right - col_inter_left)
    area_inter = height_inter * width_inter

    # Compute IoU based on the specified method
    if method == "union":
        # Union: Area of A + Area of B - Intersection
        area_union = area_bboxes[:, None] + area_bboxes[None, :] - area_inter
        iou_matrix = area_inter / area_union
    elif method == "min":
        # Minimum: Area of the smaller box between A and B
        area_min = np.minimum(area_bboxes[:, None], area_bboxes[None, :])
        iou_matrix = area_inter / area_min
    else:
        raise ValueError("Method should be either 'union' or 'min'.")

    return iou_matrix


def sort_by_scores(tensor, scores, ascending=True):
    """
    Sorts a tensor based on an array of scores, either in ascending or descending order.

    Args:
        tensor (np.ndarray): Tensor of shape (N, ...) where N is the number of elements to sort.
        scores (np.ndarray): Array of shape (N,) containing scores associated with each element in the tensor.
        ascending (bool, optional): Whether to sort in ascending order. Default is True (ascending).

    Returns:
        np.ndarray: The tensor sorted according to the scores.
    """

    # Get the sorted indices based on the scores
    sorted_indices = np.argsort(scores)

    # Sort the tensor using the sorted indices, reversing if descending
    sorted_tensor = tensor[sorted_indices[::(-2 * int(not ascending) + 1)]]

    return sorted_tensor


def nms(target_iou, threshold):
    """
    Performs Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes based on the IoU threshold.

    Args:
        target_iou (np.ndarray): A square IoU matrix of shape (N, N) where each element [i, j] represents the IoU
                                 between the i-th and j-th bounding box.
        threshold (float): IoU threshold above which boxes are considered to overlap too much and will be suppressed.

    Returns:
        np.ndarray: Array of indices of bounding boxes that are kept after NMS.
    """

    # Step 1: Create a mask for allowed comparisons (upper triangular part of the IoU matrix, excluding the diagonal)
    allowed_mask = np.triu(np.ones((target_iou.shape[0], target_iou.shape[0])), k=1)

    # Step 2: Create a mask for failed comparisons (IoU above the threshold)
    failed_mask = (target_iou > threshold).astype(int)

    # Step 3: Combine the masks and get the indices of the remaining boxes
    result_indexes = np.where((failed_mask * allowed_mask).sum(axis=0) == 0)[0]

    return result_indexes


def smart_nms_from_bboxes(bboxes, threshold, column_image_id=0, columns_bbox=slice(1, 5, None), column_confidence=5,
                          method="union", initial_sort=True):
    """
    Applies Non-Maximum Suppression (NMS) to a set of bounding boxes grouped by image ID.

    Args:
        bboxes (np.ndarray): Array of bounding boxes, where each box is represented as
                             [image_id, row1, col1, row2, col2, score].
        threshold (float): IoU threshold for NMS. Bounding boxes with IoU higher than this value will be suppressed.
        column_image_id (int): Column position in the array indicating the image id.
        columns_bbox (slice): Slice of columns containing the BBox coords.
        method (str, optional): Method for IoU calculation. Can be "union" or "min". Default is "union".
        column_confidence (int): Column containing the value of confidence for each bbox.
        initial_sort (bool): True to sort bboxes by confidence value. False otherwise.

    Returns:
        A np.ndarray containing the filtered bboxes, image-wise.
        dict: A dictionary where keys are `image_id` and values are arrays of indices of bounding boxes that are
              kept after NMS for each image.
    """
    # Step 0: Sort if required
    if initial_sort:
        bboxes = sort_by_scores(bboxes, scores=bboxes[:, column_confidence], ascending=False)

    # Step 1: Get unique image IDs
    image_ids = np.unique(bboxes[:, 0])

    result = []

    # Step 2: Apply NMS per image
    for image_id in image_ids:
        # Filter bounding boxes for the current image
        target_bboxes = bboxes[bboxes[:, column_image_id] == image_id]

        # Compute the IoU matrix for the bounding boxes
        target_iou = iou(target_bboxes[:, columns_bbox], method=method)

        # Perform NMS and get the indices of the boxes to keep
        target_indexes = nms(target_iou, threshold)

        # Filter the boxes for the image
        target_filtered_bboxes = target_bboxes[target_indexes.astype(int)]

        # Store the result
        result.append(target_filtered_bboxes)

    result = np.concatenate(result, axis=0) if len(result) > 0 else np.empty((0, 6))

    return result


def resize_to_square(bboxes):
    """
    Adjusts bounding boxes to be square by resizing them based on their largest dimension 
    (width or height). The bounding boxes are resized by expanding the smaller dimension
    to match the larger one while keeping the center of the box intact.

    Args:
        bboxes (np.ndarray): An array of bounding boxes of shape (n, 5), where each row
                             represents [batch_index, x1, y1, x2, y2].

    Returns:
        np.ndarray: An array of bounding boxes adjusted to be square, maintaining their center positions.
    """
    bboxes = bboxes.copy()
    h = bboxes[:, 4] - bboxes[:, 2]  # Height of each bounding box
    w = bboxes[:, 3] - bboxes[:, 1]  # Width of each bounding box
    largest_size = np.maximum(w, h)  # Largest dimension (width or height)

    # Adjust x1 and y1 to center the bounding box and resize to square
    bboxes[:, 1] = bboxes[:, 1] + w * 0.5 - largest_size * 0.5
    bboxes[:, 2] = bboxes[:, 2] + h * 0.5 - largest_size * 0.5
    bboxes[:, 3:5] = bboxes[:, 1:3] + np.tile(largest_size, (2, 1)).T  # Resize x2, y2

    return bboxes


def replace_confidence(bboxes_batch, new_scores):
    """
    Replaces the confidence scores of bounding boxes with new scores provided.

    Args:
        bboxes_batch (np.ndarray): An array of bounding boxes of shape (n, m), where each row
                                   contains the bounding box coordinates and the confidence score.
                                   The confidence score is expected to be in the last column.
        new_scores (np.ndarray): An array of new confidence scores of shape (n, m), where the 
                                 confidence score is also expected to be in the last column.

    Returns:
        np.ndarray: The bounding boxes array with updated confidence scores from `new_scores`.
    """
    bboxes_batch[:, -1] = new_scores[:, -1]
    return bboxes_batch


def adjust_bboxes(bboxes_batch, bboxes_offsets):
    """
    Adjusts the bounding box coordinates by applying the provided offsets.
    
    The offsets are applied to resize and shift the bounding boxes based on their width and height.

    Args:
        bboxes_batch (np.ndarray): An array of bounding boxes of shape (n, m), where each row contains
                                   the batch index, bounding box coordinates [x1, y1, x2, y2], and
                                   potentially additional data such as scores.
        bboxes_offsets (np.ndarray): An array of offsets for adjusting the bounding boxes. The shape should be
                                     (n, 4), where each row contains offsets for [dx1, dy1, dx2, dy2].

    Returns:
        np.ndarray: The adjusted bounding boxes with updated coordinates, maintaining any additional columns 
                    beyond the bounding box coordinates (such as scores).
    """
    bboxes_batch = bboxes_batch.copy()
    w = bboxes_batch[:, 3] - bboxes_batch[:, 1] + 1  # Calculate width of each bounding box
    h = bboxes_batch[:, 4] - bboxes_batch[:, 2] + 1  # Calculate height of each bounding box

    sizes = np.stack([w, h, w, h], axis=-1)  # Stack width and height to match bbox_offsets
    bboxes_batch[:, 1:5] += bboxes_offsets * sizes  # Apply offsets to the coordinates

    return bboxes_batch


def pick_matches(bboxes_batch, scores_column=-1, score_threshold=0.7):
    """
    Filters bounding boxes based on the confidence score threshold.
    
    Only bounding boxes with a confidence score higher than the specified threshold are returned.

    Args:
        bboxes_batch (np.ndarray): An array of bounding boxes of shape (n, m), where each row contains
                                   bounding box coordinates and confidence scores. The confidence scores 
                                   are expected to be in the column specified by `scores_column`.
        scores_column (int): The index of the column that contains the confidence scores. Default is -1 (last column).
        score_threshold (float): The minimum confidence score threshold to select bounding boxes. 
                                 Default is 0.7.

    Returns:
        np.ndarray: An array of bounding boxes that have confidence scores greater than `score_threshold`.
    """
    return bboxes_batch[np.where(bboxes_batch[:, scores_column] > score_threshold)[0]]


def to_json(bboxes_batch, images_count, input_as_width_height=False, output_as_width_height=True):
    """
    Converts a batch of bounding boxes and facial keypoints into a JSON-friendly format.
    
    This function processes the bounding boxes grouped by unique image IDs, and formats each bounding box
    and its associated keypoints (facial landmarks) into a dictionary structure suitable for JSON serialization.
    
    Args:
        bboxes_batch (np.ndarray): An array of shape (n, 16) where each row represents a bounding box 
                                   and associated keypoints in the following format:
                                   [image_id, x1, y1, x2, y2, confidence, left_eye_x, left_eye_y, right_eye_x, 
                                   right_eye_y, nose_x, nose_y, mouth_left_x, mouth_left_y, mouth_right_x, mouth_right_y].
        images_count (int): Number of different images composed by the batch.
        input_as_width_height (bool, optional): True if format of input bounding boxes is [x1, x2, width, height].
                                                 False if format is [x1, y1, x2, y2].
        output_as_width_height (bool, optional): True to format bounding boxes as [x1, x2, width, height].
                                                 False to format as [x1, y1, x2, y2].
        
    Returns:
        list: A list of lists, where each inner list contains dictionaries for bounding boxes and keypoints 
              for a specific image. Each dictionary has the following structure:
              {
                "box": [x, y, width, height],
                "keypoints": {
                    "nose": [nose_x, nose_y],
                    "mouth_right": [mouth_right_x, mouth_right_y],
                    "right_eye": [right_eye_x, right_eye_y],
                    "left_eye": [left_eye_x, left_eye_y],
                    "mouth_left": [mouth_left_x, mouth_left_y]
                },
                "confidence": confidence_score
              }
    """
    single_element = len(bboxes_batch.shape) == 1

    if single_element:
        bboxes_batch = np.expand_dims(bboxes_batch, axis=0)

    #unique_ids = np.unique(bboxes_batch[:, 0])

    result_batch = []

    # Loop over each unique image ID
    for unique_id in range(images_count):
        result = []
        bboxes_subset = bboxes_batch[bboxes_batch[:, 0] == unique_id]

        # Loop over each bounding box in the subset
        for bbox in bboxes_subset:
            row = {
                "box": parse_bbox(bbox, 
                                  output_as_width_height=output_as_width_height,
                                  input_as_width_height=input_as_width_height).tolist(),
                "confidence": bbox[5]
            }
            result.append(row)

            # If the stages combination allows landmarks, then we append them. Otherwise we don't
            try:
                row["keypoints"] = parse_landmarks(bbox)
            except IndexError:
                pass

        result_batch.append(result)

    return result_batch


def limit_bboxes(bboxes_batch, images_shapes, limit_landmarks=True):
    """
    Adjusts bounding boxes so that they fit within the boundaries of their corresponding images.
    If any bounding box exceeds the image dimensions, it will be corrected to stay within the limits.

    Args:
        bboxes_batch (np.ndarray): An array of bounding boxes of shape (n, 5), where each row
                                   represents [batch_index, x1, y1, x2, y2].
        images_shapes (np.ndarray): A tensor of image shapes of shape (batch, 3), where each row
                                    represents [width, height, channels] of each image in the batch.
        limit_landmarks (bool): A flag to specify whether the limit should also apply to landmarks or not.

    Returns:
        np.ndarray: The adjusted bounding boxes where no coordinate exceeds the image dimensions.
    """
    bboxes_batch_fitted = bboxes_batch.copy()

    # Get the original shapes (height, width) for each image in the batch
    expected_shapes = images_shapes[bboxes_batch_fitted[:, 0].astype(int)]

    # Adjust x1 and x2 to be within [0, width-1]
    bboxes_batch_fitted[:, 1] = np.minimum(np.maximum(bboxes_batch_fitted[:, 1], 0), expected_shapes[:, 1] - 1)
    bboxes_batch_fitted[:, 3] = np.minimum(np.maximum(bboxes_batch_fitted[:, 3], 0), expected_shapes[:, 1] - 1)

    # Adjust y1 and y2 to be within [0, height-1]
    bboxes_batch_fitted[:, 2] = np.minimum(np.maximum(bboxes_batch_fitted[:, 2], 0), expected_shapes[:, 0] - 1)
    bboxes_batch_fitted[:, 4] = np.minimum(np.maximum(bboxes_batch_fitted[:, 4], 0), expected_shapes[:, 0] - 1)

    if limit_landmarks:
        # Adjust x1..x5 of the landmarks to not surpass boundaries
        bboxes_batch_fitted[:, 6:11] = np.minimum(np.maximum(bboxes_batch_fitted[:, 6:11], 0), expected_shapes[:, 1:2] - 1)

        # Adjust y1..y5 of the landmarks to not surpass boundaries
        bboxes_batch_fitted[:, 11:16] = np.minimum(np.maximum(bboxes_batch_fitted[:, 11:16], 0), expected_shapes[:, 0:1] - 1)

    return bboxes_batch_fitted


def parse_bbox(bbox, output_as_width_height=True, input_as_width_height=True):
    """
    Parses a bounding box from different formats (dict, list, or ndarray) into a standardized format.
    
    Args:
        bbox (dict, list, np.ndarray): Bounding box in one of the following formats:
                                       - dict with key 'box': [x1, y1, x2, y2]
                                       - list: [x1, y1, x2, y2] or [x1, y1, width, height]
                                       - np.ndarray: Shape (4,) or (5,) where the first value might be an index.
        output_as_width_height (bool): Whether to return the bounding box as [x1, y1, width, height] (default True) or [x1, y1, x2, y2] if False.
        input_as_width_height (bool): Whether the input format of the bounding box is [x1, y1, width, height] (default True) or 
                                      [x1, y1, x2, y2] if False.
                                
    
    Returns:
        np.ndarray: Parsed bounding box in format [x1, y1, width, height] or [x1, y1, x2, y2].
    """
    # Extract box if input is a dict
    if isinstance(bbox, dict):
        bbox = bbox['box']

    # Parse list format
    if isinstance(bbox, list):
        x1, y1, width, height = bbox

        if not input_as_width_height:
            width = width - x1
            height = height - y1

        x2_or_w = width if output_as_width_height else x1 + width
        y2_or_h = height if output_as_width_height else y1 + height

        return np.asarray([x1, y1, x2_or_w, y2_or_h]).round().astype(int)

    # Parse ndarray format
    if isinstance(bbox, np.ndarray):
        offset = 1 if bbox.shape[0] > 4 else 0  # Handle optional first element

        x1, y1, width, height = bbox[offset:offset+4]

        if not input_as_width_height:
            width = width - x1
            height = height - y1

        x2_or_w = width if output_as_width_height else x1 + width
        y2_or_h = height if output_as_width_height else y1 + height

        return np.asarray([x1, y1, x2_or_w, y2_or_h]).round().astype(int)

    raise ValueError("Invalid bbox format. Expected dict, list, or ndarray.")


def fix_bboxes_offsets(bboxes_batch, pad_param):
    """
    Adjusts the bounding boxes and landmarks by subtracting the corresponding padding offsets applied during image padding.
    
    This function corrects the bounding boxes' coordinates and facial landmarks after padding the images, ensuring that 
    the boxes and landmarks are aligned with the padded images by subtracting the appropriate offsets.

    Args:
        bboxes_batch (np.ndarray): An array of bounding boxes and landmarks of shape (n, m), where each row represents 
                                   [image_id, x1, y1, x2, y2, confidence, landmark_x1, landmark_y1, ..., landmark_x5, landmark_y5].
                                   The first column (index 0) corresponds to the image ID.
        pad_param (np.ndarray): An array of padding parameters of shape (n, 2, 2), where each entry represents the amount 
                                of padding applied to each image along the width and height dimensions.

    Returns:
        np.ndarray: A modified copy of `bboxes_batch` with updated bounding box and landmark coordinates adjusted for padding.
    """
    bboxes_batch = bboxes_batch.copy()
    images_ids = np.unique(bboxes_batch[:, 0])  # Get unique image IDs

    indexes_bbox_x = [1,3]
    indexes_bbox_y = [2,4]

    indexes_landmarks_x = [6, 7, 8, 9, 10]
    indexes_landmarks_y = [11, 12, 13, 14, 15]


    # Adjust bounding boxes and landmarks for each image based on its padding parameters
    for image_id, pad in zip(images_ids, pad_param):
        selector = bboxes_batch[:, 0] == image_id

        # Adjust the x-coordinates of bounding boxes by subtracting width padding
        bboxes_batch[np.ix_(selector, indexes_bbox_x)] -= pad[1, 0]

        # Adjust the y-coordinates of bounding boxes by subtracting height padding
        bboxes_batch[np.ix_(selector, indexes_bbox_y)] -= pad[0, 0]

        # If stages combinations contain landmarks, we adjust them too
        try:
            # Adjust the x-coordinates of landmarks by subtracting width padding
            bboxes_batch[np.ix_(selector, indexes_landmarks_x)] -= pad[1, 0]

            # Adjust the y-coordinates of landmarks by subtracting height padding
            bboxes_batch[np.ix_(selector, indexes_landmarks_y)] -= pad[0, 0]

        except IndexError:
            pass


    return bboxes_batch
