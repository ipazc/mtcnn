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

from tensorflow.python.framework.errors_impl import NotFoundError


def build_scale_pyramid(width, height, min_face_size, scale_factor, min_size=12):
    """
    Builds a scale pyramid for detecting objects (e.g., faces) at different sizes in an image.
    
    Args:
        width (int): The width of the input image, in pixels.
        height (int): The height of the input image, in pixels.
        min_face_size (int): The minimum size (in pixels) of the object (e.g., face) that should be detectable.
        scale_factor (float): The factor by which the image is downscaled at each level (e.g., 0.7 means 70% of the previous scale).
        min_size (int, optional): The smallest size (in pixels) to which the image can be downscaled. Default is 12 pixels.
    
    Returns:
        np.ndarray: Array of scales to apply to the image at each level of the pyramid.
    """

    # Find the smallest dimension of the image
    min_dim = min(width, height)

    # Calculate how many scales are needed based on the smallest dimension and the scale factor
    scales_count = round(-((np.log(min_dim / min_size) / np.log(scale_factor)) + 1))

    # Calculate the base scale value (based on the smallest detectable face size)
    m = min_size / min_face_size

    # Generate an array of scales for the pyramid
    return m * scale_factor ** np.arange(scales_count)


def scale_images(images, scale: float=None, new_shape: tuple=None):
    """
    Scales the input images either by a given factor or to a specified new shape.

    Args:
        images (np.ndarray or tf.Tensor): A batch of images or a single image. The expected input should have 
                                          the shape (..., height, width, channels), where the last three dimensions
                                          represent the height, width, and color channels of the images.
        scale (float, optional): A scaling factor to resize the images. For example, a value of 0.5 will reduce 
                                 the image to 50% of its original size, while 2.0 will double its size. 
                                 This parameter is ignored if `new_shape` is provided.
        new_shape (tuple, optional): A tuple specifying the new shape (height, width) to resize the images to. 
                                     If provided, this will override the scaling factor.

    Returns:
        tf.Tensor: The scaled images as a tensor with resized dimensions, determined either by the scaling factor 
                   or the new shape provided.
    """

    # Extract the shape from the images
    shape = np.asarray(images.shape[-3:-1])

    if scale is None and new_shape is None:
        new_shape = shape

    new_shape = shape * scale if new_shape is None else new_shape

    # Resize the images using the specified scaling factor
    images_scaled = tf.image.resize(images, new_shape, method=tf.image.ResizeMethod.AREA)

    return images_scaled


def normalize_images(images):
    """
    Normalizes the input images to the range [-1, 1].

    Args:
        images (np.ndarray or tf.Tensor): A batch of images or a single image. The expected input should have 
                                          pixel values in the range [0, 255].

    Returns:
        np.ndarray or tf.Tensor: The normalized images, where pixel values are rescaled to the range [-1, 1].
    """
    # Normalize the images to the range [-1, 1]
    return (images - 127.5) / 128


def pad_stack_np(images, justification="center"):
    """
    Pads and stacks a list of images to ensure they all have the same size, based on the specified justification.
    
    Args:
        images (list of np.ndarray): A list of images with varying shapes. Each image is expected to be a NumPy array.
        justification (str, optional): Specifies the justification (alignment) for padding. 
                                       Available options are: "center", "top", "topleft", "topright", 
                                       "bottom", "bottomleft", "bottomright", "left", "right". 
                                       Default is "center".
    
    Returns:
        np.ndarray: A stacked NumPy array where all images have been padded to the same size, based on the chosen justification.
        np.ndarray: A stacked NumPy array of each original shape.
        np.ndarray: A NumPy array containing the padding parameters applied to each image.
    """

    # Stack the shapes of all images into an array
    sizes_stack = np.stack([img.shape for img in images], axis=0)

    # Find the maximum shape along each dimension
    sizes_max = sizes_stack.max(axis=0, keepdims=True)

    # Calculate the difference in size for padding
    sizes_diff = sizes_max - sizes_stack

    # Calculate if any padding size is odd, to adjust padding
    sizes_mod = sizes_diff % 2
    sizes_diff = sizes_diff - sizes_mod

    # Justification masks for padding alignment
    justification_mask = {
        "top": np.asarray([[[0, 1], [0.5, 0.5], [0, 0]]]),
        "topleft": np.asarray([[[0, 1], [0, 1], [0, 0]]]),
        "topright": np.asarray([[[0, 1], [1, 0], [0, 0]]]),
        "bottom": np.asarray([[[1, 0], [0.5, 0.5], [0, 0]]]),
        "bottomleft": np.asarray([[[1, 0], [0, 1], [0, 0]]]),
        "bottomright": np.asarray([[[1, 0], [1, 0], [0, 0]]]),
        "left": np.asarray([[[0.5, 0.5], [0, 1], [0, 0]]]),
        "right": np.asarray([[[0.5, 0.5], [1, 0], [0, 0]]]),
        "center": np.asarray([[[0.5, 0.5], [0.5, 0.5], [0, 0]]]),
    }

    # Justification adjustments for padding if needed
    justification_pad_mask = {
        "top": "topleft",
        "bottom": "bottomleft",
        "left": "topleft",
        "right": "topright",
        "center": "topleft"
    }

    # Get the correct padding mask based on justification
    pad_mask = justification_mask[justification]
    mod_mask = justification_mask[justification_pad_mask.get(justification, justification)]

    # Calculate the exact padding parameters
    pad_param = (pad_mask * sizes_diff[:,:,None] + mod_mask * sizes_mod[:,:,None]).astype(int)

    # Apply the calculated padding to each image and stack them into a single array
    images_padded = np.stack([np.pad(img, pad) for img, pad in zip(images, pad_param)], axis=0)

    # We keep the original faces to return as extra info
    original_shapes = np.stack([img.shape for img in images], axis=0)

    return images_padded, original_shapes, pad_param


def pad_stack_tf(images, justification="center"):
    """
    Pads and stacks a list of images to ensure they all have the same size, based on the specified justification.

    Args:
        images (list of tf.Tensor): A list of images with varying shapes. Each image is expected to be a TensorFlow tensor.
        justification (str, optional): Specifies the justification (alignment) for padding. 
                                       Available options are: "center", "top", "topleft", "topright", 
                                       "bottom", "bottomleft", "bottomright", "left", "right". 
                                       Default is "center".

    Returns:
        tf.Tensor: A stacked TensorFlow tensor where all images have been padded to the same size, based on the chosen justification.
        tf.Tensor: A TensorFlow tensor of the original shapes of each image, for reference.
        tf.Tensor: A TensorFlow tensor containing the padding parameters applied to each image.
    """
    # Stack the shapes of all images into a tensor
    sizes_stack = tf.stack([tf.shape(img) for img in images], axis=0)

    # Find the maximum shape along each dimension
    sizes_max = tf.reduce_max(sizes_stack, axis=0, keepdims=True)

    # Calculate the difference in size for padding
    sizes_diff = sizes_max - sizes_stack

    # Calculate if any padding size is odd, to adjust padding
    sizes_mod = tf.cast(sizes_diff % 2, tf.float32)
    sizes_diff = tf.cast(sizes_diff, tf.float32) - sizes_mod

    # Justification masks for padding alignment
    justification_mask = {
        "top": tf.constant([[[0, 1.], [0.5, 0.5], [0, 0]]]),
        "topleft": tf.constant([[[0, 1.], [0, 1.], [0, 0]]]),
        "topright": tf.constant([[[0, 1.], [1., 0], [0, 0]]]),
        "bottom": tf.constant([[[1., 0], [0.5, 0.5], [0, 0]]]),
        "bottomleft": tf.constant([[[1., 0], [0, 1.], [0, 0]]]),
        "bottomright": tf.constant([[[1., 0], [1., 0], [0, 0]]]),
        "left": tf.constant([[[0.5, 0.5], [0, 1.], [0, 0]]]),
        "right": tf.constant([[[0.5, 0.5], [1., 0], [0, 0]]]),
        "center": tf.constant([[[0.5, 0.5], [0.5, 0.5], [0, 0]]]),
    }

    # Justification adjustments for padding if needed
    justification_pad_mask = {
        "top": "topleft",
        "bottom": "bottomleft",
        "left": "topleft",
        "right": "topright",
        "center": "topleft"
    }

    # Get the correct padding mask based on justification
    pad_mask = justification_mask[justification]
    mod_mask = justification_mask[justification_pad_mask.get(justification, justification)]

    # Calculate the exact padding parameters
    pad_param = (pad_mask * sizes_diff[:,:,None] + mod_mask * sizes_mod[:,:,None])
    pad_param = tf.cast(pad_param, tf.int32)

    # Apply the calculated padding to each image and stack them into a single tensor
    images_padded = tf.stack([tf.pad(img, paddings=pad) for img, pad in zip(images, pad_param)], axis=0)

    # We keep the original faces to return as extra info
    original_shapes = tf.stack([tf.shape(img) for img in images], axis=0)

    return images_padded, original_shapes, pad_param


def ensure_stack(images):
    """
    Ensures that the input is a properly stacked array of images.
    This function should be called to format the input for a given model.
    
    Args:
        images (list or np.ndarray): A list of images or a NumPy array of images. 
                                     If it's a list, images will be padded and stacked.
                                     If it is a single image, the image's dimension will be expanded as 
                                     if it were a list of a single image.
    
    Returns:
        np.ndarray: A properly stacked array of images, ensuring they have the same shape.
    """

    # If images is a list, pad and stack them
    if isinstance(images, list):
        images = pad_stack_np(images)

    # Broadcast to ensure the images have a consistent shape (batch dimension)
    return np.broadcast_to(images,
                           [(len(images.shape) < 4) + (len(images.shape) >= 4) * images.shape[0],] + list(images.shape[len(images.shape) >= 4:]))


def load_image(image, dtype=tf.float32, device="CPU:0"):
    """
    Loads an image and decodes it into a TensorFlow tensor, optionally normalizing it.

    Args:
        image (str, np.ndarray, or tf.Tensor): The input image. It can be:
                                               - A file path to the image as a string.
                                               - A TensorFlow tensor or a NumPy array representing an image.
        dtype (tf.DType, optional): The desired data type for the decoded image. Default is tf.float32.
        device (str, optional): the target device for the operation. Using CPU most of the times should be fine.
    
    Returns:
        tf.Tensor: The decoded image tensor, with shape (height, width, channels) and dtype `dtype`. If 
                   `normalize=True`, the image values will be scaled to the range [0, 1].
    """

    with tf.device(device):
        is_tensor = tf.is_tensor(image) or isinstance(image, np.ndarray)

        if is_tensor:
            decoded_image = image
        else:
            try:
                if isinstance(image, str):
                    image_data = tf.io.read_file(image)  # Read image from file
                else:
                    image_data = image  # Assume image data is provided directly
            except NotFoundError:
                image_data = image  # If file not found, use the input directly

            # Decode the image with 3 channels (RGB)
            decoded_image = tf.image.decode_image(image_data, channels=3, dtype=dtype).numpy()

        # If dtype is float, adjust the image scale
        if dtype in [tf.float16, tf.float32]:
            decoded_image *= 255  # Convert pixel values to [0, 255] if using float data type

    return decoded_image


def load_images_batch(images, dtype=tf.float32, device="CPU:0"):
    """
    Loads a batch of images into memory. If the images are not already in tensor or NumPy array format, 
    they are loaded from their file paths.

    Args:
        images (list of str, np.ndarray, or tf.Tensor): A list of images, where each image can either be 
                                                        a TensorFlow tensor, NumPy array, or a file path (string).
        dtype (tf.DType, optional): The data type for the loaded images. Default is tf.float32.
        device (str, optional): the target device for the operation. Using CPU most of the times should be fine.

    Returns:
        list of tf.Tensor: A list of TensorFlow tensors representing the raw images.
    """
    is_tensor = tf.is_tensor(images[0]) or isinstance(images[0], np.ndarray)
    images_raw = images if is_tensor else [load_image(img, dtype=dtype, device=device) for img in images]
    return images_raw


def standarize_batch(images_raw, normalize=True, justification="center"):
    """
    Pads and stacks a batch of images to ensure they all have the same size, with an option to normalize them.

    Args:
        images_raw (list of tf.Tensor or np.ndarray): A list of raw images, each either as a TensorFlow tensor or 
                                                      a NumPy array.
        normalize (bool, optional): Whether to normalize the images after stacking. Default is True.
        justification (str, optional): The alignment for padding the images. Available options are: "center", 
                                       "top", "topleft", "topright", "bottom", "bottomleft", "bottomright", 
                                       "left", "right". Default is "center".

    Returns:
        np.ndarray: A stacked array of images, padded to the same shape based on the chosen justification.
        np.ndarray: An array containing the original shapes of each image before padding.
        np.ndarray: The padding parameters applied to each image.
    """
    images_result, images_oshapes, pad_param = pad_stack_np(images_raw, justification=justification)

    if normalize:
        images_result = normalize_images(images_result)

    return images_result, images_oshapes, pad_param


def apply_scales(images_normalized, scales_groups):
    """
    Applies scales to the normalized images based on the largest group of scales.
    
    Args:
        images_normalized (np.ndarray): A normalized image or batch of images.
        scales_groups (list of np.ndarray): A list of different scale groups, where each group contains multiple possible scales.

    Returns:
        tuple: 
            - list of np.ndarray: A list of images scaled according to the largest group of scales.
            - np.ndarray: The largest group of scales that was applied to the images.
    """
    # Select the scale group with the largest number of elements
    selected_scaleset_as_index = np.argmax([x.shape[0] for x in scales_groups])
    largest_scale_group_set = scales_groups[selected_scaleset_as_index]

    # Apply the scales from the largest scale group to the normalized images
    result = [scale_images(images_normalized, scale) for scale in largest_scale_group_set]

    return result, largest_scale_group_set


def extract_patches(images_normalized, bboxes_batch, expected_size=(24, 24)):
    """
    Extracts patches from a batch of normalized images based on bounding boxes, and resizes each patch to a specified size.

    Args:
        images_normalized (tf.Tensor): A batch of images or a single image, normalized, with the shape (batch_size, height, width, channels).
        bboxes_batch (np.ndarray): An array of bounding boxes of shape (n, 5), where each row represents 
                                   [batch_index, x1, y1, x2, y2]. The coordinates are in pixel format, and the first 
                                   column indicates the image index in the batch.
        expected_size (tuple, optional): A tuple specifying the size (height, width) to resize each extracted patch. 
                                         Defaults to (24, 24).

    Returns:
        tf.Tensor: A tensor of extracted patches with shape (n, height, width, channels), where each patch corresponds 
                   to a bounding box in `bboxes_batch`, resized to `expected_size`.
    """
    # Get the shape of the input images
    shape = images_normalized.shape

    # Normalize the bounding box coordinates to be within [0, 1] relative to image dimensions
    selector = [2, 1, 4, 3]

    bboxes_batch_coords = bboxes_batch[:, selector] / np.asarray([[shape[selector[1]], shape[selector[0]], shape[selector[1]], shape[selector[0]]]])

    # Extract patches from the images using the bounding boxes, resizing them to `expected_size`
    result = tf.image.crop_and_resize(
        images_normalized,                 # Input image tensor
        bboxes_batch_coords,               # Bounding boxes in format [y1, x1, y2, x2], normalized to [0.0, 1.0]
        bboxes_batch[:, 0].astype(int),    # Indices of the images in the batch corresponding to the bounding boxes
        expected_size                      # Size to resize the cropped patches (height, width)
    )

    return result
