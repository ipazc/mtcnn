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

from .bboxes import parse_bbox
from .landmarks import parse_landmarks


def plot_bbox(image, bbox, color="#FFFF00", normalize_color=False, input_as_width_height=True):
    """
    Draws a bounding box on the given image.
    
    Args:
        image (np.ndarray): The input image on which to draw the bounding box.
        bbox (list, dict, or np.ndarray): The bounding box coordinates in one of the following formats:
                                          - List or array [x1, y1, x2, y2].
                                          - Dict with a key 'box' that contains the bounding box.
        color (str): The color of the bounding box in hex format (default is yellow, "#FFFF00").
        normalize_color (bool): True if color should be in [0..1]. False to make it between [0..255]
        input_as_width_height (bool): True if input `bbox` parameter follows format [x1, y1, width, height]. False if follows format [x1, y1, x2, y2]
    Returns:
        np.ndarray: The image with the bounding box drawn.
    """
    color = parse_color(color)  # Convert color to RGB
    color = color if normalize_color else (color * 255).astype(np.uint8)

    # Parse the bounding box coordinates
    bbox = parse_bbox(bbox, input_as_width_height=input_as_width_height, output_as_width_height=False)

    image = image.copy()  # Copy the image to avoid modifying the original

    # Draw the vertical sides of the box (left and right)
    image[bbox[1]:bbox[3], bbox[0], :] = color  # Left side
    image[bbox[1]:bbox[3], bbox[2], :] = color  # Right side

    # Draw the horizontal sides of the box (top and bottom)
    image[bbox[1], bbox[0]:bbox[2], :] = color  # Top side
    image[bbox[3], bbox[0]:bbox[2], :] = color  # Bottom side

    result = image if normalize_color else image.astype(np.uint8)
    return result


def plot_landmarks(image, landmarks, color="#FFFF00", keypoints="nose,mouth_right,right_eye,left_eye,mouth_left",
                   brush_size=2, normalize_color=False):
    """
    Plots facial landmarks on the given image.
    
    Args:
        image (np.ndarray): The input image on which to draw the landmarks.
        landmarks (dict or np.ndarray): The facial landmarks to plot, either as a dictionary or an array.
        color (str): The color of the landmarks in hex format (default is yellow, "#FFFF00").
        keypoints (str): A comma-separated list of keypoints to plot (default includes all facial landmarks).
        brush_size (int): The size of the brush used to draw the keypoints (default is 2).
        normalize_color (bool): True if color should be in [0..1]. False to make it between [0..255]

    Returns:
        np.ndarray: The image with the landmarks drawn.
    """
    keypoints = [k.strip() for k in keypoints.split(",")]  # Parse the keypoints list
    color = parse_color(color)  # Convert color to RGB
    color = color if normalize_color else (color * 255).astype(np.uint8)

    try:
        landmarks = parse_landmarks(landmarks)  # Parse the landmarks

    except IndexError:  # No landmarks available
        return image

    image = image.copy()  # Copy the image to avoid modifying the original

    # Draw each landmark as a small circle
    for key in keypoints:
        if key in landmarks:
            x, y = landmarks[key]
            image[y-brush_size:y+brush_size+1, x-brush_size:x+brush_size+1, :] = color  # Draw the landmark

    result = image if normalize_color else image.astype(np.uint8)
    return result


def plot(image, detection, input_as_width_height=True):
    """
    Plots a single or multiple facial detection results on the given image.

    Args:
        image (np.ndarray): The input image on which to draw the detections.
        detection (list, dict, or np.ndarray): A single detection or a list/array of detections to plot. 
            Each detection contains facial landmarks and/or bounding box information.
        input_as_width_height (bool): Whether the input bounding box format is (width, height) instead of 
            the default (x1, y1, x2, y2) (default is True).

    Returns:
        np.ndarray or None: The image with the detection(s) plotted, or None if no detection is present.
    """
    if len(detection) == 0:
        return None

    if isinstance(detection, list) or (isinstance(detection, np.ndarray) and len(detection.shape) > 1):
        return plot_all(image, detection, input_as_width_height=input_as_width_height)

    return plot_landmarks(plot_bbox(image, detection, input_as_width_height=input_as_width_height), detection)


def plot_all(image, detections, input_as_width_height=True):
    """
    Plots multiple facial detection results on the given image.

    Args:
        image (np.ndarray): The input image on which to draw the detections.
        detections (list or np.ndarray): A list or array of detections, where each detection contains 
            facial landmarks and/or bounding box information.
        input_as_width_height (bool): Whether the input bounding box format is (width, height) instead of 
            the default (x1, y1, x2, y2) (default is True).

    Returns:
        np.ndarray: The image with all detections plotted.
    """
    for detection in detections:
        image = plot_landmarks(plot_bbox(image, detection, input_as_width_height=input_as_width_height), detection)

    return image


def parse_color(color):
    """
    Parses a color from a string in various formats (e.g., hex, RGB) into a normalized RGB array.
    
    The color can be provided in the following formats:
    * Hexadecimal string (e.g., "#RRGGBB" or "#RGB")
    * Hexadecimal string with prefix "0x" (e.g., "0xRRGGBB")
    
    Args:
        color (str): A color string in hex format (e.g., "#RRGGBB", "#RGB", "0xRRGGBB").
    
    Returns:
        np.ndarray: A numpy array of normalized RGB values (between 0 and 1) representing the color.
    """
    if isinstance(color, str):
        if color.startswith("#"):
            color = color[1:]  # Remove '#' prefix
        if color.startswith("0x"):
            color = color[2:]  # Remove '0x' prefix
        if len(color) == 3:  # Short form hex color (#RGB)
            color = np.asarray([int(f"{color[0]}{color[0]}", base=16),
                                int(f"{color[1]}{color[1]}", base=16),
                                int(f"{color[2]}{color[2]}", base=16)]) / 255
        if len(color) == 6:  # Full form hex color (#RRGGBB)
            color = np.asarray([int(f"{color[0]}{color[1]}", base=16),
                                int(f"{color[2]}{color[3]}", base=16),
                                int(f"{color[4]}{color[5]}", base=16)]) / 255

    return color
