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

import os
import importlib

import joblib


def load_weights(weights_name):
    """
    Attempts to load weights from a user-provided file path or, if not found, from the package's assets.
    
    Args:
        weights_name (str): The name of the weights file to load (can be a file path provided by the user 
                            or just the name of the weights file for fallback).
    
    Returns:
        The loaded weights if found, otherwise raises an exception.
    """
    # Define possible paths: first check user-provided path, then fallback to package assets
    paths = [
        os.path.abspath(weights_name),  # Check if it's a user-provided file path
        importlib.resources.files('mtcnn.assets.weights') / weights_name  # Fallback to package's assets
    ]

    # Try to load weights from the first valid path
    for path in paths:
        if os.path.exists(path):  # First checks the local filesystem
            return joblib.load(path)

    # If no file is found, raise an error
    raise FileNotFoundError(f"Weights file '{weights_name}' not found in the system or in the package assets.")


def set_gpu_memory_growth():
    """
    Configures TensorFlow to allocate only the required amount of GPU memory instead of 
    allocating all available GPU memory at once. The memory usage will grow dynamically 
    as needed during model execution.

    This should be called before any TensorFlow or Keras operations are initialized to 
    ensure proper memory management.

    Raises:
        RuntimeError: If the GPUs have already been initialized or if memory growth cannot be set.
    """
    import tensorflow as tf

    # List available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            # Set memory growth for each GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            # Error occurs if GPUs have already been initialized
            print(f"Error setting memory growth: {e}")
