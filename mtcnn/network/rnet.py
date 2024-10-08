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

import tensorflow as tf


L = tf.keras.layers


class RNet(tf.keras.Model):
    """
    Definition of RNet (Refinement Network) for MTCNN.

    This network takes as input an image of size 24x24 with 3 channels, and outputs:

    * The regression of the bounding boxes (x1, y1, x2, y2) with a linear activation.
    * The classification of the area as a softmax operation ([1, 0] -> Not face; [0, 1] -> Face).
    """
    def __init__(self, **kwargs):
        super(RNet, self).__init__(**kwargs)

        # Defining the layers according to the provided architecture
        self.conv1 = L.Conv2D(28, kernel_size=(3,3), strides=(1,1), padding="valid", activation="linear", name="conv1")
        self.prelu1 = L.PReLU(shared_axes=[1, 2], name="prelu1")
        self.maxpool1 = L.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same", name="maxpooling1")

        self.conv2 = L.Conv2D(48, kernel_size=(3,3), strides=(1,1), padding="valid", activation="linear", name="conv2")
        self.prelu2 = L.PReLU(shared_axes=[1, 2], name="prelu2")
        self.maxpool2 = L.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid", name="maxpooling2")

        self.conv3 = L.Conv2D(64, kernel_size=(2,2), strides=(1,1), padding="valid", activation="linear", name="conv3")
        self.prelu3 = L.PReLU(shared_axes=[1, 2], name="prelu3")

        self.permute = L.Permute((2, 1, 3), name="permute")
        self.flatten = L.Flatten(name="flatten3")

        self.fc4 = L.Dense(128, activation="linear", name="fc4")
        self.prelu4 = L.PReLU(name="prelu4")

        self.fc5_1 = L.Dense(4, activation="linear", name="fc5-1")
        self.fc5_2 = L.Dense(2, activation="softmax", name="fc5-2")

    def build(self, input_shape=(None, 24, 24, 3)):
        """
        Build the network by defining the input and manually creating each layer step by step, computing output shapes.
        This method mirrors the layer initialization in the functional API.
        """
        # Build conv1 block
        self.conv1.build(input_shape)
        output_shape = self.conv1.compute_output_shape(input_shape)
        self.prelu1.build(output_shape)
        output_shape = self.prelu1.compute_output_shape(output_shape)
        self.maxpool1.build(output_shape)
        output_shape = self.maxpool1.compute_output_shape(output_shape)

        # Build conv2 block
        self.conv2.build(output_shape)
        output_shape = self.conv2.compute_output_shape(output_shape)
        self.prelu2.build(output_shape)
        output_shape = self.prelu2.compute_output_shape(output_shape)
        self.maxpool2.build(output_shape)
        output_shape = self.maxpool2.compute_output_shape(output_shape)

        # Build conv3 block
        self.conv3.build(output_shape)
        output_shape = self.conv3.compute_output_shape(output_shape)
        self.prelu3.build(output_shape)
        output_shape = self.prelu3.compute_output_shape(output_shape)

        # Permute and flatten
        self.permute.build(output_shape)
        output_shape = self.permute.compute_output_shape(output_shape)
        self.flatten.build(output_shape)
        output_shape = self.flatten.compute_output_shape(output_shape)

        # Fully connected layers
        self.fc4.build(output_shape)
        output_shape = self.fc4.compute_output_shape(output_shape)
        self.prelu4.build(output_shape)
        output_shape = self.prelu4.compute_output_shape(output_shape)

        # Outputs (classification and regression)
        self.fc5_1.build(output_shape)
        self.fc5_2.build(output_shape)

        # Call the super build to finalize the model building
        super(RNet, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x = inputs

        # First conv block
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.maxpool1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.maxpool2(x)

        # Third conv block
        x = self.conv3(x)
        x = self.prelu3(x)

        # Permute, flatten, and fully connected layers
        x = self.permute(x)
        x = self.flatten(x)
        x = self.fc4(x)
        x = self.prelu4(x)

        # Outputs
        bbox_reg = self.fc5_1(x)  # Regression of bounding boxes
        bbox_class = self.fc5_2(x)  # Classification (face or not)

        return [bbox_reg, bbox_class]
