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


class PNet(tf.keras.Model):
    """
    Definition of PNet (Proposal Network)

    This network takes as input an image with variable width and height, and generates two outputs:

     * The regression of the bounding boxes (x1, y1, x2, y2) with a linear activation.
     * The classification of the area as a softmax operation ([1,0] -> Not face; [0,1] -> Face)
    """
    def __init__(self, **kwargs):
        super(PNet, self).__init__(**kwargs)

        # Definir las capas
        self.conv1 = L.Conv2D(10, kernel_size=(3,3), strides=(1,1), padding="valid", activation="linear", name="conv1")
        self.prelu1 = L.PReLU(shared_axes=[1, 2], name="prelu1")
        self.maxpool1 = L.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same", name="maxpooling1")
        self.conv2 = L.Conv2D(16, kernel_size=(3,3), strides=(1,1), padding="valid", activation="linear", name="conv2")
        self.prelu2 = L.PReLU(shared_axes=[1, 2], name="prelu2")
        self.conv3 = L.Conv2D(32, kernel_size=(3,3), strides=(1,1), padding="valid", activation="linear", name="conv3")
        self.prelu3 = L.PReLU(shared_axes=[1, 2], name="prelu3")
        self.conv4_1 = L.Conv2D(4, kernel_size=(1,1), strides=(1,1), padding="valid", activation="linear", name="conv4-1")
        self.conv4_2 = L.Conv2D(2, kernel_size=(1,1), strides=(1,1), padding="valid", activation="softmax", name="conv4-2")

    def build(self, input_shape=(None, None, None, 3)):
        self.conv1.build(input_shape)
        output_shape = self.conv1.compute_output_shape(input_shape)

        self.prelu1.build(output_shape)
        output_shape = self.prelu1.compute_output_shape(output_shape)

        self.maxpool1.build(output_shape)
        output_shape = self.maxpool1.compute_output_shape(output_shape)

        self.conv2.build(output_shape)
        output_shape = self.conv2.compute_output_shape(output_shape)

        self.prelu2.build(output_shape)
        output_shape = self.prelu2.compute_output_shape(output_shape)

        self.conv3.build(output_shape)
        output_shape = self.conv3.compute_output_shape(output_shape)

        self.prelu3.build(output_shape)
        output_shape = self.prelu3.compute_output_shape(output_shape)

        self.conv4_1.build(output_shape)
        self.conv4_2.build(output_shape)

        super(PNet, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x = inputs

        # First conv block
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.maxpool1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.prelu2(x)

        # Third conv block
        x = self.conv3(x)
        x = self.prelu3(x)

        # Outputs
        bbox_reg = self.conv4_1(x)
        bbox_class = self.conv4_2(x)

        return [bbox_reg, bbox_class]
