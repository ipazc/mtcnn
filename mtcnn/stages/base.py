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

from abc import ABC, abstractmethod


class StageBase(ABC):
    """
    Base class for defining a stage in a processing pipeline.

    This class serves as an abstract base for stages in a model's processing pipeline. Each stage
    is defined by a name, an identifier, and an optional model associated with it. Subclasses must
    implement the `__call__` method to define the specific functionality of the stage.
    """

    def __init__(self, stage_name, stage_id, model=None, **kwargs):
        """
        Initializes a StageBase object with a name, ID, and optional model.

        Args:
            stage_name (str): The name of the stage.
            stage_id (str or int): The identifier of the stage.
            model (object, optional): The model associated with this stage.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self._name = stage_name
        self._id = stage_id
        self._model = model

    @property
    def model(self):
        """
        Returns the model associated with this stage.

        Returns:
            object: The model associated with this stage, or None if no model is set.
        """
        return self._model

    @property
    def id(self):
        """
        Returns the identifier of the stage.

        Returns:
            str or int: The identifier of the stage.
        """
        return self._id

    @property
    def name(self):
        """
        Returns the name of the stage.

        Returns:
            str: The name of the stage.
        """
        return self._name

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Abstract method that must be implemented by subclasses to define the functionality of the stage.

        Args:
            *args: Positional arguments for the stage's functionality.
            **kwargs: Keyword arguments for the stage's functionality.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
