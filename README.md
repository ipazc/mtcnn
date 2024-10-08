# MTCNN - Multitask Cascaded Convolutional Networks for Face Detection and Alignment

[![PyPI version](https://badge.fury.io/py/mtcnn.svg)](https://badge.fury.io/py/mtcnn)
[![Documentation Status](https://readthedocs.org/projects/mtcnn/badge/?version=latest)](https://mtcnn.readthedocs.io/en/latest/?badge=latest)
![Test Status](https://github.com/ipazc/mtcnn/actions/workflows/tests.yml/badge.svg)
![Pylint Check](https://github.com/ipazc/mtcnn/actions/workflows/pylint.yml/badge.svg)
![PyPI Downloads](https://img.shields.io/pypi/dm/mtcnn)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13901378.svg)](https://doi.org/10.5281/zenodo.13901378)



## Overview

![Example](resources/result.jpg)

MTCNN is a robust face detection and alignment library implemented for Python >= 3.10 and TensorFlow >= 2.12, designed to detect faces and their landmarks using a multitask cascaded convolutional network. This library improves on the original implementation by offering a complete refactor, simplifying usage, improving performance, and providing support for batch processing.

This library is ideal for applications requiring face detection and alignment, with support for both bounding box and landmark prediction.

## Installation

MTCNN can be installed via pip:

```bash
pip install mtcnn
```

MTCNN requires Tensorflow >= 2.12. This external dependency can be installed manually or automatically along with MTCNN via:

```bash
pip install mtcnn[tensorflow]
```

## Usage Example

```python
from mtcnn import MTCNN
from mtcnn.utils.images import load_image

# Create a detector instance
detector = MTCNN(device="CPU:0")

# Load an image
image = load_image("ivan.jpg")

# Detect faces in the image
result = detector.detect_faces(image)

# Display the result
print(result)
```

Output example:

```json
[
    {
        "box": [277, 90, 48, 63],
        "keypoints": {
            "nose": [303, 131],
            "mouth_right": [313, 141],
            "right_eye": [314, 114],
            "left_eye": [291, 117],
            "mouth_left": [296, 143]
        },
        "confidence": 0.9985
    }
]
```

## Models Overview

MTCNN uses a cascade of three networks to detect faces and facial landmarks:

- **PNet (Proposal Network)**: Scans the image and proposes candidate face regions. 
- **RNet (Refine Network)**: Refines the face proposals from PNet.
- **ONet (Output Network)**: Detects facial landmarks (eyes, nose, mouth) and provides a final refinement of the bounding boxes.

All networks are implemented using TensorFlow’s functional API and optimized to avoid unnecessary operations, such as transpositions, ensuring faster and more efficient execution.

# Documentation

The full documentation for this project is available at [Read the Docs](http://mtcnn.readthedocs.io/).


## Citation

If you use this library implementation for your research or projects, please consider using this cite:

```
@software{ivan_de_paz_centeno_2024_13901378,
  author       = {Iván de Paz Centeno},
  title        = {ipazc/mtcnn: v1.0.0},
  month        = oct,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.13901378},
  url          = {https://doi.org/10.5281/zenodo.13901378}
}
```

And the original research work from Kaipeng Zhang:

```
@article{7553523,
    author={K. Zhang and Z. Zhang and Z. Li and Y. Qiao}, 
    journal={IEEE Signal Processing Letters}, 
    title={Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks}, 
    year={2016}, 
    volume={23}, 
    number={10}, 
    pages={1499-1503}, 
    keywords={Benchmark testing;Computer architecture;Convolution;Detectors;Face;Face detection;Training;Cascaded convolutional neural network (CNN);face alignment;face detection}, 
    doi={10.1109/LSP.2016.2603342}, 
    ISSN={1070-9908}, 
    month={Oct}
}
```

You may also reference the original GitHub repository that this project was based on (including the networks weights):  
[Original MTCNN Implementation by Kaipeng Zhang](https://github.com/kpzhang93/MTCNN_face_detection_alignment/tree/master/code)

And the FaceNet's implementation that served as inspiration:
[Facenet's MTCNN implementation](https://github.com/davidsandberg/facenet/tree/master/src/align)


## About the Author

This project is developed and maintained by [Iván de Paz Centeno](https://ipazc.com), with the goal of standardizing face detection and providing an easy-to-use framework to help the research community push the boundaries of AI knowledge.

If you find this project useful, please consider supporting it through GitHub Sponsors. Your support will help cover costs related to improving the codebase, adding new features, and providing better documentation.

[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-brightgreen)](https://github.com/sponsors/ipazc)


## Acknowledgments

This project has evolved over time with contributions from multiple developers. While the current codebase has been completely rewritten, we acknowledge and appreciate the valuable input and collaboration from past contributors.

A special thanks to everyone who has submitted pull requests, reported issues, or provided feedback to make this project better. 

For a full list of contributors, please visit the [GitHub contributors page](https://github.com/ipazc/mtcnn/graphs/contributors).


## License

This project is licensed under the [MIT License](LICENSE).
