# References

This document provides a detailed list of references, including the original research papers and projects that served as the foundation for this MTCNN implementation. Additionally, it includes information on how to properly cite this work if used in your research or projects.

## Citation

If you use this library in your research or projects, please consider citing the original paper where the MTCNN model was introduced. This paper presents the Joint Face Detection and Alignment using Multitask Cascaded Convolutional Networks, a groundbreaking approach for face detection and landmark alignment.

### Original Paper:

- **Authors**: K. Zhang, Z. Zhang, Z. Li, and Y. Qiao
- **Journal**: IEEE Signal Processing Letters
- **Title**: Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks
- **Year**: 2016
- **Volume**: 23
- **Number**: 10
- **Pages**: 1499-1503
- **Keywords**: Benchmark testing, Computer architecture, Convolution, Detectors, Face, Face detection, Training, Cascaded convolutional neural network (CNN), Face alignment
- **DOI**: [10.1109/LSP.2016.2603342](https://doi.org/10.1109/LSP.2016.2603342)

```bibtex
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

### Abstract of the Original Paper:
The paper presents a multitask cascaded convolutional network (MTCNN) for joint face detection and alignment. This method integrates the detection and alignment process into a unified architecture, which significantly enhances the accuracy and speed of facial landmark localization. The system uses a three-stage network to predict face locations and landmarks iteratively, improving the results progressively across each stage.

## Original MTCNN Repository

This library is based on the original implementation by Kaipeng Zhang, who made the pretrained networks and the code available for the research community. If you are using the models or weights provided in this library, you may also consider citing the original GitHub repository:

- **[Original MTCNN Implementation by Kaipeng Zhang](https://github.com/kpzhang93/MTCNN_face_detection_alignment/tree/master/code)**

This repository includes the source code, pre-trained weights, and additional information related to the original MTCNN framework, all released under the MIT license.

## Related Work

This project also draws inspiration from the **FaceNet's MTCNN implementation** by David Sandberg. This implementation is part of a larger face recognition framework called FaceNet, which uses the MTCNN architecture to handle the task of face alignment prior to recognition. You may also want to refer to this project if you are using concepts from this work:

- **[Facenet's MTCNN implementation](https://github.com/davidsandberg/facenet/tree/master/src/align)**


## About this project

The code for this project was created to standardize face detection and provide an easy-to-use framework that helps the research community push the boundaries of AI knowledge. Learn more about the author of this code on [Iván de Paz Centeno's website](https://ipazc.com)

