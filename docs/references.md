# References

This document provides references to the foundational work that inspired this implementation of the MTCNN (Multitask Cascaded Convolutional Networks) model, as well as guidelines on how to properly cite this library if used in your research or projects.

## Citation for this MTCNN Library

If you use this implementation of the MTCNN library in your research or projects, please consider citing the following Zenodo entry:

```bibtex
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

This citation provides proper credit for this specific implementation of the MTCNN library, as hosted on Zenodo.

## Original Research Paper

The development of the MTCNN model is based on the original paper titled **"Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks"** by K. Zhang, Z. Zhang, Z. Li, and Y. Qiao. If your work benefits from the concepts or algorithms presented in this library, please consider citing this paper:

- **Authors**: K. Zhang, Z. Zhang, Z. Li, Y. Qiao
- **Journal**: IEEE Signal Processing Letters
- **Title**: Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks
- **Year**: 2016
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
  doi={10.1109/LSP.2016.2603342},
  month={Oct}
}
```

### Summary of the Original Paper

The paper introduces a multitask cascaded convolutional network (MTCNN) designed for joint face detection and alignment. This architecture improves accuracy and speed by integrating both tasks into a unified process, using a three-stage network to progressively refine facial landmark predictions.

## Original MTCNN Implementation

This library builds upon the original implementation by Kaipeng Zhang, who developed the MTCNN model and released the code for the research community. If you are using the models or weights provided in this library, you may also want to reference the original implementation:

- **[Original MTCNN Implementation by Kaipeng Zhang](https://github.com/kpzhang93/MTCNN_face_detection_alignment/tree/master/code)**

The original implementation includes the source code, pre-trained weights, and additional information related to the MTCNN framework.

## Related Work

This project also draws inspiration from the **FaceNet's MTCNN implementation** by David Sandberg, which incorporates the MTCNN architecture into the FaceNet framework for face alignment prior to recognition. You may refer to this project if you use related concepts:

- **[Facenet's MTCNN implementation](https://github.com/davidsandberg/facenet/tree/master/src/align)**

## Acknowledgments

This implementation of the MTCNN library was developed by Iván de Paz Centeno, with contributions from various developers over the history of the project. Special thanks to all the contributors for their valuable input, which has helped improve the library. You can view the full list of contributors on the [GitHub contributors page](https://github.com/ipazc/mtcnn/graphs/contributors).

For more information about the author, visit [Iván de Paz Centeno's website](https://ipazc.com).

