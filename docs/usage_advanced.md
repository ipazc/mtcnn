## Advanced Usage: Batch Processing with MTCNN

MTCNN supports batch processing, allowing you to detect faces in multiple images at once. This feature is especially useful for speeding up detection when processing a large number of images. In batch mode, MTCNN handles the padding and justification of smaller images internally, allowing the user to input a list of images directly or load them from URIs.

### Key Differences in Batch Processing

1. **Image Loading and Padding**: Images are passed as a list and then internally padded by MTCNN to match the size of the largest image in the batch. By default, smaller images are **centered** within the padded tensor, but this behavior can be customized using the `batch_stack_justification` parameter.
2. **Batching Across Scales**: When processing multiple images, MTCNN applies the same set of scales across all images in the batch. For example, if there are 10 images and 5 scales, MTCNN processes 10 images at a time for each scale. 
3. **NMS and Postprocessing**: Non-Maximum Suppression (NMS) operates on each batch of images independently, filtering overlapping and low-confidence detections per image. Postprocessing ensures that bounding boxes are correctly adjusted to account for the padding added during tensor construction.

### Workflow Overview

1. Load a batch of images using `load_images_batch` or pass a list of image URIs directly to MTCNN.
2. MTCNN automatically standardizes the images by padding them and applying justification.
3. The detector processes each batch of images through the three stages (PNet, RNet, ONet) while applying NMS after each stage.
4. The final results include face detections and landmarks for all images in the batch, with postprocessing to adjust bounding box coordinates for padding.

### Example: Detecting Faces in a Batch of Images

In this example, we will:

- Load a batch of images from disk.
- Detect faces and landmarks across the entire batch using MTCNN.
- Plot the results for each image in the batch.

#### 1. Importing Required Modules

First, import the necessary functions for loading images in batches, initializing the detector, and plotting the results:

```python
from mtcnn import MTCNN
from mtcnn.utils.images import load_images_batch
from mtcnn.utils.plotting import plot
import matplotlib.pyplot as plt
```

#### 2. Loading a Batch of Images

You can load multiple images into a list using `load_images_batch`. This function simply reads and returns the images without padding or standardizing them:

```python
image_paths = ["../resources/image1.jpg", "../resources/image2.jpg", "../resources/image3.jpg"]
images = load_images_batch(image_paths)
```

At this point, `images` is just a list of loaded image tensors, with no padding or justification applied.

#### 3. Initializing the MTCNN Detector

As in single image processing, we initialize the MTCNN detector. The detector will automatically handle batch processing if you pass a list of images:

```python
mtcnn = MTCNN(device="CPU:0")
```

You can specify `"GPU:0"` or another device if you want to leverage GPU acceleration.

#### 4. Detecting Faces in Batch Mode

The `detect_faces` method supports batch input and performs all necessary padding and justification internally. You can control how smaller images are aligned within the padded tensor using the `batch_stack_justification` parameter. The default is `"center"`, which centers smaller images within the padded tensor.

```python
results = mtcnn.detect_faces(images, batch_stack_justification="center")
```

MTCNN will:

- Pad each image to match the size of the largest image in the batch.
- Group images by scale and process them through PNet, RNet, and ONet.
- Apply Non-Maximum Suppression (NMS) after each stage.

The `results` will be a list where each element corresponds to the detection result of one image in the batch. Each result will contain bounding boxes, landmarks, and confidence scores, as in single-image detection.

#### 5. Plotting Results for Each Image

To visualize the detections for each image, you can loop through the results and plot the bounding boxes and landmarks on each image:

```python
for i, image in enumerate(images):
    plt.figure()
    plt.imshow(plot(image, results[i]))
    plt.title(f"Results for image {i+1}")
    plt.show()
```

This will display each image with its corresponding detections, including bounding boxes around the faces and landmarks for each facial feature.

---

### Using URIs Instead of Loading Images Manually

MTCNN also supports passing image URIs directly to the `detect_faces` function, bypassing the need for manual image loading. This method is especially useful when you do not need to manipulate or plot the original image tensors.

Here’s how you can detect faces by providing image paths or URIs directly to MTCNN:

```python
image_uris = ["../resources/image1.jpg", "../resources/image2.jpg", "../resources/image3.jpg"]
results = mtcnn.detect_faces(image_uris)
```

In this case, MTCNN will automatically load the images from the URIs, standardize them (by padding smaller images), and perform face detection. However, since the original image tensors are not returned, plotting the results using the original images won’t be possible without loading them manually.

### How Batch Processing Works Internally

The following steps describe the internal workings of MTCNN during batch processing:

1. **Padding and Justification**: After loading a list of images, MTCNN pads them internally to match the size of the largest image in the batch. The smaller images are aligned within the tensor according to the `batch_stack_justification` parameter (default is `"center"`).

2. **Image Scaling (Image Pyramid)**: MTCNN applies a set of scales to each image in the batch, creating a pyramid of resized images. The images are processed in groups by scale, with the same set of scales applied to all images.

3. **PNet Stage**: For each scale, PNet processes the batch of images, generating bounding box proposals and confidence scores. After this, **Non-Maximum Suppression (NMS)** is applied to each image independently to remove overlapping or low-confidence proposals.

4. **RNet and ONet Stages**: The bounding boxes from PNet are processed by RNet and then ONet. For each batch of images, the networks refine the proposals and detect facial landmarks. NMS is applied after each stage to refine the results.

5. **Postprocessing**: After the final stage, MTCNN adjusts the bounding box coordinates to account for the padding applied during tensor creation. This ensures that bounding boxes are accurate relative to the original image dimensions.

---

### Full Batch Processing Code Example

```python
from mtcnn import MTCNN
from mtcnn.utils.images import load_images_batch
from mtcnn.utils.plotting import plot
import matplotlib.pyplot as plt

# Load a batch of images
image_paths = ["../resources/image1.jpg", "../resources/image2.jpg", "../resources/image3.jpg"]
images = load_images_batch(image_paths)

# Initialize MTCNN detector for batch processing
mtcnn = MTCNN(device="CPU:0")

# Detect faces and landmarks in the batch
results = mtcnn.detect_faces(images, batch_stack_justification="center")

# Plot results for each image in the batch
for i, image in enumerate(images):
    plt.figure()
    plt.imshow(plot(image, results[i]))
    plt.title(f"Results for image {i+1}")
    plt.show()
```

### Conclusion

Batch processing in MTCNN allows you to efficiently detect faces and facial landmarks across multiple images. By passing a list of images or URIs directly to the detector, MTCNN handles padding and justification internally, making the process seamless. This feature is ideal for applications that require large-scale face detection, such as video processing or image batch analysis.
