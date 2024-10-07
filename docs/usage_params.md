The `mtcnn.detect_faces()` method in MTCNN provides a powerful and flexible way to detect faces and facial landmarks. While the method is easy to use out of the box, it also offers a variety of parameters that allow you to fine-tune the detection process based on your specific needs. This guide explains each parameter in detail, how they influence the results, and the impact of adjusting them.

---

## Key Parameters

### `image` (Required)

This is the primary input to the method. You can provide:

- A single image.
- A batch of images (as a list).
- URIs or paths to image files.

The `mtcnn.detect_faces()` method is capable of working with individual images or batches of images, allowing flexible input types.

### `fit_to_image` (Default: `True`)

This option ensures that the detected bounding boxes fit within the boundaries of the image. When set to `True`, any bounding box that extends beyond the edges of the image will be adjusted to stay within the visible area. This is useful when faces near the edges of the image are detected.

**When to adjust**: Set this to `False` if you want to allow detections that might go beyond the image (for example, when faces are partially outside the frame).

### `limit_boundaries_landmarks` (Default: `False`)

Similar to `fit_to_image`, but specific to facial landmarks. When enabled, landmarks (like eyes, nose, mouth corners) are adjusted so that they remain within the image boundaries. 

**When to adjust**: If you're working with images where facial features could be near or beyond the edge of the image, set this to `True` to ensure all landmarks stay visible.

### `box_format` (Default: `"xywh"`)

Determines the format of the bounding boxes in the output. You can choose between:

- `"xywh"`: `[X1, Y1, width, height]`, where `X1, Y1` are the top-left corner coordinates and `width, height` represent the size.
- `"xyxy"`: `[X1, Y1, X2, Y2]`, where `X1, Y1` are the top-left corner coordinates, and `X2, Y2` are the bottom-right corner coordinates.

**When to adjust**: Change to `"xyxy"` if you need to work with absolute coordinates for both corners of the box instead of width and height.

### `output_type` (Default: `"json"`)

This defines the format in which the detection results are returned. You can choose between:

- `"json"`: The output is a list of dictionaries, each containing:
    - `"box"`: The bounding box coordinates.
    - `"keypoints"`: A dictionary with the detected landmarks.
    - `"confidence"`: The confidence score of the detection.
- `"numpy"`: The output is a NumPy array with structured data.

**When to adjust**: Use `"numpy"` if you are processing the results programmatically and prefer working with NumPy arrays.

### `postprocess` (Default: `True`)

Enabling this option ensures that several postprocessing steps are applied to the results:

- Bounding boxes and landmarks are adjusted to fit within the image, based on the `fit_to_image` and `limit_boundaries_landmarks` settings.
- Padding from batch processing is removed, ensuring clean output for images of different sizes.

**When to adjust**: Set this to `False` if you want raw outputs from each stage of the network without any adjustments.

### `batch_stack_justification` (Default: `"center"`)

When processing a batch of images, smaller images are padded to match the largest image in the batch. This parameter controls how these smaller images are aligned in the padded tensor:

- **`"top"`**: Aligns smaller images to the top edge of the padded area, centered horizontally.
- **`"topleft"`**: Aligns smaller images to the top-left corner of the padded area.
- **`"topright"`**: Aligns smaller images to the top-right corner of the padded area.
- **`"bottom"`**: Aligns smaller images to the bottom edge of the padded area, centered horizontally.
- **`"bottomleft"`**: Aligns smaller images to the bottom-left corner of the padded area.
- **`"bottomright"`**: Aligns smaller images to the bottom-right corner of the padded area.
- **`"left"`**: Aligns smaller images to the left edge of the padded area, centered vertically.
- **`"right"`**: Aligns smaller images to the right edge of the padded area, centered vertically.
- **`"center"`**: Centers smaller images both vertically and horizontally within the padded area.

**When to adjust**: Use different justifications if you want to control how images are aligned during batch processing.

---

## Fine-Tuning Parameters for Each Detection Stage

MTCNN detects faces through three stages: **PNet**, **RNet**, and **ONet**. Each stage has its own set of parameters that you can adjust to control detection sensitivity, scaling, and thresholds.

### StagePNet (Proposal Network)

**PNet** is the first network in the MTCNN pipeline. It quickly scans the image at multiple scales to propose candidate face regions.

- **`min_face_size`** *(Default: 20)*: This controls the minimum size of the face (in pixels) that the detector will consider. Faces smaller than this will be ignored.
    - **When to adjust**: Lower this value if you're working with images where faces are very small, or increase it if you want to ignore smaller faces for performance reasons.
- **`min_size`** *(Default: 12)*: Defines the minimum size for the smallest scale in the image pyramid. Smaller values will lead to a finer scan at smaller face sizes.
    - **When to adjust**: Lowering this can detect smaller faces but may slow down detection.
- **`scale_factor`** *(Default: 0.709)*: This controls the scaling factor for the image pyramid, determining how much the image is resized at each level.
    - **When to adjust**: A smaller value creates more image scales, leading to more detailed detections but slower performance.
- **`threshold_pnet`** *(Default: 0.6)*: The confidence threshold for accepting face proposals from PNet. Lower thresholds result in more proposals, while higher thresholds discard more uncertain detections.
    - **When to adjust**: Lower it to catch more potential face candidates (at the cost of more false positives).
- **`nms_pnet1`** *(Default: 0.5)* and **`nms_pnet2`** *(Default: 0.7)*: These are the IoU (Intersection over Union) thresholds for Non-Maximum Suppression (NMS), a technique used to remove overlapping bounding boxes.
    - **nms_pnet1**: Applied per scale.
    - **nms_pnet2**: Applied across all scales.
    - **When to adjust**: Increase if too many boxes overlap, decrease to retain more overlapping proposals.

### StageRNet (Refinement Network)

**RNet** refines the face proposals from PNet, removing false positives and further improving the bounding box quality.

- **`threshold_rnet`** *(Default: 0.7)*: The confidence threshold for accepting face proposals in RNet. Higher values make the network more conservative.
    - **When to adjust**: Lower if RNet is rejecting too many proposals, raise if it’s accepting too many false positives.

- **`nms_rnet`** *(Default: 0.7)*: The IoU threshold for NMS after RNet processing.
    - **When to adjust**: Adjust as needed to control overlap in bounding boxes.

### StageONet (Output Network)

**ONet** is the final stage of the MTCNN pipeline. It provides the most refined bounding boxes and predicts facial landmarks (eyes, nose, mouth corners).

- **`threshold_onet`** *(Default: 0.8)*: The confidence threshold for face proposals in ONet. Like the earlier thresholds, higher values make the detector more conservative.
    - **When to adjust**: If landmarks are too sparse, lower this value. If you're getting too many incorrect faces, raise it.

- **`nms_onet`** *(Default: 0.7)*: The IoU threshold for NMS after ONet.
    - **When to adjust**: Fine-tune to remove overlapping boxes while keeping enough valid face detections.

---

## Practical Examples

### Adjusting Detection Sensitivity

If you're looking to detect smaller faces or fine-tune the detection sensitivity, adjusting `min_face_size` and the stage thresholds can help.

```python
results = mtcnn.detect_faces(
    image,
    min_face_size=15,  # Detect smaller faces
    threshold_pnet=0.5,  # More proposals from PNet
    threshold_rnet=0.6,  # Loosen RNet filtering
    threshold_onet=0.7   # More final faces accepted by ONet
)
```

### Batch Processing with Custom Padding

When processing a batch of images, you can control how smaller images are padded relative to larger ones.

```python
results = mtcnn.detect_faces(
    images_list,
    batch_stack_justification="topleft"  # Align smaller images to the top-left
)
```

### Disabling Postprocessing

If you need the raw output directly from the network stages without any adjustments, you can disable postprocessing.

```python
results = mtcnn.detect_faces(
    image,
    postprocess=False  # Get raw detection results
)
```

### Changing Bounding Box Format

To return bounding boxes in `[X1, Y1, X2, Y2]` format instead of the default `[X1, Y1, width, height]`:

```python
results = mtcnn.detect_faces(
    image,
    box_format="xyxy"
)
```

---

## Summary

- **Single Image**: Returns a list of detected faces, with each face represented by a dictionary containing the bounding box, landmarks, and confidence score.
- **Batch of Images**: Returns a list of lists, where each sublist contains the detections for one image.
- The default parameters provide good results in most cases, but you may need to adjust thresholds, face size settings, and scaling factors depending on the specifics of your task. Fine-tuning these parameters will allow you to balance detection accuracy, speed, and sensitivity.
- Increasing thresholds generally makes the detector more conservative (fewer false positives but potentially missing some faces), while decreasing thresholds makes it more aggressive (detecting more faces but possibly increasing false positives).
- Adjusting the `scale_factor` affects the number of scales in the image pyramid and can impact detection performance and speed.
- When processing large batches or high-resolution images, consider running the detector on a GPU for better performance.
