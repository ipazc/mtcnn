## Networks and Stages in MTCNN

MTCNN (Multitask Cascaded Convolutional Networks) is a powerful framework for face detection and alignment, built around three main networks: **PNet**, **RNet**, and **ONet**. These networks are organized into distinct *stages*, each refining the output of the previous one. Together, they enable MTCNN to achieve high accuracy in face detection and landmark alignment.

### Overview of Stages and Networks

The MTCNN pipeline consists of three stages:

- **Stage 1 (PNet)**: The Proposal Network stage, where initial candidate face regions are generated.
- **Stage 2 (RNet)**: The Refinement Network stage, where these proposals are refined and filtered.
- **Stage 3 (ONet)**: The Output Network stage, where the final bounding boxes and facial landmarks are predicted.

Each stage includes the following key operations:

1. **Image pyramid scaling** (in Stage 1 only).
2. **Face detection and bounding box regression**.
3. **Non-Maximum Suppression (NMS)** with thresholds to filter out overlapping and low-confidence boxes.
4. **Landmark regression** (in Stage 3).

Now, let’s break down each stage and its corresponding network.

---

### 1. Stage 1: PNet and Image Pyramid Construction

#### Function of Stage 1

The first stage of MTCNN uses the **Proposal Network (PNet)** to scan the image at multiple scales. Since faces can appear at different sizes, the input image is **scaled down progressively** to create an *image pyramid*. This allows PNet to detect faces at various sizes across the image.

At each scale, PNet slides over the image and generates **bounding box proposals** for regions that might contain faces. These proposals include:

- **Bounding Box Regressions**: Initial estimates for the bounding boxes.
- **Face/Non-Face Classification**: A score indicating whether a region contains a face or not.

#### Image Pyramid and Proposal Generation

1. **Image Pyramid Construction**: The input image is scaled down multiple times, forming an image pyramid. Each scale produces a resized image, and the smallest scale ensures that even small faces are detected.
   
2. **PNet Processing**: For each scaled image, PNet scans regions using a sliding window, proposing candidate face regions and outputting bounding boxes and confidence scores.
   
3. **Scale-Specific NMS**: For each scale, PNet outputs a set of candidate regions. These are processed with **Non-Maximum Suppression (NMS)** to remove overlapping boxes that likely represent the same face. A **threshold** controls how aggressive the NMS is at filtering boxes.

4. **Aggregate Proposals Across Scales**: The candidate boxes from all scales are combined into a single list. NMS is applied again to merge overlapping detections across scales, ensuring that only the best bounding boxes remain.

#### Strengths of PNet
- The image pyramid ensures detection of faces at multiple scales.
- PNet is fast and efficient, generating many face proposals in a short amount of time.

```text
Input Image -> Image Pyramid -> PNet -> Scale-specific NMS -> Combined Proposals -> Final NMS
```

---

### 2. Stage 2: RNet (Refinement Network)

#### Function of Stage 2

After the proposals from PNet are filtered through NMS, they are passed to the **Refinement Network (RNet)**. The purpose of RNet is to further refine these bounding boxes, rejecting **false positives** and improving the precision of the face regions. Like PNet, RNet performs:

- **Bounding Box Regression**: Adjusts the bounding boxes to better fit the faces.
- **Face/Non-Face Classification**: Classifies whether each region contains a face or not.

#### Key Operations in RNet

1. **Input from PNet**: The refined proposals from PNet are cropped from the original image and resized to a standard size before being fed into RNet.
   
2. **Bounding Box Refinement**: RNet processes these regions and further refines the bounding box coordinates, producing a more accurate estimate of where the face is located.
   
3. **Face Classification and NMS**: RNet classifies each region as face or non-face and applies another round of **NMS** to filter out overlapping or low-confidence detections. This stage also has a specific **NMS threshold**, which controls how strictly overlapping boxes are filtered.

#### Strengths of RNet
- RNet provides more accurate bounding box predictions and reduces false positives.
- The additional round of NMS refines the proposals from PNet, resulting in better precision.

```text
Refined Proposals -> RNet -> Bounding Box Refinement -> NMS -> Refined Detections
```

---

### 3. Stage 3: ONet (Output Network)

#### Function of Stage 3

In the final stage, the **Output Network (ONet)** refines the bounding boxes even further and detects **five facial landmarks** (eyes, nose, and mouth corners). ONet provides three outputs:

- **Bounding Box Regression**: Final adjustments to the bounding boxes.
- **Face/Non-Face Classification**: Classifies whether a region contains a face or not.
- **Landmark Regression**: Predicts the positions of five facial landmarks for each face.

#### Key Operations in ONet

1. **Input from RNet**: The refined regions from RNet are again cropped and resized to the appropriate input size for ONet.
   
2. **Final Bounding Box Refinement**: ONet produces the final adjustments to the bounding boxes, ensuring maximum accuracy in detecting the face regions.

3. **Facial Landmark Detection**: In addition to bounding boxes, ONet predicts the coordinates of five key landmarks (left eye, right eye, nose, left mouth corner, right mouth corner).

4. **NMS with Landmark Consideration**: The final round of **NMS** is applied, but this time the landmark predictions are also taken into account when merging overlapping boxes. The NMS threshold is tuned to preserve the best bounding boxes and corresponding landmarks.

#### Strengths of ONet
- ONet provides highly accurate face detection results, as well as landmark predictions that are essential for facial alignment tasks.
- The final NMS ensures that the best bounding boxes and landmarks are kept while filtering redundant detections.

```text
Final Proposals -> ONet -> Bounding Box Refinement -> Landmark Detection -> NMS -> Final Bounding Boxes + Landmarks
```

---

### Thresholds and Non-Maximum Suppression (NMS)

Throughout the MTCNN pipeline, **Non-Maximum Suppression (NMS)** is a key operation used to filter overlapping bounding boxes. Each stage of the network applies NMS after detecting face proposals. NMS removes redundant boxes by keeping only the box with the highest confidence score when there are multiple overlapping boxes representing the same face. 

At each stage, a **threshold** is applied to control how aggressively NMS filters the proposals:

- **PNet NMS Threshold**: This threshold is more lenient to keep as many proposals as possible in the early stage.
- **RNet NMS Threshold**: A stricter threshold is used to discard false positives and refine the bounding boxes.
- **ONet NMS Threshold**: The strictest threshold is used to produce the final high-confidence detections.

