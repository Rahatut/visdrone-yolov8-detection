# Drone-Based Human and Car Detection using YOLOv8

## Overview

This project implements a drone-based object detection and human counting system using YOLOv8 and the VisDrone Dataset dataset. The system is designed to detect pedestrians and cars from aerial imagery and estimate the total number of detected humans in each frame.

The project demonstrates a complete computer vision workflow including:

* Dataset understanding and preprocessing
* Class filtering and annotation remapping
* Model training using transfer learning
* Human and car detection
* Human counting
* Visualization and evaluation

The model was trained using a pretrained YOLOv8s detector and evaluated on validation images from the VisDrone dataset.

---

# Features

* Real-time pedestrian and car detection
* Human counting from detection outputs
* YOLOv8-based transfer learning pipeline
* Custom dataset preprocessing and class filtering
* Bounding box visualization with confidence scores
* Batch inference support
* FPS and inference-time evaluation
* Precision, Recall, mAP50, and mAP50-95 evaluation
* Visualization of prediction outputs and training metrics

---

# Dataset

Dataset used: VisDrone Dataset

The VisDrone dataset contains aerial images captured using drones across various urban environments and traffic scenes. The dataset includes annotations for multiple object categories such as:

* Pedestrians
* Cars
* Buses
* Trucks
* Bicycles
* Motorcycles
* Vans
* Tricycles

For this project:

* Only pedestrian and car classes were selected
* Original class labels were filtered and remapped into a two-class setup
* YOLO-format annotations were used directly for training

## Dataset Structure

```text
VisDrone_Dataset/
│
├── VisDrone2019-DET-train/
│   ├── images/
│   └── labels/
│
├── VisDrone2019-DET-val/
│   ├── images/
│   └── labels/
│
└── VisDrone2019-DET-test-dev/
```

---

# Technologies Used

* Python
* YOLOv8
* PyTorch
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Google Colab

---

# Dataset Download

## Configure Kaggle API

```python
import os

os.environ['KAGGLE_USERNAME'] = 'YOUR_USERNAME'
os.environ['KAGGLE_KEY'] = 'YOUR_KAGGLE_KEY'
```

## Download Dataset

```bash
kaggle datasets download -d banuprasadb/visdrone-dataset
```

## Extract Dataset

```python
import zipfile

with zipfile.ZipFile('/content/visdrone-dataset.zip', 'r') as z:
    z.extractall('/content/VisDrone')
```

---

# Preprocessing

The preprocessing pipeline performs:

* Class filtering
* Label remapping
* Dataset restructuring
* Annotation verification

## Selected Classes

| Original Class | New Class |
| -------------- | --------- |
| Pedestrian (0) | 0         |
| Car (3)        | 1         |

Images without pedestrian or car annotations were excluded from the final training dataset.

Bounding boxes were visualized during preprocessing to verify annotation correctness and label alignment.

## Dataset Challenges

Several challenges were observed during dataset analysis:

* Very small pedestrians due to aerial viewpoints
* Dense urban traffic scenes
* Partial object occlusion
* Scale variation between near and distant objects
* Complex backgrounds and illumination conditions

---

# Model Training

The project uses a pretrained YOLOv8s model for transfer learning.

## Training Configuration

```python
model.train(
    data="/content/visdrone.yaml",
    epochs=37,
    imgsz=640,
    batch=16,
    device=0,
    project="/content/runs",
    name="visdrone_humans_cars",
    save=True,
    save_period=5
)
```

## Training Features

The training pipeline includes:

* Transfer learning using YOLOv8s
* Mosaic augmentation
* Random scaling
* Horizontal flipping
* HSV color augmentation

A higher image resolution improves detection performance for very small aerial objects such as distant pedestrians and vehicles.

---

# Inference Pipeline

The trained model performs:

* Pedestrian detection
* Car detection
* Bounding box generation
* Confidence score prediction
* Human counting

Each prediction result includes:

* Bounding boxes
* Predicted class labels
* Confidence scores
* Total detected human count

Different bounding box colors are used for pedestrians and cars to improve visualization clarity.

---

# Evaluation Metrics

The model was evaluated using:

* Precision
* Recall
* mAP50
* mAP50-95
* FPS (Frames Per Second)
* Average inference time

Metrics were extracted directly from YOLOv8 training logs.

## Results

The YOLOv8s model showed stable convergence throughout training, with both training and validation losses decreasing consistently over time. Detection performance improved steadily across epochs, particularly for pedestrian detection, which is challenging in aerial imagery due to small object sizes and crowd density.

| Metric                         | Epoch 1 | Epoch 37 | Observation                                                      |
| ------------------------------ | ------- | -------- | ---------------------------------------------------------------- |
| Box Loss                       | 1.60    | 1.30     | Significant reduction, indicating improved localization accuracy |
| Classification Loss            | 1.16    | 0.73     | Strong improvement in class prediction capability                |
| DFL Loss                       | 0.96    | 0.86     | Better bounding box refinement during training                   |
| Precision                      | 0.54    | 0.70     | Reduction in false positive detections                           |
| Recall                         | 0.41    | 0.56     | Improved object detection coverage                               |
| mAP50                          | 0.41    | 0.61     | Strong improvement in detection performance                      |
| mAP50-95                       | 0.23    | 0.36     | Improved localization quality across IoU thresholds              |
| Validation Box Loss            | 1.46    | 1.17     | Validation performance improved consistently                     |
| Validation Classification Loss | 0.99    | 0.69     | Better generalization on validation data                         |
| Validation DFL Loss            | 0.93    | 0.86     | Stable validation localization refinement                        |
| Learning Rate                  | 0.00055 | 0.00048  | Scheduler gradually reduced learning rate for stable convergence |

### Training Analysis

Several important trends were observed during training:

* Loss curves consistently decreased, indicating stable optimization and effective transfer learning from the pretrained YOLOv8s weights.
* Precision improved steadily, suggesting that the model learned to reduce incorrect detections over time.
* Recall increased gradually, meaning the detector became more capable of identifying pedestrians and cars across complex aerial scenes.
* mAP50 and mAP50-95 improved throughout training, demonstrating stronger overall detection quality and bounding box localization.
* Validation losses followed similar trends to training losses, indicating that the model generalized reasonably well without severe overfitting.

---

# Sample Output

The system generates:

* Detected pedestrians and cars
* Bounding boxes with confidence scores
* Human count overlay
* Saved inference visualizations

## Example Output

```text
Humans detected: 14
FPS: 28.7
mAP50: 0.59
```

---

# Applications

Possible applications include:

* Drone surveillance
* Smart traffic monitoring
* Crowd analysis
* Urban monitoring
* Intelligent transportation systems
* Public safety systems

---

# Challenges Faced

Major implementation challenges included:

* Small object detection from drone viewpoints
* Dense crowd occlusion
* GPU runtime limitations
* Memory usage during high-resolution training
* Balancing detection accuracy and inference speed

---

### Observed Strengths

* Effective detection of cars and pedestrians from aerial viewpoints.
* Good performance on small and medium-sized objects after high-resolution training.
* Stable convergence despite dense urban scenes and occlusions.
* Real-time capable inference performance using YOLOv8s.

### Observed Limitations

* Detection quality decreases in highly crowded scenes with severe occlusion.
* Very small pedestrians at long distances are occasionally missed.
* Human counting accuracy depends directly on detection accuracy.
* Only two object classes were trained, limiting scene understanding capability.

### Overall Outcome

The final model successfully achieved the project objectives by:

* Detecting pedestrians and cars from drone imagery
* Performing automated human counting
* Generating visualized prediction outputs with confidence scores
* Achieving stable detection performance with efficient inference speed

The results demonstrate that YOLOv8s is effective for lightweight aerial object detection tasks under constrained GPU training environments.

---

# License

This project was developed for academic and educational purposes only.

