# Drone-Based Human and Car Detection using YOLOv8

## Overview

This project implements a real-time aerial object detection and human counting system using YOLOv8 and the VisDrone dataset. The system is designed to detect pedestrians and cars from drone-captured imagery and estimate the number of detected humans in each frame.

The project focuses on:

* Human detection from aerial viewpoints
* Car detection in urban traffic scenes
* Human counting from detection outputs
* Real-time inference and visualization
* Performance evaluation using standard object detection metrics

The model was trained using transfer learning with a pretrained YOLOv8s architecture and evaluated on validation images from the VisDrone dataset.

---

# Features

* Real-time pedestrian and car detection
* Human counting using detected pedestrian instances
* YOLOv8-based transfer learning pipeline
* Custom preprocessing and class filtering
* Bounding box visualization 
* Batch inference support
* FPS and inference-time evaluation
* Precision, Recall, mAP50, and mAP50-95 evaluation

---

# Dataset

Dataset used: VisDrone Dataset

The VisDrone dataset contains aerial images captured from drone viewpoints across different urban environments. The dataset includes annotations for multiple object categories such as pedestrians, cars, buses, trucks, bicycles, and motorcycles.

For this project:

* Only pedestrian and car categories were selected
* Original class labels were filtered and remapped into a 2-class setup
* YOLO-format annotations were used for training

Dataset structure:

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
* YOLOv8 (Ultralytics)
* OpenCV
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Google Colab

---

# Dataset Download

Configure Kaggle credentials:

```python
import os

os.environ['KAGGLE_USERNAME'] = 'YOUR_USERNAME'
os.environ['KAGGLE_KEY'] = 'YOUR_KAGGLE_KEY'
```

Download dataset:

```bash
kaggle datasets download -d banuprasadb/visdrone-dataset
```

Extract dataset:

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

Selected classes:

| Original Class | New Class |
| -------------- | --------- |
| Pedestrian (0) | 0         |
| Car (3)        | 1         |

Bounding boxes were visualized to verify annotation correctness and label alignment.

---

# Model Training

The project uses a pretrained YOLOv8s model for transfer learning.

Training configuration:

```python
model.train(
    data="/content/visdrone.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    project="/content/runs",
    name="visdrone_humans_cars"
)
```

Training features:

* Transfer learning with YOLOv8s
* Mosaic augmentation
* Random scaling
* Horizontal flipping
* Color-space augmentation

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

---

# Sample Output

The system generates:

* Detected pedestrians and cars
* Bounding boxes with confidence scores
* Human count overlay
* Saved inference visualizations

Example output:

```text
Humans detected: 14
FPS: 28.7
mAP50: 0.81
```

---

# Applications

Possible applications of this system include:

* Drone surveillance
* Smart traffic monitoring
* Crowd analysis
* Urban monitoring
* Intelligent transportation systems
* Public safety systems

---

# Results Summary

The project successfully demonstrates:

* Accurate aerial pedestrian and car detection
* Human counting capability
* Real-time inference performance
* Effective transfer learning using YOLOv8
* Reliable detection visualization and evaluation

---

## Challenges Faced

* Small object detection due to aerial viewpoint
* Dense crowd occlusion in busy scenes
* Limited GPU runtime constraints
* High-resolution training increases memory usage

## Strengths and Limitations

Strengths:

* Detects small pedestrians reasonably well
* Fast inference using YOLOv8
* Real-time capable on GPU
* Human counting integrated directly into inference

Limitations:

* Counting accuracy decreases in crowded scenes
* Occluded pedestrians may be missed
* Model is trained on two classes only
* No tracking between frames


# License

This project is intended for academic and educational purposes.

---
