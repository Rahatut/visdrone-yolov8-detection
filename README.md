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
    epochs=50,
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

| Metric              | Epoch 1 | Epoch 30 | Observation                              |
| ------------------- | ------- | -------- | ---------------------------------------- |
| Box Loss            | 1.60    | 1.32     | Strong decrease → localization improving |
| Classification Loss | 1.16    | 0.75     | Major improvement in class prediction    |
| DFL Loss            | 0.96    | 0.87     | Bounding box refinement improving        |
| Precision           | 0.54    | 0.70     | Fewer false positives over time          |
| Recall              | 0.41    | 0.55     | Model detecting more objects             |
| mAP50               | 0.41    | 0.59     | Significant detection improvement        |
| mAP50-95            | 0.23    | 0.35     | Better localization quality              |
| Validation Box Loss | 1.46    | 1.18     | Validation performance improving         |
| Learning Rate       | 0.00055 | 0.00070  | Scheduler behaving normally              |


The training results indicate stable convergence of the YOLOv8s model throughout the training process. Training losses consistently decreased while validation precision, recall, and mAP metrics improved over successive epochs, demonstrating effective learning and generalization.

By epoch 30, the model achieved approximately 0.59 mAP50 and 0.35 mAP50-95 on the validation set. Considering the challenges associated with aerial imagery, including very small object sizes, occlusion, and dense urban scenes, the obtained performance is considered reasonable for a lightweight real-time detector.

The results also indicate that the model maintained relatively strong precision while recall remained comparatively lower, suggesting that some distant or highly occluded pedestrians were still missed during detection. This behavior is common in drone-based small-object detection tasks.

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

# Results Summary

The project successfully demonstrates:

* Aerial pedestrian and car detection
* Human counting capability
* Real-time inference performance
* Effective transfer learning using YOLOv8
* Detection visualization and evaluation

The system performs reasonably well despite the challenges associated with aerial imagery and small object detection.

---

# Challenges Faced

Major implementation challenges included:

* Small object detection from drone viewpoints
* Dense crowd occlusion
* GPU runtime limitations
* Memory usage during high-resolution training
* Balancing detection accuracy and inference speed

---

# Strengths and Limitations

## Strengths

* Detects pedestrians and cars effectively
* Fast inference using YOLOv8
* Real-time capable on GPU
* Human counting integrated into inference pipeline
* Lightweight and easy to deploy

## Limitations

* Counting accuracy decreases in crowded scenes
* Very small or heavily occluded pedestrians may be missed
* Model is trained on only two classes
* No multi-object tracking between frames

---

# License

This project was developed for academic and educational purposes only.

