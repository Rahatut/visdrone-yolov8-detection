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

# Project Structure

The repo is kept lightweight. Store datasets and full training runs outside Git, and keep only curated outputs and README assets in the repo.

```text
<repo-root>/
│
├── notebook/
│   └── visdrone_yolov8_detection.ipynb
│
├── outputs/
│   ├── sample_predictions/
│   └── training_results/
│
├── assets/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Post-Training Workflow (Colab)

Use this exact workflow after training finishes.

1. Clean your `runs/` folder

Inside Colab:

```python
import os

os.listdir("/content/runs/visdrone_humans_cars")
```

You mainly need:

```text
weights/
results.csv
results.png
confusion_matrix.png
PR_curve.png
F1_curve.png
P_curve.png
R_curve.png
```

Ignore the rest.

---

2. Create GitHub-ready folders

Run:

```python
import os

folders = [
    "/content/github_repo/assets",
    "/content/github_repo/outputs/sample_predictions",
    "/content/github_repo/outputs/training_results"
]

for f in folders:
    os.makedirs(f, exist_ok=True)

print("Folders created.")
```

---

3. Save sample prediction images

Use 3-5 clean outputs only.

Modify your inference save path:

```python
output_dir = "/content/github_repo/outputs/sample_predictions"
```

Then run inference again on a few images.

Recommended:

* one crowded scene
* one sparse scene
* one difficult small-object scene
* one balanced scene

---

4. Copy YOLO training plots

Run:

```python
import shutil

source = "/content/runs/visdrone_humans_cars"
target = "/content/github_repo/outputs/training_results"

files_to_copy = [
    "results.png",
    "confusion_matrix.png",
    "PR_curve.png",
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png"
]

for file in files_to_copy:
    src = os.path.join(source, file)

    if os.path.exists(src):
        shutil.copy(src, target)

print("Training plots copied.")
```

---

5. Create README assets

Add:

* one architecture screenshot
* one sample output
* maybe training curve preview

Copy them:

```python
shutil.copy(
    "/content/github_repo/outputs/sample_predictions/sample1.jpg",
    "/content/github_repo/assets/sample_output.jpg"
)
```

---

6. Download best weights separately

DO NOT upload huge training folders.

Only keep:

```text
best.pt
```

Optional Google Drive upload:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then copy:

```python
shutil.copy(
    "/content/runs/visdrone_humans_cars/weights/best.pt",
    "/content/drive/MyDrive/best.pt"
)
```

Put Drive link in README.

---

7. Create requirements.txt

In Colab:

```python
requirements = """
ultralytics
torch
opencv-python
matplotlib
numpy
pandas
supervision
roboflow
lapx
"""

with open("/content/github_repo/requirements.txt", "w") as f:
    f.write(requirements)

print("requirements.txt created")
```

---

8. Create `.gitignore`

```python
gitignore = """
dataset/
runs/
*.pt
*.zip
__pycache__/
.ipynb_checkpoints/
"""

with open("/content/github_repo/.gitignore", "w") as f:
    f.write(gitignore)

print(".gitignore created")
```

---

9. Final repo should look like:

```text
drone-human-car-detection-yolov8/
│
├── notebook/
│   └── visdrone_yolov8_detection.ipynb
│
├── outputs/
│   ├── sample_predictions/
│   └── training_results/
│
├── assets/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

10. Final GitHub polish checklist

Before pushing:

* notebook runs top-to-bottom
* remove unnecessary outputs
* rename notebook cleanly
* add screenshots in README
* add metrics table
* add sample output image near top
* ensure repo size stays small

Good ML repos are:

* readable
* reproducible
* visually clear
* not cluttered with raw training artifacts.

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

# License

This project is intended for academic and educational purposes.

---
