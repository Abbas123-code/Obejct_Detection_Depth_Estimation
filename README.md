# 🚗 Object Detection and Depth Estimation using YOLOv8 & KITTI Dataset

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-red)
![Status](https://img.shields.io/badge/Status-Academic%20report-blue)

---

##  Table of Contents
1. [Project Overview](#-project-overview)
2. [Authors](#-authors)
3. [Approach](#-approach)
4. [Depth Estimation Method](#-depth-estimation-method)
5. [Results](#-results)
6. [Technologies & Libraries](#-technologies--libraries)
7. [Sample Outputs](#-sample-outputs)
8. [How to Run](#-how-to-run)

---

##  Project Overview
This project uses **YOLOv8x** for **2D object detection** and leverages **intrinsic camera parameters** from the **KITTI dataset** to estimate the **3D depth of detected cars**.  
The main goal is to analyze how accurately depth estimation corresponds to ground-truth values and identify factors affecting discrepancies.

---

##  Authors
- **Manoj Nagendrakumar** (Master’s Mechatronics)  
- **Mohammed Kumail Abbas** (Master’s Mechatronics)  

**Guided by:** Prof. Dr. Stefan Elser

---

## 🔥 Approach
1. **Object Detection**  
   - **YOLOv8x** was used to detect cars and draw bounding boxes.  
   - Bounding box coordinates (`x1, y1, x2, y2`) were saved for each detected object.

2. **Ground Truth Comparison**  
   - Ground truth boxes were extracted from KITTI labels.
   - **Intersection over Union (IoU)** was used to match predictions with ground truth.  

   **Precision & Recall:**

   \[
   Precision = \frac{TP}{TP + FP}, \quad Recall = \frac{TP}{TP + FN}
   \]

3. **Filtering False Positives**  
   - Predictions with IoU < 0.5 were treated as false positives.

---

## 🧠 Depth Estimation Method
Depth estimation was done using **camera intrinsic parameters**.

### **Formulas**

> Note: notation follows the report. This README does not state the camera height or the ground-plane assumption behind `Z = h / dz`; see `object_detection.pdf` for the derivation.

1. **Intrinsic Matrix**  
K = | fx   0   cx |  
   | 0    fy  cy |  
   | 0    0    1 |

2. **3D Direction Vector**  
d = inverse(K) · p

3. **Depth & 3D Coordinates**  
Z = h / dz  
X = Z · dx  
Y = Z · dy

4. **Distance from Camera**  
Distance = √(X² + Y² + Z²)

---

##  Results

###  Images with Proper Detection
- Accurate depth estimation for clear, unobstructed cars.  
- Good performance up to **40 meters**.

###  Complex Scenarios
- Errors due to:
  - Occlusions
  - False positives
  - Poor lighting/weather conditions

###  No Detection
- Cases where no cars were detected in the frame.

### **Overall**
- Depth estimation error increases as objects move further away.

---

##  Technologies & Libraries
- **Python 3.x**
- **YOLOv8 (Ultralytics)**
- **OpenCV (cv2)**
- **NumPy**
- **Matplotlib**

---

##  Sample Outputs

| Detection | Depth Estimation | Accuracy Plot |
|-----------|-----------------|---------------|
| <img width="1116" height="328" alt="image" src="https://github.com/user-attachments/assets/3a761c86-1e6e-4042-b7f9-bff1b384a0ea" /> | <img width="685" height="655" alt="image" src="https://github.com/user-attachments/assets/ab5e1150-2100-43f6-88bb-87cf2bf20e88" /> |

---

## Repository contents and status
This repository currently contains the project **report** (`object_detection.pdf`) and this summary only.
The Python source, model weights and the KITTI data are **not** included, so the results cannot be re-run from this repository.
The YOLOv8 model is the pretrained Ultralytics checkpoint; KITTI data must be obtained from the official KITTI site under its own terms.
