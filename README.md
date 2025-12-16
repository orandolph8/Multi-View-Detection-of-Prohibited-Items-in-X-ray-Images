# Multi-View Detection of Prohibited Items in X-ray Images

## Overview
This repository contains a deep learning–based computer vision project for **automated detection of prohibited items in X-ray baggage images**. The system combines multi-label classification, object detection, and **dual-view (multi-view) fusion** to improve detection reliability in cluttered security screening environments.

The project demonstrates how leveraging **orthogonal X-ray views** significantly increases recall for occluded or ambiguous threat items.

---

## Key Components
- **Multi-label classification** using ResNet-18 for global threat detection
- **Object detection** with YOLOv8-Nano (single-stage) and Faster R-CNN (two-stage)
- **Dual-view late fusion** across Optical Level (OL) and Side View (SD) images
- **Explainability** using Grad-CAM to validate model focus
- Automated data pipeline supporting YOLO, COCO, and classification formats

---

## Dataset
- **Dual-View X-ray (DvXray) Dataset**
- Each sample includes two orthogonal X-ray projections (OL and SD)
- 15 prohibited item categories with bounding-box annotations

---

## Results (Highlights)
- YOLOv8 mAP@0.50: **84.81%**
- Recall after multi-view fusion: **91.01%**
- Multi-label classifier mAP: **0.8122**
- Strong localization with Faster R-CNN, but lower class discrimination under imbalance

---

## Technologies
- Python, PyTorch
- YOLOv8, Faster R-CNN
- ResNet-18 / ResNet-50
- Grad-CAM
- HPC GPU training environment

---

## Course Context
Presented by **Owen Randolph, Soon-Hyuck Lee, Marcos Fernandez, and Pratham Dedhiya**  
as part of **E533 – Computer Vision**  
Indiana University, Luddy School of Informatics, Computing, and Engineering
