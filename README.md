# 🌲 Forest Analysis System

AI-powered forest monitoring and geospatial analysis platform built using YOLOv8 segmentation, satellite imagery, and computer vision techniques for vegetation analysis, deforestation detection, and environmental monitoring.

---

# 🚀 Features

- Tree region segmentation using YOLOv8
- Forest coverage estimation from satellite imagery
- Deforestation change detection
- Tree density heatmap visualization
- Carbon sequestration estimation
- Coordinate-based forest analysis using Google Maps API
- Real-time image analysis through Streamlit dashboard
- Binary mask generation and overlay visualization

---

# 🛠️ Tech Stack

## Machine Learning & Computer Vision
- YOLOv8 Segmentation
- Python
- OpenCV
- NumPy
- PyTorch

## Web Application
- Streamlit

## APIs & Tools
- Google Maps Static API
- Roboflow
- Matplotlib
- Seaborn

---

# 📊 System Workflow

Satellite Image → YOLOv8 Segmentation → Tree Mask Generation → Coverage Analysis → Heatmaps & Deforestation Detection → Visualization Dashboard

---

# 📸 Screenshots

## Forest Segmentation
![Segmentation](assets/segmentation.png)

## Tree Density Heatmap
![Heatmap](assets/heatmap.png)

## Deforestation Detection
![Deforestation](assets/deforestation.png)

---

# 🌍 Key Functionalities

## Tree Segmentation
Performs pixel-level forest region segmentation using YOLOv8 trained on custom annotated datasets exported in COCO segmentation format.

---

## Forest Coverage Analysis
Calculates vegetation density by comparing segmented tree pixels against total image pixels.

---

## Deforestation Detection
Compares satellite images captured at different time periods to estimate forest loss or gain.

---

## Tree Density Heatmaps
Divides images into smaller regions to visualize dense and sparse vegetation zones.

---

## Coordinate-Based Analysis
Fetches satellite imagery using latitude and longitude coordinates through Google Maps Static API for automated analysis.

---

## Carbon Sequestration Estimation
Estimates stored carbon and annual carbon sequestration capacity using detected forest coverage.

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/gaganpraj7-wq/forest-analysis-system.git
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Application

```bash
streamlit run app.py
```

---

# 📁 Dataset

- Custom forest analysis dataset
- Annotated using Roboflow
- Exported in COCO segmentation format

---

# 🎯 Purpose

This project was developed to automate environmental monitoring and forest analysis using artificial intelligence, satellite imagery, and computer vision workflows.

---

# 🔮 Future Improvements

- Tree species classification
- Drone-based real-time monitoring
- Forest fire prediction
- Biodiversity assessment
- Large-scale ecological monitoring
- Real-time satellite integration
