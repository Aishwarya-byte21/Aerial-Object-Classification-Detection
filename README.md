# 🛰️ Aerial Object Classification & Detection

An end-to-end Deep Learning project that performs:

- 🐦 Bird vs Drone Classification (CNN + Transfer Learning)
- 🎯 Object Detection using YOLOv8
- 🌐 Streamlit Web Deployment

---

## 📌 Project Overview

This project builds a complete aerial object recognition system capable of:

1. Classifying images as **Bird or Drone**
2. Detecting objects with bounding boxes using YOLOv8
3. Deploying the solution via Streamlit for real-time predictions

---

## 🏗️ Project Architecture

Image → Preprocessing →  
Classification Model (Transfer Learning) →  
YOLOv8 Detection Model →  
Streamlit UI → Prediction Output

---

## 🚀 Features

- Custom CNN model
- Transfer Learning model (higher accuracy)
- YOLOv8 object detection
- Confusion Matrix & Classification Report
- Streamlit Web App
- Clean GitHub repository structure

---

## 📊 Model Performance

### 🔹 Transfer Learning Model
- Accuracy: ~97%
- F1 Score: ~0.97

### 🔹 YOLOv8 Detection
- mAP50: ~97%
- Precision: ~96%
- Recall: ~94%

---

## 🖥️ Streamlit Demo

Upload image →  
✔️ Classification result (Bird/Drone)  
✔️ Confidence score  
✔️ YOLO bounding box detection  

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- Scikit-learn

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Aishwarya-byte21/Aerial-Object-Classification-Detection.git
cd Aerial-Object-Classification-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit app:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Aerial_Object_Classification/
│
├── app.py
├── cnn_train.py
├── transfer_train.py
├── evaluate_model.py
├── evaluate_transfer.py
├── yolo_dataset/
├── requirements.txt
└── README.md
```

---

## 👩‍💻 Author

**Aishwarya J**  
Deep Learning & Computer Vision Enthusiast

---

## ⭐ If You Like This Project

Give it a star ⭐ on GitHub!