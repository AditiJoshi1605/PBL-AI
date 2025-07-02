
# Human Activity Anomaly Detection System

This project implements an unsupervised anomaly detection system for identifying abnormal human activities using the **Isolation Forest** algorithm. It leverages time-series data representing human activity patterns and detects deviations that may indicate unusual or suspicious behavior.

---

## 🚀 Project Overview

Traditional supervised learning methods rely on labeled datasets to detect anomalies. However, our approach uses **unsupervised learning** to model and detect anomalies based solely on the statistical properties of the input data.

Key highlights:
- Uses `Isolation Forest` for detecting anomalies.
- Works with unlabelled `.npy` formatted human activity sequences.
- Complete pipeline: Preprocessing ➝ Model Training ➝ Evaluation ➝ Visualization ➝ Real-time Inference.
- Deployable for real-time monitoring via webcam or video streams.

---

## 🧠 Approach & Architecture

### System Pipeline:
1. Load human activity sequences (`.npy` format).
2. Flatten the time-series data into 2D vectors.
3. **Apply PCA** to reduce dimensionality while preserving key features.
4. Standardize features using `StandardScaler`.
5. Train `IsolationForest` model on the normalized data.
6. Save model and scaler using `pickle`.
7. Generate and visualize anomaly scores.
8. Real-time detection with GUI/Web interface.

### Libraries Used:
- `NumPy`: Numerical operations and reshaping
- `Matplotlib`: Anomaly score visualizations
- `Scikit-learn`: PCA, Isolation Forest, scaling, evaluation
- `OpenCV` – Real-time video capture and processing
- `Pickle`: Model serialization
- `OS`: Directory and file handling

---

## ✅ Project Status

| Task                                       | Status     |
|--------------------------------------------|------------|
| Data Preprocessing & Feature Scaling       | ✅ Completed |
| Model Training with Isolation Forest       | ✅ Completed |
| Saving Models & Visualizations             | ✅ Completed |
| Validation & Anomaly Score Analysis        | ✅ Completed |
| Real-time Video Detection Interface        | ✅ Completed |
| GUI/Web Interface for Demo                 | ✅ Completed |
| Anomaly Labeling and Categorization        | ⏳ In Progress |
| Backend Integration                        | ⏳ In Progress |

---

## 📊 Testing & Validation

| Test Type                            |   Status   |             Notes                      |
|--------------------------------------|-------------|---------------------------------------|
| Model Training & Save                |   ✅ Pass  | Model and scaler saved successfully    |
| Anomaly Score Evaluation             |   ✅ Pass  | Score graphs and distributions created |
| Manual Inspection of Anomaly Ratio   |   ✅ Pass  | Approx. 60% anomalies identified       |
| PCA Component Visualization          |   ✅ Pass  | PCA variance plot reviewed             |
| Real-Time Frame Evaluation (OpenCV)  | ⏳ Testing | Testing on video input                 |

---

## 📦 Deliverables

- ✅ `isolation_forest_model.pkl`: Trained anomaly detection model.
- ✅ `scaler.pkl`: Feature scaling pipeline.
- ✅ `anomaly_scores.png`: Distribution of detected anomalies.
- ✅ `model_train.py`: Full training and validation pipeline.
- 🔜 Real-time video anomaly detection module.

---

## 📌 Future Scope

- Improve real-time detection performance.
- Add a dashboard for anomaly visualization.
- Evaluate with larger, more diverse datasets.
- Compare with other unsupervised methods (e.g., Autoencoders, One-Class SVM).
- Integrate REST API for mobile/edge usage.

---

## 🛠️ Requirements

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

Recommended contents for `requirements.txt`:
```
numpy
matplotlib
scikit-learn
pickle-mixin
```
