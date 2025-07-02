
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
3. Standardize features using `StandardScaler`.
4. Train `IsolationForest` model on the normalized data.
5. Save model and scaler using `pickle`.
6. Generate and visualize anomaly scores.
7. Real-time detection with GUI/Web interface.

### Libraries Used:
- `NumPy`: Numerical operations and reshaping
- `Matplotlib`: Anomaly score visualizations
- `Scikit-learn`: Isolation Forest, scaling, evaluation
- `Pickle`: Model serialization
- `OS`: Directory and file handling

---

## 📂 Repository Structure

```
📁 PBL-AI/
├── model_train.py                # Training script for Isolation Forest
├── scaler.pkl                   # Serialized StandardScaler
├── isolation_forest_model.pkl   # Trained Isolation Forest model
├── anomaly_scores.png           # Visualization of anomaly scores
├── requirements.txt             # (Optional) Required Python packages
└── README.md                    # Project documentation (this file)
```

---

## ✅ Project Status

| Task                                       | Status     |
|--------------------------------------------|------------|
| Data Preprocessing & Feature Scaling       | ✅ Completed |
| Model Training with Isolation Forest       | ✅ Completed |
| Saving Models & Visualizations             | ✅ Completed |
| Validation & Anomaly Score Analysis        | ✅ Completed |
| Real-time Video Detection Interface        | ⏳ In Progress |
| GUI/Web Interface for Demo                 | ⏳ In Progress |
| Anomaly Labeling and Categorization        | ⏳ In Progress |
| Backend Integration                        | ⏳ In Progress |

---

## 📊 Testing & Validation

| Test Type                            | Status | Notes                                |
|--------------------------------------|--------|--------------------------------------|
| Model Training & Save                | ✅ Pass | Model and scaler saved successfully  |
| Anomaly Score Evaluation             | ✅ Pass | Score graphs and distributions created |
| Manual Inspection of Anomaly Ratio   | ✅ Pass | Approx. 60% anomalies identified     |

---

## 📦 Deliverables

- ✅ `isolation_forest_model.pkl`: Trained anomaly detection model.
- ✅ `scaler.pkl`: Feature scaling pipeline.
- ✅ `anomaly_scores.png`: Distribution of detected anomalies.
- ✅ `model_train.py`: Full training and validation pipeline.
- 🔜 Real-time video anomaly detection module.

---

## 👥 Team Members

| Name                     | Student ID   | Role             |
|--------------------------|--------------|------------------|
| Aditi Joshi              | 23012630     | Team Lead        |
| Akshat Bansal            | 230111983    | Model Training   |
| Sarthak Singh Choudhary  | 230111855    | Integration & UI |

---

## 🔗 Useful Links

- [Main Repository (MeSarthak/AI)](https://github.com/MeSarthak/AI)
- [Forked/Personal Repo](https://github.com/AditiJoshi1605/PBL-AI)

---

## 📌 Future Work

- Improve real-time detection performance.
- Add a dashboard for anomaly visualization.
- Evaluate with larger, more diverse datasets.
- Compare with other unsupervised methods (e.g., Autoencoders, One-Class SVM).

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

---

## 📞 Contact

Feel free to reach out for collaboration or questions:

- Aditi Joshi – [ADITIJOSHI.23012630@gehu.ac.in](mailto:ADITIJOSHI.23012630@gehu.ac.in)
- Akshat Bansal – [AKSHATBANSAL.230111983@gehu.ac.in](mailto:AKSHATBANSAL.230111983@gehu.ac.in)
- Sarthak Singh Choudhary – [SARTHAKSINGHCHOUDHARY.230111855@gehu.ac.in](mailto:SARTHAKSINGHCHOUDHARY.230111855@gehu.ac.in)

---
