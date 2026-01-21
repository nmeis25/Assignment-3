# Assignment 3: Network Activity Type Classification

## Project Overview

This project implements a machine learning model for classifying network activity types from the Darknet dataset. The model predicts **8 different network activities** with **93.81% accuracy**, demonstrating strong multi-class classification performance.

---

## Key Results

### Model Performance Summary

| Metric | Value |
|------|------|
| Overall Accuracy | **93.81%** |
| Cross-Validation Accuracy | **93.50% ± 0.16%** |
| Number of Classes | 8 |
| Best Model | Random Forest Classifier |

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|------|---------|-----------|--------|---------|
| Random Forest | 93.81% | 0.873 | 0.899 | 0.883 |
| Neural Network | 69.68% | - | - | - |

---

## Activity Classification Performance

| Activity Type | Precision | Recall | F1-Score | Support |
|-------------|----------|--------|---------|---------|
| P2P | 99.9% | 99.9% | 99.9% | 9,704 |
| Browsing | 97.5% | 97.4% | 97.5% | 9,292 |
| Audio-Streaming | 95.5% | 88.8% | 92.0% | 4,267 |
| File-Transfer | 91.0% | 89.3% | 90.1% | 2,235 |
| VOIP | 74.5% | 92.8% | 82.7% | 713 |
| Chat | 89.0% | 76.3% | 82.1% | 2,325 |
| Video-Streaming | 75.8% | 89.2% | 82.0% | 1,950 |
| Email | 75.4% | 85.3% | 80.1% | 1,228 |

---

## Dataset Information

### Original Dataset

| Property | Value |
|--------|------|
| Total Samples | 158,616 |
| Features | 85 |
| Original Labels | 11 |

### After Preprocessing

| Property | Value |
|--------|------|
| Clean Samples | 158,566 |
| Features Used | 81 |
| Activity Types | 8 |

### Standardized Labels

- Audio-Streaming  
- Browsing  
- Chat  
- Email  
- File-Transfer  
- P2P  
- VOIP  
- Video-Streaming  

---

## Class Distribution

| Activity | Count | Percentage |
|--------|-------|------------|
| P2P | 48,520 | 30.6% |
| Browsing | 46,457 | 29.3% |
| Audio-Streaming | 21,336 | 13.5% |
| Chat | 11,623 | 7.3% |
| File-Transfer | 11,173 | 7.0% |
| Video-Streaming | 9,748 | 6.1% |
| Email | 6,143 | 3.9% |
| VOIP | 3,566 | 2.3% |

---

## Model Implementation

**Random Forest Parameters**

| Parameter | Value |
|---------|------|
| Trees | 200 |
| Max Depth | 20 |
| Min Samples Split | 5 |
| Min Samples Leaf | 2 |
| Class Weight | Balanced |
| Random State | 42 |

### Preprocessing Steps

- Label standardization
- NaN row removal (50 rows)
- IP encoding (factorization)
- RobustScaler feature scaling
- Stratified train-test split

---

## Feature Importance (Top 15)

| Feature | Importance |
|--------|-----------|
| Src IP | 12.94% |
| Dst IP | 8.77% |
| Idle Max | 6.60% |
| Dst Port | 5.22% |
| Idle Min | 4.49% |
| Idle Mean | 4.28% |
| Flow IAT Min | 2.86% |
| Flow IAT Max | 2.66% |
| Flow IAT Mean | 2.48% |
| Flow Bytes/s | 2.39% |
| Src Port | 2.39% |
| Flow Packets/s | 2.30% |
| Flow Duration | 2.28% |
| Fwd Packets/s | 2.17% |
| Bwd Packets/s | 2.12% |

**Key Insight:** Timing features and IP information are the strongest predictors.

---

## Performance Insights

### Excellent (F1 > 90%)

- P2P
- Browsing
- Audio-Streaming

### Good (F1 80–90%)

- File-Transfer
- VOIP
- Chat
- Video-Streaming

### Acceptable

- Email

---

## Sample Predictions

| Sample | True | Predicted | Confidence | Correct |
|------|------|----------|-----------|--------|
| 1 | Video-Streaming | Audio-Streaming | 47.17% | ❌ |
| 2 | Audio-Streaming | Audio-Streaming | 98.57% | ✅ |
| 3 | P2P | P2P | 99.83% | ✅ |
| 4 | P2P | P2P | 100.00% | ✅ |
| 5 | Chat | Chat | 82.39% | ✅ |

**Success Rate:** 80%

---

## Comparison with Assignment 2

| Feature | Assignment 2 | Assignment 3 |
|-------|-------------|-------------|
| Classes | 4 | 8 |
| Accuracy | 99.89% | 93.81% |
| Task | Traffic Type | Activity Type |
| Best Model | Random Forest | Random Forest |

---

## Generated Files

### Model Files

- activity_model_20260121_224846.pkl  ( I can't upload it due to size.)
- activity_scaler_20260121_224846.pkl  
- activity_encoder_20260121_224846.pkl  

### Data Files

- activity_feature_importance.csv  
- activity_metadata_20260121_224846.json  

### Reports

- activity_classification_report_20260121_224846.txt  

---

## Technical Architecture

**Pipeline**

Data → Preprocessing → Feature Engineering → Scaling
→ Training → Cross-Validation → Evaluation → Saving


---

## Business Applications

- Network Security
- Traffic Analysis
- QoS Optimization
- Anomaly Detection

---

## Conclusion

### Achievements

- 93.81% accuracy
- Excellent cross-validation stability
- Near-perfect P2P detection
- Interpretable feature importance
- Production-ready artifacts

### Future Improvements

- Improve Email classification
- Reduce streaming confusion
- Balance rare classes

---

##  How to Use

### Requirements

pip install pandas numpy scikit-learn

Run assignment3.py

Load Model
import joblib

- model = joblib.load('activity_model_20260121_224846.pkl')
- scaler = joblib.load('activity_scaler_20260121_224846.pkl')
- encoder = joblib.load('activity_encoder_20260121_224846.pkl')

- new_data_scaled = scaler.transform(new_data)
- predictions = model.predict(new_data_scaled)
- activity_names = encoder.inverse_transform(predictions)




