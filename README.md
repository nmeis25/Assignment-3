# Assignment-3

Network Activity Type Classification 
Project Overview
This project implements a machine learning model for classifying network activity types from the Darknet dataset. The model successfully predicts 8 different network activities with 93.81% accuracy, demonstrating exceptional performance in multi-class classification.

-------------

Key Results
Model Performance Summary
Overall Accuracy: 93.81% (0.9381)

Cross-Validation Accuracy: 93.50% (± 0.16%)

Number of Activity Classes: 8

Best Model: Random Forest Classifier

------------------------------------------------




### **Performance Comparison**
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **93.81%** | 0.873 | 0.899 | 0.883 |
| Neural Network | 69.68% | - | - | - |


------------------------------------------------

### **Activity Classification Performance**
| Activity Type | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **P2P** | 99.9% | 99.9% | 99.9% | 9,704 |
| **Browsing** | 97.5% | 97.4% | 97.5% | 9,292 |
| **Audio-Streaming** | 95.5% | 88.8% | 92.0% | 4,267 |
| **File-Transfer** | 91.0% | 89.3% | 90.1% | 2,235 |
| **VOIP** | 74.5% | 92.8% | 82.7% | 713 |
| **Chat** | 89.0% | 76.3% | 82.1% | 2,325 |
| **Video-Streaming** | 75.8% | 89.2% | 82.0% | 1,950 |
| **Email** | 75.4% | 85.3% | 80.1% | 1,228 |

## Dataset Information

### **Original Dataset**
- **Total Samples**: 158,616
- **Features**: 85 columns
- **Original Activity Types**: 11 unique labels

### **After Preprocessing & Standardization**
- **Clean Samples**: 158,566 (50 removed due to NaN)
- **Features Used**: 81
- **Activity Types**: **8** (after standardization)

### **Class Distribution (After Standardization)**
| Activity Type | Count | Percentage |
|---------------|-------|------------|
| P2P | 48,520 | 30.6% |
| Browsing | 46,457 | 29.3% |
| Audio-Streaming | 21,336 | 13.5% |
| Chat | 11,623 | 7.3% |
| File-Transfer | 11,173 | 7.0% |
| Video-Streaming | 9,748 | 6.1% |
| Email | 6,143 | 3.9% |
| VOIP | 3,566 | 2.3% |

##  Model Implementation

### **Best Model: Random Forest Classifier**
- **Number of Trees**: 200
- **Max Depth**: 20
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2
- **Class Weight**: Balanced (for imbalanced data)
- **Random State**: 42

## Feature Importance Analysis

### **Top 15 Most Important Features**
| Rank | Feature | Importance |
|:----:|---------|------------|
| 1 | **Src IP** | 12.94% |
| 2 | **Dst IP** | 8.77% |
| 3 | **Idle Max** | 6.60% |
| 4 | Dst Port | 5.22% |
| 5 | Idle Min | 4.49% |
| 6 | Idle Mean | 4.28% |
| 7 | Flow IAT Min | 2.86% |
| 8 | Flow IAT Max | 2.66% |
| 9 | Flow IAT Mean | 2.48% |
| 10 | Flow Bytes/s | 2.39% |
| 11 | Src Port | 2.39% |
| 12 | Flow Packets/s | 2.30% |
| 13 | Flow Duration | 2.28% |
| 14 | Fwd Packets/s | 2.17% |
| 15 | Bwd Packets/s | 2.12% |

## Model Performance Insights

### **Excellent Performance (F1 > 90%)**
- **P2P**: **99.9%** F1-score - Near perfect classification
- **Browsing**: **97.5%** F1-score - Excellent performance
- **Audio-Streaming**: **92.0%** F1-score - Very good

### **Good Performance (F1 80-90%)**
- **File-Transfer**: **90.1%** F1-score - Good
- **VOIP**: **82.7%** F1-score - Good (high recall)
- **Chat**: **82.1%** F1-score - Good
- **Video-Streaming**: **82.0%** F1-score - Good

### **Acceptable Performance (F1 < 80%)**
- **Email**: **80.1%** F1-score - Acceptable

##  Sample Predictions
| Sample | True Activity | Predicted | Confidence | Correct |
|--------|---------------|-----------|------------|---------|
| 1 | Video-Streaming | Audio-Streaming | 47.17% | ❌ |
| 2 | Audio-Streaming | Audio-Streaming | 98.57% | ✅ |
| 3 | P2P | P2P | 99.83% | ✅ |
| 4 | P2P | P2P | 100.00% | ✅ |
| 5 | Chat | Chat | 82.39% | ✅ |

##Comparison with Assignment 2

### **Assignment 2 vs Assignment 3**
| Aspect | Assignment 2 | Assignment 3 |
|--------|--------------|--------------|
| **Target** | Label1 (Traffic Type) | Label2 (Activity Type) |
| **Classes** | 4 | 8 |
| **Accuracy** | 99.89% | 93.81% |
| **Top Features** | Src IP, Dst IP, Idle Max | Src IP, Dst IP, Idle Max |
| **Best Model** | Random Forest | Random Forest |

### **Key Differences**
1. **Complexity**: Assignment 3 is more complex (8 classes vs 4)
2. **Performance**: Slightly lower accuracy due to complexity
3. **Feature Importance**: Similar top features across both tasks
4. **Model Choice**: Random Forest outperforms Neural Network in both assignments

##  Technical Architecture

### **Data Pipeline**
| Step | Description | Tools Used |
|------|-------------|------------|
| 1 | Data Loading | pandas |
| 2 | Preprocessing | pandas, numpy |
| 3 | Feature Engineering | scikit-learn |
| 4 | Model Training | scikit-learn |
| 5 | Evaluation | scikit-learn |
| 6 | Deployment | joblib |

##  Generated Files

Model Files
activity_model_20260121_224846.pkl - Trained Random Forest model ( I can't upload it due to size.)

activity_scaler_20260121_224846.pkl - Feature scaler

activity_encoder_20260121_224846.pkl - Label encoder

Data Files
activity_feature_importance.csv - Feature importance rankings

activity_metadata_20260121_224846.json - Model metadata

Report Files
activity_classification_report_20260121_224846.txt - Comprehensive summary report

Technical Architecture
Data Pipeline
Data Loading → Preprocessing → Feature Engineering → Scaling

Model Training → Cross-Validation → Evaluation → Saving

Key Technical Decisions
RobustScaler: Chosen over StandardScaler for better outlier handling

Class Weighting: Applied to handle imbalanced classes

Stratified Splitting: Ensures representative class distribution

Cross-Validation: 3-fold CV for reliable performance estimation



Business/Research Implications
High-Value Applications
Network Security: Detect unauthorized activities

Quality of Service: Optimize bandwidth for different activities

Traffic Analysis: Understand network usage patterns

Anomaly Detection: Identify unusual activity patterns

Key Findings for Practitioners
P2P Detection: Near perfect (99.9% accuracy)

Streaming vs Browsing: High accuracy differentiation

Timing Features Critical: Idle times and IAT are highly predictive

IP Information Important: Source/Dest IPs provide strong signals


Conclusion
Success Metrics Achieved

93.81% accuracy on 8-class classification


Excellent cross-validation results (93.50% ± 0.16%)


Near-perfect P2P detection (99.9% F1-score)


Robust feature importance analysis


Production-ready model with all artifacts saved

Model Strengths
High Accuracy: 93.81% is excellent for 8-class problem

Consistent Performance: Low variance in cross-validation

Interpretable: Clear feature importance insights

Scalable: Can handle large datasets efficiently

Areas for Improvement
Email Classification: Lowest F1-score (80.1%)

Streaming Confusion: Some confusion between Audio/Video streaming

Rare Classes: Could benefit from more balanced sampling


How to Use
Requirements ->
pip install pandas numpy scikit-learn
Run the Model
python assignment3_fixed.py
Load and Predict

import joblib

Load saved model
model = joblib.load('activity_model_20260121_224846.pkl')
scaler = joblib.load('activity_scaler_20260121_224846.pkl')
encoder = joblib.load('activity_encoder_20260121_224846.pkl')

Prepare new data
new_data = ... (must match training features)

Scale and predict
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
activity_names = encoder.inverse_transform(predictions)
