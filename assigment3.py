# assignment3_activity_classification.py
"""
Assignment 3: Network Activity Type Classification
Predicts Label2 (specific network activities) from Darknet dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("=" * 70)
print("ASSIGNMENT 3: NETWORK ACTIVITY TYPE CLASSIFICATION")
print("=" * 70)
print("Predicting Label2 (specific network activities)")
print("=" * 70)

# ============================================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================================
print("\n1. LOADING AND EXPLORING DATA")
print("-" * 40)

df = pd.read_csv("Darknet.csv")
print(f"Original dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)[:10]}...")

# Show Label2 distribution (this is our target)
print("\nLabel2 (Activity Type) Distribution:")
label2_counts = df['Label2'].value_counts()
print(label2_counts)

print(f"\nTotal unique activity types: {df['Label2'].nunique()}")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n2. PREPROCESSING DATA")
print("-" * 40)

# Keep both labels for now
df_clean = df.copy()

# Drop unnecessary columns
df_clean = df_clean.drop(["Flow ID", "Timestamp"], axis=1)

# Check for and handle infinite/NaN values
print("Checking for data issues...")
df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
nan_counts = df_clean.isna().sum()
print(f"NaN counts:\n{nan_counts[nan_counts > 0]}")

# Drop rows with NaN
df_clean = df_clean.dropna()
print(f"Shape after cleaning: {df_clean.shape}")

# Store Label1 for potential use
print(f"\nLabel1 distribution (traffic type):")
print(df_clean['Label1'].value_counts())

print(f"\nLabel2 distribution (activity type):")
activity_counts = df_clean['Label2'].value_counts()
print(activity_counts)

# Check if we have enough samples per activity
min_samples = 100
rare_activities = activity_counts[activity_counts < min_samples]
if len(rare_activities) > 0:
    print(f"\nWarning: {len(rare_activities)} activities have less than {min_samples} samples:")
    print(rare_activities)
    print("\nConsider grouping rare activities or using class weights.")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n3. ENGINEERING FEATURES")
print("-" * 40)

# Separate features and target
X = df_clean.drop(['Label1', 'Label2'], axis=1)
y = df_clean['Label2']

# Process IP addresses (same as Assignment 2)
print("Processing IP addresses...")
X['Src IP'] = pd.factorize(X['Src IP'])[0]
X['Dst IP'] = pd.factorize(X['Dst IP'])[0]

# Encode target labels (activity types)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
activity_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

print(f"\nActivity type mapping:")
for activity, code in activity_mapping.items():
    print(f"  {activity}: {code}")

print(f"\nFeatures shape: {X.shape}")
print(f"Number of activity classes: {len(activity_mapping)}")

# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================
print("\n4. SPLITTING DATA")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Check class distribution in splits
print("\nClass distribution in training set:")
train_counts = pd.Series(y_train).value_counts().sort_index()
for idx, count in train_counts.items():
    activity_name = le.inverse_transform([idx])[0]
    print(f"  {activity_name}: {count} samples")

# ============================================================================
# 5. FEATURE SCALING
# ============================================================================
print("\n5. SCALING FEATURES")
print("-" * 40)

# Use RobustScaler to handle outliers
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed.")

# ============================================================================
# 6. MODEL TRAINING - RANDOM FOREST
# ============================================================================
print("\n" + "=" * 70)
print("6. TRAINING RANDOM FOREST CLASSIFIER")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")
rf_model.fit(X_train_scaled, y_train)

# Cross-validation
print("\nPerforming cross-validation...")
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train,
                            cv=3, scoring='accuracy', n_jobs=-1)
print(f"CV Accuracy scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Predictions
y_pred_rf = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"\nRandom Forest Test Accuracy: {rf_accuracy:.4f}")

# Detailed classification report
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf,
                            target_names=le.classes_,
                            zero_division=0))

# Confusion matrix
print("\nConfusion Matrix (first 10x10 for readability):")
cm = confusion_matrix(y_test, y_pred_rf)
print("Rows = True labels, Columns = Predicted labels")
print("Activity order:", le.classes_[:10])
print(cm[:10, :10])

# ============================================================================
# 7. MODEL TRAINING - NEURAL NETWORK
# ============================================================================
print("\n" + "=" * 70)
print("7. TRAINING NEURAL NETWORK")
print("=" * 70)

try:
    # Calculate class weights for neural network
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, class_weights))

    nn_model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=256,
        learning_rate='adaptive',
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )

    print("Training Neural Network...")
    nn_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_nn = nn_model.predict(X_test_scaled)
    nn_accuracy = accuracy_score(y_test, y_pred_nn)

    print(f"\nNeural Network Test Accuracy: {nn_accuracy:.4f}")

    print("\nClassification Report (Neural Network - top classes):")
    report = classification_report(y_test, y_pred_nn,
                                   target_names=le.classes_,
                                   output_dict=True,
                                   zero_division=0)

    # Show top 5 classes by F1-score
    f1_scores = {}
    for class_name in le.classes_:
        if class_name in report:
            f1_scores[class_name] = report[class_name]['f1-score']

    top_classes = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 best predicted activities:")
    for activity, f1 in top_classes:
        print(f"  {activity}: F1-score = {f1:.3f}")

except Exception as e:
    print(f"Neural Network training failed: {e}")
    print("Using Random Forest only...")
    nn_accuracy = 0

# ============================================================================
# 8. MODEL COMPARISON AND SELECTION
# ============================================================================
print("\n" + "=" * 70)
print("8. MODEL COMPARISON")
print("=" * 70)

print(f"Random Forest Accuracy:    {rf_accuracy:.4f}")
if nn_accuracy > 0:
    print(f"Neural Network Accuracy:   {nn_accuracy:.4f}")

# Select best model
if nn_accuracy > 0 and nn_accuracy > rf_accuracy:
    best_model = nn_model
    best_accuracy = nn_accuracy
    best_model_name = "Neural Network"
else:
    best_model = rf_model
    best_accuracy = rf_accuracy
    best_model_name = "Random Forest"

print(f"\nBest Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy:.4f}")

# ============================================================================
# 9. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("9. FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

if best_model_name == "Random Forest":
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Top 15 Most Important Features for Activity Prediction:")
    print(feature_importance.head(15).to_string(index=False))

    # Save feature importance
    feature_importance.to_csv('activity_feature_importance.csv', index=False)
    print("\nFeature importance saved to 'activity_feature_importance.csv'")

# ============================================================================
# 10. SAVE MODELS AND RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("10. SAVING MODELS AND RESULTS")
print("=" * 70)

import joblib
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save models
joblib.dump(best_model, f'activity_model_{timestamp}.pkl')
joblib.dump(scaler, f'activity_scaler_{timestamp}.pkl')
joblib.dump(le, f'activity_encoder_{timestamp}.pkl')

print(f"Saved files:")
print(f"- activity_model_{timestamp}.pkl (trained model)")
print(f"- activity_scaler_{timestamp}.pkl (feature scaler)")
print(f"- activity_encoder_{timestamp}.pkl (label encoder)")

# Save metadata
metadata = {
    'timestamp': timestamp,
    'best_model': best_model_name,
    'accuracy': float(best_accuracy),
    'num_classes': len(activity_mapping),
    'num_features': X.shape[1],
    'activity_mapping': activity_mapping,
    'class_distribution': dict(zip(le.classes_, np.bincount(y_encoded)))
}

import json

with open(f'activity_metadata_{timestamp}.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"- activity_metadata_{timestamp}.json (metadata)")

# ============================================================================
# 11. TEST PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("11. SAMPLE PREDICTIONS")
print("=" * 70)

# Make sample predictions
print("\nSample predictions from test set:")
sample_indices = np.random.choice(len(X_test), 5, replace=False)

for i, idx in enumerate(sample_indices, 1):
    true_label = le.inverse_transform([y_test[idx]])[0]
    prediction = best_model.predict(X_test_scaled[idx:idx + 1])
    pred_label = le.inverse_transform(prediction)[0]

    if hasattr(best_model, 'predict_proba'):
        probabilities = best_model.predict_proba(X_test_scaled[idx:idx + 1])
        confidence = np.max(probabilities)
    else:
        confidence = 1.0

    print(f"\nSample {i}:")
    print(f"  True activity:  {true_label}")
    print(f"  Predicted:      {pred_label}")
    print(f"  Confidence:     {confidence:.2%}")
    print(f"  Correct:        {true_label == pred_label}")

# ============================================================================
# 12. GENERATE FINAL REPORT
# ============================================================================
print("\n" + "=" * 70)
print("12. FINAL REPORT SUMMARY")
print("=" * 70)

# Performance by class for best model
y_pred_best = best_model.predict(X_test_scaled)
report_dict = classification_report(y_test, y_pred_best,
                                    target_names=le.classes_,
                                    output_dict=True,
                                    zero_division=0)

# Calculate metrics
num_classes = len(activity_mapping)
avg_precision = np.mean([report_dict[cls]['precision'] for cls in le.classes_ if cls in report_dict])
avg_recall = np.mean([report_dict[cls]['recall'] for cls in le.classes_ if cls in report_dict])
avg_f1 = np.mean([report_dict[cls]['f1-score'] for cls in le.classes_ if cls in report_dict])

print(f"\nOVERALL PERFORMANCE:")
print(f"  Accuracy:          {best_accuracy:.4f}")
print(f"  Avg Precision:     {avg_precision:.4f}")
print(f"  Avg Recall:        {avg_recall:.4f}")
print(f"  Avg F1-Score:      {avg_f1:.4f}")
print(f"  Number of classes: {num_classes}")
print(f"  Total samples:     {len(df_clean)}")
print(f"  Features used:     {X.shape[1]}")

# Best and worst performing classes
class_performance = []
for class_name in le.classes_:
    if class_name in report_dict and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
        class_performance.append((
            class_name,
            report_dict[class_name]['precision'],
            report_dict[class_name]['recall'],
            report_dict[class_name]['f1-score'],
            report_dict[class_name]['support']
        ))

# Sort by F1-score
class_performance.sort(key=lambda x: x[3], reverse=True)

print(f"\nTOP 5 BEST PREDICTED ACTIVITIES:")
for i, (name, prec, rec, f1, supp) in enumerate(class_performance[:5], 1):
    print(f"  {i}. {name:20s} F1={f1:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, Samples={supp}")

print(f"\nTOP 5 WORST PREDICTED ACTIVITIES:")
for i, (name, prec, rec, f1, supp) in enumerate(class_performance[-5:], 1):
    print(f"  {i}. {name:20s} F1={f1:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, Samples={supp}")

print("\n" + "=" * 70)
print("ASSIGNMENT 3 COMPLETED SUCCESSFULLY!")
print("=" * 70)

# Save final report
report_text = f"""
ASSIGNMENT 3 - ACTIVITY TYPE CLASSIFICATION REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

MODEL: {best_model_name}
ACCURACY: {best_accuracy:.4f}

DATASET:
- Total samples: {len(df_clean)}
- Number of activity types: {num_classes}
- Number of features: {X.shape[1]}

PERFORMANCE METRICS:
- Accuracy: {best_accuracy:.4f}
- Average Precision: {avg_precision:.4f}
- Average Recall: {avg_recall:.4f}
- Average F1-Score: {avg_f1:.4f}

BEST PERFORMING ACTIVITIES:
"""
for i, (name, prec, rec, f1, supp) in enumerate(class_performance[:5], 1):
    report_text += f"{i}. {name}: F1={f1:.3f} (samples={supp})\n"

report_text += f"\nMODEL FILES SAVED:\n"
report_text += f"- activity_model_{timestamp}.pkl\n"
report_text += f"- activity_scaler_{timestamp}.pkl\n"
report_text += f"- activity_encoder_{timestamp}.pkl\n"
report_text += f"- activity_metadata_{timestamp}.json\n"

with open(f'activity_classification_report_{timestamp}.txt', 'w') as f:
    f.write(report_text)

print(f"\nFinal report saved to: activity_classification_report_{timestamp}.txt")