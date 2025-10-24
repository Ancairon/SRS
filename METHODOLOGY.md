# Methodology Comparison: Original vs Holdout Detection

## Overview
Both scripts use a **single Random Forest classifier**, but the holdout version adds uncertainty detection mechanisms on top of the standard RF predictions.

---

## Original Script (`classification-rf.py`)

### What It Does:
- **Standard supervised classification** with Random Forest
- Trains on 70% of data, tests on 30% from **all classes**
- Makes predictions using `rf.predict()` 
- Reports accuracy and classification metrics

### Methodology:
1. Load all classes with train/test split
2. Train single RF model on all classes
3. Predict on test set
4. Report accuracy

### Output:
- Single accuracy score
- Classification report (precision/recall/f1)
- Training/inference time

**Key Point**: This is a **closed-set** classifier - assumes all test samples belong to one of the known classes.

---

## Holdout Script (`classification-rf-holdout.py`)

### What It Does:
- **Open-set classification** experiment with Random Forest
- Still uses a **single RF model** per iteration, BUT...
- Adds **uncertainty detection layer** on top of RF predictions
- Tests if the model can identify when it sees an unknown class

### Methodology:

#### 1. **One-Class-Out Cross-Validation**
   - Iterates through each class
   - Holds one class completely out of training (the "unknown" class)
   - Trains RF on remaining 6 classes only
   - Tests detection on the held-out class

#### 2. **Single RF Model Per Iteration**
   ```python
   rf = RandomForestClassifier(
       n_estimators=100,
       random_state=42,
       max_depth=None,
       min_samples_split=2,
       n_jobs=-1
   )
   rf.fit(X_train, y_train)  # Train on 6 known classes
   ```
   - **Same RF architecture** as original (just enhanced with more parameters)
   - No ensemble of multiple RFs
   - No separate novelty detection models

#### 3. **Uncertainty Detection Layer (NEW)**
   After getting RF predictions, we compute multiple signals:
   
   **A. Confidence-based signals:**
   - **Max Probability**: `rf.predict_proba()` - highest class probability
   - **Margin**: Difference between top-2 probabilities
   - **Entropy**: Uncertainty across all class probabilities
   
   **B. Distance-based signal:**
   - **Euclidean distance**: Minimum distance from test sample to any training sample
   - This is the **key innovation** - doesn't rely on RF output
   
   **C. Combined flagging logic:**
   ```python
   flag_as_uncertain = (
       max_proba < threshold OR
       margin < threshold OR
       distance > threshold
   )
   ```

#### 4. **Adaptive Threshold Computation**
   - Thresholds are computed from **known test data** (not training data)
   - Uses percentiles: 5th for confidence, 95th for distance
   - Minimum floor values prevent over-flagging

---

## Key Differences

| Aspect | Original | Holdout |
|--------|----------|---------|
| **Model** | Single RF | Single RF (per iteration) |
| **Training** | All 7 classes | 6 classes (1 held out) |
| **Test Set** | All 7 classes | 6 known + 1 unknown class |
| **Prediction** | `rf.predict()` only | `rf.predict()` + uncertainty detection |
| **Output** | Accuracy | Detection rate + FPR + accuracy |
| **Scenario** | Closed-set | Open-set |
| **Innovation** | Standard RF | **Distance metric** + confidence combo |

---

## What We Changed - The Innovations

### 1. **Distance Metric (Most Important)**
   ```python
   def compute_distance_to_training(X_test, X_train):
       distances = cdist(X_test, X_train, metric='euclidean')
       min_distances = np.min(distances, axis=1)
       return min_distances
   ```
   - Computes how far each test sample is from the nearest training sample
   - **This is independent of the RF model** - pure geometric measure
   - **Why it works**: Unknown classes are typically far from training data

### 2. **Multi-Signal Uncertainty Detection**
   - Don't rely solely on RF confidence (overconfident!)
   - Combine multiple signals with OR logic
   - Any signal can trigger "uncertain" flag

### 3. **Adaptive Thresholding**
   - Thresholds learned from known test data
   - Different percentiles for different metrics
   - Prevents over-fitting to training distribution

### 4. **Comprehensive Evaluation**
   - Test each class as "unknown"
   - Measure both detection rate AND false positive rate
   - Identify which classes are distinguishable

---

## Why This Approach Works

### ✅ Strengths:
1. **No additional models needed** - uses same RF, just adds post-processing
2. **Distance metric is model-agnostic** - would work with any classifier
3. **Interpretable** - can see why samples are flagged (low confidence OR far from training)
4. **Comprehensive evaluation** - tests all classes systematically

### ⚠️ Limitations:
1. **Still one RF model** - not an ensemble or multi-model approach
2. **Distance computation is expensive** - O(n*m) for n test × m train samples
3. **Thresholds are heuristic** - 5th/95th percentiles are chosen empirically
4. **Euclidean distance** - may not capture complex feature relationships

---

## What We Did NOT Do

❌ **Create multiple RF models** (it's still a single RF per experiment)  
❌ **Use DBSCAN** (mentioned in requirements but distance metric proved sufficient)  
❌ **Use Local Outlier Factor extensively** (imported but not used in final approach)  
❌ **Train a separate novelty detector** (one-class SVM, isolation forest, etc.)  

---

## The Core Innovation

### **Post-hoc Uncertainty Quantification**

Instead of:
```python
# Standard approach
prediction = rf.predict(X)
```

We do:
```python
# Our approach
prediction = rf.predict(X)
confidence = rf.predict_proba(X).max(axis=1)
distance = min_distance_to_training(X)

if confidence < threshold OR distance > threshold:
    flag_as_uncertain(prediction)
```

**Bottom Line**: Same RF classifier, but we add a **safety check** using multiple signals to detect when predictions might be unreliable because the input is too different from training data.

This is conceptually similar to:
- **Conformal prediction** (without the mathematical guarantees)
- **Out-of-distribution detection** in deep learning
- **Anomaly detection** but applied post-classification

The key insight from our experiments: **Distance matters more than confidence** for Random Forests in open-set scenarios!
