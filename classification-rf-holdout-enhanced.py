"""
Enhanced Holdout Detection with Multiple Model Architectures

This script aims for 100% holdout detection accuracy by combining:
1. Multiple model types (RF, CNN)
2. Advanced uncertainty quantification
3. Learned feature embeddings
4. Ensemble voting
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("TensorFlow not available - using RF only")


def load_time_split_sequences(base_dir, seq_len, step, train_frac, excluded_class=None):
    """Load sequences with time-based train/test split."""
    X_tr, y_tr, X_te, y_te = [], [], [], []
    X_holdout, y_holdout = [], []

    for label in sorted(os.listdir(base_dir)):
        label_dir = os.path.join(base_dir, label)
        csv_path = os.path.join(label_dir, f"{label}.csv")
        if not os.path.isfile(csv_path) or "idle" in label.lower():
            continue

        data = pd.read_csv(csv_path).drop(columns="Target")
        n_rows = data.shape[0]
        split = int(n_rows * train_frac)

        if label == excluded_class:
            for start in range(0, n_rows - seq_len + 1, step):
                X_holdout.append(data[start:start+seq_len].to_numpy())
                y_holdout.append(label)
            continue

        for start in range(0, split - seq_len + 1, step):
            X_tr.append(data[start:start+seq_len].to_numpy())
            y_tr.append(label)

        for start in range(split, n_rows - seq_len + 1, step):
            X_te.append(data[start:start+seq_len].to_numpy())
            y_te.append(label)

    return (
        np.array(X_tr) if X_tr else np.array([]),
        np.array(y_tr),
        np.array(X_te) if X_te else np.array([]),
        np.array(y_te),
        np.array(X_holdout) if X_holdout else np.array([]),
        np.array(y_holdout)
    )


def create_cnn_model(input_shape, num_classes):
    """Create a 1D CNN for time series classification."""
    model = keras.Sequential([
        # Conv block 1
        layers.Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Conv block 2
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Conv block 3
        layers.Conv1D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_embedding_extractor(model):
    """Extract embeddings (pre-softmax features) from CNN."""
    return keras.Model(
        inputs=model.input,
        outputs=model.layers[-2].output  # Before softmax layer
    )


def compute_mahalanobis_distance(X_test, X_train):
    """
    Compute Mahalanobis distance - better than Euclidean for correlated features.
    Falls back to Euclidean if singular covariance matrix.
    """
    try:
        # Compute mean and covariance of training data
        mean = np.mean(X_train, axis=0)
        cov = np.cov(X_train, rowvar=False)
        
        # Add small regularization to prevent singular matrix
        cov += np.eye(cov.shape[0]) * 1e-6
        
        # Compute inverse covariance
        cov_inv = np.linalg.inv(cov)
        
        # Compute Mahalanobis distance for each test sample
        distances = []
        for x in X_test:
            diff = x - mean
            distance = np.sqrt(diff.T @ cov_inv @ diff)
            distances.append(distance)
        
        return np.array(distances)
    except:
        # Fallback to Euclidean if Mahalanobis fails
        return compute_min_distance(X_test, X_train, 'euclidean')


def compute_min_distance(X_test, X_train, metric='euclidean'):
    """Compute minimum distance from each test sample to training samples."""
    if len(X_train) > 5000:
        indices = np.random.choice(len(X_train), 5000, replace=False)
        X_train_sample = X_train[indices]
    else:
        X_train_sample = X_train
    
    batch_size = 1000
    min_distances = []
    
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        distances = cdist(batch, X_train_sample, metric=metric)
        min_distances.extend(np.min(distances, axis=1))
    
    return np.array(min_distances)


def compute_knn_distance(X_test, X_train, k=5):
    """
    Compute average distance to k nearest neighbors.
    More robust than single nearest neighbor.
    """
    if len(X_train) > 5000:
        indices = np.random.choice(len(X_train), 5000, replace=False)
        X_train_sample = X_train[indices]
    else:
        X_train_sample = X_train
    
    batch_size = 1000
    knn_distances = []
    
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        distances = cdist(batch, X_train_sample, metric='euclidean')
        # Get k smallest distances for each sample
        k_smallest = np.partition(distances, k, axis=1)[:, :k]
        knn_distances.extend(np.mean(k_smallest, axis=1))
    
    return np.array(knn_distances)


def compute_confidence_features(model, X, model_type='rf'):
    """Compute confidence metrics from model predictions."""
    if model_type == 'rf':
        pred_proba = model.predict_proba(X)
    else:  # CNN
        pred_proba = model.predict(X, verbose=0)
    
    max_proba = np.max(pred_proba, axis=1)
    
    # Entropy
    epsilon = 1e-10
    entropy = -np.sum(pred_proba * np.log(pred_proba + epsilon), axis=1)
    
    # Margin
    sorted_proba = np.sort(pred_proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    
    # Prediction variance across top-k predictions
    top_k_std = np.std(sorted_proba[:, -3:], axis=1)
    
    return {
        'max_proba': max_proba,
        'entropy': entropy,
        'margin': margin,
        'top_k_std': top_k_std,
        'pred_proba': pred_proba
    }


def flag_uncertain_multi_metric(conf_features, dist_euclidean, dist_knn, 
                                 dist_mahalanobis=None,
                                 proba_thresh=0.6, margin_thresh=0.3,
                                 euclidean_thresh=None, knn_thresh=None,
                                 mahalanobis_thresh=None,
                                 voting='any'):
    """
    Advanced flagging with multiple metrics and voting strategies.
    
    voting: 'any' (flag if ANY metric triggers), 
            'majority' (flag if MAJORITY trigger),
            'all' (flag if ALL metrics trigger)
    """
    n_samples = len(conf_features['max_proba'])
    flags = np.zeros((n_samples, 5), dtype=bool)  # 5 different signals
    
    # Signal 1: Low confidence
    flags[:, 0] = conf_features['max_proba'] < proba_thresh
    
    # Signal 2: Low margin
    flags[:, 1] = conf_features['margin'] < margin_thresh
    
    # Signal 3: High Euclidean distance
    if euclidean_thresh is not None:
        flags[:, 2] = dist_euclidean > euclidean_thresh
    
    # Signal 4: High KNN distance
    if knn_thresh is not None:
        flags[:, 3] = dist_knn > knn_thresh
    
    # Signal 5: High Mahalanobis distance
    if mahalanobis_thresh is not None and dist_mahalanobis is not None:
        flags[:, 4] = dist_mahalanobis > mahalanobis_thresh
    
    # Apply voting strategy
    if voting == 'any':
        final_flags = np.any(flags, axis=1)
    elif voting == 'majority':
        final_flags = np.sum(flags, axis=1) >= (flags.shape[1] / 2)
    elif voting == 'all':
        final_flags = np.all(flags, axis=1)
    else:
        final_flags = np.any(flags, axis=1)
    
    return final_flags, flags


def evaluate_holdout_enhanced(model, X_train_flat, X_test, y_test, 
                              X_holdout, y_holdout, excluded_class,
                              model_type='rf'):
    """Enhanced evaluation with multiple distance metrics."""
    
    # Flatten for distance computation
    X_test_flat = X_test.reshape(len(X_test), -1)
    X_holdout_flat = X_holdout.reshape(len(X_holdout), -1)
    
    # Get predictions
    if model_type == 'rf':
        y_pred_test = model.predict(X_test_flat)
        y_pred_holdout = model.predict(X_holdout_flat)
    else:
        y_pred_test = np.argmax(model.predict(X_test, verbose=0), axis=1)
        y_pred_holdout = np.argmax(model.predict(X_holdout, verbose=0), axis=1)
    
    # Compute confidence features
    if model_type == 'rf':
        conf_test = compute_confidence_features(model, X_test_flat, 'rf')
        conf_holdout = compute_confidence_features(model, X_holdout_flat, 'rf')
    else:
        conf_test = compute_confidence_features(model, X_test, 'cnn')
        conf_holdout = compute_confidence_features(model, X_holdout, 'cnn')
    
    # Compute multiple distance metrics
    print("\n  Computing distance metrics...")
    
    # Euclidean distance
    dist_euclidean_test = compute_min_distance(X_test_flat, X_train_flat, 'euclidean')
    dist_euclidean_holdout = compute_min_distance(X_holdout_flat, X_train_flat, 'euclidean')
    
    # KNN distance (more robust)
    dist_knn_test = compute_knn_distance(X_test_flat, X_train_flat, k=5)
    dist_knn_holdout = compute_knn_distance(X_holdout_flat, X_train_flat, k=5)
    
    # Mahalanobis distance
    try:
        dist_maha_test = compute_mahalanobis_distance(X_test_flat, X_train_flat)
        dist_maha_holdout = compute_mahalanobis_distance(X_holdout_flat, X_train_flat)
        use_mahalanobis = True
    except:
        dist_maha_test = dist_maha_holdout = None
        use_mahalanobis = False
    
    # Compute adaptive thresholds (using stricter percentiles)
    proba_thresh = max(0.7, np.percentile(conf_test['max_proba'], 3))
    margin_thresh = max(0.4, np.percentile(conf_test['margin'], 3))
    euclidean_thresh = np.percentile(dist_euclidean_test, 97)
    knn_thresh = np.percentile(dist_knn_test, 97)
    maha_thresh = np.percentile(dist_maha_test, 97) if use_mahalanobis else None
    
    print(f"\n  Adaptive thresholds (stricter - 3rd/97th percentile):")
    print(f"    Probability: {proba_thresh:.3f}")
    print(f"    Margin: {margin_thresh:.3f}")
    print(f"    Euclidean dist: {euclidean_thresh:.3f}")
    print(f"    KNN dist: {knn_thresh:.3f}")
    if use_mahalanobis:
        print(f"    Mahalanobis dist: {maha_thresh:.3f}")
    
    # Flag uncertain predictions with multiple voting strategies
    flags_test, signals_test = flag_uncertain_multi_metric(
        conf_test, dist_euclidean_test, dist_knn_test, dist_maha_test,
        proba_thresh, margin_thresh, euclidean_thresh, knn_thresh, maha_thresh,
        voting='any'  # Flag if ANY metric triggers
    )
    
    flags_holdout, signals_holdout = flag_uncertain_multi_metric(
        conf_holdout, dist_euclidean_holdout, dist_knn_holdout, dist_maha_holdout,
        proba_thresh, margin_thresh, euclidean_thresh, knn_thresh, maha_thresh,
        voting='any'
    )
    
    # Calculate metrics
    detection_rate = np.sum(flags_holdout) / len(flags_holdout) if len(flags_holdout) > 0 else 0
    fpr = np.sum(flags_test) / len(flags_test) if len(flags_test) > 0 else 0
    
    non_flagged_mask = ~flags_test
    if np.sum(non_flagged_mask) > 0:
        accuracy_non_flagged = accuracy_score(y_test[non_flagged_mask], y_pred_test[non_flagged_mask])
    else:
        accuracy_non_flagged = 0.0
    
    # Analyze which signals are most effective
    print(f"\n  Signal Analysis (% triggered):")
    signal_names = ['Low Prob', 'Low Margin', 'High Euclid', 'High KNN', 'High Maha']
    for i, name in enumerate(signal_names):
        if i == 4 and not use_mahalanobis:
            continue
        known_trigger = np.sum(signals_test[:, i]) / len(signals_test) * 100
        holdout_trigger = np.sum(signals_holdout[:, i]) / len(signals_holdout) * 100
        print(f"    {name:12s}: Known={known_trigger:5.1f}%, Holdout={holdout_trigger:5.1f}%")
    
    results = {
        'excluded_class': excluded_class,
        'model_type': model_type,
        'detection_rate': detection_rate,
        'fpr': fpr,
        'accuracy_non_flagged': accuracy_non_flagged,
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'avg_proba_known': np.mean(conf_test['max_proba']),
        'avg_proba_holdout': np.mean(conf_holdout['max_proba']),
        'avg_dist_euclidean_known': np.mean(dist_euclidean_test),
        'avg_dist_euclidean_holdout': np.mean(dist_euclidean_holdout),
        'avg_dist_knn_known': np.mean(dist_knn_test),
        'avg_dist_knn_holdout': np.mean(dist_knn_holdout),
    }
    
    return results


def main():
    base_dir = "custom_datasets"
    seq_len = 3  # 3 seconds at 1Hz sampling
    step = 3
    train_frac = 0.7
    use_cnn = HAS_TF  # Use CNN if TensorFlow available
    
    # Get all classes
    all_classes = [
        d for d in sorted(os.listdir(base_dir))
        if os.path.isdir(os.path.join(base_dir, d)) and "idle" not in d.lower()
    ]
    
    print(f"Found {len(all_classes)} classes: {all_classes}")
    print(f"Window size: {seq_len} samples = 3 seconds at 1Hz")
    print(f"Model type: {'CNN' if use_cnn else 'Random Forest'}")
    print("=" * 80)
    
    all_results = []
    
    for excluded_class in all_classes:
        print(f"\n{'='*80}")
        print(f"Experiment: Holding out class '{excluded_class}'")
        print(f"{'='*80}")
        
        # Load data
        X_train, y_train, X_test, y_test, X_holdout, y_holdout = load_time_split_sequences(
            base_dir, seq_len, step, train_frac, excluded_class=excluded_class
        )
        
        if len(X_train) == 0 or len(X_holdout) == 0:
            print(f"  Skipping {excluded_class}: insufficient data")
            continue
        
        print(f"\n  Train samples: {len(y_train)} | Test: {len(y_test)} | Holdout: {len(y_holdout)}")
        print(f"  Input shape: {X_train.shape}")
        
        # Prepare data based on model type
        if use_cnn:
            # CNN uses 3D input (samples, timesteps, features)
            # Create label encoding
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
            
            # Build and train CNN
            model = create_cnn_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                num_classes=len(np.unique(y_train))
            )
            
            print(f"\n  Training CNN...")
            start_time = time.time()
            
            history = model.fit(
                X_train, y_train_encoded,
                validation_split=0.2,
                epochs=50,
                batch_size=32,
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                ]
            )
            
            train_time = time.time() - start_time
            print(f"  Training time: {train_time:.2f}s")
            
            X_train_flat = X_train.reshape(len(X_train), -1)
            model_type = 'cnn'
            
        else:
            # RF uses flattened input
            X_train_flat = X_train.reshape(len(X_train), -1)
            X_test_flat = X_test.reshape(len(X_test), -1)
            
            model = RandomForestClassifier(
                n_estimators=200,  # Increased from 100
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            
            print(f"\n  Training Random Forest...")
            start_time = time.time()
            model.fit(X_train_flat, y_train)
            train_time = time.time() - start_time
            print(f"  Training time: {train_time:.2f}s")
            
            model_type = 'rf'
        
        # Evaluate
        results = evaluate_holdout_enhanced(
            model, X_train_flat, X_test, y_test,
            X_holdout, y_holdout, excluded_class,
            model_type=model_type
        )
        results['train_time'] = train_time
        
        print(f"\n  Results:")
        print(f"    Detection rate: {results['detection_rate']:.1%}")
        print(f"    False positive rate: {results['fpr']:.1%}")
        print(f"    Test accuracy: {results['test_accuracy']:.4f}")
        print(f"    Accuracy (non-flagged): {results['accuracy_non_flagged']:.4f}")
        
        all_results.append(results)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}\n")
    
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('detection_rate', ascending=False)
    
    print(df_results[[
        'excluded_class', 'model_type', 'detection_rate', 'fpr',
        'accuracy_non_flagged', 'test_accuracy'
    ]].to_string(index=False))
    
    print(f"\n\nAverage Performance:")
    print(f"  Detection rate: {df_results['detection_rate'].mean():.1%} (±{df_results['detection_rate'].std():.1%})")
    print(f"  False positive rate: {df_results['fpr'].mean():.1%} (±{df_results['fpr'].std():.1%})")
    print(f"  Test accuracy: {df_results['test_accuracy'].mean():.4f} (±{df_results['test_accuracy'].std():.4f})")
    
    # Save
    output_file = f"eval_results/rf_holdout_enhanced_{'cnn' if use_cnn else 'rf'}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
