"""
Direction finding methods for identifying internal correlates.

Two fundamentally different approaches:
1. probe: Linear regression to find direction that best predicts target metric
2. mean_diff: mean(high_metric_samples) - mean(low_metric_samples)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr


def probe_direction(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1000.0,
    use_pca: bool = True,
    pca_components: int = 100,
    bootstrap_splits: List[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Find direction via Ridge regression probe.

    Args:
        X: (n_samples, hidden_dim) activation matrix
        y: (n_samples,) target metric values
        alpha: Ridge regularization strength
        use_pca: Whether to use PCA dimensionality reduction
        pca_components: Number of PCA components
        bootstrap_splits: Pre-generated list of (train_idx, test_idx) tuples for bootstrap.
                         If provided, computes confidence intervals. If None, fits on all data.

    Returns:
        direction: Normalized direction vector (hidden_dim,)
        info: Dict with r2, mae, correlation, and fit details (with std if bootstrap)
    """
    from sklearn.metrics import mean_absolute_error

    def _fit_and_eval(X_train, y_train, X_test, y_test):
        """Fit probe and return metrics and direction."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if use_pca:
            n_comp = min(pca_components, X_train.shape[0], X_train.shape[1])
            pca_model = PCA(n_components=n_comp)
            X_train_final = pca_model.fit_transform(X_train_scaled)
            X_test_final = pca_model.transform(X_test_scaled)
            variance_explained = float(pca_model.explained_variance_ratio_.sum())
        else:
            pca_model = None
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled
            variance_explained = None

        probe = Ridge(alpha=alpha)
        probe.fit(X_train_final, y_train)

        y_pred = probe.predict(X_test_final)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        corr, _ = pearsonr(y_test, y_pred)

        # Extract direction in original space
        if pca_model is not None:
            direction = pca_model.inverse_transform(probe.coef_.reshape(1, -1)).flatten()
            direction = direction / scaler.scale_
        else:
            direction = probe.coef_.copy()
            direction = direction / scaler.scale_

        direction = direction / np.linalg.norm(direction)

        return r2, mae, corr, direction, variance_explained

    if bootstrap_splits is not None and len(bootstrap_splits) > 0:
        # Use pre-generated bootstrap splits
        test_r2s, test_maes, test_corrs = [], [], []

        for train_idx, test_idx in bootstrap_splits:
            r2, mae, corr, _, _ = _fit_and_eval(
                X[train_idx], y[train_idx], X[test_idx], y[test_idx]
            )
            test_r2s.append(r2)
            test_maes.append(mae)
            test_corrs.append(corr)

        # Final direction from first split's training data (canonical)
        train_idx = bootstrap_splits[0][0]
        _, _, _, direction, variance_explained = _fit_and_eval(
            X[train_idx], y[train_idx], X[train_idx], y[train_idx]
        )

        info = {
            "r2": float(np.mean(test_r2s)),
            "r2_std": float(np.std(test_r2s)),
            "mae": float(np.mean(test_maes)),
            "mae_std": float(np.std(test_maes)),
            "corr": float(np.mean(test_corrs)),
            "corr_std": float(np.std(test_corrs)),
            "pca_variance_explained": variance_explained,
            "alpha": alpha,
            "n_components": pca_components if use_pca else X.shape[1],
            "n_bootstrap": len(bootstrap_splits),
        }
    else:
        # No bootstrap - fit on all data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if use_pca:
            n_components = min(pca_components, X.shape[0], X.shape[1])
            pca = PCA(n_components=n_components)
            X_final = pca.fit_transform(X_scaled)
            variance_explained = float(pca.explained_variance_ratio_.sum())
        else:
            pca = None
            X_final = X_scaled
            variance_explained = None

        probe = Ridge(alpha=alpha)
        probe.fit(X_final, y)

        y_pred = probe.predict(X_final)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        corr, _ = pearsonr(y, y_pred)

        if pca is not None:
            direction = pca.inverse_transform(probe.coef_.reshape(1, -1)).flatten()
            direction = direction / scaler.scale_
        else:
            direction = probe.coef_.copy()
            direction = direction / scaler.scale_

        direction = direction / np.linalg.norm(direction)

        info = {
            "r2": float(r2),
            "mae": float(mae),
            "corr": float(corr),
            "pca_variance_explained": variance_explained,
            "alpha": alpha,
            "n_components": pca_components if use_pca else X.shape[1],
        }

    return direction, info


def mean_diff_direction(
    X: np.ndarray,
    y: np.ndarray,
    quantile: float = 0.25,
    bootstrap_splits: List[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Find direction via mean difference between high and low metric samples.

    Direction = mean(top_quantile) - mean(bottom_quantile)

    Args:
        X: (n_samples, hidden_dim) activation matrix
        y: (n_samples,) target metric values
        quantile: Fraction of samples to use for each group (top and bottom)
        bootstrap_splits: Pre-generated list of (train_idx, test_idx) tuples for bootstrap.
                         If provided, computes confidence intervals. If None, fits on all data.

    Returns:
        direction: Normalized direction vector (hidden_dim,)
        info: Dict with group statistics and fit metrics (with std if bootstrap)
    """
    def _fit_and_eval(X_data, y_data, X_test, y_test):
        """Compute mean_diff direction on data, evaluate on test."""
        n = len(y_data)
        n_group = max(1, int(n * quantile))

        sorted_idx = np.argsort(y_data)
        low_idx = sorted_idx[:n_group]
        high_idx = sorted_idx[-n_group:]

        mean_low = X_data[low_idx].mean(axis=0)
        mean_high = X_data[high_idx].mean(axis=0)

        direction = mean_high - mean_low
        magnitude = np.linalg.norm(direction)
        direction = direction / (magnitude + 1e-10)

        # Evaluate on test set
        projections = X_test @ direction
        corr, _ = pearsonr(y_test, projections)
        r2 = float(corr ** 2)

        # Compute MAE
        proj_mean, y_mean = projections.mean(), y_test.mean()
        proj_std, y_std = projections.std(), y_test.std()
        if proj_std > 0:
            y_pred = y_mean + (projections - proj_mean) * (y_std / proj_std) * np.sign(corr)
            mae = float(np.abs(y_test - y_pred).mean())
        else:
            mae = float(np.abs(y_test - y_mean).mean())

        return r2, mae, corr, direction, magnitude, n_group

    if bootstrap_splits is not None and len(bootstrap_splits) > 0:
        # Bootstrap for confidence intervals
        test_r2s, test_maes, test_corrs = [], [], []

        for train_idx, test_idx in bootstrap_splits:
            r2, mae, corr, _, _, _ = _fit_and_eval(
                X[train_idx], y[train_idx], X[test_idx], y[test_idx]
            )
            test_r2s.append(r2)
            test_maes.append(mae)
            test_corrs.append(corr)

        # Final direction from first split's training data (canonical)
        train_idx = bootstrap_splits[0][0]
        _, _, _, direction, magnitude, n_group = _fit_and_eval(
            X[train_idx], y[train_idx], X[train_idx], y[train_idx]
        )

        info = {
            "r2": float(np.mean(test_r2s)),
            "r2_std": float(np.std(test_r2s)),
            "mae": float(np.mean(test_maes)),
            "mae_std": float(np.std(test_maes)),
            "corr": float(np.mean(test_corrs)),
            "corr_std": float(np.std(test_corrs)),
            "quantile": quantile,
            "n_group": n_group,
            "direction_magnitude": float(magnitude),
            "n_bootstrap": len(bootstrap_splits),
        }
    else:
        # No bootstrap - fit on all data
        n = len(y)
        n_group = max(1, int(n * quantile))

        sorted_idx = np.argsort(y)
        low_idx = sorted_idx[:n_group]
        high_idx = sorted_idx[-n_group:]

        mean_low = X[low_idx].mean(axis=0)
        mean_high = X[high_idx].mean(axis=0)

        direction = mean_high - mean_low
        magnitude = np.linalg.norm(direction)
        direction = direction / (magnitude + 1e-10)

        projections = X @ direction
        corr, _ = pearsonr(y, projections)
        r2 = float(corr ** 2)

        proj_mean, y_mean = projections.mean(), y.mean()
        proj_std, y_std = projections.std(), y.std()
        if proj_std > 0:
            y_pred = y_mean + (projections - proj_mean) * (y_std / proj_std) * np.sign(corr)
            mae = float(np.abs(y - y_pred).mean())
        else:
            mae = float(np.abs(y - y_mean).mean())

        info = {
            "r2": r2,
            "mae": mae,
            "corr": float(corr),
            "quantile": quantile,
            "n_low": n_group,
            "n_high": n_group,
            "metric_mean_low": float(y[low_idx].mean()),
            "metric_mean_high": float(y[high_idx].mean()),
            "direction_magnitude": float(magnitude),
        }

    return direction, info


def _process_layer(
    layer: int,
    X: np.ndarray,
    y: np.ndarray,
    methods: List[str],
    probe_alpha: float,
    probe_pca_components: int,
    bootstrap_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]],
    mean_diff_quantile: float
) -> Tuple[int, Dict[str, np.ndarray], Dict[str, Dict]]:
    """Process a single layer - used for parallel execution."""
    layer_directions = {}
    layer_fits = {}

    for method in methods:
        if method == "probe":
            direction, info = probe_direction(
                X, y,
                alpha=probe_alpha,
                pca_components=probe_pca_components,
                bootstrap_splits=bootstrap_splits,
            )
        elif method == "mean_diff":
            direction, info = mean_diff_direction(
                X, y,
                quantile=mean_diff_quantile,
                bootstrap_splits=bootstrap_splits,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        layer_directions[method] = direction
        layer_fits[method] = info

    return layer, layer_directions, layer_fits


def find_directions(
    activations_by_layer: Dict[int, np.ndarray],
    target_values: np.ndarray,
    methods: List[str] = None,
    probe_alpha: float = 1000.0,
    probe_pca_components: int = 100,
    probe_n_bootstrap: int = 0,
    probe_train_split: float = 0.8,
    mean_diff_quantile: float = 0.25,
    seed: int = 42,
    n_jobs: int = -1
) -> Dict:
    """
    Find directions using multiple methods across all layers.

    Args:
        activations_by_layer: {layer_idx: (n_samples, hidden_dim)}
        target_values: (n_samples,) metric values to predict
        methods: Which methods to use. Default: ["probe", "mean_diff"]
        probe_alpha: Ridge regularization for probe method
        probe_pca_components: PCA components for probe method
        probe_n_bootstrap: Bootstrap iterations for probe (0 = no bootstrap)
        probe_train_split: Train/test split ratio for bootstrap
        mean_diff_quantile: Quantile for mean_diff method
        seed: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 = all cores, 1 = sequential)

    Returns:
        {
            "directions": {method: {layer: direction_vector}},
            "fits": {method: {layer: {"r2": float, "corr": float, ...}}},
            "comparison": {layer: {"cosine_sim": float, ...}}
        }
    """
    from joblib import Parallel, delayed

    if methods is None:
        methods = ["probe", "mean_diff"]

    layers = sorted(activations_by_layer.keys())
    y = np.asarray(target_values)
    n_samples = len(y)

    # Pre-generate bootstrap splits ONCE (shared across all layers)
    bootstrap_splits = None
    if probe_n_bootstrap > 0 and "probe" in methods:
        rng = np.random.RandomState(seed)
        bootstrap_splits = []
        for _ in range(probe_n_bootstrap):
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            split_idx = int(n_samples * probe_train_split)
            bootstrap_splits.append((indices[:split_idx].copy(), indices[split_idx:].copy()))

    # Run layers in parallel (or sequential with progress bar)
    if n_jobs == 1:
        # Sequential with tqdm progress
        from tqdm import tqdm
        layer_results = []
        for layer in tqdm(layers, desc="Processing layers"):
            result = _process_layer(
                layer,
                activations_by_layer[layer],
                y,
                methods,
                probe_alpha,
                probe_pca_components,
                bootstrap_splits,
                mean_diff_quantile
            )
            layer_results.append(result)
    else:
        # Parallel processing
        layer_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_process_layer)(
                layer,
                activations_by_layer[layer],
                y,
                methods,
                probe_alpha,
                probe_pca_components,
                bootstrap_splits,
                mean_diff_quantile
            )
            for layer in layers
        )

    # Collect results
    results = {
        "directions": {m: {} for m in methods},
        "fits": {m: {} for m in methods},
        "comparison": {},
    }

    for layer, layer_directions, layer_fits in layer_results:
        for method in methods:
            results["directions"][method][layer] = layer_directions[method]
            results["fits"][method][layer] = layer_fits[method]

        # Compare methods at this layer
        if len(methods) == 2:
            d1, d2 = layer_directions[methods[0]], layer_directions[methods[1]]
            cosine_sim = float(np.dot(d1, d2))
            results["comparison"][layer] = {
                "cosine_sim": cosine_sim,
                "methods": methods,
            }

    return results


def apply_direction(
    activations: np.ndarray,
    direction: np.ndarray
) -> np.ndarray:
    """
    Project activations onto direction to get scalar predictions.

    Args:
        activations: (n_samples, hidden_dim) or (hidden_dim,)
        direction: (hidden_dim,) normalized direction vector

    Returns:
        Projections: (n_samples,) or scalar
    """
    return np.dot(activations, direction)


def evaluate_transfer(
    activations: np.ndarray,
    direction: np.ndarray,
    target_values: np.ndarray
) -> Dict:
    """
    Evaluate how well a direction predicts targets on new data.

    Args:
        activations: (n_samples, hidden_dim)
        direction: (hidden_dim,) direction found on training data
        target_values: (n_samples,) ground truth values

    Returns:
        Dict with r2, correlation, and predictions
    """
    projections = apply_direction(activations, direction)
    corr, p_value = pearsonr(target_values, projections)
    # R² = correlation² for simple bivariate relationship
    r2 = float(corr ** 2)

    return {
        "r2": r2,
        "corr": float(corr),
        "corr_pvalue": float(p_value),
        "projections": projections,
    }
