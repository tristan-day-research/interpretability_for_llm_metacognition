"""
Ablation causality test for uncertainty directions.

Tests whether directions from identify_mc_correlate.py are causally necessary
for the model's meta-judgments. If ablating a direction degrades the correlation
between stated confidence and actual uncertainty, that's evidence the direction
is causally involved in introspection.

Key features:
- Tests ALL layers by default (no pre-filtering by transfer R²)
- Tests BOTH probe and mean_diff methods in a single run for comparison
- Uses pooled null distribution + FDR correction for robust statistics

Usage:
    python run_ablation_causality.py

Expects outputs from identify_mc_correlate.py:
    outputs/{INPUT_BASE_NAME}_mc_{METRIC}_directions.npz
    outputs/{INPUT_BASE_NAME}_mc_dataset.json
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

from core.model_utils import (
    load_model_and_tokenizer,
    should_use_chat_template,
    get_model_short_name,
    DEVICE,
)
from core.steering import generate_orthogonal_directions
from core.steering_experiments import (
    SteeringExperimentConfig,
    BatchAblationHook,
    pretokenize_prompts,
    build_padded_gpu_batches,
    get_kv_cache,
    create_fresh_cache,
    precompute_direction_tensors,
)
from core.metrics import metric_sign_for_confidence
from tasks import (
    format_stated_confidence_prompt,
    get_stated_confidence_signal,
    format_answer_or_delegate_prompt,
    get_answer_or_delegate_signal,
    STATED_CONFIDENCE_OPTIONS,
    ANSWER_OR_DELEGATE_OPTIONS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
INPUT_BASE_NAME = "Llama-3.3-70B-Instruct_TriviaMC"
METRIC = "top_logit"  # Which metric's directions to test
META_TASK = "confidence"  # "confidence" or "delegate"

# Experiment settings
NUM_QUESTIONS = 100  # How many questions to test
NUM_CONTROLS = 25    # Random orthogonal directions per layer for null distribution
BATCH_SIZE = 8
SEED = 42

# Expanded batch target for batched ablation.
# When ablating k directions (1 primary + NUM_CONTROLS), we expand each base batch by k.
# This sets the TARGET total expanded batch size (base_batch * directions_per_pass).
# Higher values = better GPU utilization but more memory.
# With NUM_CONTROLS=25 (26 total) and EXPANDED_BATCH_TARGET=52, base batch = 2, we do 26 in one pass.
# Set to None to disable batched ablation (run each direction separately).
EXPANDED_BATCH_TARGET = 52

# Optional: specify layers to test (None = all layers from directions file)
LAYERS = None  # e.g., [20, 25, 30] for quick testing

# Quantization (for large models)
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False

# Output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# DIRECTION LOADING
# =============================================================================

def load_directions(base_name: str, metric: str) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Load all direction methods from npz file.

    Returns:
        Dict mapping method name -> {layer: direction_vector}
        e.g., {"probe": {0: arr, 1: arr, ...}, "mean_diff": {0: arr, 1: arr, ...}}
    """
    path = OUTPUT_DIR / f"{base_name}_mc_{metric}_directions.npz"
    if not path.exists():
        raise FileNotFoundError(f"Directions file not found: {path}")

    data = np.load(path)

    methods: Dict[str, Dict[int, np.ndarray]] = {}
    for key in data.files:
        if key.startswith("_"):
            continue  # Skip metadata keys

        # Keys are like "probe_layer_0", "mean_diff_layer_5"
        parts = key.rsplit("_layer_", 1)
        if len(parts) != 2:
            continue

        method, layer_str = parts
        try:
            layer = int(layer_str)
        except ValueError:
            continue

        if method not in methods:
            methods[method] = {}

        # Normalize direction
        direction = data[key].astype(np.float32)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        methods[method][layer] = direction

    return methods


def load_dataset(base_name: str) -> Dict:
    """Load dataset with questions and metric values."""
    path = OUTPUT_DIR / f"{base_name}_mc_dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


# =============================================================================
# META-TASK HELPERS
# =============================================================================

def get_format_fn(meta_task: str):
    """Get prompt formatting function for meta-task."""
    if meta_task == "confidence":
        return format_stated_confidence_prompt
    elif meta_task == "delegate":
        return format_answer_or_delegate_prompt
    else:
        raise ValueError(f"Unknown meta_task: {meta_task}")


def get_signal_fn(meta_task: str):
    """Get signal extraction function for meta-task.

    Returns a function with signature (probs, mapping) -> float.
    For confidence task, mapping is ignored.
    """
    if meta_task == "confidence":
        # Wrap to match (probs, mapping) signature
        return lambda p, m: get_stated_confidence_signal(p)
    elif meta_task == "delegate":
        return get_answer_or_delegate_signal
    else:
        raise ValueError(f"Unknown meta_task: {meta_task}")


def get_options(meta_task: str) -> List[str]:
    """Get response options for meta-task."""
    if meta_task == "confidence":
        return list(STATED_CONFIDENCE_OPTIONS.keys())
    elif meta_task == "delegate":
        return ANSWER_OR_DELEGATE_OPTIONS
    else:
        raise ValueError(f"Unknown meta_task: {meta_task}")


# =============================================================================
# ABLATION EXPERIMENT
# =============================================================================

def run_ablation_for_method(
    model,
    tokenizer,
    questions: List[Dict],
    metric_values: np.ndarray,
    directions: Dict[int, np.ndarray],
    num_controls: int,
    meta_task: str,
    use_chat_template: bool,
    layers: Optional[List[int]] = None,
) -> Dict:
    """
    Run ablation experiment for a single direction method.

    Uses batched ablation when EXPANDED_BATCH_TARGET is set: multiple directions
    are ablated in a single forward pass by expanding the batch.

    Returns dict with per-layer results including baseline, ablated, and controls.
    """
    if layers is None:
        layers = sorted(directions.keys())
    else:
        layers = [l for l in layers if l in directions]

    if not layers:
        return {"error": "No layers to test"}

    metric_mean = float(np.mean(metric_values))
    metric_std = float(np.std(metric_values))
    if metric_std < 1e-10:
        metric_std = 1.0

    # Get formatting functions and options
    format_fn = get_format_fn(meta_task)
    signal_fn = get_signal_fn(meta_task)
    options = get_options(meta_task)

    # Tokenize options
    option_token_ids = [
        tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options
    ]

    # Format and tokenize prompts
    prompts = []
    mappings = []
    for q_idx, question in enumerate(questions):
        if meta_task == "delegate":
            prompt, _, mapping = format_fn(question, tokenizer, trial_index=q_idx, use_chat_template=use_chat_template)
        else:
            prompt, _ = format_fn(question, tokenizer, use_chat_template=use_chat_template)
            mapping = None
        prompts.append(prompt)
        mappings.append(mapping)

    cached_inputs = pretokenize_prompts(prompts, tokenizer, DEVICE)
    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, BATCH_SIZE)

    # Generate control directions for each layer
    print(f"  Generating {num_controls} control directions per layer...")
    controls_by_layer = {}
    for layer in layers:
        controls_by_layer[layer] = generate_orthogonal_directions(
            directions[layer], num_controls, seed=SEED + layer
        )

    # Precompute direction tensors
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    cached_directions = {}
    for layer in layers:
        dir_tensor = torch.tensor(directions[layer], dtype=dtype, device=DEVICE)
        ctrl_tensors = [torch.tensor(c, dtype=dtype, device=DEVICE) for c in controls_by_layer[layer]]
        # Stack all directions: [primary, control_0, control_1, ..., control_N-1]
        all_dirs = torch.stack([dir_tensor] + ctrl_tensors, dim=0)  # (1 + num_controls, hidden_dim)
        cached_directions[layer] = {
            "direction": dir_tensor,
            "controls": ctrl_tensors,
            "all_stacked": all_dirs,
        }

    # Initialize results
    baseline_results = [None] * len(questions)
    layer_results = {}
    for layer in layers:
        layer_results[layer] = {
            "baseline": baseline_results,
            "ablated": [None] * len(questions),
            "controls_ablated": {f"control_{i}": [None] * len(questions) for i in range(num_controls)}
        }

    # Determine batching strategy
    total_directions = 1 + num_controls  # primary + controls
    if EXPANDED_BATCH_TARGET is not None and EXPANDED_BATCH_TARGET > 0:
        # Batch multiple directions together
        # directions_per_pass = how many directions we can fit given EXPANDED_BATCH_TARGET
        # We want: base_batch_size * directions_per_pass <= EXPANDED_BATCH_TARGET
        directions_per_pass = max(1, EXPANDED_BATCH_TARGET // BATCH_SIZE)
        # Don't exceed total directions
        directions_per_pass = min(directions_per_pass, total_directions)
        use_batched = directions_per_pass > 1
    else:
        directions_per_pass = 1
        use_batched = False

    if use_batched:
        num_passes = (total_directions + directions_per_pass - 1) // directions_per_pass
        print(f"  Batched ablation: {directions_per_pass} directions per pass, {num_passes} passes per layer")
        total_forward_passes = len(gpu_batches) * len(layers) * num_passes
    else:
        print(f"  Sequential ablation: 1 direction per pass")
        total_forward_passes = len(gpu_batches) * len(layers) * total_directions

    print(f"  Total forward passes: {total_forward_passes}")

    pbar = tqdm(total=total_forward_passes, desc="  Forward passes")

    for batch_idx, (batch_indices, batch_inputs) in enumerate(gpu_batches):
        B = len(batch_indices)

        # Compute KV cache once per batch
        base_step_data = get_kv_cache(model, batch_inputs)
        keys_snapshot, values_snapshot = base_step_data["past_key_values_data"]

        inputs_template = {
            "input_ids": base_step_data["input_ids"],
            "attention_mask": base_step_data["attention_mask"],
            "use_cache": True
        }
        if "position_ids" in base_step_data:
            inputs_template["position_ids"] = base_step_data["position_ids"]

        # Compute baseline (no ablation)
        if baseline_results[batch_indices[0]] is None:
            fresh_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)
            baseline_inputs = inputs_template.copy()
            baseline_inputs["past_key_values"] = fresh_cache

            with torch.inference_mode():
                out = model(**baseline_inputs)
                logits = out.logits[:, -1, :][:, option_token_ids]
                probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

            for i, q_idx in enumerate(batch_indices):
                p = probs[i]
                resp = options[np.argmax(p)]
                conf = signal_fn(p, mappings[q_idx])
                m_val = metric_values[q_idx]
                baseline_results[q_idx] = {
                    "question_idx": q_idx,
                    "response": resp,
                    "confidence": float(conf),
                    "metric": float(m_val),
                }

        # Run ablation for each layer
        for layer in layers:
            if hasattr(model, 'get_base_model'):
                layer_module = model.get_base_model().model.layers[layer]
            else:
                layer_module = model.model.layers[layer]

            all_dirs = cached_directions[layer]["all_stacked"]  # (total_directions, hidden_dim)

            hook = BatchAblationHook()
            hook.register(layer_module)

            try:
                if use_batched:
                    # Process directions in batched passes
                    for pass_start in range(0, total_directions, directions_per_pass):
                        pass_end = min(pass_start + directions_per_pass, total_directions)
                        k_dirs = pass_end - pass_start  # directions in this pass

                        # Expand inputs for this pass
                        expanded_input_ids = inputs_template["input_ids"].repeat_interleave(k_dirs, dim=0)
                        expanded_attention_mask = inputs_template["attention_mask"].repeat_interleave(k_dirs, dim=0)
                        expanded_inputs = {
                            "input_ids": expanded_input_ids,
                            "attention_mask": expanded_attention_mask,
                            "use_cache": True
                        }
                        if "position_ids" in inputs_template:
                            expanded_inputs["position_ids"] = inputs_template["position_ids"].repeat_interleave(k_dirs, dim=0)

                        # Create expanded cache
                        pass_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=k_dirs)
                        expanded_inputs["past_key_values"] = pass_cache

                        # Build directions tensor: for each question, apply each direction in pass
                        # Shape: (B * k_dirs, hidden_dim)
                        dirs_for_pass = all_dirs[pass_start:pass_end]  # (k_dirs, hidden_dim)
                        # Tile: for each of B questions, repeat the k_dirs directions
                        dirs_batch = dirs_for_pass.unsqueeze(0).expand(B, -1, -1).reshape(B * k_dirs, -1)
                        hook.set_directions(dirs_batch)

                        # Run model
                        with torch.inference_mode():
                            out = model(**expanded_inputs)
                            logits = out.logits[:, -1, :][:, option_token_ids]
                            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                        # Store results
                        for i, q_idx in enumerate(batch_indices):
                            for j in range(k_dirs):
                                dir_idx = pass_start + j  # global direction index
                                prob_idx = i * k_dirs + j
                                p = probs[prob_idx]
                                resp = options[np.argmax(p)]
                                conf = signal_fn(p, mappings[q_idx])
                                m_val = metric_values[q_idx]
                                data = {
                                    "question_idx": q_idx,
                                    "response": resp,
                                    "confidence": float(conf),
                                    "metric": float(m_val),
                                }

                                if dir_idx == 0:
                                    # Primary direction
                                    layer_results[layer]["ablated"][q_idx] = data
                                else:
                                    # Control direction
                                    ctrl_key = f"control_{dir_idx - 1}"
                                    layer_results[layer]["controls_ablated"][ctrl_key][q_idx] = data

                        pbar.update(1)

                else:
                    # Sequential: one direction per forward pass
                    def run_single_ablation(direction_tensor, result_list, key=None):
                        pass_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)
                        current_inputs = inputs_template.copy()
                        current_inputs["past_key_values"] = pass_cache

                        dirs_batch = direction_tensor.unsqueeze(0).expand(B, -1)
                        hook.set_directions(dirs_batch)

                        with torch.inference_mode():
                            out = model(**current_inputs)
                            logits = out.logits[:, -1, :][:, option_token_ids]
                            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                        for i, q_idx in enumerate(batch_indices):
                            p = probs[i]
                            resp = options[np.argmax(p)]
                            conf = signal_fn(p, mappings[q_idx])
                            m_val = metric_values[q_idx]
                            data = {
                                "question_idx": q_idx,
                                "response": resp,
                                "confidence": float(conf),
                                "metric": float(m_val),
                            }
                            if key:
                                result_list[key][q_idx] = data
                            else:
                                result_list[q_idx] = data
                        pbar.update(1)

                    # Ablate primary direction
                    run_single_ablation(cached_directions[layer]["direction"], layer_results[layer]["ablated"])
                    # Ablate each control direction
                    for i_c, ctrl_dir in enumerate(cached_directions[layer]["controls"]):
                        run_single_ablation(ctrl_dir, layer_results[layer]["controls_ablated"], key=f"control_{i_c}")

            finally:
                hook.remove()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    pbar.close()
    return {
        "layers": layers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": layer_results,
    }


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_correlation(confidences: np.ndarray, metric_values: np.ndarray) -> float:
    """Compute Pearson correlation between confidence and metric."""
    if len(confidences) < 2 or np.std(confidences) < 1e-10 or np.std(metric_values) < 1e-10:
        return 0.0
    return float(np.corrcoef(confidences, metric_values)[0, 1])


def analyze_ablation_results(results: Dict, metric: str) -> Dict:
    """
    Compute ablation effect statistics with pooled null + FDR correction.

    Returns analysis dict with per-layer stats and summary.
    """
    layers = results["layers"]
    num_controls = results["num_controls"]

    # Get metric sign (for interpretation)
    metric_sign = metric_sign_for_confidence(metric)

    analysis = {
        "layers": layers,
        "num_questions": results["num_questions"],
        "num_controls": num_controls,
        "metric": metric,
        "metric_sign": metric_sign,
        "per_layer": {},
    }

    # First pass: collect all control effects for pooled null
    all_control_corr_changes = []
    layer_data = {}

    for layer in layers:
        lr = results["layer_results"][layer]

        # Extract data
        baseline_conf = np.array([r["confidence"] for r in lr["baseline"]])
        baseline_metric = np.array([r["metric"] for r in lr["baseline"]])
        ablated_conf = np.array([r["confidence"] for r in lr["ablated"]])
        ablated_metric = np.array([r["metric"] for r in lr["ablated"]])

        # Compute correlations
        baseline_corr = compute_correlation(baseline_conf, baseline_metric)
        ablated_corr = compute_correlation(ablated_conf, ablated_metric)

        # Control ablations
        control_corrs = []
        for ctrl_key in lr["controls_ablated"]:
            ctrl_conf = np.array([r["confidence"] for r in lr["controls_ablated"][ctrl_key]])
            ctrl_metric = np.array([r["metric"] for r in lr["controls_ablated"][ctrl_key]])
            control_corrs.append(compute_correlation(ctrl_conf, ctrl_metric))

        corr_change = ablated_corr - baseline_corr
        control_corr_changes = [c - baseline_corr for c in control_corrs]

        all_control_corr_changes.extend(control_corr_changes)

        layer_data[layer] = {
            "baseline_corr": baseline_corr,
            "baseline_conf_mean": float(np.mean(baseline_conf)),
            "ablated_corr": ablated_corr,
            "ablated_conf_mean": float(np.mean(ablated_conf)),
            "corr_change": corr_change,
            "control_corrs": control_corrs,
            "control_corr_changes": control_corr_changes,
        }

    # Convert pooled null to array
    pooled_null = np.array(all_control_corr_changes)

    # Second pass: compute p-values
    raw_p_values = []

    for layer in layers:
        ld = layer_data[layer]

        # Per-layer statistics
        ctrl_mean = float(np.mean(ld["control_corr_changes"]))
        ctrl_std = float(np.std(ld["control_corr_changes"]))

        # Pooled p-value: how many pooled controls have effect >= ours
        n_pooled_worse = np.sum(pooled_null >= ld["corr_change"])
        p_value_pooled = (n_pooled_worse + 1) / (len(pooled_null) + 1)

        # Effect size (Z-score vs controls)
        if ctrl_std > 1e-10:
            effect_size_z = (ld["corr_change"] - ctrl_mean) / ctrl_std
        else:
            effect_size_z = 0.0

        raw_p_values.append((layer, p_value_pooled))

        analysis["per_layer"][layer] = {
            "baseline_correlation": ld["baseline_corr"],
            "baseline_confidence_mean": ld["baseline_conf_mean"],
            "ablated_correlation": ld["ablated_corr"],
            "ablated_confidence_mean": ld["ablated_conf_mean"],
            "correlation_change": ld["corr_change"],
            "control_correlation_mean": float(np.mean(ld["control_corrs"])),
            "control_correlation_change_mean": ctrl_mean,
            "control_correlation_change_std": ctrl_std,
            "p_value_pooled": float(p_value_pooled),
            "effect_size_z": float(effect_size_z),
        }

    # FDR correction (Benjamini-Hochberg)
    sorted_pvals = sorted(raw_p_values, key=lambda x: x[1])
    n_tests = len(sorted_pvals)
    fdr_adjusted = {}

    for rank, (layer, p_val) in enumerate(sorted_pvals, 1):
        adjusted = min(1.0, p_val * n_tests / rank)
        fdr_adjusted[layer] = adjusted

    # Make monotonic
    prev_adjusted = 0.0
    for layer, _ in sorted(sorted_pvals, key=lambda x: x[1]):
        fdr_adjusted[layer] = max(fdr_adjusted[layer], prev_adjusted)
        prev_adjusted = fdr_adjusted[layer]

    # Add FDR p-values
    for layer in layers:
        analysis["per_layer"][layer]["p_value_fdr"] = float(fdr_adjusted[layer])

    # Summary
    significant_pooled = [l for l in layers if analysis["per_layer"][l]["p_value_pooled"] < 0.05]
    significant_fdr = [l for l in layers if analysis["per_layer"][l]["p_value_fdr"] < 0.05]

    # Best layer by effect size
    best_layer = max(layers, key=lambda l: analysis["per_layer"][l]["effect_size_z"])

    analysis["summary"] = {
        "pooled_null_size": len(pooled_null),
        "significant_layers_pooled": significant_pooled,
        "significant_layers_fdr": significant_fdr,
        "n_significant_pooled": len(significant_pooled),
        "n_significant_fdr": len(significant_fdr),
        "best_layer": best_layer,
        "best_effect_z": analysis["per_layer"][best_layer]["effect_size_z"],
    }

    return analysis


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_ablation_results(analysis: Dict, method: str, output_path: Path):
    """
    Create 3-panel ablation visualization for a single method.
    """
    layers = analysis["layers"]

    if not layers:
        print(f"  Skipping plot for {method} - no layers")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Ablation Results: {method.upper()} directions ({analysis['metric']})", fontsize=14)

    x = np.arange(len(layers))

    # Panel 1: Absolute correlations
    ax1 = axes[0]
    baseline_corrs = [analysis["per_layer"][l]["baseline_correlation"] for l in layers]
    ablated_corrs = [analysis["per_layer"][l]["ablated_correlation"] for l in layers]
    ctrl_corrs = [analysis["per_layer"][l]["control_correlation_mean"] for l in layers]

    ax1.plot(x, baseline_corrs, 'o-', label='Baseline', color='blue', linewidth=2, markersize=8)
    ax1.plot(x, ablated_corrs, 's-', label=f'{method} ablated', color='red', linewidth=2, markersize=8)
    ax1.plot(x, ctrl_corrs, '^--', label='Control ablated', color='gray', linewidth=1.5, markersize=7, alpha=0.8)

    ax1.axhline(y=0, color='black', linestyle=':', alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Correlation (confidence vs metric)")
    ax1.set_title("Correlation by Condition")
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Effect size with significance
    ax2 = axes[1]
    effect_sizes = [analysis["per_layer"][l]["effect_size_z"] for l in layers]
    p_values_fdr = [analysis["per_layer"][l]["p_value_fdr"] for l in layers]

    colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'gray' for p in p_values_fdr]
    bars = ax2.bar(x, effect_sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Effect Size (Z-score)")
    ax2.set_title("Effect Size vs Controls")
    ax2.grid(True, alpha=0.3, axis='y')

    legend_elements = [
        Patch(facecolor='red', alpha=0.7, edgecolor='black', label='p < 0.05 (FDR)'),
        Patch(facecolor='orange', alpha=0.7, edgecolor='black', label='p < 0.10 (FDR)'),
        Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='n.s.'),
    ]
    ax2.legend(handles=legend_elements, loc='best', fontsize=9)

    # Panel 3: Summary text
    ax3 = axes[2]
    ax3.axis('off')

    summary = analysis["summary"]
    baseline_corr = np.mean(baseline_corrs)

    summary_text = f"""
ABLATION ANALYSIS: {method.upper()}

Metric: {analysis['metric']}
Layers tested: {len(layers)}
Questions: {analysis['num_questions']}
Controls per layer: {analysis['num_controls']}
Pooled null size: {summary['pooled_null_size']}

Results:
  Baseline correlation: {baseline_corr:.4f}
  Significant layers (p<0.05 pooled): {summary['n_significant_pooled']}
  Significant layers (FDR<0.05): {summary['n_significant_fdr']}

Best layer: {summary['best_layer']}
  Effect size (Z): {summary['best_effect_z']:.2f}
  p-value (FDR): {analysis['per_layer'][summary['best_layer']]['p_value_fdr']:.4f}

Interpretation:
"""
    if summary['n_significant_fdr'] > 0:
        summary_text += f"""  ✓ SIGNIFICANT after FDR correction
  {summary['n_significant_fdr']} layer(s) show that ablating the
  {method} direction degrades calibration more
  than random directions.
  Evidence for causal role."""
    elif summary['n_significant_pooled'] > 0:
        summary_text += """  ⚠ Nominally significant (not FDR-corrected)
  Suggestive but not definitive evidence."""
    else:
        summary_text += """  ✗ No significant effect detected
  Direction may not be causally involved."""

    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved {output_path}")
    plt.close()


def plot_method_comparison(analyses: Dict[str, Dict], output_path: Path):
    """
    Create comparison plot of different direction methods.
    """
    methods = list(analyses.keys())
    if len(methods) < 2:
        print("  Skipping comparison plot - need at least 2 methods")
        return

    # Use layers from first method
    layers = analyses[methods[0]]["layers"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Method Comparison: Ablation Effects", fontsize=14)

    x = np.arange(len(layers))
    width = 0.35
    colors = {'probe': 'tab:blue', 'mean_diff': 'tab:orange'}

    # Panel 1: Effect sizes by layer
    ax1 = axes[0]
    for i, method in enumerate(methods):
        effect_sizes = [analyses[method]["per_layer"][l]["effect_size_z"] for l in layers]
        offset = (i - len(methods)/2 + 0.5) * width
        ax1.bar(x + offset, effect_sizes, width, label=method, color=colors.get(method, f'C{i}'), alpha=0.7)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Effect Size (Z-score)")
    ax1.set_title("Effect Size by Method")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Summary comparison
    ax2 = axes[1]
    ax2.axis('off')

    comparison_text = "METHOD COMPARISON\n" + "=" * 40 + "\n\n"

    for method in methods:
        summary = analyses[method]["summary"]
        comparison_text += f"{method.upper()}:\n"
        comparison_text += f"  Significant layers (FDR): {summary['n_significant_fdr']}\n"
        comparison_text += f"  Best layer: {summary['best_layer']} (Z={summary['best_effect_z']:.2f})\n\n"

    # Winner
    best_method = max(methods, key=lambda m: analyses[m]["summary"]["n_significant_fdr"])
    comparison_text += f"Method with more FDR-significant layers: {best_method.upper()}\n"

    ax2.text(0.1, 0.9, comparison_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved {output_path}")
    plt.close()


def print_summary(analyses: Dict[str, Dict]):
    """Print summary of ablation results."""
    print("\n" + "=" * 70)
    print("ABLATION CAUSALITY TEST RESULTS")
    print("=" * 70)

    for method, analysis in analyses.items():
        summary = analysis["summary"]
        print(f"\n{method.upper()} directions:")
        print(f"  Layers tested: {len(analysis['layers'])}")
        print(f"  Significant (pooled p<0.05): {summary['n_significant_pooled']}")
        print(f"  Significant (FDR p<0.05): {summary['n_significant_fdr']}")
        print(f"  Best layer: {summary['best_layer']} (Z={summary['best_effect_z']:.2f})")

        if summary['significant_layers_fdr']:
            print(f"  FDR-significant layers: {summary['significant_layers_fdr']}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ABLATION CAUSALITY TEST")
    print("=" * 70)
    print(f"\nModel: {MODEL}")
    print(f"Input: {INPUT_BASE_NAME}")
    print(f"Metric: {METRIC}")
    print(f"Meta-task: {META_TASK}")
    print(f"Questions: {NUM_QUESTIONS}")
    print(f"Controls per layer: {NUM_CONTROLS}")

    # Load directions
    print("\nLoading directions...")
    all_directions = load_directions(INPUT_BASE_NAME, METRIC)
    methods = list(all_directions.keys())
    print(f"  Found methods: {methods}")

    for method in methods:
        layers = sorted(all_directions[method].keys())
        print(f"  {method}: {len(layers)} layers ({min(layers)}-{max(layers)})")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(INPUT_BASE_NAME)
    data_items = dataset["data"][:NUM_QUESTIONS]
    # Extract questions (each item has question, options, correct_answer, etc.)
    questions = data_items
    # Extract metric values from each item
    metric_values = np.array([item[METRIC] for item in data_items])
    print(f"  Questions: {len(questions)}")
    print(f"  {METRIC}: mean={metric_values.mean():.3f}, std={metric_values.std():.3f}")

    # Determine layers to test
    if LAYERS is not None:
        test_layers = LAYERS
    else:
        # Use all layers from first method
        test_layers = sorted(all_directions[methods[0]].keys())
    print(f"\nLayers to test: {len(test_layers)}")

    # Load model
    print("\nLoading model...")
    model, tokenizer, num_layers = load_model_and_tokenizer(
        MODEL,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )
    use_chat_template = should_use_chat_template(MODEL, tokenizer)
    print(f"  Use chat template: {use_chat_template}")
    print(f"  Device: {DEVICE}")

    # Run ablation for each method
    all_results = {}
    all_analyses = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"ABLATION EXPERIMENT: {method.upper()}")
        print(f"{'='*60}")

        results = run_ablation_for_method(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            metric_values=metric_values,
            directions=all_directions[method],
            num_controls=NUM_CONTROLS,
            meta_task=META_TASK,
            use_chat_template=use_chat_template,
            layers=test_layers,
        )
        all_results[method] = results

        # Analyze results
        print(f"\n  Analyzing results...")
        analysis = analyze_ablation_results(results, METRIC)
        all_analyses[method] = analysis

        summary = analysis["summary"]
        print(f"  Significant layers (FDR): {summary['n_significant_fdr']}")
        print(f"  Best layer: {summary['best_layer']} (Z={summary['best_effect_z']:.2f})")

    # Generate output filename
    model_short = get_model_short_name(MODEL)
    base_output = f"{model_short}_{INPUT_BASE_NAME.split('_')[-1]}_ablation_{META_TASK}_{METRIC}"

    # Save JSON results
    print("\nSaving results...")
    results_path = OUTPUT_DIR / f"{base_output}_results.json"

    output_json = {
        "config": {
            "model": MODEL,
            "input_base_name": INPUT_BASE_NAME,
            "metric": METRIC,
            "meta_task": META_TASK,
            "num_questions": NUM_QUESTIONS,
            "num_controls": NUM_CONTROLS,
            "layers_tested": test_layers,
            "methods_tested": methods,
        },
    }

    for method, analysis in all_analyses.items():
        output_json[method] = {
            "per_layer": analysis["per_layer"],
            "summary": analysis["summary"],
        }

    # Comparison summary
    if len(methods) >= 2:
        output_json["comparison"] = {
            method: {
                "n_significant_fdr": all_analyses[method]["summary"]["n_significant_fdr"],
                "best_layer": all_analyses[method]["summary"]["best_layer"],
                "best_effect_z": all_analyses[method]["summary"]["best_effect_z"],
            }
            for method in methods
        }
        best_method = max(methods, key=lambda m: all_analyses[m]["summary"]["n_significant_fdr"])
        output_json["comparison"]["method_with_more_effect"] = best_method

    with open(results_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"  Saved {results_path}")

    # Generate plots
    print("\nGenerating plots...")
    for method in methods:
        plot_path = OUTPUT_DIR / f"{base_output}_{method}.png"
        plot_ablation_results(all_analyses[method], method, plot_path)

    if len(methods) >= 2:
        comparison_path = OUTPUT_DIR / f"{base_output}_comparison.png"
        plot_method_comparison(all_analyses, comparison_path)

    # Print summary
    print_summary(all_analyses)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  {results_path.name}")
    for method in methods:
        print(f"  {base_output}_{method}.png")
    if len(methods) >= 2:
        print(f"  {base_output}_comparison.png")


if __name__ == "__main__":
    main()
