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
from scipy.stats import pearsonr, spearmanr, norm

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
    find_mc_positions,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
INPUT_BASE_NAME = "Llama-3.3-70B-Instruct_TriviaMC"
METRIC = "top_logit"  # Which metric's directions to test
META_TASK = "delegate"  # "confidence" or "delegate"

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
EXPANDED_BATCH_TARGET = 192

# Optional: specify layers to test (None = all layers from directions file)
# For non-final positions, layers are auto-selected based on transfer R² if None
LAYERS = None  # e.g., [20, 25, 30] for quick testing

# Optional: specify which direction methods to test (None = both probe and mean_diff)
METHODS = ["mean_diff"]  # e.g., ["mean_diff"] or ["probe"] to test just one

# Positions to test (multi-position extraction)
# These are token positions within the meta-task prompt where we can ablate
PROBE_POSITIONS = ["options_newline"]  # ["question_mark", "question_newline", "options_newline", "final"]

# Extra diagnostics: understand cases where transfer is strong but ablation has weak/odd corr effects
PRINT_DELTA_DIAGNOSTICS = True
DELTA_DIAGNOSTIC_TOPK = 5  # print this many layers for biggest +Δcorr and biggest -Δcorr

# Layer selection from transfer results (for non-final positions)
# When LAYERS is None and position != "final", auto-select layers with transfer R² >= threshold
TRANSFER_R2_THRESHOLD = 0.3  # Layers with R² >= this are tested for non-final positions
TRANSFER_RESULTS_PATH = None  # Auto-detect from INPUT_BASE_NAME if None

# Control count for non-final positions (final uses NUM_CONTROLS)
# Fewer controls = faster, but need enough for valid p-values
# 10 controls × N layers gives pooled null with 10*N samples
NUM_CONTROLS_NONFINAL = 10

# Quantization (for large models)
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False

# Output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# TRANSFER RESULTS LOADING (for layer selection)
# =============================================================================

def load_transfer_results(base_name: str, meta_task: str) -> Optional[Dict]:
    """
    Load transfer results JSON to get per-layer R² values.

    Returns None if file not found.
    """
    path = TRANSFER_RESULTS_PATH
    if path is None:
        path = OUTPUT_DIR / f"{base_name}_transfer_{meta_task}_results.json"
    else:
        path = Path(path)

    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


def get_layers_from_transfer(
    transfer_data: Dict,
    metric: str,
    position: str,
    r2_threshold: float,
    method: str = "probe",
) -> List[int]:
    """
    Get layers with transfer R² >= threshold for a given metric and position.

    Args:
        transfer_data: Loaded transfer results JSON
        metric: Which metric to check (e.g., "top_logit", "entropy")
        position: Token position (e.g., "final", "question_mark")
        r2_threshold: Minimum R² to include layer
        method: Direction method - "probe" uses transfer_by_position, "mean_diff" uses mean_diff_by_position

    Returns:
        Sorted list of layer indices meeting threshold
    """
    # Select the appropriate section based on method
    if method == "mean_diff":
        section_key = "mean_diff_by_position"
        legacy_key = None  # No legacy fallback for mean_diff
    else:
        section_key = "transfer_by_position"
        legacy_key = "transfer"

    # Try position-specific data first
    if section_key in transfer_data and position in transfer_data[section_key]:
        pos_data = transfer_data[section_key][position]
    elif legacy_key and legacy_key in transfer_data:
        # Fall back to legacy format (final position only, probe only)
        pos_data = transfer_data[legacy_key]
    else:
        return []

    if metric not in pos_data:
        return []

    metric_data = pos_data[metric]
    per_layer = metric_data.get("per_layer", {})

    selected = []
    for layer_str, layer_data in per_layer.items():
        # Check for centered R² (preferred) or d2m_centered_r2 (legacy)
        r2 = layer_data.get("centered_r2") or layer_data.get("d2m_centered_r2", 0)
        if r2 >= r2_threshold:
            selected.append(int(layer_str))

    return sorted(selected)


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
    position: str = "final",
) -> Dict:
    """
    Run ablation experiment for a single direction method at a specific position.

    Uses batched ablation when EXPANDED_BATCH_TARGET is set: multiple directions
    are ablated in a single forward pass by expanding the batch.

    For position="final", uses KV cache optimization.
    For other positions, uses full forward passes with indexed ablation.

    Args:
        position: Token position to ablate at. One of PROBE_POSITIONS:
            - "final": Last token (uses KV cache)
            - "question_mark": Token after "?" in question
            - "question_newline": Newline after question
            - "options_newline": Newline after MC options

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

    # Format prompts and find token positions
    prompts = []
    mappings = []
    position_indices = []  # Per-prompt token index for intervention
    for q_idx, question in enumerate(questions):
        if meta_task == "delegate":
            prompt, _, mapping = format_fn(question, tokenizer, trial_index=q_idx, use_chat_template=use_chat_template)
        else:
            prompt, _ = format_fn(question, tokenizer, use_chat_template=use_chat_template)
            mapping = None
        prompts.append(prompt)
        mappings.append(mapping)

        # Find token positions for this prompt
        positions = find_mc_positions(prompt, tokenizer, question)
        pos_idx = positions.get(position, -1)
        position_indices.append(pos_idx)

    # Warn if some positions weren't found (will fall back to final token)
    # Note: "final" position is always -1 by design, so don't warn for it
    if position != "final":
        n_valid = sum(1 for idx in position_indices if idx >= 0)
        n_total = len(position_indices)
        if n_valid < n_total:
            print(f"  Warning: {position} position found for {n_valid}/{n_total} prompts (others fall back to final)")

    cached_inputs = pretokenize_prompts(prompts, tokenizer, DEVICE)
    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, BATCH_SIZE)

    # Check if we can use KV cache (only for "final" position)
    use_kv_cache = (position == "final")

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
        directions_per_pass = max(1, EXPANDED_BATCH_TARGET // BATCH_SIZE)
        directions_per_pass = min(directions_per_pass, total_directions)
        use_batched = directions_per_pass > 1
    else:
        directions_per_pass = 1
        use_batched = False

    # Calculate number of passes (same formula for both paths)
    num_passes = (total_directions + directions_per_pass - 1) // directions_per_pass if use_batched else total_directions

    if use_kv_cache:
        # KV cache path: efficient but only works for final position
        if use_batched:
            print(f"  Batched ablation (KV cache): {directions_per_pass} directions per pass, {num_passes} passes per layer")
            total_forward_passes = len(gpu_batches) * len(layers) * num_passes
        else:
            print(f"  Sequential ablation (KV cache): 1 direction per pass")
            total_forward_passes = len(gpu_batches) * len(layers) * total_directions
    else:
        # Full forward path: required for non-final positions (also supports batching)
        if use_batched:
            print(f"  Batched ablation (full forward) at '{position}': {directions_per_pass} dirs/pass, {num_passes} passes/layer")
            total_forward_passes = len(gpu_batches) * len(layers) * num_passes
        else:
            print(f"  Sequential ablation (full forward) at '{position}': {total_directions} directions per layer")
            total_forward_passes = len(gpu_batches) * len(layers) * total_directions

    print(f"  Total forward passes: {total_forward_passes}")

    pbar = tqdm(total=total_forward_passes, desc=f"  Ablation ({position})")

    for batch_idx, (batch_indices, batch_inputs) in enumerate(gpu_batches):
        B = len(batch_indices)

        if use_kv_cache:
            # KV cache path (position == "final")
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

            # Run ablation for each layer (KV cache path)
            for layer in layers:
                if hasattr(model, 'get_base_model'):
                    layer_module = model.get_base_model().model.layers[layer]
                else:
                    layer_module = model.model.layers[layer]

                all_dirs = cached_directions[layer]["all_stacked"]
                hook = BatchAblationHook()
                hook.register(layer_module)

                try:
                    if use_batched:
                        for pass_start in range(0, total_directions, directions_per_pass):
                            pass_end = min(pass_start + directions_per_pass, total_directions)
                            k_dirs = pass_end - pass_start

                            expanded_input_ids = inputs_template["input_ids"].repeat_interleave(k_dirs, dim=0)
                            expanded_attention_mask = inputs_template["attention_mask"].repeat_interleave(k_dirs, dim=0)
                            expanded_inputs = {
                                "input_ids": expanded_input_ids,
                                "attention_mask": expanded_attention_mask,
                                "use_cache": True
                            }
                            if "position_ids" in inputs_template:
                                expanded_inputs["position_ids"] = inputs_template["position_ids"].repeat_interleave(k_dirs, dim=0)

                            pass_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=k_dirs)
                            expanded_inputs["past_key_values"] = pass_cache

                            dirs_for_pass = all_dirs[pass_start:pass_end]
                            dirs_batch = dirs_for_pass.unsqueeze(0).expand(B, -1, -1).reshape(B * k_dirs, -1)
                            hook.set_directions(dirs_batch)

                            with torch.inference_mode():
                                out = model(**expanded_inputs)
                                logits = out.logits[:, -1, :][:, option_token_ids]
                                probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                            for i, q_idx in enumerate(batch_indices):
                                for j in range(k_dirs):
                                    dir_idx = pass_start + j
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
                                        layer_results[layer]["ablated"][q_idx] = data
                                    else:
                                        ctrl_key = f"control_{dir_idx - 1}"
                                        layer_results[layer]["controls_ablated"][ctrl_key][q_idx] = data

                            pbar.update(1)
                    else:
                        # Sequential KV cache path
                        def run_single_ablation_kv(direction_tensor, result_list, key=None):
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

                        run_single_ablation_kv(cached_directions[layer]["direction"], layer_results[layer]["ablated"])
                        for i_c, ctrl_dir in enumerate(cached_directions[layer]["controls"]):
                            run_single_ablation_kv(ctrl_dir, layer_results[layer]["controls_ablated"], key=f"control_{i_c}")
                finally:
                    hook.remove()

        else:
            # Full forward path (position != "final")
            # Build position indices for this batch (adjusted for left-padding)
            batch_pos_indices = []
            seq_len = batch_inputs["input_ids"].shape[1]
            for i, q_idx in enumerate(batch_indices):
                pos = position_indices[q_idx]
                if pos >= 0:
                    # Adjust for left-padding
                    actual_len = int(batch_inputs["attention_mask"][i].sum())
                    pad_offset = seq_len - actual_len
                    adjusted_pos = pos + pad_offset
                else:
                    adjusted_pos = seq_len - 1  # fallback to final
                batch_pos_indices.append(adjusted_pos)
            batch_pos_tensor = torch.tensor(batch_pos_indices, dtype=torch.long, device=DEVICE)

            # Compute baseline (no ablation) - full forward
            if baseline_results[batch_indices[0]] is None:
                with torch.inference_mode():
                    out = model(**batch_inputs, use_cache=False)
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

            # Run ablation for each layer (full forward path with batched directions)
            for layer in layers:
                if hasattr(model, 'get_base_model'):
                    layer_module = model.get_base_model().model.layers[layer]
                else:
                    layer_module = model.model.layers[layer]

                all_dirs = cached_directions[layer]["all_stacked"]

                if use_batched:
                    # Batched ablation: expand batch by k_dirs directions per pass
                    for pass_start in range(0, total_directions, directions_per_pass):
                        pass_end = min(pass_start + directions_per_pass, total_directions)
                        k_dirs = pass_end - pass_start

                        # Expand inputs by k_dirs
                        expanded_input_ids = batch_inputs["input_ids"].repeat_interleave(k_dirs, dim=0)
                        expanded_attention_mask = batch_inputs["attention_mask"].repeat_interleave(k_dirs, dim=0)
                        expanded_inputs = {
                            "input_ids": expanded_input_ids,
                            "attention_mask": expanded_attention_mask,
                        }

                        # Expand position indices to match expanded batch
                        expanded_pos_tensor = batch_pos_tensor.repeat_interleave(k_dirs)

                        # Build direction tensor: (B * k_dirs, hidden_dim)
                        dirs_for_pass = all_dirs[pass_start:pass_end]
                        dirs_batch = dirs_for_pass.unsqueeze(0).expand(B, -1, -1).reshape(B * k_dirs, -1)

                        hook = BatchAblationHook(intervention_position="indexed")
                        hook.set_position_indices(expanded_pos_tensor)
                        hook.set_directions(dirs_batch)
                        hook.register(layer_module)

                        try:
                            with torch.inference_mode():
                                out = model(**expanded_inputs, use_cache=False)
                                logits = out.logits[:, -1, :][:, option_token_ids]
                                probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                            # Store results
                            for i, q_idx in enumerate(batch_indices):
                                for j in range(k_dirs):
                                    dir_idx = pass_start + j
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
                                        layer_results[layer]["ablated"][q_idx] = data
                                    else:
                                        ctrl_key = f"control_{dir_idx - 1}"
                                        layer_results[layer]["controls_ablated"][ctrl_key][q_idx] = data
                        finally:
                            hook.remove()

                        pbar.update(1)
                else:
                    # Sequential ablation (one direction per pass)
                    hook = BatchAblationHook(intervention_position="indexed")
                    hook.set_position_indices(batch_pos_tensor)
                    hook.register(layer_module)

                    try:
                        # Primary direction
                        dirs_batch = cached_directions[layer]["direction"].unsqueeze(0).expand(B, -1)
                        hook.set_directions(dirs_batch)

                        with torch.inference_mode():
                            out = model(**batch_inputs, use_cache=False)
                            logits = out.logits[:, -1, :][:, option_token_ids]
                            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                        for i, q_idx in enumerate(batch_indices):
                            p = probs[i]
                            resp = options[np.argmax(p)]
                            conf = signal_fn(p, mappings[q_idx])
                            m_val = metric_values[q_idx]
                            layer_results[layer]["ablated"][q_idx] = {
                                "question_idx": q_idx,
                                "response": resp,
                                "confidence": float(conf),
                                "metric": float(m_val),
                            }
                        pbar.update(1)

                        # Control directions
                        for i_c, ctrl_dir in enumerate(cached_directions[layer]["controls"]):
                            dirs_batch = ctrl_dir.unsqueeze(0).expand(B, -1)
                            hook.set_directions(dirs_batch)

                            with torch.inference_mode():
                                out = model(**batch_inputs, use_cache=False)
                                logits = out.logits[:, -1, :][:, option_token_ids]
                                probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                            for i, q_idx in enumerate(batch_indices):
                                p = probs[i]
                                resp = options[np.argmax(p)]
                                conf = signal_fn(p, mappings[q_idx])
                                m_val = metric_values[q_idx]
                                layer_results[layer]["controls_ablated"][f"control_{i_c}"][q_idx] = {
                                    "question_idx": q_idx,
                                    "response": resp,
                                    "confidence": float(conf),
                                    "metric": float(m_val),
                                }
                            pbar.update(1)
                    finally:
                        hook.remove()

    pbar.close()
    return {
        "layers": layers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": layer_results,
        "position": position,
    }


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_correlation(confidences: np.ndarray, metric_values: np.ndarray) -> float:
    """Compute Pearson correlation between confidence and metric."""
    if len(confidences) < 2 or np.std(confidences) < 1e-10 or np.std(metric_values) < 1e-10:
        return 0.0
    return float(np.corrcoef(confidences, metric_values)[0, 1])


def compute_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman (rank) correlation."""
    if len(x) < 2 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    return float(spearmanr(x, y).correlation)



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

        # Use signed metric for interpretation-consistent diagnostics
        metric_signed = baseline_metric * metric_sign

        # Control ablations (and Δconf diagnostics for controls)
        control_corrs = []
        control_delta_corrs = []
        control_delta_means = []

        for ctrl_key in lr["controls_ablated"]:
            ctrl_conf = np.array([r["confidence"] for r in lr["controls_ablated"][ctrl_key]])
            ctrl_metric = np.array([r["metric"] for r in lr["controls_ablated"][ctrl_key]])
            control_corrs.append(compute_correlation(ctrl_conf, ctrl_metric))

            delta_ctrl = ctrl_conf - baseline_conf
            control_delta_means.append(float(np.mean(delta_ctrl)))
            control_delta_corrs.append(compute_correlation(delta_ctrl, metric_signed))

        corr_change = ablated_corr - baseline_corr
        control_corr_changes = [c - baseline_corr for c in control_corrs]

        # -------- Δconf diagnostics --------
        # 1) Is the change mostly a uniform shift/rescale?
        delta_conf = ablated_conf - baseline_conf
        delta_conf_mean = float(np.mean(delta_conf))
        delta_conf_std = float(np.std(delta_conf))

        # 2) Is the change metric-dependent?
        delta_corr_metric = compute_correlation(delta_conf, metric_signed)
        delta_spearman_metric = compute_spearman(delta_conf, metric_signed)

        # Affine fit: ablated ≈ a * baseline + b
        if np.std(baseline_conf) > 1e-10:
            affine_slope, affine_intercept = np.polyfit(baseline_conf, ablated_conf, 1)
        else:
            affine_slope, affine_intercept = 0.0, float(np.mean(ablated_conf))

        baseline_to_ablated_corr = compute_correlation(baseline_conf, ablated_conf)

        # Residual after best affine transform: if this correlates with metric, it's not "just a shift/rescale"
        resid = ablated_conf - (affine_slope * baseline_conf + affine_intercept)
        residual_corr_metric = compute_correlation(resid, metric_signed)

        # Pooled p-value for delta_corr_metric vs controls (two-tailed)
        pooled_delta_corr = np.array(control_delta_corrs, dtype=np.float32)
        if len(pooled_delta_corr) > 0:
            n_worse = np.sum(np.abs(pooled_delta_corr) >= np.abs(delta_corr_metric))
            p_value_delta_corr_pooled = float((n_worse + 1) / (len(pooled_delta_corr) + 1))
        else:
            p_value_delta_corr_pooled = 1.0

        # Mean Δconf by metric decile (helps distinguish "bias shift" vs metric-targeted change)
        if np.std(metric_signed) < 1e-10:
            delta_by_decile = [None] * 10
        else:
            edges = np.quantile(metric_signed, np.linspace(0, 1, 11))
            if np.unique(edges).size < 3:
                delta_by_decile = [None] * 10
            else:
                bin_idx = np.digitize(metric_signed, edges[1:-1], right=True)  # 0..9
                delta_by_decile = [
                    float(np.mean(delta_conf[bin_idx == k])) if np.any(bin_idx == k) else None
                    for k in range(10)
                ]

        all_control_corr_changes.extend(control_corr_changes)

        layer_data[layer] = {
            "baseline_corr": baseline_corr,
            "baseline_conf_mean": float(np.mean(baseline_conf)),
            "ablated_corr": ablated_corr,
            "ablated_conf_mean": float(np.mean(ablated_conf)),
            "corr_change": corr_change,
            "control_corrs": control_corrs,
            "control_corr_changes": control_corr_changes,

            # Δconf diagnostics
            "delta_conf_mean": delta_conf_mean,
            "delta_conf_std": delta_conf_std,
            "delta_conf_corr_metric": float(delta_corr_metric),
            "delta_conf_spearman_metric": float(delta_spearman_metric),
            "baseline_to_ablated_conf_corr": float(baseline_to_ablated_corr),
            "affine_slope": float(affine_slope),
            "affine_intercept": float(affine_intercept),
            "residual_corr_metric": float(residual_corr_metric),
            "control_delta_conf_corr_metric_mean": float(np.mean(control_delta_corrs)) if len(control_delta_corrs) else 0.0,
            "control_delta_conf_corr_metric_std": float(np.std(control_delta_corrs)) if len(control_delta_corrs) else 0.0,
            "p_value_delta_corr_pooled": float(p_value_delta_corr_pooled),
            "delta_conf_mean_by_metric_decile": delta_by_decile,
        }

    # Convert pooled null to array
    pooled_null = np.array(all_control_corr_changes)

    # Second pass: compute p-values
    raw_p_values = []

    for layer in layers:
        ld = layer_data[layer]

        # Per-layer statistics (handle empty controls)
        if len(ld["control_corr_changes"]) > 0:
            ctrl_mean = float(np.mean(ld["control_corr_changes"]))
            ctrl_std = float(np.std(ld["control_corr_changes"]))
        else:
            ctrl_mean = 0.0
            ctrl_std = 0.0

        # Pooled p-value: two-tailed test (how many controls have |effect| >= |ours|)
        if len(pooled_null) > 0:
            n_pooled_worse = np.sum(np.abs(pooled_null) >= np.abs(ld["corr_change"]))
            p_value_pooled = (n_pooled_worse + 1) / (len(pooled_null) + 1)
        else:
            # No controls - can't compute p-value
            p_value_pooled = 1.0

        # Effect size (Z-score vs controls)
        if ctrl_std > 1e-10:
            effect_size_z = (ld["corr_change"] - ctrl_mean) / ctrl_std
            # Parametric p-value from Z-score (two-tailed)
            p_value_parametric = 2 * norm.sf(abs(effect_size_z))
        else:
            effect_size_z = 0.0
            p_value_parametric = 1.0

        raw_p_values.append((layer, p_value_pooled))

        # Handle case with no controls
        if len(ld["control_corrs"]) > 0:
            ctrl_corr_mean = float(np.mean(ld["control_corrs"]))
            ctrl_corr_std = float(np.std(ld["control_corrs"]))
        else:
            ctrl_corr_mean = ld["baseline_corr"]  # No control = baseline as reference
            ctrl_corr_std = 0.0

        analysis["per_layer"][layer] = {
            "baseline_correlation": ld["baseline_corr"],
            "baseline_confidence_mean": ld["baseline_conf_mean"],
            "ablated_correlation": ld["ablated_corr"],
            "ablated_confidence_mean": ld["ablated_conf_mean"],
            "correlation_change": ld["corr_change"],
            "control_correlation_mean": ctrl_corr_mean,
            "control_correlation_std": ctrl_corr_std,
            "control_correlation_change_mean": ctrl_mean,
            "control_correlation_change_std": ctrl_std,
            "p_value_pooled": float(p_value_pooled),
            "p_value_parametric": float(p_value_parametric),
            "effect_size_z": float(effect_size_z),

            # Δconf diagnostics
            "delta_conf_mean": ld["delta_conf_mean"],
            "delta_conf_std": ld["delta_conf_std"],
            "delta_conf_corr_metric": ld["delta_conf_corr_metric"],
            "delta_conf_spearman_metric": ld["delta_conf_spearman_metric"],
            "baseline_to_ablated_conf_corr": ld["baseline_to_ablated_conf_corr"],
            "affine_slope": ld["affine_slope"],
            "affine_intercept": ld["affine_intercept"],
            "residual_corr_metric": ld["residual_corr_metric"],
            "control_delta_conf_corr_metric_mean": ld["control_delta_conf_corr_metric_mean"],
            "control_delta_conf_corr_metric_std": ld["control_delta_conf_corr_metric_std"],
            "p_value_delta_corr_pooled": ld["p_value_delta_corr_pooled"],
            "delta_conf_mean_by_metric_decile": ld["delta_conf_mean_by_metric_decile"],
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
    significant_parametric = [l for l in layers if analysis["per_layer"][l]["p_value_parametric"] < 0.05]

    # Best layer by effect size magnitude (most extreme effect in either direction)
    best_layer = max(layers, key=lambda l: abs(analysis["per_layer"][l]["effect_size_z"]))

    analysis["summary"] = {
        "pooled_null_size": len(pooled_null),
        "significant_layers_pooled": significant_pooled,
        "significant_layers_fdr": significant_fdr,
        "significant_layers_parametric": significant_parametric,
        "n_significant_pooled": len(significant_pooled),
        "n_significant_fdr": len(significant_fdr),
        "n_significant_parametric": len(significant_parametric),
        "best_layer": best_layer,
        "best_effect_z": analysis["per_layer"][best_layer]["effect_size_z"],
    }


    # Optional: print extra diagnostics for layers with biggest corr changes
    if PRINT_DELTA_DIAGNOSTICS and len(layers) > 0:
        per = analysis["per_layer"]

        top_inc = sorted(layers, key=lambda l: per[l]["correlation_change"], reverse=True)[:DELTA_DIAGNOSTIC_TOPK]
        top_dec = sorted(layers, key=lambda l: per[l]["correlation_change"])[:DELTA_DIAGNOSTIC_TOPK]

        def _fmt_deciles(arr):
            def _fmt_one(x):
                if x is None:
                    return "None"
                try:
                    if x != x:  # NaN
                        return "nan"
                except Exception:
                    return "None"
                return f"{x:+.3f}"

            return "[" + ", ".join(_fmt_one(x) for x in arr) + "]"

        def _print_layer(l):
            d = per[l]
            print(f"    L{l:>3}  corr {d['baseline_correlation']:+.3f} -> {d['ablated_correlation']:+.3f}  (Δ={d['correlation_change']:+.3f}, Z={d['effect_size_z']:+.2f}, FDR={d['p_value_fdr']:.3g})")
            print(f"         conf mean {d['baseline_confidence_mean']:.3f} -> {d['ablated_confidence_mean']:.3f}  (Δmean={d['delta_conf_mean']:+.4f}, Δstd={d['delta_conf_std']:.4f})")
            print(f"         corr(Δconf, metric*s) pearson={d['delta_conf_corr_metric']:+.3f} spearman={d['delta_conf_spearman_metric']:+.3f}  (ctrl mean={d['control_delta_conf_corr_metric_mean']:+.3f}±{d['control_delta_conf_corr_metric_std']:.3f}, p_pooled={d['p_value_delta_corr_pooled']:.3g})")
            print(f"         ablated≈{d['affine_slope']:.3f}*baseline+{d['affine_intercept']:+.3f}  corr(baseline,ablated)={d['baseline_to_ablated_conf_corr']:.4f}  corr(resid, metric*s)={d['residual_corr_metric']:+.3f}")
            print(f"         mean Δconf by metric decile: {_fmt_deciles(d['delta_conf_mean_by_metric_decile'])}")

        print("\n  [Δconf diagnostics] Layers with biggest +Δcorr:")
        for l in top_inc:
            _print_layer(l)

        print("\n  [Δconf diagnostics] Layers with biggest -Δcorr:")
        for l in top_dec:
            _print_layer(l)

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

    fig, axes = plt.subplots(3, 1, figsize=(20, 14))
    fig.suptitle(f"Ablation Results: {method.upper()} directions ({analysis['metric']})", fontsize=14)

    x = np.arange(len(layers))

    # Panel 1: Absolute correlations
    ax1 = axes[0]
    baseline_corrs = np.array([analysis["per_layer"][l]["baseline_correlation"] for l in layers])
    ablated_corrs = np.array([analysis["per_layer"][l]["ablated_correlation"] for l in layers])
    ctrl_corrs = np.array([analysis["per_layer"][l]["control_correlation_mean"] for l in layers])
    ctrl_corr_stds = np.array([analysis["per_layer"][l]["control_correlation_std"] for l in layers])

    # Smaller markers, thinner lines for 80 layers
    ax1.plot(x, baseline_corrs, '-', label='Baseline', color='blue', linewidth=1.5, marker='o', markersize=3)
    ax1.plot(x, ablated_corrs, '-', label=f'{method} ablated', color='red', linewidth=1.5, marker='s', markersize=3)
    # Control with ±1 SD band (SD of control correlations, not changes)
    ax1.plot(x, ctrl_corrs, '--', label='Control ablated (mean)', color='gray', linewidth=1, alpha=0.8)
    ax1.fill_between(x, ctrl_corrs - ctrl_corr_stds, ctrl_corrs + ctrl_corr_stds,
                     color='gray', alpha=0.2, label='Control ±1σ')

    ax1.axhline(y=0, color='black', linestyle=':', alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Correlation (confidence vs metric)")
    ax1.set_title("Correlation by Condition")
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Auto-zoom y-axis to data range with padding (handle NaN/Inf)
    all_vals = np.concatenate([baseline_corrs, ablated_corrs, ctrl_corrs - ctrl_corr_stds, ctrl_corrs + ctrl_corr_stds])
    all_vals = all_vals[np.isfinite(all_vals)]  # Filter out NaN/Inf
    if len(all_vals) > 0:
        ymin, ymax = np.min(all_vals), np.max(all_vals)
        padding = (ymax - ymin) * 0.1 if ymax > ymin else 0.1
        ax1.set_ylim(ymin - padding, ymax + padding)

    # Panel 2: Effect size with significance and CIs
    ax2 = axes[1]
    effect_sizes = [analysis["per_layer"][l]["effect_size_z"] for l in layers]
    p_values_fdr = [analysis["per_layer"][l]["p_value_fdr"] for l in layers]

    colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'gray' for p in p_values_fdr]
    bars = ax2.bar(x, effect_sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add error bars showing ±1 SD of control distribution (in Z-score units, this is ±1)
    ax2.errorbar(x, effect_sizes, yerr=1.0, fmt='none', ecolor='black', capsize=2, alpha=0.4, linewidth=1)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Effect Size (Z-score)")
    ax2.set_title("Effect Size vs Controls (error bars = ±1 SD of null)")
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

Note: Negative Z = ablation decreased correlation
      (expected if direction is causally necessary)
      Positive Z = ablation increased correlation
      (opposite of expected causal effect)

Interpretation:
"""
    if summary['n_significant_fdr'] > 0:
        # Check direction of effect
        best_z = summary['best_effect_z']
        if best_z < 0 and baseline_corr > 0:
            direction_text = "Ablation DECREASED correlation (expected causal effect)"
        elif best_z > 0 and baseline_corr > 0:
            direction_text = "Ablation INCREASED correlation (opposite of expected)"
        else:
            direction_text = f"Effect direction: Z={best_z:.2f}"

        summary_text += f"""  ✓ SIGNIFICANT after FDR correction
  {summary['n_significant_fdr']} layer(s) show effects beyond
  random direction ablations.
  {direction_text}"""
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

    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    fig.suptitle("Method Comparison: Ablation Effects", fontsize=14)

    x = np.arange(len(layers))
    colors = {'probe': 'tab:blue', 'mean_diff': 'tab:orange'}

    # Panel 1: Effect sizes by layer (line plot)
    ax1 = axes[0]
    for method in methods:
        effect_sizes = [analyses[method]["per_layer"][l]["effect_size_z"] for l in layers]
        p_values_fdr = [analyses[method]["per_layer"][l]["p_value_fdr"] for l in layers]
        color = colors.get(method, 'gray')

        # Line plot
        ax1.plot(x, effect_sizes, '-', label=method, color=color, linewidth=1.5, alpha=0.8)

        # Mark significant layers with filled markers
        sig_x = [i for i, p in enumerate(p_values_fdr) if p < 0.05]
        sig_y = [effect_sizes[i] for i in sig_x]
        ax1.scatter(sig_x, sig_y, color=color, s=40, zorder=5, edgecolor='black', linewidth=0.5)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Effect Size (Z-score)")
    ax1.set_title("Effect Size by Method (filled markers = FDR p<0.05)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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
    print("\nKey question: Does ablating the direction HURT calibration?")
    print("Expected: correlation should DECREASE (negative change)")

    for method, analysis in analyses.items():
        layers = analysis['layers']
        print(f"\n{method.upper()} directions ({len(layers)} layers):")
        print("-" * 70)
        print(f"{'Layer':>5}  {'Baseline':>8}  {'Ablated':>8}  {'Change':>8}  {'p-value':>8}  {'Hurt?':>6}")
        print("-" * 70)

        # Sort by effect magnitude for easier reading
        sorted_layers = sorted(layers, key=lambda l: analysis['per_layer'][l]['correlation_change'])

        for layer in sorted_layers:
            ld = analysis['per_layer'][layer]
            baseline = ld['baseline_correlation']
            ablated = ld['ablated_correlation']
            change = ld['correlation_change']
            p_val = ld['p_value_parametric']

            # "Hurt" = significant decrease in correlation (ablation impaired calibration)
            hurt = "YES" if (change < 0 and p_val < 0.05) else "no"
            sig = "*" if p_val < 0.05 else ""

            print(f"{layer:>5}  {baseline:>8.4f}  {ablated:>8.4f}  {change:>+8.4f}  {p_val:>8.2e}{sig:1}  {hurt:>6}")

        # Summary
        n_hurt = sum(1 for l in layers
                     if analysis['per_layer'][l]['correlation_change'] < 0
                     and analysis['per_layer'][l]['p_value_parametric'] < 0.05)
        n_helped = sum(1 for l in layers
                       if analysis['per_layer'][l]['correlation_change'] > 0
                       and analysis['per_layer'][l]['p_value_parametric'] < 0.05)

        print("-" * 70)
        print(f"Summary: {n_hurt} layers where ablation HURT calibration (p<0.05)")
        if n_helped > 0:
            print(f"         {n_helped} layers where ablation HELPED calibration (unexpected)")


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
    print(f"Controls: {NUM_CONTROLS} (final), {NUM_CONTROLS_NONFINAL} (non-final)")

    # Load directions
    print("\nLoading directions...")
    all_directions = load_directions(INPUT_BASE_NAME, METRIC)
    available_methods = list(all_directions.keys())
    print(f"  Found methods: {available_methods}")

    # Filter to requested methods
    if METHODS is not None:
        methods = [m for m in METHODS if m in available_methods]
        if not methods:
            raise ValueError(f"No matching methods found. Available: {available_methods}, requested: {METHODS}")
        print(f"  Using methods: {methods}")
    else:
        methods = available_methods

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

    # Load transfer results for layer selection (non-final positions)
    transfer_data = load_transfer_results(INPUT_BASE_NAME, META_TASK)
    if transfer_data is not None:
        print(f"\nLoaded transfer results for layer selection")
        # Preview what layers would be selected FOR EACH (POSITION, METHOD) combination
        for pos in PROBE_POSITIONS:
            if pos == "final":
                print(f"  {pos}: all layers (no R² filter)")
            else:
                for method in methods:
                    pos_layers = get_layers_from_transfer(transfer_data, METRIC, pos, TRANSFER_R2_THRESHOLD, method)
                    if pos_layers:
                        print(f"  {pos}/{method}: {len(pos_layers)} layers with {METRIC} R²≥{TRANSFER_R2_THRESHOLD}: {pos_layers}")
                    else:
                        # Try fallback to final
                        fallback_layers = get_layers_from_transfer(transfer_data, METRIC, "final", TRANSFER_R2_THRESHOLD, method)
                        if fallback_layers:
                            print(f"  {pos}/{method}: no position-specific data, using final: {len(fallback_layers)} layers")
                        else:
                            print(f"  {pos}/{method}: WARNING - no layers found, will use ALL layers")
    else:
        expected_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_transfer_{META_TASK}_results.json"
        print(f"\nNo transfer results found - will use all layers for all positions")
        print(f"  Expected: {expected_path}")

    # Determine base layers (all available)
    all_available_layers = sorted(all_directions[methods[0]].keys())

    # Layer selection depends on position - will be set per-position below
    if LAYERS is not None:
        print(f"\nExplicit LAYERS override: {len(LAYERS)} layers")
    else:
        print(f"\nLayer selection: all layers for final, R²≥{TRANSFER_R2_THRESHOLD} for non-final")

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

    # Run ablation for each method and position
    # Structure: {position: {method: analysis}}
    all_results_by_pos = {pos: {} for pos in PROBE_POSITIONS}
    all_analyses_by_pos = {pos: {} for pos in PROBE_POSITIONS}

    for position in PROBE_POSITIONS:
        print(f"\n{'='*70}")
        print(f"POSITION: {position}")
        print(f"{'='*70}")

        # Determine number of controls for this position
        position_num_controls = NUM_CONTROLS if position == "final" else NUM_CONTROLS_NONFINAL

        for method in methods:
            print(f"\n{'='*60}")
            print(f"ABLATION EXPERIMENT: {method.upper()} @ {position}")
            print(f"{'='*60}")

            # Determine layers for this position AND method
            if LAYERS is not None:
                # Explicit override applies to all positions/methods
                method_layers = LAYERS
            elif position == "final":
                # Final position: use all layers
                method_layers = all_available_layers
            else:
                # Non-final position: select based on transfer R² for THIS method
                if transfer_data is not None:
                    method_layers = get_layers_from_transfer(
                        transfer_data, METRIC, position, TRANSFER_R2_THRESHOLD, method
                    )
                    if not method_layers:
                        # Fall back to "final" position transfer data if position-specific not available
                        method_layers = get_layers_from_transfer(
                            transfer_data, METRIC, "final", TRANSFER_R2_THRESHOLD, method
                        )
                else:
                    method_layers = all_available_layers

                if not method_layers:
                    print("\n" + "!"*70)
                    print("!!! WARNING: FALLING BACK TO ALL LAYERS !!!")
                    print(f"!!! No layers meet R²≥{TRANSFER_R2_THRESHOLD} threshold for {method}/{METRIC}")
                    print(f"!!! This will test {len(all_available_layers)} layers instead of ~50")
                    print(f"!!! Check that METRIC and method match transfer results")
                    print("!"*70)
                    print("Continuing in 3 seconds (Ctrl+C to abort)...")
                    import time
                    time.sleep(3)
                    method_layers = all_available_layers

            print(f"  Layers: {len(method_layers)} (range {min(method_layers)}-{max(method_layers)})")
            print(f"  Controls: {position_num_controls}")

            results = run_ablation_for_method(
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                metric_values=metric_values,
                directions=all_directions[method],
                num_controls=position_num_controls,
                meta_task=META_TASK,
                use_chat_template=use_chat_template,
                layers=method_layers,
                position=position,
            )
            all_results_by_pos[position][method] = results

            # Analyze results
            print(f"\n  Analyzing results...")
            analysis = analyze_ablation_results(results, METRIC)
            all_analyses_by_pos[position][method] = analysis

            summary = analysis["summary"]
            print(f"  Significant layers (FDR): {summary['n_significant_fdr']}")
            print(f"  Best layer: {summary['best_layer']} (Z={summary['best_effect_z']:.2f})")

        # Incremental save after each position completes (crash protection)
        model_short = get_model_short_name(MODEL)
        base_output = f"{model_short}_{INPUT_BASE_NAME.split('_')[-1]}_ablation_{META_TASK}_{METRIC}"
        checkpoint_path = OUTPUT_DIR / f"{base_output}_checkpoint.json"
        checkpoint_json = {
            "config": {
                "model": MODEL,
                "input_base_name": INPUT_BASE_NAME,
                "metric": METRIC,
                "meta_task": META_TASK,
                "num_questions": NUM_QUESTIONS,
                "positions_completed": [p for p in PROBE_POSITIONS if all_analyses_by_pos[p]],
            },
            "by_position": {},
        }
        for pos in PROBE_POSITIONS:
            if all_analyses_by_pos[pos]:
                checkpoint_json["by_position"][pos] = {}
                for m, analysis in all_analyses_by_pos[pos].items():
                    checkpoint_json["by_position"][pos][m] = {
                        "per_layer": analysis["per_layer"],
                        "summary": analysis["summary"],
                    }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_json, f, indent=2)
        print(f"  Checkpoint saved: {checkpoint_path.name}")

    # For backward compatibility, keep "final" as the default if it was tested
    if "final" in all_analyses_by_pos:
        all_analyses = all_analyses_by_pos["final"]
    else:
        # Use first available position
        all_analyses = all_analyses_by_pos[PROBE_POSITIONS[0]]

    # Generate output filename
    model_short = get_model_short_name(MODEL)
    base_output = f"{model_short}_{INPUT_BASE_NAME.split('_')[-1]}_ablation_{META_TASK}_{METRIC}"

    # Save JSON results
    print("\nSaving results...")
    results_path = OUTPUT_DIR / f"{base_output}_results.json"

    # Load existing results if present, otherwise create new
    if results_path.exists():
        with open(results_path, "r") as f:
            output_json = json.load(f)
        print(f"  Merging with existing results: {results_path.name}")
        # Ensure by_position exists (for older format files)
        if "by_position" not in output_json:
            output_json["by_position"] = {}
    else:
        output_json = {
            "config": {
                "model": MODEL,
                "input_base_name": INPUT_BASE_NAME,
                "metric": METRIC,
                "meta_task": META_TASK,
                "num_questions": NUM_QUESTIONS,
                "num_controls_final": NUM_CONTROLS,
                "num_controls_nonfinal": NUM_CONTROLS_NONFINAL,
                "transfer_r2_threshold": TRANSFER_R2_THRESHOLD,
                "methods_tested": [],
                "positions_tested": [],
            },
            "by_position": {},
        }

    # Update config with current run's positions/methods (accumulate)
    existing_positions = set(output_json["config"].get("positions_tested", []))
    existing_methods = set(output_json["config"].get("methods_tested", []))
    output_json["config"]["positions_tested"] = sorted(existing_positions | set(PROBE_POSITIONS))
    output_json["config"]["methods_tested"] = sorted(existing_methods | set(methods))

    # Legacy format: only update top-level keys when "final" position was tested
    if "final" in PROBE_POSITIONS:
        for method, analysis in all_analyses.items():
            output_json[method] = {
                "layers": analysis["layers"],
                "num_questions": analysis["num_questions"],
                "num_controls": analysis["num_controls"],
                "metric": analysis["metric"],
                "per_layer": analysis["per_layer"],
                "summary": analysis["summary"],
            }

    # Merge new results into by_position (overwrites same position/method)
    for position in PROBE_POSITIONS:
        if position not in output_json["by_position"]:
            output_json["by_position"][position] = {}
        for method, analysis in all_analyses_by_pos[position].items():
            output_json["by_position"][position][method] = {
                "layers": analysis["layers"],
                "num_questions": analysis["num_questions"],
                "num_controls": analysis["num_controls"],
                "metric": analysis["metric"],
                "per_layer": analysis["per_layer"],
                "summary": analysis["summary"],
            }

    # Comparison summary (for "final" position)
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

    # Generate plots for each position
    print("\nGenerating plots...")
    for position in PROBE_POSITIONS:
        for method in methods:
            plot_path = OUTPUT_DIR / f"{base_output}_{method}_{position}.png"
            plot_ablation_results(all_analyses_by_pos[position][method], method, plot_path)

        if len(methods) >= 2:
            comparison_path = OUTPUT_DIR / f"{base_output}_comparison_{position}.png"
            plot_method_comparison(all_analyses_by_pos[position], comparison_path)

    # Print summary for each position
    for position in PROBE_POSITIONS:
        print(f"\n--- Position: {position} ---")
        print_summary(all_analyses_by_pos[position])

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  {results_path.name}")
    for position in PROBE_POSITIONS:
        for method in methods:
            print(f"  {base_output}_{method}_{position}.png")
        if len(methods) >= 2:
            print(f"  {base_output}_comparison_{position}.png")


if __name__ == "__main__":
    main()
