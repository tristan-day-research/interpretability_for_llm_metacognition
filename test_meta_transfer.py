"""
Test whether directions found in identify_mc_correlate.py transfer to meta-tasks.

This tests the core introspection hypothesis: if the model encodes uncertainty
during direct MC answering, does the same representation appear when the model
reports its confidence?

Loads from identify_mc_correlate.py outputs:
- {model}_{dataset}_mc_{metric}_directions.npz: Direction vectors
- {model}_{dataset}_mc_dataset.json: Questions and metric values

Configuration is set at the top of the script - no CLI args needed.
"""

from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

from core import (
    load_model_and_tokenizer,
    get_model_short_name,
    should_use_chat_template,
    BatchedExtractor,
    evaluate_transfer,
    metric_sign_for_confidence,
)
from tasks import (
    format_stated_confidence_prompt,
    format_answer_or_delegate_prompt,
    get_stated_confidence_signal,
    get_answer_or_delegate_signal,
    STATED_CONFIDENCE_OPTIONS,
    ANSWER_OR_DELEGATE_OPTIONS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base name for input files from identify_mc_correlate.py
# Will load: {BASE_NAME}_mc_{metric}_directions.npz and {BASE_NAME}_mc_dataset.json
INPUT_BASE_NAME = "Llama-3.1-8B-Instruct_SimpleMC"

# Which metrics to test transfer for
METRICS = ["entropy", "logit_gap"]

# Model (must match the one used in identify)
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = None  # Optional: path to PEFT/LoRA adapter (must match identify step)

# Meta task to test
META_TASK = "confidence"  # "confidence" or "delegate"

# Processing
BATCH_SIZE = 8

# Output
OUTPUT_DIR = Path(__file__).parent / "outputs"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def load_directions(directions_path: Path) -> dict:
    """Load directions from a _directions.npz file."""
    data = np.load(directions_path, allow_pickle=True)

    # Get metadata
    metadata = {
        "dataset": str(data.get("_metadata_dataset", "")),
        "model": str(data.get("_metadata_model", "")),
        "metric": str(data.get("_metadata_metric", "")),
    }

    # Extract directions for each method
    directions = {"probe": {}, "mean_diff": {}}

    for key in data.files:
        if key.startswith("_"):
            continue
        # Key format: {method}_layer_{layer}
        parts = key.split("_layer_")
        if len(parts) == 2:
            method, layer = parts[0], int(parts[1])
            directions[method][layer] = data[key]

    num_layers = max(max(directions["probe"].keys()), max(directions["mean_diff"].keys())) + 1

    return {
        "metadata": metadata,
        "directions": directions,
        "num_layers": num_layers,
    }


def load_dataset(dataset_path: Path) -> dict:
    """Load dataset JSON with questions and metric values."""
    with open(dataset_path) as f:
        data = json.load(f)

    # Extract questions
    questions = data["data"]

    # Extract metric values as arrays
    metric_values = {}
    for item in questions:
        for key, val in item.items():
            if key in ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]:
                if key not in metric_values:
                    metric_values[key] = []
                metric_values[key].append(val)

    # Convert to numpy
    for key in metric_values:
        metric_values[key] = np.array(metric_values[key])

    return {
        "config": data["config"],
        "stats": data["stats"],
        "questions": questions,
        "metric_values": metric_values,
    }


def get_meta_format_fn(meta_task: str):
    """Get the prompt formatting function for a meta task."""
    if meta_task == "confidence":
        return format_stated_confidence_prompt
    elif meta_task == "delegate":
        return format_answer_or_delegate_prompt
    else:
        raise ValueError(f"Unknown meta task: {meta_task}")


def get_meta_signal_fn(meta_task: str):
    """Get the signal extraction function for a meta task."""
    if meta_task == "confidence":
        return lambda probs, mapping: get_stated_confidence_signal(probs)
    elif meta_task == "delegate":
        return get_answer_or_delegate_signal
    else:
        raise ValueError(f"Unknown meta task: {meta_task}")


def get_meta_options(meta_task: str):
    """Get option tokens for a meta task."""
    if meta_task == "confidence":
        return list(STATED_CONFIDENCE_OPTIONS.keys())
    elif meta_task == "delegate":
        return ANSWER_OR_DELEGATE_OPTIONS
    else:
        raise ValueError(f"Unknown meta task: {meta_task}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load dataset
    dataset_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_mc_dataset.json"
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)

    print(f"  Model: {dataset['config']['base_model']}")
    print(f"  Dataset: {dataset['config']['dataset']}")
    print(f"  Questions: {len(dataset['questions'])}")
    print(f"  Metrics available: {list(dataset['metric_values'].keys())}")

    # Verify model matches
    if get_model_short_name(MODEL) != get_model_short_name(dataset['config']['base_model']):
        raise ValueError(f"Model mismatch: {MODEL} vs {dataset['config']['base_model']}")

    # Load directions for each metric
    all_directions = {}
    num_layers = None

    for metric in METRICS:
        directions_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_mc_{metric}_directions.npz"
        if not directions_path.exists():
            print(f"  Warning: {directions_path} not found, skipping {metric}")
            continue

        print(f"Loading directions for {metric} from {directions_path}...")
        dir_data = load_directions(directions_path)
        all_directions[metric] = dir_data["directions"]
        if num_layers is None:
            num_layers = dir_data["num_layers"]

    if not all_directions:
        raise ValueError("No direction files found!")

    print(f"  Layers: {num_layers}")

    model_short = get_model_short_name(MODEL)
    if ADAPTER:
        adapter_short = get_model_short_name(ADAPTER)
        output_path = OUTPUT_DIR / f"{model_short}_adapter-{adapter_short}_{dataset['config']['dataset']}_transfer_{META_TASK}.npz"
    else:
        output_path = OUTPUT_DIR / f"{model_short}_{dataset['config']['dataset']}_transfer_{META_TASK}.npz"
    print(f"\nMeta task: {META_TASK}")
    print(f"Output: {output_path}")

    # Load model
    print("\nLoading model...")
    model, tokenizer, num_layers_model = load_model_and_tokenizer(MODEL, adapter_path=ADAPTER)
    use_chat_template = should_use_chat_template(MODEL, tokenizer)

    if num_layers_model != num_layers:
        print(f"  Warning: model has {num_layers_model} layers but directions have {num_layers}")
        num_layers = min(num_layers, num_layers_model)

    # Get questions
    questions = dataset["questions"]

    # Get meta task setup
    format_fn = get_meta_format_fn(META_TASK)
    signal_fn = get_meta_signal_fn(META_TASK)
    meta_options = get_meta_options(META_TASK)
    option_token_ids = [tokenizer.encode(k, add_special_tokens=False)[0] for k in meta_options]

    print(f"  Meta options: {meta_options}")
    print(f"  Option token IDs: {option_token_ids}")

    # Extract meta activations
    print(f"\nExtracting meta activations (batch_size={BATCH_SIZE})...")

    all_activations = {layer: [] for layer in range(num_layers)}
    all_confidences = []
    all_mappings = []

    with BatchedExtractor(model, num_layers) as extractor:
        for batch_start in tqdm(range(0, len(questions), BATCH_SIZE)):
            batch_questions = questions[batch_start:batch_start + BATCH_SIZE]

            prompts = []
            batch_mappings = []
            for i, q in enumerate(batch_questions):
                trial_idx = batch_start + i
                if META_TASK == "delegate":
                    prompt, _, mapping = format_fn(q, tokenizer, trial_index=trial_idx, use_chat_template=use_chat_template)
                    batch_mappings.append(mapping)
                else:
                    prompt, _ = format_fn(q, tokenizer, use_chat_template=use_chat_template)
                    batch_mappings.append(None)
                prompts.append(prompt)

            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)

            layer_acts, probs, _, _ = extractor.extract_batch(input_ids, attention_mask, option_token_ids)

            for item_acts in layer_acts:
                for layer, act in item_acts.items():
                    all_activations[layer].append(act)

            for p, mapping in zip(probs, batch_mappings):
                confidence = signal_fn(p, mapping)
                all_confidences.append(confidence)
                all_mappings.append(mapping)

    # Stack activations
    print("\nStacking activations...")
    meta_activations = {
        layer: np.stack(acts) for layer, acts in all_activations.items()
    }
    print(f"  Shape per layer: {meta_activations[0].shape}")

    confidences = np.array(all_confidences)
    print(f"\nStated confidence: mean={confidences.mean():.3f}, std={confidences.std():.3f}")

    # Test transfer for each metric and method
    print("\n" + "=" * 60)
    print("TESTING TRANSFER")
    print("=" * 60)

    transfer_results = {}
    metrics_tested = list(all_directions.keys())

    for metric in metrics_tested:
        print(f"\n--- {metric.upper()} ---")

        direct_values = dataset["metric_values"][metric]
        sign = metric_sign_for_confidence(metric)

        transfer_results[metric] = {}

        for method in ["probe", "mean_diff"]:
            transfer_results[metric][method] = {}

            for layer in range(num_layers):
                direction = all_directions[metric][method][layer]
                result = evaluate_transfer(
                    meta_activations[layer],
                    direction,
                    direct_values
                )
                transfer_results[metric][method][layer] = result

            # Find best layer
            best_layer = max(
                range(num_layers),
                key=lambda l: transfer_results[metric][method][l]["r2"]
            )
            best_r2 = transfer_results[metric][method][best_layer]["r2"]
            best_corr = transfer_results[metric][method][best_layer]["corr"]

            avg_r2 = np.mean([transfer_results[metric][method][l]["r2"] for l in range(num_layers)])

            print(f"  {method:12s}: transfer R²={best_r2:.3f} at layer {best_layer}, avg R²={avg_r2:.3f}")

    # Behavioral correlation
    print("\n" + "-" * 40)
    print("BEHAVIORAL CORRELATION")
    print("-" * 40)

    behavioral = {}
    for metric in metrics_tested:
        direct_values = dataset["metric_values"][metric]
        sign = metric_sign_for_confidence(metric)

        corr, p_value = pearsonr(direct_values * sign, confidences)
        spearman_corr, spearman_p = spearmanr(direct_values * sign, confidences)

        behavioral[metric] = {
            "pearson_r": float(corr),
            "pearson_p": float(p_value),
            "spearman_r": float(spearman_corr),
            "spearman_p": float(spearman_p),
        }

        sign_str = "(inverted)" if sign < 0 else ""
        print(f"  {metric} {sign_str}: r={corr:.3f} (p={p_value:.2e}), ρ={spearman_corr:.3f}")

    # Save results
    print(f"\nSaving to {output_path}...")

    save_dict = {
        "model": MODEL,
        "dataset": dataset['config']['dataset'],
        "meta_task": META_TASK,
        "metrics": metrics_tested,
        "num_questions": len(questions),
        "num_layers": num_layers,
        "confidences": confidences,
    }

    for metric in metrics_tested:
        for method in ["probe", "mean_diff"]:
            for layer in range(num_layers):
                r2 = transfer_results[metric][method][layer]["r2"]
                corr = transfer_results[metric][method][layer]["corr"]
                save_dict[f"transfer_{metric}_{method}_layer{layer}_r2"] = r2
                save_dict[f"transfer_{metric}_{method}_layer{layer}_corr"] = corr

        save_dict[f"behavioral_{metric}_pearson_r"] = behavioral[metric]["pearson_r"]
        save_dict[f"behavioral_{metric}_spearman_r"] = behavioral[metric]["spearman_r"]

    np.savez(output_path, **save_dict)
    print("Done!")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for metric in metrics_tested:
        print(f"\n{metric}:")

        for method in ["probe", "mean_diff"]:
            best_layer = max(
                range(num_layers),
                key=lambda l: transfer_results[metric][method][l]["r2"]
            )
            transfer_r2 = transfer_results[metric][method][best_layer]["r2"]
            print(f"  {method}: transfer R²={transfer_r2:.3f} (layer {best_layer})")

        sign_str = "(inv)" if metric_sign_for_confidence(metric) < 0 else ""
        print(f"  behavioral{sign_str}: r={behavioral[metric]['pearson_r']:.3f}")


if __name__ == "__main__":
    main()
