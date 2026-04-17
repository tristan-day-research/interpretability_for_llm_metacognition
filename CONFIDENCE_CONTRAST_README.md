# Confidence Contrast Analysis

## Overview

`confidence_contrast.py` computes two types of contrast vectors in residual-stream space for uncertainty analysis:

1. **Entropy-based mean-diff contrast**
   - Compares high vs low output entropy from direct MC task
   - Uses quantile-based mean difference: `mean(high_entropy) - mean(low_entropy)`
   
2. **Confidence-based mean-diff contrast**
   - Compares high vs low stated confidence from confidence task
   - Uses quantile-based mean difference: `mean(high_confidence) - mean(low_confidence)`
   - Same robust approach as entropy, avoids overfitting from imbalanced data

## What It Does

### Part 1: Extract Direct MC Activations
- Loads model and questions
- Extracts activations at final token position for direct MC task
- Computes output entropy for each question

### Part 2: Extract Confidence Task Activations
- Extracts activations at final token for confidence task
- Records model's stated confidence tokens (S-Z scale)

### Part 3: Compute Contrasts

#### Entropy Mean-Diff Contrast
- Groups samples into high/low entropy (top/bottom 25%)
- Computes direction: `mean(high) - mean(low)`
- Normalizes to unit norm
- Evaluates via correlation with entropy values

#### Confidence Mean-Diff Contrast
- Maps confidence tokens to scalar values (midpoints):
  - S: 0.025 (<5%)
  - T: 0.075 (5-10%)
  - U: 0.15 (10-20%)
  - V: 0.3 (20-40%)
  - W: 0.5 (40-60%)
  - X: 0.7 (60-80%)
  - Y: 0.85 (80-90%)
  - Z: 0.95 (>90%)
- Groups samples into high/low confidence (top/bottom 25% by value)
- Computes direction: `mean(high_conf) - mean(low_conf)`
- Normalizes to unit norm
- Evaluates via correlation with confidence values

**Why mean-diff instead of ridge regression?**
- More robust to severe data imbalance (e.g., 88% "Z" tokens)
- Doesn't overfit to dominant class
- No hyperparameter tuning needed
- Same proven method that works for entropy

### Part 4: Cosine Similarity Analysis

Computes cosine similarity between entropy and confidence directions for each layer:
- `cos(d_entropy_l, d_conf_l)` for each layer l
- Both directions are unit-normalized, so cosine = dot product
- Tracks positive/negative alignment across layers
- Identifies layers with maximum alignment

### Part 5: Save Results

Saves 6 files per run:
1. `{model}_{dataset}_entropy_contrast.json` - Metadata and per-layer stats
2. `{model}_{dataset}_entropy_contrast_directions.pt` - PyTorch tensor (layers × hidden_dim)
3. `{model}_{dataset}_confidence_contrast.json` - Metadata and per-layer stats
4. `{model}_{dataset}_confidence_contrast_directions.pt` - PyTorch tensor (layers × hidden_dim)
5. `{model}_{dataset}_cosine_similarity_entropy_vs_confidence.json` - Similarity analysis
6. `{model}_{dataset}_cosine_similarity_entropy_vs_confidence.png` - Similarity plot

### Part 6: Validation

Automatically validates:
- All values are finite (no NaN/Inf)
- All directions are unit norm (||d|| = 1)
- Correct shape (hidden_dim for model)

## Configuration

Edit the top of `confidence_contrast.py`:

```python
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = None  # Or path to adapter
DATASET = "TriviaMC"
NUM_QUESTIONS = 500
SEED = 42
BATCH_SIZE = 8

# Quantization (for large models)
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False

# Contrast parameters
ENTROPY_QUANTILE = 0.25  # Top/bottom 25% for mean-diff
CONFIDENCE_ALPHA_CANDIDATES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]  # Ridge CV
CONFIDENCE_TRAIN_SPLIT = 0.8  # Train/val split
```

## Usage

```bash
python confidence_contrast.py
```

The script will:
1. Load model and extract activations for both tasks
2. Compute both contrasts per layer
3. Save results to `outputs/` directory
4. Print summary statistics

## Cosine Similarity Analysis

The script automatically analyzes the relationship between the two contrast directions:

### What It Measures

For each layer l, computes:
```
cosine_similarity_l = dot(d_entropy_l, d_conf_l)
```

Since both directions are unit-normalized, this is simply their dot product.

### Interpretation

- **Positive cosine**: Directions point in similar directions
  - Higher entropy correlates with higher confidence (counterintuitive!)
  - Or lower entropy correlates with lower confidence
  
- **Negative cosine**: Directions point in opposite directions
  - Higher entropy correlates with lower confidence (expected!)
  - This is the typical uncertainty relationship

- **Near-zero cosine**: Directions are orthogonal
  - Entropy and confidence encode different aspects
  - May operate in different subspaces

### Quality Checks

The analysis includes automatic validation:
- Verifies vector dimensions match per layer
- Ensures near-unit norms (||d|| ≈ 1.0)
- Normalizes if needed and logs warnings
- Identifies layers with max/min/max-abs cosine

### Plot Format

The generated plot shows:
- X-axis: Layer index (0 to num_layers-1)
- Y-axis: Cosine similarity (-1 to +1)
- Horizontal reference lines at 0, ±0.2
- Statistics box showing mean, std, max, min
- Clean matplotlib style (no seaborn)

## Output Format

### JSON Metadata

**Entropy and Confidence Contrast JSONs** contain:
- `method`: "mean_diff" or "ordinal_ridge"
- `target`: "entropy" or "stated_confidence"
- `num_samples`, `num_layers`
- `per_layer`: List of dicts with:
  - `layer`: Layer index
  - `r2`, `corr`: Fit quality metrics
  - `norm`: Direction magnitude (should be 1.0)
  - `alpha`: Ridge regularization (confidence only)
  - Additional method-specific stats

**Cosine Similarity JSON** contains:
- `analysis`: Type identifier ("cosine_similarity_entropy_vs_confidence")
- `entropy_direction_source`: Method used ("entropy_mean_diff")
- `confidence_direction_source`: Method used ("confidence_ordinal_ridge")
- `layer_count`: Number of layers analyzed
- `layers`: List of layer indices
- `cosine`: List of cosine values per layer
- `abs_cosine`: List of absolute cosine values
- `max_abs_cosine`: Layer and value with maximum absolute cosine
- `max_cosine`: Layer and value with maximum cosine
- `min_cosine`: Layer and value with minimum cosine
- `statistics`: Mean, std, mean_abs, median of cosine values
- `notes`: Metadata about methods and missing layers

### PyTorch Directions

Shape: `[num_layers, hidden_dim]`

Load with:
```python
import torch
directions = torch.load("path/to/directions.pt")
layer_5_direction = directions[5]  # Get specific layer
```

Example loading and using cosine similarity:
```python
import json
import torch

# Load cosine analysis
with open("outputs/Model_Dataset_cosine_similarity_entropy_vs_confidence.json") as f:
    cosine_data = json.load(f)

# Find layer with strongest alignment (positive or negative)
max_layer = cosine_data["max_abs_cosine"]["layer"]
max_value = cosine_data["max_abs_cosine"]["value"]
print(f"Strongest alignment at layer {max_layer}: {max_value:.3f}")

# Check if entropy and confidence are aligned or opposed
mean_cosine = cosine_data["statistics"]["mean_cosine"]
if mean_cosine > 0:
    print("On average, entropy and confidence point in SAME direction")
else:
    print("On average, entropy and confidence point in OPPOSITE directions")
```

## Key Differences from Existing Code

1. **Separate from identify_mc_correlate.py**: 
   - This is a standalone script focusing only on contrast computation
   - Does not compute probe-based directions (only contrasts)

2. **Confidence task included**:
   - Extracts activations from confidence prompts
   - Uses ordinal regression (not just mean-diff)

3. **Cross-validation for alpha**:
   - Automatically selects best regularization per layer
   - More robust than fixed alpha

4. **PyTorch format**:
   - Saves directions as tensors for easy use in steering/ablation
   - Compatible with `run_steering_causality.py` and `run_ablation_causality.py`

## Validation Checks

The script performs automatic validation:

```
✓ All 32 layers valid (finite, unit norm, correct shape)
```

If validation fails, check:
1. Model loaded correctly
2. Questions contain valid data
3. Activations extracted at correct positions
4. No numerical instability (very small/large values)

## Integration with Existing Code

To use these directions in other scripts:

```python
import torch
import json

# Load directions
entropy_dirs = torch.load("outputs/Model_Dataset_entropy_contrast_directions.pt")
conf_dirs = torch.load("outputs/Model_Dataset_confidence_contrast_directions.pt")

# Load metadata
with open("outputs/Model_Dataset_entropy_contrast.json") as f:
    entropy_meta = json.load(f)

# Find best layer
best_layer = max(
    range(len(entropy_meta["per_layer"])),
    key=lambda l: entropy_meta["per_layer"][l]["r2"]
)

# Get direction for best layer
best_direction = entropy_dirs[best_layer]
```

## Troubleshooting

**ImportError**: Make sure all dependencies are installed:
```bash
pip install torch numpy scipy scikit-learn tqdm
```

**CUDA OOM**: Reduce `BATCH_SIZE` or enable quantization:
```python
BATCH_SIZE = 4
LOAD_IN_4BIT = True  # For 70B+ models
```

**Low R² values**: 
- Check if model is answering questions correctly
- Try different `CONFIDENCE_ALPHA_CANDIDATES` values
- Increase `NUM_QUESTIONS` for more stable estimates

**Different hidden dimensions**: 
- Entropy contrast uses direct MC activations (at answer token)
- Confidence contrast uses confidence task activations (at confidence token)
- These may have different shapes if using different token positions
- Current implementation assumes both use final token position
