# Contrast JSON Analysis: What Insights Can You Extract?

This guide explains all the valuable insights hidden in the contrast JSON files beyond just the headline R² values.

## Quick Start

Run the deep analysis script:
```bash
python analyze_contrast_insights.py
```

This will generate a comprehensive 9-panel visualization and detailed statistics.

## Key Insights Available

### 1. **Layer-wise Best Performance**
**What it tells you:** Where in the network does each type of uncertainty emerge?

**From the JSON:**
- Best entropy layer vs best confidence layer
- Do they peak at the same depth?
- Or process at different stages?

**Why it matters:**
- **Same layer**: Both uncertainties computed together (shared representation)
- **Different layers**: Processed at different depths (separate mechanisms)
- **Early layers**: Low-level features
- **Late layers**: High-level semantic understanding

**Example insight:**
> "Entropy peaks at layer 30, confidence at layer 28 → both emerge late, suggesting high-level semantic processing"

---

### 2. **Layer-wise Progression Patterns**
**What it tells you:** How does the signal evolve through the network?

**From the JSON:**
- R² trajectory across layers
- Growth rate (derivative)
- Early vs middle vs late layer performance

**Patterns to look for:**
- **Monotonic increase**: Signal builds gradually
- **Sudden jumps**: Discrete computation stages
- **Plateau**: Information saturates
- **Similar progression**: Spearman correlation between entropy/confidence evolution

**Why it matters:**
- Reveals computational stages
- Shows when uncertainty becomes "knowable" to the model
- Similar progressions suggest shared mechanisms

**Example insight:**
> "Both show fastest growth at layers 10-15, suggesting this is where uncertainty representations crystallize"

---

### 3. **Correlation Strength Patterns**
**What it tells you:** How reliably can you predict the target from activations?

**From the JSON:**
- Pearson r values (not just R²)
- Distribution across layers
- How many layers have strong signal (r > 0.5)

**Why R² vs correlation matters:**
- **R² = 0.25, r = 0.5**: Moderate linear relationship
- **R² = 0.04, r = 0.2**: Weak but still significant
- Sign tells you direction (positive/negative relationship)

**Example insight:**
> "Only 5/32 layers have r > 0.5 for confidence, suggesting weak encodings throughout"

---

### 4. **Statistical Significance**
**What it tells you:** Is the signal real or just noise?

**From the JSON:**
- P-values for each layer
- How many layers reach p < 0.001 (very strong)
- How many reach p < 0.05 (standard threshold)

**Interpretation:**
- **p < 0.001**: Very confident the signal is real
- **0.001 < p < 0.05**: Significant but less strong
- **p > 0.05**: Could be noise

**Why it matters:**
- With 500 samples, p-values help validate findings
- Multiple significant layers = robust signal
- Few significant layers = questionable results

**Example insight:**
> "Entropy: 28/32 layers with p < 0.001 (robust). Confidence: 12/32 layers (weaker but still real)"

---

### 5. **Direction Magnitude (Signal Strength)**
**What it tells you:** How far apart are high vs low groups in activation space?

**From the JSON:**
- Magnitude before normalization
- How it changes across layers
- Entropy magnitude vs confidence magnitude

**Interpretation:**
- **Large magnitude**: Groups well-separated in activation space
- **Small magnitude**: Groups overlap (weak contrast)
- **Decreasing magnitude**: Signal compresses in later layers

**Why it matters:**
- Independent of R² (geometric vs predictive quality)
- Shows raw separation strength
- Can compare across different contrasts

**Example insight:**
> "Entropy magnitude 10x larger than confidence → entropy creates stronger activation differences"

---

### 6. **Group Separation Quality**
**What it tells you:** How different are the high/low groups you're comparing?

**From the JSON:**
- Mean value in low group
- Mean value in high group
- Gap between them
- Gap as % of total range

**Interpretation:**
- **Large gap**: Clear distinction (e.g., 0.2 vs 0.9 confidence)
- **Small gap**: Subtle distinction (e.g., 0.85 vs 0.95 confidence)
- **% of range**: Context-dependent measure

**Why it matters:**
- Small gaps are hard to learn from
- Explains why R² might be low
- Shows if you're really measuring different states

**Example insight:**
> "Confidence gap only 0.20 (75% vs 95%) while entropy gap is 0.87 → entropy has clearer contrast"

---

### 7. **Cosine Similarity Patterns Across Layers**
**What it tells you:** Does alignment change through the network?

**From the JSON:**
- Cosine at each layer
- Early vs middle vs late trends
- Layers with strongest alignment (|cos| > 0.1)

**Patterns to look for:**
- **Constant near zero**: Always orthogonal (independent mechanisms)
- **Increasing**: Converge in later layers
- **Decreasing**: Diverge in later layers
- **Fluctuating**: Complex interaction

**Why it matters:**
- Shows if representations merge or separate
- Identifies specific layers where they interact
- Reveals processing hierarchy

**Example insight:**
> "Cosine increases from -0.05 to +0.02 in late layers → slight convergence but still mostly independent"

---

### 8. **Token Distribution Quality**
**What it tells you:** Is your confidence data balanced enough to learn from?

**From the JSON:**
- Count per token (S, T, U, V, W, X, Y, Z)
- Number of tokens actually used
- Entropy-based diversity measure

**Metrics:**
- **Tokens used**: How many of 8 possible levels
- **Diversity ratio**: 1.0 = perfect balance, <0.3 = severe imbalance
- **Dominant class %**: What % is most common token

**Why it matters:**
- Severe imbalance (>80% one class) → can't learn good predictor
- Low diversity → limited signal
- Explains poor performance

**Example insight:**
> "Diversity ratio 0.15 (88.8% are 'Z') → too imbalanced for reliable learning"

---

### 9. **Cross-Contrast Comparisons**

#### Progression Similarity (Spearman ρ)
**What it tells you:** Do entropy and confidence follow the same developmental arc?

- **ρ > 0.7**: Very similar evolution → likely shared mechanism
- **ρ = 0.3-0.7**: Moderate similarity → some overlap
- **ρ < 0.3**: Different paths → independent mechanisms

#### Magnitude Ratio
**What it tells you:** Which contrast creates stronger activation patterns?

- **Ratio > 2x**: One contrast much stronger
- **Ratio ~1x**: Similar strength
- **Useful for**: Understanding relative importance

#### Best Layer Alignment
**What it tells you:** Do they peak together?

- **Within 2 layers**: Processed at same stage
- **5+ layers apart**: Different processing stages

---

## Practical Use Cases

### Use Case 1: Validating Your Contrast
**Check these metrics:**
1. Statistical significance (p-values)
2. Group separation quality
3. Token distribution diversity
4. R² progression (should increase, not random)

**Red flags:**
- Most layers not significant (p > 0.05)
- Group gap < 10% of range
- Diversity ratio < 0.3
- Random R² pattern

### Use Case 2: Comparing Two Contrasts
**Compare:**
1. Best R² values
2. Number of significant layers
3. Magnitude ratios
4. Cosine similarity patterns

**Tells you:**
- Which is stronger
- Whether they're related
- If one is just noise

### Use Case 3: Understanding Model Representations
**Analyze:**
1. Layer-wise progression
2. Where each peaks
3. Growth rate patterns
4. Cosine evolution

**Reveals:**
- Computational stages
- When uncertainty is computed
- Whether mechanisms are shared

### Use Case 4: Diagnosing Problems
**If R² is low, check:**
1. **Statistical significance**: Real signal or noise?
2. **Group separation**: Is the contrast meaningful?
3. **Token distribution**: Data quality issue?
4. **Magnitude**: Are groups actually different in activation space?

---

## Running the Analysis

The script `analyze_contrast_insights.py` extracts all of these automatically:

```bash
python analyze_contrast_insights.py
```

**Outputs:**
1. Console: Detailed statistics for all 9 insight categories
2. PNG: 9-panel visualization showing all patterns
3. Saved to: `outputs/.../ANALYSIS_deep_insights.png`

**Interpreting the 9-panel plot:**
1. **R² comparison**: Which is stronger across layers
2. **Correlation strength**: Raw correlation values
3. **Cosine similarity**: Direction alignment
4. **Magnitude**: Signal strength (log scale)
5. **Significance**: -log10(p-value), higher = more significant
6. **Growth rate**: Where signals emerge
7. **Layer similarity**: Scatter plot showing progression correlation
8. **Signal vs alignment**: Relationship between strength and cosine
9. **Summary stats**: Key numbers in text box

---

## Advanced: Combining Multiple JSON Files

If you have contrasts from multiple models or conditions:

```python
# Compare base model vs fine-tuned
base_entropy = json.load(open("base_model_entropy_contrast.json"))
ft_entropy = json.load(open("finetuned_model_entropy_contrast.json"))

# Extract R² values
base_r2 = [layer["r2"] for layer in base_entropy["per_layer"]]
ft_r2 = [layer["r2"] for layer in ft_entropy["per_layer"]]

# Compare
improvement = np.array(ft_r2) - np.array(base_r2)
print(f"Fine-tuning improved R² by {improvement.mean():.4f} on average")
```

---

## Key Takeaways

**The JSON files contain much more than just headline R² values:**

1. ✅ **Layer-wise evolution** → computational stages
2. ✅ **Statistical validity** → is the signal real?
3. ✅ **Signal strength** → how robust is it?
4. ✅ **Data quality** → can you trust the results?
5. ✅ **Cross-contrast relationships** → how do they relate?
6. ✅ **Geometric properties** → activation space structure
7. ✅ **Progression patterns** → developmental trajectories
8. ✅ **Group separation** → meaningfulness of contrast
9. ✅ **Direction alignment** → shared vs independent mechanisms

**Don't just look at the best R²!** The progression patterns, significance levels, and cross-layer relationships tell a much richer story about how uncertainty is represented in the model.
