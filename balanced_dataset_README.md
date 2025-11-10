# Balanced Dataset for KTO Training

## File Information

**File**: `syngen_toolset_v1.0.0_claude_balanced_254.jsonl`
**Location**: https://huggingface.co/datasets/professorsynapse/claudesidian-synthetic-dataset
**Size**: 703 KB
**Examples**: 508 (254 desirable + 254 undesirable)
**Ratio**: 1:1 (perfectly balanced)
**Format**: ChatML with boolean labels

## Purpose

This balanced subset was created to work around a bug in TRL's KTOTrainer where batches containing only one label type (all True or all False) cause CUDA indexing errors.

## Creation Process

1. **Source**: Full `syngen_toolset_v1.0.0_claude.jsonl` dataset (1000 examples)
2. **Separation**: 746 desirable vs 254 undesirable examples
3. **Sampling**: Randomly selected 254 desirable examples (seed=42) to match 254 undesirable
4. **Shuffling**: Combined and shuffled all 508 examples
5. **Upload**: Published to HuggingFace dataset repository

## Usage in Notebook

```python
raw_datasets = load_dataset(
    "professorsynapse/claudesidian-synthetic-dataset",
    data_files="syngen_toolset_v1.0.0_claude_balanced_254.jsonl"
)
```

## Label Distribution

- **Total**: 508 examples
- **Desirable (True)**: 254 examples (50%)
- **Undesirable (False)**: 254 examples (50%)
- **Ratio**: 1.00:1

First 50 labels distribution:
- True: 26
- False: 24

This shows good mixing throughout the dataset.

## Benefits

✅ **Pre-balanced**: No runtime sampling needed
✅ **Pre-shuffled**: Labels well-mixed throughout
✅ **Hosted on HF**: Easy to load, version controlled
✅ **Reproducible**: Same seed=42 used for sampling
✅ **Avoids bug**: Reduces chance of homogeneous batches

## Configuration

When using this balanced dataset, set equal weights:

```python
KTOConfig(
    desirable_weight=1.0,
    undesirable_weight=1.0,
    # ... other params
)
```

## Trade-offs

**Pros:**
- Eliminates most CUDA indexing errors
- Faster training (fewer examples)
- Matches TRL official example pattern

**Cons:**
- Loses 492 desirable examples from full dataset
- May underfit on desirable response patterns
- Not using complete training data

## Next Steps

Once KTO training is confirmed working with balanced data:
1. Implement custom batch sampler for full dataset
2. Or report bug to TRL and wait for fix
3. Or switch to DPO/ORPO (paired preference methods)

## Related Documentation

- `kto_bug_workaround.md` - Detailed explanation of the TRL bug
- `dataset_comparison.md` - Comparison with TRL's kto-mix-14k example
- Original full dataset: `syngen_toolset_v1.0.0_claude.jsonl`
