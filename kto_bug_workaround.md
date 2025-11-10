# KTO Training Bug Workaround: Balanced Subset Approach

## Problem Summary

### The Bug
TRL's KTOTrainer has a bug in the `forward` method where it assumes every batch contains **both** desirable (True) and undesirable (False) examples:

```python
# From trl/trainer/kto_trainer.py line ~800
chosen_idx = [i for i in range(completion_logps.shape[0]) if batch["label"][i] is True]
rejected_idx = [i for i in range(completion_logps.shape[0]) if batch["label"][i] is False]

chosen_logps = completion_logps[chosen_idx, ...]
rejected_logps = completion_logps[rejected_idx, ...]
```

**When a batch has only one label type:**
- If batch = `[True, True]`: `rejected_idx = []` → Empty tensor indexing → CUDA error
- If batch = `[False, False]`: `chosen_idx = []` → Empty tensor indexing → CUDA error

### Error Message
```
AcceleratorError: CUDA error: invalid configuration argument
```

### Why It Happens
With our imbalanced dataset (746 desirable : 254 undesirable), PyTorch's random sampling can create batches with all True or all False labels. The more imbalanced the data, the more likely this occurs.

### Why Weights Don't Fix It
The `desirable_weight` and `undesirable_weight` parameters are applied **during loss computation**, but the bug occurs **during the forward pass** before weights are even considered.

## Workaround: Balanced Subset

### Approach
Create a perfectly balanced subset for testing:
- Keep all 254 undesirable examples
- Randomly sample 254 desirable examples from the 746 available
- Result: 508 total examples with 1:1 ratio

### Implementation Changes

**Cell-6: Dataset Preparation**
```python
# Separate by label
desirable_examples = [ex for ex in processed_dataset if ex["label"] == True]
undesirable_examples = [ex for ex in processed_dataset if ex["label"] == False]

# Create balanced subset
import random
random.seed(42)  # Reproducibility

balanced_desirable = random.sample(desirable_examples, len(undesirable_examples))
balanced_dataset = balanced_desirable + undesirable_examples
random.shuffle(balanced_dataset)

# Result: 254 + 254 = 508 examples, 1:1 ratio
```

**Cell-8: Training Configuration**
```python
# Use equal weights for balanced data
desirable_weight = 1.0
undesirable_weight = 1.0
```

### Why This Works
1. **Balanced sampling**: With 1:1 ratio, batches are much more likely to contain mixed labels
2. **No data manipulation**: Still using natural shuffling, just with balanced input
3. **Matches TRL example**: The `trl-lib/kto-mix-14k` dataset uses the same 1:1 approach
4. **Quick test**: We can verify KTO training works before dealing with the full imbalanced dataset

## Next Steps

### If This Works
1. **Report the bug** to HuggingFace TRL repository with minimal reproduction case
2. **Consider solutions** for using full dataset:
   - Custom batch sampler ensuring mixed labels per batch
   - Patch TRL's forward method to handle empty indices gracefully
   - Use DPO/ORPO instead (paired preference methods)

### If This Still Fails
The bug is deeper than label distribution - investigate:
- Model architecture compatibility
- Tokenizer issues
- Unsloth-specific patches
- PyTorch/CUDA version conflicts

## Impact on Training Quality

### Trade-offs
**Pros:**
- ✅ Should eliminate CUDA indexing errors
- ✅ Faster iteration (508 vs 1000 examples)
- ✅ Follows same pattern as official TRL examples
- ✅ Equal representation of both classes

**Cons:**
- ❌ Loses 492 desirable examples (66% of desirable data)
- ❌ May underfit on desirable patterns
- ❌ Not using full dataset

### Mitigation
Once we confirm KTO training works with balanced data, we can:
1. Implement proper batch sampling for full dataset
2. Or train multiple models on different random subsets
3. Or switch to a different preference optimization method (DPO, ORPO)

## Technical Context

### TRL KTO Implementation
- **File**: `trl/trainer/kto_trainer.py`
- **Method**: `forward()` around line 800
- **Issue**: No validation that both label types exist in batch before indexing

### Unsloth's Role
- **File**: `unsloth/models/dpo.py`
- **Function**: `PatchKTOTrainer()` is **empty** - just returns
- **Conclusion**: Unsloth doesn't patch KTO, uses standard TRL implementation
- **Therefore**: This is a pure TRL bug, not Unsloth-specific

## References

- **Dataset comparison**: `dataset_comparison.md`
- **TRL KTO Docs**: https://huggingface.co/docs/trl/main/en/kto_trainer
- **TRL Source**: https://github.com/huggingface/trl/blob/main/trl/trainer/kto_trainer.py
- **Official Example**: `trl-lib/kto-mix-14k` (13.5k examples, perfectly balanced 1:1)

---

**Status**: Testing balanced subset approach to verify KTO training works before addressing imbalanced data handling.
