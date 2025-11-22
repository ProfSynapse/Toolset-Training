# error recovery Dataset

**Behavior:** error_recovery
**Rubric:** `../../behavior_rubrics/error_recovery.yaml`

## Quick Reference

See the main rubric file for:
- Detailed positive/negative indicators
- Tool patterns and sequences
- Trigger scenarios
- Example pairs
- Validation criteria

## Files

- `seed_pairs_v1.0.jsonl` - Manual seed examples (10-20 pairs)
- `pairs_v1.0.jsonl` - Full dataset (generated + seed)
- `interleaved_v1.0.jsonl` - Ready for training
- `validation_report_v1.0.md` - Validation results

## Validation

```bash
# Schema validation
python tools/validate_syngen.py \
  Datasets/behavior_datasets/error_recovery/pairs_v1.0.jsonl
```

## Status

- [ ] Seed pairs created
- [ ] Full generation complete
- [ ] Schema validation passed
- [ ] Manual review complete
- [ ] Interleaved for training
- [ ] Ready for KTO

## Notes

_Add generation notes and observations here_
