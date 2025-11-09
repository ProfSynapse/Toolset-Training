# MLX Fine-Tuning Architecture Documentation

## Overview

This directory contains the complete architectural design for fine-tuning Mistral-7B-Instruct-v0.3 on Mac M4 (24GB) using Apple's MLX framework with LoRA (Low-Rank Adaptation) on the Claudesidian synthetic dataset.

## Quick Start

**New to this architecture?** Start here:

1. Read `00_INDEX.md` for navigation guidance
2. Review `01_EXECUTIVE_SUMMARY.md` for high-level overview
3. Dive into specific documents based on your role

## Architecture at a Glance

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                  MLX Fine-Tuning System                      │
│                                                               │
│  Input:  syngen_toolset_v1.0.0_claude.jsonl (1000 examples) │
│  Output: Fine-tuned Mistral-7B with LoRA adapters           │
│  Time:   4-6 hours training                                  │
│  Memory: 14-16GB peak usage                                  │
└─────────────────────────────────────────────────────────────┘

Core Modules:
  1. Configuration Manager - Parameter management
  2. Data Pipeline        - JSONL → MLX batches
  3. Model Manager        - Mistral-7B + LoRA layers
  4. Training Engine      - Optimization loop
  5. Evaluation Module    - Validation and metrics
  6. Utilities            - Logging, monitoring, checkpointing
```

### Key Specifications

| Aspect | Specification |
|--------|--------------|
| **Model** | Mistral-7B-Instruct-v0.3 |
| **Framework** | MLX (Apple Silicon optimized) |
| **Method** | LoRA (rank=16, alpha=32) |
| **Hardware** | Mac M4, 24GB unified memory |
| **Dataset** | 1000 examples (746/254 split) |
| **Training Time** | 4-6 hours (3 epochs) |
| **Memory Usage** | 14-16GB peak |
| **Trainable Params** | ~8M (0.1% of 7B) |

## Document Roadmap

### For Quick Reference

| Document | Purpose | Key Info |
|----------|---------|----------|
| `00_INDEX.md` | Navigation guide | Document overview, quick links |
| `01_EXECUTIVE_SUMMARY.md` | High-level overview | Architecture philosophy, key decisions |

### For Implementation

| Document | Purpose | When to Read |
|----------|---------|-------------|
| `02_SYSTEM_ARCHITECTURE.md` | Component design | Before coding any module |
| `03_DATA_PIPELINE.md` | Data processing | Implementing data loading |
| `04_TRAINING_PIPELINE.md` | Training loop | Implementing training engine |
| `05_CONFIGURATION_SCHEMA.md` | Configuration | Setting up parameters |
| `06_ERROR_HANDLING.md` | Error management | Adding error handling |
| `07_IMPLEMENTATION_ROADMAP.md` | Development plan | Planning implementation |

## Key Design Decisions

### Why This Architecture?

1. **Modular Design**: Each component has single responsibility
   - Easy to test, debug, and extend
   - Clear interfaces between modules
   - No circular dependencies

2. **Memory Efficiency**: Optimized for 24GB constraint
   - LoRA reduces trainable params by 99.9%
   - Batch size=2 with gradient accumulation
   - float16 precision throughout

3. **Configuration-Driven**: All behavior externalized
   - Easy experimentation
   - Reproducible runs
   - No hard-coded parameters

4. **Production-Ready**: Comprehensive error handling
   - Graceful degradation
   - Checkpoint recovery
   - Detailed logging and monitoring

### Technical Highlights

**LoRA Configuration**:
```yaml
rank: 16              # Low-rank adapter dimension
alpha: 32             # Scaling factor
target_modules:       # Apply to attention layers
  - q_proj            # Query projection
  - v_proj            # Value projection
```

**Training Configuration**:
```yaml
batch_size: 2                      # Memory constraint
gradient_accumulation_steps: 4     # Effective batch = 8
learning_rate: 2e-4                # Standard for LoRA
max_grad_norm: 1.0                 # Gradient clipping
warmup_steps: 100                  # LR warmup
```

**Memory Budget**:
```
Base Model (frozen):     ~7GB
LoRA Parameters:         ~32MB
Optimizer State:         ~64MB
Activations (batch=2):   ~4-6GB
System/MLX Overhead:     ~2-3GB
─────────────────────────────────
Total Peak:              ~14-16GB
```

## Implementation Timeline

### 6-Week Development Plan

**Week 1**: Foundation + Data Pipeline
- Project setup and configuration management
- Data loading, validation, and preprocessing
- Tokenization and batching

**Week 2-3**: Model & LoRA
- Model loading from Hugging Face
- MLX conversion
- LoRA layer injection

**Week 3-4**: Training Engine
- Loss computation and gradient handling
- Optimizer and scheduler setup
- Training loop implementation
- Checkpointing

**Week 4-5**: Integration & Testing
- End-to-end integration
- Performance testing
- Error handling validation

**Week 5-6**: Polish & Documentation
- Code quality improvements
- User documentation
- Example notebooks

## Testing Strategy

### Test Pyramid

```
     E2E Tests (10%)
    ───────────────
   Integration (30%)
  ───────────────────
  Unit Tests (60%)
 ───────────────────────
```

**Coverage Goals**:
- Unit tests: 80%+ coverage
- Integration: All critical paths
- E2E: Complete training pipeline

## Success Criteria

The implementation succeeds when:

1. **Functionality**
   - [x] Fine-tunes Mistral-7B successfully
   - [x] Produces improved model on validation
   - [x] Saves/loads checkpoints reliably

2. **Performance**
   - [x] Trains in 4-6 hours on M4
   - [x] Uses < 16GB peak memory
   - [x] Achieves expected loss reduction

3. **Usability**
   - [x] Clear documentation
   - [x] Intuitive CLI
   - [x] Helpful error messages

4. **Maintainability**
   - [x] 80%+ test coverage
   - [x] Clean, documented code
   - [x] Modular architecture

5. **Robustness**
   - [x] Handles errors gracefully
   - [x] Recovers from interruptions
   - [x] Validates inputs comprehensively

## Architecture Principles

All design decisions follow these principles:

1. **Modularity**: Single responsibility per component
2. **Memory Efficiency**: Optimized for 24GB constraint
3. **Configurability**: Externalized parameters
4. **Observability**: Comprehensive logging/monitoring
5. **Maintainability**: Clear interfaces, documentation
6. **Extensibility**: Easy to add features
7. **Robustness**: Error handling and recovery
8. **Testability**: Unit and integration testing

## File Organization

```
docs/architecture/
├── README.md                          # This file
├── 00_INDEX.md                        # Navigation guide
├── 01_EXECUTIVE_SUMMARY.md            # High-level overview
├── 02_SYSTEM_ARCHITECTURE.md          # Component design
├── 03_DATA_PIPELINE.md                # Data processing
├── 04_TRAINING_PIPELINE.md            # Training loop
├── 05_CONFIGURATION_SCHEMA.md         # Configuration
├── 06_ERROR_HANDLING.md               # Error management
└── 07_IMPLEMENTATION_ROADMAP.md       # Development plan
```

## Getting Started with Implementation

### Prerequisites

1. Review architecture documents (start with `00_INDEX.md`)
2. Set up development environment (Python 3.9+, MLX)
3. Create project structure per roadmap
4. Install dependencies

### First Steps

1. **Set up project** (Task 1.1 in roadmap)
   ```bash
   mkdir -p src/{config,data,model,training,evaluation,utils}
   touch src/__init__.py
   ```

2. **Create configuration** (Task 1.2)
   - Implement configuration data classes
   - Add YAML loading
   - Validate parameters

3. **Build data pipeline** (Task 2.1-2.4)
   - JSONL parser
   - Conversation formatter
   - Tokenization
   - DataLoader

4. **Follow roadmap** (Tasks 3.1-6.3)
   - Implement in dependency order
   - Test each component
   - Integrate progressively

## Common Questions

### Q: Why LoRA instead of full fine-tuning?
**A**: LoRA reduces trainable parameters by 99.9%, enabling training on 24GB memory while maintaining quality.

### Q: Why batch_size=2?
**A**: Memory constraint. We use gradient accumulation (4 steps) for effective batch_size=8.

### Q: Why MLX instead of PyTorch?
**A**: MLX is optimized for Apple Silicon, uses unified memory efficiently, and provides better performance on M4.

### Q: Can I use a different model?
**A**: Yes, the architecture is designed to be extensible. See "Extension Points" in `02_SYSTEM_ARCHITECTURE.md`.

### Q: How do I tune hyperparameters?
**A**: See `05_CONFIGURATION_SCHEMA.md` for all parameters and recommended ranges.

## Resources

### Documentation
- All architecture docs in this directory
- Implementation roadmap with detailed tasks
- Configuration examples and templates

### External References
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Mistral Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## Version Information

- **Architecture Version**: 1.0.0
- **Date**: 2025-11-09
- **Status**: Design Phase Complete
- **Next Phase**: Implementation (Code phase)

## Contact and Support

For architecture questions:
- Review relevant architecture document
- Check implementation roadmap for guidance
- Consult error handling strategies

For implementation support:
- Follow task breakdown in roadmap
- Reference component specifications
- Use testing strategy for validation

---

**Ready to implement?** Start with `07_IMPLEMENTATION_ROADMAP.md` for the detailed development sequence.

**Need clarification?** See `00_INDEX.md` for navigation to specific topics.

**Architecture questions?** Review `01_EXECUTIVE_SUMMARY.md` for key decisions and rationale.
