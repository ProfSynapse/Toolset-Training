# Implementation Roadmap

## 1. Overview

This document provides a detailed, sequenced implementation plan for building the MLX fine-tuning system. It specifies the order of module development, dependencies, testing strategies, and integration points.

## 2. Development Phases

### Phase 1: Foundation (Week 1)
**Goal**: Set up project structure and core utilities

### Phase 2: Data Pipeline (Week 1-2)
**Goal**: Complete data loading and preprocessing

### Phase 3: Model & LoRA (Week 2-3)
**Goal**: Model initialization with LoRA adapters

### Phase 4: Training Engine (Week 3-4)
**Goal**: Core training loop and optimization

### Phase 5: Integration & Testing (Week 4-5)
**Goal**: End-to-end integration and validation

### Phase 6: Polish & Documentation (Week 5-6)
**Goal**: Refinement, optimization, and documentation

## 3. Detailed Implementation Sequence

### 3.1 Phase 1: Foundation (Days 1-3)

#### Task 1.1: Project Setup
**Priority**: Critical
**Estimated Time**: 2 hours

**Deliverables**:
```
project_root/
├── src/
│   ├── __init__.py
│   ├── config/
│   ├── data/
│   ├── model/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── config/
│   └── default_config.yaml
├── tests/
├── outputs/
└── requirements.txt
```

**Implementation Steps**:
1. Create directory structure
2. Initialize git repository
3. Create virtual environment
4. Set up requirements.txt with dependencies:
   ```
   mlx>=0.0.8
   transformers>=4.35.0
   pyyaml>=6.0
   tqdm>=4.66.0
   pandas>=2.0.0
   matplotlib>=3.7.0
   pytest>=7.4.0
   ```

**Testing**:
- Verify directory structure
- Test package imports
- Run `pytest --collect-only` to verify test discovery

---

#### Task 1.2: Configuration Manager
**Priority**: Critical
**Estimated Time**: 6 hours
**Dependencies**: Task 1.1

**Files to Create**:
- `src/config/__init__.py`
- `src/config/config_manager.py`
- `config/default_config.yaml`
- `tests/test_config.py`

**Implementation Steps**:
1. Create configuration data classes (ModelConfig, LoRAConfig, etc.)
2. Implement ConfigurationManager with YAML loading
3. Add validation logic for each config section
4. Implement configuration override mechanism
5. Add environment variable support

**Testing Strategy**:
```python
def test_load_default_config():
    """Test loading default configuration."""
    config_manager = ConfigurationManager()
    config = config_manager.load('config/default_config.yaml')
    assert config.model.name == "mistralai/Mistral-7B-Instruct-v0.3"
    assert config.lora.rank == 16

def test_config_validation():
    """Test configuration validation."""
    config = Config(...)
    warnings = config.validate()
    assert isinstance(warnings, list)

def test_config_override():
    """Test parameter override."""
    config_manager = ConfigurationManager()
    config = config_manager.load('config/default_config.yaml')
    config = config_manager.override(**{'training.batch_size': 4})
    assert config.training.batch_size == 4

def test_invalid_config():
    """Test invalid configuration raises error."""
    with pytest.raises(ConfigurationError):
        config = ModelConfig(dtype='invalid')
```

**Acceptance Criteria**:
- Configuration loads from YAML without errors
- All validation rules enforced
- Override mechanism works correctly
- Invalid configurations raise descriptive errors

---

#### Task 1.3: Logging Infrastructure
**Priority**: High
**Estimated Time**: 4 hours
**Dependencies**: Task 1.2

**Files to Create**:
- `src/utils/__init__.py`
- `src/utils/logger.py`
- `tests/test_logger.py`

**Implementation Steps**:
1. Implement StructuredLogger class
2. Add console, file, and JSON log handlers
3. Create progress reporter with tqdm
4. Add metrics tracking functionality

**Testing Strategy**:
```python
def test_logger_creation():
    """Test logger initialization."""
    config = LoggingConfig()
    logger = StructuredLogger('test', config)
    assert logger is not None

def test_log_levels():
    """Test different log levels."""
    logger = StructuredLogger('test', config)
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    # Verify logs written to file

def test_metrics_logging():
    """Test metrics logging."""
    logger = StructuredLogger('test', config)
    logger.metrics(100, {'loss': 0.5, 'lr': 1e-4})
    # Verify metrics in JSON log
```

**Acceptance Criteria**:
- Logs written to console and file
- JSON logs properly formatted
- Different log levels work correctly
- Metrics logging structured and parsable

---

#### Task 1.4: Memory Monitor
**Priority**: High
**Estimated Time**: 3 hours
**Dependencies**: Task 1.3

**Files to Create**:
- `src/utils/memory_monitor.py`
- `tests/test_memory_monitor.py`

**Implementation Steps**:
1. Implement MemoryMonitor class
2. Add MLX-specific memory tracking (mx.metal.get_active_memory)
3. Add system memory tracking (psutil)
4. Create memory logging and alert mechanisms

**Testing Strategy**:
```python
def test_memory_stats():
    """Test memory statistics retrieval."""
    monitor = MemoryMonitor(logger)
    stats = monitor.get_current_usage()
    assert stats.used_gb >= 0
    assert stats.available_gb >= 0
    assert 0 <= stats.percent_used <= 100

def test_memory_check():
    """Test memory availability check."""
    monitor = MemoryMonitor(logger)
    assert monitor.check_available(1.0)  # 1GB should be available
```

**Acceptance Criteria**:
- Accurate memory statistics
- Peak memory tracking works
- Memory warnings triggered appropriately

---

### 3.2 Phase 2: Data Pipeline (Days 4-7)

#### Task 2.1: JSONL Parser and Validator
**Priority**: Critical
**Estimated Time**: 6 hours
**Dependencies**: Task 1.3

**Files to Create**:
- `src/data/__init__.py`
- `src/data/data_validator.py`
- `tests/test_data_validator.py`

**Implementation Steps**:
1. Create DataValidator class
2. Implement JSONL line-by-line parsing
3. Add schema validation for each example
4. Implement content validation (roles, labels)
5. Add error tracking and reporting

**Testing Strategy**:
```python
def test_valid_example():
    """Test validation of valid example."""
    validator = DataValidator(logger)
    valid_data = {
        'conversations': [
            {'from': 'user', 'value': 'Hello'},
            {'from': 'assistant', 'value': 'Hi'},
        ],
        'label': 'desirable'
    }
    result = validator._validate_schema(valid_data, 0)
    # Should not raise

def test_invalid_example():
    """Test validation of invalid example."""
    validator = DataValidator(logger)
    invalid_data = {'conversations': [], 'label': 'invalid'}
    with pytest.raises(ValidationError):
        validator._validate_schema(invalid_data, 0)

def test_error_threshold():
    """Test error rate threshold."""
    validator = DataValidator(logger)
    # Simulate many errors
    validator.error_counts['total'] = 100
    with pytest.raises(DataError):
        validator.check_error_threshold(1000)  # 10% error rate
```

**Acceptance Criteria**:
- Valid examples pass validation
- Invalid examples rejected with clear errors
- Error threshold enforced
- Comprehensive error reporting

---

#### Task 2.2: Conversation Formatting
**Priority**: Critical
**Estimated Time**: 4 hours
**Dependencies**: Task 2.1

**Files to Create**:
- `src/data/formatters.py`
- `tests/test_formatters.py`

**Implementation Steps**:
1. Implement Mistral Instruct formatting function
2. Handle system messages
3. Handle multi-turn conversations
4. Add format verification

**Testing Strategy**:
```python
def test_simple_conversation():
    """Test formatting of simple conversation."""
    conversation = [
        {'from': 'user', 'value': 'Hello'},
        {'from': 'assistant', 'value': 'Hi'},
    ]
    formatted = format_conversation_for_mistral(conversation)
    assert '<s>[INST]' in formatted
    assert '[/INST]' in formatted
    assert '</s>' in formatted

def test_system_message():
    """Test formatting with system message."""
    conversation = [
        {'from': 'system', 'value': 'You are helpful'},
        {'from': 'user', 'value': 'Hello'},
        {'from': 'assistant', 'value': 'Hi'},
    ]
    formatted = format_conversation_for_mistral(conversation)
    assert 'You are helpful' in formatted

def test_multiturn():
    """Test multi-turn conversation formatting."""
    conversation = [
        {'from': 'user', 'value': 'First question'},
        {'from': 'assistant', 'value': 'First answer'},
        {'from': 'user', 'value': 'Second question'},
        {'from': 'assistant', 'value': 'Second answer'},
    ]
    formatted = format_conversation_for_mistral(conversation)
    # Verify proper turn structure
    assert formatted.count('<s>') == 2
    assert formatted.count('</s>') == 2
```

**Acceptance Criteria**:
- Correctly formats single-turn conversations
- Handles system messages
- Formats multi-turn conversations
- Output matches Mistral Instruct format

---

#### Task 2.3: Tokenization
**Priority**: Critical
**Estimated Time**: 5 hours
**Dependencies**: Task 2.2

**Files to Create**:
- `src/data/tokenizer.py`
- `tests/test_tokenizer.py`

**Implementation Steps**:
1. Load Mistral tokenizer from Hugging Face
2. Implement tokenization function
3. Add padding and truncation
4. Create label generation (shifted input_ids)
5. Handle special tokens correctly

**Testing Strategy**:
```python
def test_tokenizer_loading():
    """Test tokenizer loads correctly."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3"
    )
    assert tokenizer is not None

def test_tokenization():
    """Test text tokenization."""
    tokenizer = load_tokenizer()
    text = "<s>[INST] Hello [/INST] Hi</s>"
    result = tokenize_example(text, tokenizer)
    assert len(result.input_ids) == 2048  # Padded to max_length
    assert len(result.attention_mask) == 2048
    assert len(result.labels) == 2048

def test_label_creation():
    """Test label creation from input IDs."""
    input_ids = [1, 2, 3, 0, 0]
    attention_mask = [1, 1, 1, 0, 0]
    labels = create_labels(input_ids, attention_mask)
    assert labels[-2:] == [-100, -100]  # Padding masked
```

**Acceptance Criteria**:
- Tokenizer loads successfully
- Text tokenized to correct length
- Padding/truncation works correctly
- Labels created properly with masking

---

#### Task 2.4: Dataset and DataLoader
**Priority**: Critical
**Estimated Time**: 8 hours
**Dependencies**: Task 2.1, 2.2, 2.3

**Files to Create**:
- `src/data/dataset.py`
- `src/data/data_loader.py`
- `src/data/data_pipeline.py`
- `tests/test_dataset.py`
- `tests/test_data_loader.py`
- `tests/test_data_pipeline.py`

**Implementation Steps**:
1. Implement MLXDataset class
2. Implement MLXDataLoader with batching
3. Add train/validation splitting
4. Create DataPipeline orchestrator
5. Add dataset statistics computation

**Testing Strategy**:
```python
def test_dataset_creation():
    """Test dataset creation."""
    examples = create_test_examples(10)
    dataset = MLXDataset(examples, tokenizer)
    assert len(dataset) == 10

def test_dataset_indexing():
    """Test dataset indexing."""
    dataset = MLXDataset(examples, tokenizer)
    example = dataset[0]
    assert isinstance(example.input_ids, mx.array)
    assert example.input_ids.shape == (2048,)

def test_data_loader():
    """Test data loader batching."""
    dataset = MLXDataset(examples, tokenizer)
    loader = MLXDataLoader(dataset, batch_size=2, shuffle=False)

    batch = next(iter(loader))
    assert batch.input_ids.shape == (2, 2048)
    assert batch.attention_mask.shape == (2, 2048)
    assert batch.labels.shape == (2, 2048)

def test_data_pipeline_end_to_end():
    """Test complete data pipeline."""
    config = DataConfig(dataset_path='test_data.jsonl')
    pipeline = DataPipeline(config, logger)
    pipeline.initialize()

    train_loader, val_loader = pipeline.load_and_prepare()

    assert len(train_loader) > 0
    assert len(val_loader) > 0

    # Test batch
    batch = next(iter(train_loader))
    assert batch.input_ids.shape[0] == config.batch_size
```

**Acceptance Criteria**:
- Dataset loads and indexes correctly
- DataLoader produces valid batches
- Train/val split works correctly
- Full pipeline runs end-to-end
- Dataset statistics accurate

---

### 3.3 Phase 3: Model & LoRA (Days 8-12)

#### Task 3.1: Model Loading
**Priority**: Critical
**Estimated Time**: 8 hours
**Dependencies**: Task 1.2, 1.4

**Files to Create**:
- `src/model/__init__.py`
- `src/model/model_loader.py`
- `tests/test_model_loader.py`

**Implementation Steps**:
1. Implement model download from Hugging Face
2. Convert model to MLX format
3. Add model verification
4. Implement retry logic for downloads
5. Add memory monitoring during load

**Testing Strategy**:
```python
def test_model_download():
    """Test model download (may be slow)."""
    loader = ModelLoader()
    tokenizer, model = loader.download_model(
        "mistralai/Mistral-7B-Instruct-v0.3",
        cache_dir="~/.cache/test"
    )
    assert tokenizer is not None
    assert model is not None

def test_mlx_conversion():
    """Test conversion to MLX format."""
    loader = ModelLoader()
    # Load small test model
    mlx_model = loader.convert_to_mlx(test_model)
    assert mlx_model is not None

def test_model_verification():
    """Test model integrity check."""
    loader = ModelLoader()
    # Should pass for valid model
    loader.verify_model_integrity(mlx_model)
```

**Acceptance Criteria**:
- Model downloads successfully
- Conversion to MLX works
- Model verification passes
- Memory usage within limits

---

#### Task 3.2: LoRA Implementation
**Priority**: Critical
**Estimated Time**: 10 hours
**Dependencies**: Task 3.1

**Files to Create**:
- `src/model/lora.py`
- `src/model/model_manager.py`
- `tests/test_lora.py`
- `tests/test_model_manager.py`

**Implementation Steps**:
1. Implement LoRA layer class
2. Create layer injection logic
3. Implement trainable parameter extraction
4. Add LoRA adapter save/load
5. Create ModelManager orchestrator

**Testing Strategy**:
```python
def test_lora_layer():
    """Test LoRA layer creation."""
    lora = LoRALayer(in_features=4096, out_features=4096, rank=16)
    assert lora.lora_A.shape == (4096, 16)
    assert lora.lora_B.shape == (16, 4096)

def test_lora_forward():
    """Test LoRA layer forward pass."""
    lora = LoRALayer(in_features=128, out_features=128, rank=8)
    x = mx.random.normal((2, 128))
    output = lora(x)
    assert output.shape == (2, 128)

def test_lora_injection():
    """Test LoRA layer injection into model."""
    model_manager = ModelManager(model_config, lora_config)
    model = model_manager.load_base_model()
    model_with_lora = model_manager.apply_lora()

    # Verify LoRA layers added
    trainable_params = model_manager.get_trainable_params()
    assert len(trainable_params) > 0

    # Check parameter count
    stats = model_manager.count_parameters()
    assert stats.trainable_params < stats.total_params * 0.01  # <1%

def test_lora_save_load():
    """Test LoRA adapter save/load."""
    model_manager = ModelManager(model_config, lora_config)
    model = model_manager.apply_lora()

    # Save adapters
    model_manager.save_adapters('test_adapters.npz')

    # Load adapters
    model_manager.load_adapters('test_adapters.npz')

    # Verify parameters unchanged
```

**Acceptance Criteria**:
- LoRA layers created correctly
- Injection into model works
- Only LoRA parameters trainable
- Save/load functionality works
- Parameter count < 1% of total

---

### 3.4 Phase 4: Training Engine (Days 13-18)

#### Task 4.1: Loss and Gradients
**Priority**: Critical
**Estimated Time**: 6 hours
**Dependencies**: Task 3.2

**Files to Create**:
- `src/training/__init__.py`
- `src/training/loss.py`
- `src/training/gradients.py`
- `tests/test_loss.py`
- `tests/test_gradients.py`

**Implementation Steps**:
1. Implement cross-entropy loss function
2. Add label smoothing support
3. Create gradient computation function
4. Implement gradient accumulator
5. Add gradient clipping

**Testing Strategy**:
```python
def test_loss_computation():
    """Test loss function."""
    logits = mx.random.normal((2, 10, 100))
    labels = mx.random.randint(0, 100, (2, 10))
    attention_mask = mx.ones((2, 10))

    loss = compute_loss(logits, labels, attention_mask)
    assert not mx.isnan(loss)
    assert loss > 0

def test_gradient_computation():
    """Test gradient computation."""
    model = create_test_model()
    batch = create_test_batch()

    loss, grads = compute_gradients(model, batch, compute_loss)

    assert not mx.isnan(loss)
    assert len(grads) > 0
    assert all(not mx.isnan(g).any() for g in grads.values())

def test_gradient_clipping():
    """Test gradient clipping."""
    grads = {'layer1': mx.array([10.0, 20.0])}
    clipped, norm = clip_gradients(grads, max_norm=1.0)

    clipped_norm = mx.linalg.norm(clipped['layer1'])
    assert float(clipped_norm) <= 1.1  # Allow small numerical error

def test_gradient_accumulation():
    """Test gradient accumulation."""
    accumulator = GradientAccumulator(accumulation_steps=4)

    for i in range(4):
        grads = {'param': mx.array([1.0])}
        accumulator.accumulate(grads)

        if i < 3:
            assert not accumulator.should_update()
        else:
            assert accumulator.should_update()

    accumulated = accumulator.get_gradients()
    # Each gradient was 1.0, accumulated over 4 steps, divided by 4
    assert float(accumulated['param'][0]) == 1.0
```

**Acceptance Criteria**:
- Loss computes correctly
- Gradients computed without NaN/Inf
- Gradient clipping works
- Accumulation logic correct

---

#### Task 4.2: Optimizer and Scheduler
**Priority**: Critical
**Estimated Time**: 5 hours
**Dependencies**: Task 4.1

**Files to Create**:
- `src/training/optimizer.py`
- `src/training/scheduler.py`
- `tests/test_optimizer.py`
- `tests/test_scheduler.py`

**Implementation Steps**:
1. Create optimizer factory (AdamW)
2. Implement cosine warmup scheduler
3. Add scheduler state save/load
4. Create optimizer update logic

**Testing Strategy**:
```python
def test_optimizer_creation():
    """Test optimizer creation."""
    optimizer = create_optimizer(model, learning_rate=1e-4)
    assert optimizer is not None

def test_optimizer_step():
    """Test optimizer parameter update."""
    optimizer = create_optimizer(model)
    params_before = {k: v.copy() for k, v in model.parameters().items()}

    # Simulate gradient step
    grads = {k: mx.random.normal(v.shape) for k, v in params_before.items()}
    optimizer.update(model, grads)

    params_after = model.parameters()

    # Parameters should change
    for k in params_before:
        assert not mx.array_equal(params_before[k], params_after[k])

def test_scheduler_warmup():
    """Test learning rate warmup."""
    optimizer = create_optimizer(model, learning_rate=1e-3)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=10, total_steps=100)

    # During warmup, LR should increase
    lrs = []
    for i in range(10):
        lrs.append(scheduler.get_lr())
        scheduler.step()

    assert lrs[-1] >= lrs[0]  # LR increases during warmup

def test_scheduler_decay():
    """Test learning rate decay after warmup."""
    optimizer = create_optimizer(model, learning_rate=1e-3)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=10, total_steps=100)

    # Skip warmup
    for i in range(10):
        scheduler.step()

    lr_after_warmup = scheduler.get_lr()

    # Continue to end
    for i in range(90):
        scheduler.step()

    lr_at_end = scheduler.get_lr()

    # LR should decay
    assert lr_at_end < lr_after_warmup
```

**Acceptance Criteria**:
- Optimizer updates parameters correctly
- Warmup schedule works
- Cosine decay works
- State save/load works

---

#### Task 4.3: Checkpoint Manager
**Priority**: High
**Estimated Time**: 6 hours
**Dependencies**: Task 1.3

**Files to Create**:
- `src/utils/checkpoint_manager.py`
- `tests/test_checkpoint_manager.py`

**Implementation Steps**:
1. Implement atomic checkpoint saving
2. Add checkpoint verification
3. Create checkpoint loading with recovery
4. Implement cleanup logic (keep last N)
5. Add best checkpoint tracking

**Testing Strategy**:
```python
def test_checkpoint_save():
    """Test checkpoint saving."""
    manager = CheckpointManager('./test_checkpoints')
    checkpoint_data = {
        'model_state': {},
        'optimizer_state': {},
        'training_state': TrainingState(...),
    }

    manager.save(checkpoint_data, step=100)

    # Verify file exists
    assert os.path.exists('./test_checkpoints/checkpoint_step_100.pt')

def test_checkpoint_load():
    """Test checkpoint loading."""
    manager = CheckpointManager('./test_checkpoints')

    # Save then load
    checkpoint_data = {...}
    manager.save(checkpoint_data, step=100)

    loaded = manager.load('./test_checkpoints/checkpoint_step_100.pt')

    assert loaded is not None
    assert 'model_state' in loaded

def test_checkpoint_cleanup():
    """Test old checkpoint cleanup."""
    manager = CheckpointManager('./test_checkpoints', keep_last_n=3)

    # Save 5 checkpoints
    for i in range(5):
        manager.save({'step': i}, step=i * 100)

    # Should only keep last 3
    checkpoints = manager.list_checkpoints()
    assert len(checkpoints) == 3

def test_best_checkpoint():
    """Test best checkpoint tracking."""
    manager = CheckpointManager('./test_checkpoints')

    manager.save({'loss': 0.5}, step=100, is_best=True)

    best_path = manager.get_best_checkpoint()
    assert best_path is not None
    assert os.path.exists(best_path)
```

**Acceptance Criteria**:
- Checkpoints save atomically
- Verification catches corruption
- Loading works correctly
- Cleanup keeps last N checkpoints
- Best checkpoint tracked

---

#### Task 4.4: Training Engine
**Priority**: Critical
**Estimated Time**: 12 hours
**Dependencies**: Task 4.1, 4.2, 4.3, Task 2.4

**Files to Create**:
- `src/training/training_engine.py`
- `tests/test_training_engine.py`

**Implementation Steps**:
1. Implement TrainingEngine class
2. Create train_step method
3. Implement train_epoch method
4. Add evaluation logic
5. Integrate checkpointing
6. Add metrics logging
7. Implement training loop
8. Add error handling and recovery

**Testing Strategy**:
```python
def test_training_step():
    """Test single training step."""
    engine = create_test_engine()
    batch = create_test_batch()

    metrics = engine.train_step(batch)

    assert metrics.loss > 0
    assert not math.isnan(metrics.loss)
    assert metrics.learning_rate > 0

def test_training_epoch():
    """Test full epoch."""
    engine = create_test_engine()

    epoch_metrics = engine.train_epoch(epoch=0)

    assert epoch_metrics.train_loss > 0
    assert epoch_metrics.epoch_time > 0

def test_evaluation():
    """Test evaluation."""
    engine = create_test_engine()

    eval_metrics = engine.evaluate()

    assert eval_metrics.val_loss > 0
    assert eval_metrics.perplexity > 1.0

def test_checkpoint_save_load():
    """Test checkpoint save/load during training."""
    engine = create_test_engine()

    # Save checkpoint
    engine.state.global_step = 100
    engine.save_checkpoint()

    # Create new engine and load
    engine2 = create_test_engine()
    engine2.load_checkpoint('./checkpoints/checkpoint_step_100.pt')

    assert engine2.state.global_step == 100

def test_training_interruption_recovery():
    """Test recovery from interruption."""
    engine = create_test_engine()

    # Train for a few steps
    for i in range(5):
        batch = create_test_batch()
        engine.train_step(batch)

    # Save checkpoint
    engine.save_checkpoint()

    # Simulate crash and recovery
    engine2 = create_test_engine()
    latest = engine.checkpoint_manager.get_latest()
    engine2.load_checkpoint(latest)

    # Continue training
    batch = create_test_batch()
    metrics = engine2.train_step(batch)

    assert not math.isnan(metrics.loss)
```

**Acceptance Criteria**:
- Training step executes correctly
- Full epoch completes
- Evaluation runs without errors
- Checkpointing works during training
- Can resume from checkpoint
- Metrics logged properly

---

### 3.5 Phase 5: Integration & Testing (Days 19-23)

#### Task 5.1: End-to-End Integration
**Priority**: Critical
**Estimated Time**: 8 hours
**Dependencies**: All previous tasks

**Files to Create**:
- `src/main.py`
- `tests/test_integration.py`

**Implementation Steps**:
1. Create main entry point
2. Integrate all components
3. Add command-line interface
4. Implement training flow
5. Add evaluation-only mode
6. Create resume-from-checkpoint flow

**Testing Strategy**:
```python
def test_full_training_pipeline():
    """Test complete training pipeline."""
    # Create test config
    config = create_test_config()

    # Run training
    result = run_training(config)

    assert result.total_epochs > 0
    assert result.total_steps > 0
    assert os.path.exists(result.final_model_path)

def test_eval_only_mode():
    """Test evaluation-only mode."""
    config = create_test_config()

    # Train briefly
    result = run_training(config)

    # Run evaluation
    eval_metrics = run_evaluation(config, result.best_checkpoint_path)

    assert eval_metrics.val_loss > 0
    assert eval_metrics.perplexity > 1.0

def test_resume_training():
    """Test resuming from checkpoint."""
    config = create_test_config()

    # Train for 1 epoch
    config.training.num_epochs = 1
    result1 = run_training(config)

    # Resume for another epoch
    config.training.num_epochs = 2
    result2 = run_training(config, resume_from=result1.best_checkpoint_path)

    assert result2.total_epochs == 2
    assert result2.total_steps > result1.total_steps
```

**Acceptance Criteria**:
- End-to-end training completes
- Evaluation mode works
- Resume from checkpoint works
- CLI interface functional
- All components integrate smoothly

---

#### Task 5.2: Performance Testing
**Priority**: High
**Estimated Time**: 6 hours
**Dependencies**: Task 5.1

**Files to Create**:
- `tests/test_performance.py`
- `tests/benchmark.py`

**Implementation Steps**:
1. Create performance benchmarks
2. Measure training throughput
3. Profile memory usage
4. Test scalability with dataset size
5. Optimize bottlenecks

**Testing Strategy**:
```python
def test_training_throughput():
    """Measure training throughput."""
    engine = create_test_engine()

    start = time.time()
    for i in range(100):
        batch = create_test_batch()
        engine.train_step(batch)
    duration = time.time() - start

    steps_per_second = 100 / duration
    assert steps_per_second > 0.3  # At least 0.3 steps/sec

def test_memory_usage():
    """Test memory stays within budget."""
    engine = create_test_engine()
    monitor = MemoryMonitor(logger)

    # Train for several steps
    peak_memory = 0
    for i in range(50):
        batch = create_test_batch()
        engine.train_step(batch)

        stats = monitor.get_current_usage()
        peak_memory = max(peak_memory, stats.used_gb)

    # Should stay under 16GB
    assert peak_memory < 16.0

def test_data_loading_speed():
    """Test data loading performance."""
    pipeline = DataPipeline(config, logger)
    pipeline.initialize()

    start = time.time()
    train_loader, val_loader = pipeline.load_and_prepare()
    duration = time.time() - start

    # Should load 1000 examples in < 30 seconds
    assert duration < 30.0
```

**Acceptance Criteria**:
- Training throughput meets expectations
- Memory usage within budget
- No performance regressions
- Bottlenecks identified and addressed

---

#### Task 5.3: Error Handling Testing
**Priority**: High
**Estimated Time**: 5 hours
**Dependencies**: Task 5.1

**Files to Create**:
- `tests/test_error_handling.py`

**Implementation Steps**:
1. Test configuration error handling
2. Test data error handling
3. Test training error recovery
4. Test checkpoint corruption recovery
5. Test OOM handling

**Testing Strategy**:
```python
def test_invalid_config_error():
    """Test invalid configuration raises error."""
    with pytest.raises(ConfigurationError):
        config = ModelConfig(dtype='invalid')

def test_corrupted_data_handling():
    """Test handling of corrupted data."""
    # Create dataset with some corrupt entries
    pipeline = create_pipeline_with_corrupt_data()

    # Should handle gracefully
    train_loader, val_loader = pipeline.load_and_prepare()

    # Corrupt entries should be skipped
    assert len(train_loader) > 0

def test_nan_loss_handling():
    """Test handling of NaN loss."""
    engine = create_test_engine()

    # Inject NaN loss scenario
    # Should raise TrainingError after threshold
    with pytest.raises(TrainingError):
        for i in range(10):
            # Simulate NaN loss
            pass

def test_checkpoint_corruption_recovery():
    """Test recovery from corrupted checkpoint."""
    manager = CheckpointManager('./test_checkpoints')

    # Save checkpoint
    manager.save({...}, step=100)

    # Corrupt file
    corrupt_checkpoint('./test_checkpoints/checkpoint_step_100.pt')

    # Should raise CheckpointError
    with pytest.raises(CheckpointError):
        manager.load('./test_checkpoints/checkpoint_step_100.pt')
```

**Acceptance Criteria**:
- All error types handled gracefully
- Clear error messages provided
- Recovery mechanisms work
- No silent failures

---

### 3.6 Phase 6: Polish & Documentation (Days 24-28)

#### Task 6.1: Code Quality
**Priority**: Medium
**Estimated Time**: 6 hours
**Dependencies**: All previous tasks

**Implementation Steps**:
1. Add type hints throughout
2. Add docstrings to all functions
3. Run code formatter (black)
4. Run linter (flake8, pylint)
5. Add inline comments for complex logic

**Acceptance Criteria**:
- All functions have type hints
- All public APIs documented
- Code passes linting
- Consistent formatting

---

#### Task 6.2: User Documentation
**Priority**: Medium
**Estimated Time**: 8 hours
**Dependencies**: Task 6.1

**Files to Create**:
- `README.md`
- `docs/QUICKSTART.md`
- `docs/CONFIGURATION.md`
- `docs/TROUBLESHOOTING.md`
- `docs/API_REFERENCE.md`

**Content**:
1. README with quick overview
2. Quickstart guide for first run
3. Configuration guide with all parameters
4. Troubleshooting common issues
5. API reference for developers

---

#### Task 6.3: Example Notebooks
**Priority**: Low
**Estimated Time**: 4 hours
**Dependencies**: Task 6.2

**Files to Create**:
- `examples/basic_training.ipynb`
- `examples/resume_training.ipynb`
- `examples/inference.ipynb`

**Content**:
1. Basic training example
2. Resuming from checkpoint
3. Using fine-tuned model for inference

---

## 4. Testing Strategy Summary

### 4.1 Test Pyramid

```
        /\
       /  \
      / E2E \ (10% - Integration tests)
     /      \
    /  Integ \ (30% - Component integration)
   /          \
  /    Unit    \ (60% - Unit tests)
 /              \
/________________\
```

### 4.2 Test Coverage Goals

- **Unit tests**: 80%+ coverage
- **Integration tests**: All critical paths
- **End-to-end tests**: Full training pipeline

### 4.3 Continuous Testing

Run tests at each phase:
```bash
# After each task
pytest tests/test_<module>.py -v

# Before committing
pytest tests/ -v --cov=src --cov-report=html

# Full test suite
pytest tests/ -v --cov=src --cov-report=html --slow
```

## 5. Deployment Checklist

Before considering implementation complete:

- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] End-to-end test completes successfully
- [ ] Memory usage < 16GB peak
- [ ] Training completes in 4-6 hours
- [ ] All configuration options documented
- [ ] Error messages are clear and actionable
- [ ] Code formatted and linted
- [ ] Documentation complete
- [ ] Example notebooks work
- [ ] README accurate and helpful

## 6. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| MLX API changes | Medium | High | Pin MLX version, monitor releases |
| Memory overflow | Medium | High | Extensive memory testing, dynamic batch sizing |
| Training instability | Medium | Medium | Comprehensive monitoring, early detection |
| Data quality issues | Low | Medium | Robust validation, clear error messages |
| Integration complexity | Low | High | Incremental integration, thorough testing |

## 7. Success Metrics

The implementation is successful when:

1. **Functionality**:
   - Fine-tunes Mistral-7B successfully
   - Produces improved model on validation set
   - Saves and loads checkpoints reliably

2. **Performance**:
   - Trains in 4-6 hours on M4
   - Uses < 16GB peak memory
   - Achieves expected loss reduction

3. **Usability**:
   - Clear documentation
   - Intuitive CLI
   - Helpful error messages

4. **Maintainability**:
   - 80%+ test coverage
   - Clean, documented code
   - Modular architecture

5. **Robustness**:
   - Handles errors gracefully
   - Recovers from interruptions
   - Validates inputs comprehensively

---

**Implementation Complete**: All architecture documentation delivered. Ready for Code phase implementation.
