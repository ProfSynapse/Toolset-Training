# System Architecture: Component Design and Interactions

## 1. System Context Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                         External Dependencies                           │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ Hugging Face │  │   MLX Lib    │  │  Local FS    │                  │
│  │  Model Hub   │  │   (Apple)    │  │  (Dataset)   │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         │                 │                  │                          │
└─────────┼─────────────────┼──────────────────┼──────────────────────────┘
          │                 │                  │
          ▼                 ▼                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    MLX Fine-Tuning System                               │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    Configuration Manager                        │    │
│  │  - Load YAML/JSON config                                        │    │
│  │  - Validate parameters                                          │    │
│  │  - Provide typed config objects                                 │    │
│  └────────────────────┬───────────────────────────────────────────┘    │
│                       │                                                 │
│         ┌─────────────┼─────────────┐                                  │
│         │             │             │                                  │
│         ▼             ▼             ▼                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │   Data   │  │  Model   │  │ Training │  │Evaluation│              │
│  │ Pipeline │  │ Manager  │  │  Engine  │  │  Module  │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       │             │              │             │                     │
│       └─────────────┼──────────────┼─────────────┘                     │
│                     │              │                                   │
│                     ▼              ▼                                   │
│            ┌─────────────────────────────┐                             │
│            │   Utilities & Monitoring    │                             │
│            │  - Logger                   │                             │
│            │  - Memory Monitor           │                             │
│            │  - Checkpoint Manager       │                             │
│            │  - Metrics Tracker          │                             │
│            └─────────────────────────────┘                             │
│                                                                          │
└────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Fine-tuned    │
              │  Model Output  │
              │  + Adapters    │
              └────────────────┘
```

## 2. Module Architecture

### 2.1 Configuration Manager

**Purpose**: Centralized management of all system parameters and paths.

**Responsibilities**:
- Load configuration from YAML/JSON files
- Validate parameter ranges and dependencies
- Provide typed configuration objects to other modules
- Support configuration inheritance and overrides

**Interface**:
```python
class ConfigurationManager:
    def __init__(self, config_path: str):
        """Load and validate configuration from file."""

    def get_model_config() -> ModelConfig:
        """Returns model-specific configuration."""

    def get_training_config() -> TrainingConfig:
        """Returns training hyperparameters."""

    def get_data_config() -> DataConfig:
        """Returns data pipeline configuration."""

    def get_lora_config() -> LoRAConfig:
        """Returns LoRA-specific parameters."""

    def validate() -> ValidationResult:
        """Validates all configuration parameters."""
```

**Configuration Objects**:
```python
@dataclass
class ModelConfig:
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    cache_dir: str = "~/.cache/huggingface"
    dtype: str = "float16"  # MLX dtype
    max_seq_length: int = 2048

@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

@dataclass
class TrainingConfig:
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10

@dataclass
class DataConfig:
    dataset_path: str
    train_split: float = 0.9
    shuffle: bool = True
    seed: int = 42
    num_workers: int = 0  # MLX doesn't use multiprocessing
```

**Error Handling**:
- Raise `ConfigurationError` for invalid parameters
- Provide detailed error messages with valid ranges
- Log warnings for non-critical issues (e.g., suboptimal settings)

---

### 2.2 Data Pipeline

**Purpose**: Load, validate, and prepare training data for MLX consumption.

**Responsibilities**:
- Parse JSONL dataset files
- Validate data schema and content
- Tokenize text using Mistral tokenizer
- Create batches with proper padding/truncation
- Handle train/validation splitting
- Convert to MLX array format

**Interface**:
```python
class DataPipeline:
    def __init__(self, config: DataConfig, tokenizer):
        """Initialize with configuration and tokenizer."""

    def load_dataset() -> RawDataset:
        """Load and validate JSONL file."""

    def validate_schema() -> ValidationResult:
        """Ensure data matches expected format."""

    def prepare_train_val_split() -> Tuple[Dataset, Dataset]:
        """Split data into train/validation sets."""

    def create_data_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
        """Create iterable data loader with batching."""

class DataLoader:
    def __iter__() -> Iterator[Batch]:
        """Iterate over batches of tokenized data."""

    def __len__() -> int:
        """Number of batches in dataset."""

@dataclass
class Batch:
    input_ids: mx.array  # Shape: (batch_size, seq_length)
    attention_mask: mx.array  # Shape: (batch_size, seq_length)
    labels: mx.array  # Shape: (batch_size, seq_length)
    metadata: Dict[str, Any]  # Original labels, lengths, etc.
```

**Data Flow**:
```
JSONL File → JSON Objects → Validation → Conversation Formatting
                                              ↓
                                    Text Tokenization
                                              ↓
                                    Padding/Truncation
                                              ↓
                                    MLX Array Conversion
                                              ↓
                                    Batch Assembly
                                              ↓
                                    Shuffle (if enabled)
                                              ↓
                                    Iterator Creation
```

**Error Handling**:
- Validate each JSON line for required fields
- Skip malformed entries with logging
- Raise error if too many invalid entries (>5%)
- Check tokenization output for overflow

---

### 2.3 Model Manager

**Purpose**: Initialize Mistral model with MLX and apply LoRA layers.

**Responsibilities**:
- Load pre-trained Mistral-7B from Hugging Face
- Convert weights to MLX format
- Apply LoRA adapters to target modules
- Manage model dtype and device placement
- Provide inference interface

**Interface**:
```python
class ModelManager:
    def __init__(self, model_config: ModelConfig, lora_config: LoRAConfig):
        """Initialize with model and LoRA configurations."""

    def load_base_model() -> mx.nn.Module:
        """Load Mistral-7B and convert to MLX."""

    def apply_lora() -> mx.nn.Module:
        """Add LoRA adapters to specified layers."""

    def get_trainable_params() -> Dict[str, mx.array]:
        """Return only LoRA parameters for optimization."""

    def get_model() -> mx.nn.Module:
        """Return the complete model with LoRA."""

    def save_adapters(path: str):
        """Save only LoRA weights (not base model)."""

    def load_adapters(path: str):
        """Load LoRA weights into model."""

    def count_parameters() -> ParameterStats:
        """Return total, trainable, and frozen parameter counts."""

@dataclass
class ParameterStats:
    total_params: int
    trainable_params: int
    frozen_params: int
    trainable_percent: float
```

**LoRA Layer Application**:
```
Mistral-7B Base Model
    ├── Layer 0
    │   ├── attention
    │   │   ├── q_proj ──→ LoRA(q_proj, rank=16, alpha=32)
    │   │   ├── k_proj (frozen)
    │   │   ├── v_proj ──→ LoRA(v_proj, rank=16, alpha=32)
    │   │   └── o_proj (frozen)
    │   └── mlp (frozen)
    ├── Layer 1-31 (same pattern)
    └── lm_head (frozen)
```

**Memory Management**:
- Load model in float16 to reduce memory footprint
- Use lazy loading if available in MLX
- Clear intermediate tensors after initialization

**Error Handling**:
- Verify model download integrity
- Check MLX compatibility
- Validate LoRA layer injection
- Monitor memory during initialization

---

### 2.4 Training Engine

**Purpose**: Execute the training loop with checkpointing and optimization.

**Responsibilities**:
- Manage training state (epoch, step, best loss)
- Execute forward/backward passes
- Apply optimizer updates
- Implement gradient accumulation
- Handle checkpointing and recovery
- Track metrics and loss

**Interface**:
```python
class TrainingEngine:
    def __init__(
        self,
        model: mx.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        optimizer: Optimizer,
        scheduler: LRScheduler
    ):
        """Initialize training components."""

    def train() -> TrainingResult:
        """Execute full training loop."""

    def train_epoch(epoch: int) -> EpochMetrics:
        """Train for one epoch."""

    def train_step(batch: Batch) -> StepMetrics:
        """Execute single training step."""

    def evaluate() -> EvalMetrics:
        """Run validation evaluation."""

    def save_checkpoint(path: str, is_best: bool = False):
        """Save training state and model."""

    def load_checkpoint(path: str) -> TrainingState:
        """Resume from checkpoint."""

@dataclass
class TrainingState:
    epoch: int
    global_step: int
    best_val_loss: float
    optimizer_state: Dict
    scheduler_state: Dict
    rng_state: Any
```

**Training Loop Pseudocode**:
```python
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(train_loader):
        # Forward pass
        logits = model(batch.input_ids, batch.attention_mask)
        loss = compute_loss(logits, batch.labels)

        # Backward pass
        gradients = mx.grad(loss_fn)(model.trainable_params())

        # Gradient accumulation
        accumulated_grads.update(gradients)

        if (step + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            clip_gradients(accumulated_grads, max_norm)

            # Optimizer step
            optimizer.update(model, accumulated_grads)
            scheduler.step()
            accumulated_grads.clear()

            # Logging
            if global_step % logging_steps == 0:
                log_metrics(loss, lr, step_time)

            # Evaluation
            if global_step % eval_steps == 0:
                eval_metrics = evaluate()
                log_metrics(eval_metrics)

            # Checkpointing
            if global_step % save_steps == 0:
                save_checkpoint(global_step)

            global_step += 1
```

**Gradient Accumulation Strategy**:
- Accumulate gradients over 4 steps (effective batch size = 8)
- Reduces memory pressure while maintaining larger effective batch
- Update only after accumulation window complete

**Error Handling**:
- Check for NaN/Inf in loss or gradients
- Implement gradient clipping for stability
- Validate checkpoint writes before deleting old checkpoints
- Graceful handling of interruptions (SIGINT)

---

### 2.5 Evaluation Module

**Purpose**: Perform inference and compute metrics on validation data.

**Responsibilities**:
- Run model in evaluation mode
- Compute loss on validation set
- Generate text samples for qualitative assessment
- Track metrics over training
- Support custom evaluation metrics

**Interface**:
```python
class EvaluationModule:
    def __init__(self, model: mx.nn.Module, config: TrainingConfig):
        """Initialize with model and config."""

    def evaluate(val_loader: DataLoader) -> EvalMetrics:
        """Compute metrics on validation set."""

    def generate_samples(prompts: List[str], max_tokens: int = 100) -> List[str]:
        """Generate text for qualitative evaluation."""

    def compute_perplexity(val_loader: DataLoader) -> float:
        """Calculate perplexity on validation data."""

@dataclass
class EvalMetrics:
    val_loss: float
    perplexity: float
    num_samples: int
    avg_sequence_length: float
    samples: List[GeneratedSample]

@dataclass
class GeneratedSample:
    prompt: str
    generated_text: str
    original_label: str
```

**Evaluation Flow**:
```
Validation Data → Model (eval mode) → Logits → Loss Computation
                                          ↓
                                    Perplexity Calculation
                                          ↓
Sample Prompts → Model.generate() → Generated Text → Qualitative Review
```

**Error Handling**:
- Handle generation timeouts
- Validate output token sequences
- Skip corrupted validation samples

---

### 2.6 Utilities & Monitoring

**Purpose**: Provide cross-cutting concerns like logging, monitoring, and I/O.

**Responsibilities**:
- Structured logging with levels
- Memory usage tracking
- Checkpoint management (save/load/cleanup)
- Metrics tracking and visualization
- Progress reporting

**Interface**:
```python
class Logger:
    def info(message: str, **kwargs):
        """Log informational message."""
    def warning(message: str, **kwargs):
        """Log warning."""
    def error(message: str, **kwargs):
        """Log error."""
    def metrics(step: int, metrics: Dict[str, float]):
        """Log training metrics."""

class MemoryMonitor:
    def get_current_usage() -> MemoryStats:
        """Get current memory consumption."""
    def log_memory(context: str):
        """Log memory at specific point."""
    def check_available(required_gb: float) -> bool:
        """Check if enough memory available."""

@dataclass
class MemoryStats:
    used_gb: float
    available_gb: float
    peak_gb: float
    percent_used: float

class CheckpointManager:
    def save(state: TrainingState, path: str, keep_last_n: int = 3):
        """Save checkpoint and clean up old ones."""
    def load(path: str) -> TrainingState:
        """Load checkpoint from disk."""
    def list_checkpoints() -> List[str]:
        """List available checkpoints."""
    def get_latest() -> Optional[str]:
        """Get path to most recent checkpoint."""

class MetricsTracker:
    def log(step: int, metrics: Dict[str, float]):
        """Record metrics for a step."""
    def get_history() -> pd.DataFrame:
        """Get all recorded metrics."""
    def plot(metric_name: str, save_path: str):
        """Plot metric over time."""
    def to_tensorboard(log_dir: str):
        """Export to TensorBoard format (if available)."""
```

**Logging Strategy**:
- Structured JSON logs for machine readability
- Human-readable console output
- Separate log files for errors and metrics
- Rotation policy for large log files

**Checkpoint Strategy**:
- Save every N steps (configurable)
- Keep last 3 checkpoints + best checkpoint
- Atomic writes to prevent corruption
- Include full training state for resumption

---

## 3. Component Interaction Sequences

### 3.1 Initialization Sequence

```
main()
  │
  ├─→ ConfigurationManager.load("config.yaml")
  │     └─→ Validate all parameters
  │
  ├─→ Logger.initialize(config.logging)
  │
  ├─→ DataPipeline.load_dataset(config.data)
  │     ├─→ Parse JSONL
  │     ├─→ Validate schema
  │     ├─→ Create train/val split
  │     └─→ Initialize tokenizer
  │
  ├─→ ModelManager.load_base_model(config.model)
  │     ├─→ Download from Hugging Face
  │     ├─→ Convert to MLX format
  │     └─→ MemoryMonitor.log_memory("after_model_load")
  │
  ├─→ ModelManager.apply_lora(config.lora)
  │     ├─→ Inject LoRA layers
  │     └─→ Freeze base parameters
  │
  ├─→ Initialize Optimizer (AdamW)
  │
  ├─→ Initialize LR Scheduler (cosine with warmup)
  │
  └─→ TrainingEngine.initialize(model, data, optimizer, scheduler)
```

### 3.2 Training Step Sequence

```
TrainingEngine.train_step(batch)
  │
  ├─→ MemoryMonitor.check_available(threshold)
  │
  ├─→ model.forward(batch.input_ids, batch.attention_mask)
  │     └─→ Returns logits
  │
  ├─→ compute_loss(logits, batch.labels)
  │     ├─→ Cross-entropy loss
  │     └─→ Returns scalar loss
  │
  ├─→ mx.grad(loss_fn)(model.trainable_params())
  │     └─→ Returns gradients dict
  │
  ├─→ Accumulate gradients
  │
  ├─→ If accumulation_step_reached:
  │     ├─→ clip_gradients(grads, max_norm)
  │     ├─→ optimizer.update(model, grads)
  │     ├─→ scheduler.step()
  │     └─→ clear accumulated gradients
  │
  ├─→ MetricsTracker.log(step, {loss, lr, ...})
  │
  └─→ Return StepMetrics
```

### 3.3 Checkpoint Save Sequence

```
CheckpointManager.save(training_state)
  │
  ├─→ Create checkpoint dict:
  │     ├─→ model.state_dict() (LoRA params only)
  │     ├─→ optimizer.state_dict()
  │     ├─→ scheduler.state_dict()
  │     ├─→ training_state (epoch, step, best_loss)
  │     └─→ config snapshot
  │
  ├─→ Write to temporary file
  │     └─→ mx.save(temp_path, checkpoint)
  │
  ├─→ Verify file integrity
  │     └─→ Try loading checkpoint
  │
  ├─→ Atomic rename temp → final path
  │
  ├─→ Clean up old checkpoints
  │     └─→ Keep last N + best checkpoint
  │
  └─→ Logger.info("Checkpoint saved", path=checkpoint_path)
```

### 3.4 Evaluation Sequence

```
EvaluationModule.evaluate(val_loader)
  │
  ├─→ model.eval()  # Set to evaluation mode
  │
  ├─→ total_loss = 0.0
  │
  ├─→ For each batch in val_loader:
  │     ├─→ logits = model(batch.input_ids, batch.attention_mask)
  │     ├─→ loss = compute_loss(logits, batch.labels)
  │     └─→ total_loss += loss
  │
  ├─→ avg_loss = total_loss / len(val_loader)
  │
  ├─→ perplexity = exp(avg_loss)
  │
  ├─→ generate_samples(sample_prompts, max_tokens=100)
  │     └─→ Returns qualitative samples
  │
  ├─→ model.train()  # Restore training mode
  │
  └─→ Return EvalMetrics(val_loss, perplexity, samples)
```

## 4. Module Dependencies

```
Configuration Manager (No dependencies)
         │
         ├─────────────────┬─────────────────┬─────────────────┐
         ▼                 ▼                 ▼                 ▼
   Data Pipeline    Model Manager    Training Engine   Evaluation Module
         │                 │                 │                 │
         └─────────────────┴─────────────────┴─────────────────┘
                           │
                           ▼
                  Utilities & Monitoring
```

**Dependency Rules**:
1. Configuration Manager has zero dependencies (pure data)
2. All modules depend on Configuration Manager
3. Training Engine depends on Data Pipeline and Model Manager
4. Evaluation Module depends on Model Manager
5. All modules can use Utilities & Monitoring
6. No circular dependencies allowed

## 5. File System Organization

```
project_root/
├── config/
│   ├── default_config.yaml          # Default hyperparameters
│   └── experiment_configs/          # Experiment-specific overrides
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config_manager.py        # Configuration Manager
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_pipeline.py         # Data Pipeline
│   │   └── data_loader.py           # DataLoader implementation
│   ├── model/
│   │   ├── __init__.py
│   │   ├── model_manager.py         # Model Manager
│   │   └── lora.py                  # LoRA layer implementation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── training_engine.py       # Training Engine
│   │   ├── optimizer.py             # Optimizer configuration
│   │   └── scheduler.py             # Learning rate scheduling
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluation_module.py     # Evaluation Module
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py                # Logging utilities
│   │   ├── memory_monitor.py        # Memory monitoring
│   │   ├── checkpoint_manager.py    # Checkpoint I/O
│   │   └── metrics_tracker.py       # Metrics tracking
│   └── main.py                      # Entry point
├── outputs/
│   ├── checkpoints/                 # Training checkpoints
│   ├── final_model/                 # Final LoRA adapters
│   ├── logs/                        # Log files
│   └── metrics/                     # Metrics plots and data
├── tests/
│   ├── test_config.py
│   ├── test_data_pipeline.py
│   ├── test_model_manager.py
│   └── test_training_engine.py
└── docs/
    └── architecture/                # This documentation
```

## 6. Design Patterns Applied

### 6.1 Builder Pattern (Configuration)
- Complex configuration objects built step-by-step
- Validation at each stage
- Immutable once built

### 6.2 Factory Pattern (Model/Optimizer Creation)
- ModelManager acts as factory for model+LoRA
- Optimizer factory based on config
- Encapsulates creation complexity

### 6.3 Strategy Pattern (Data Loading)
- Different batching strategies possible
- Pluggable tokenization approaches
- Flexible data augmentation

### 6.4 Observer Pattern (Metrics Tracking)
- Training engine notifies metrics tracker
- Logger observes training events
- Decoupled monitoring

### 6.5 State Pattern (Training State)
- Training state encapsulated
- Easy checkpointing and recovery
- State transitions well-defined

## 7. Concurrency and Parallelism

**MLX Considerations**:
- MLX handles GPU parallelism internally
- No explicit threading needed for model execution
- Data loading is single-threaded (num_workers=0)
- Unified memory architecture simplifies synchronization

**Async Operations**:
- Checkpoint saves can be async to avoid blocking
- Logging can be buffered and flushed periodically
- Memory monitoring runs in background

## 8. Extension Points

The architecture supports these future extensions:

1. **Multiple Dataset Support**: Add dataset registry in DataPipeline
2. **Different Base Models**: Generalize ModelManager for other architectures
3. **Custom LoRA Configurations**: Support QLoRA, AdaLoRA variants
4. **Advanced Schedulers**: Pluggable LR scheduling strategies
5. **Distributed Training**: Add multi-GPU support when MLX supports it
6. **Experiment Tracking**: Integration with Weights & Biases, MLflow
7. **Hyperparameter Tuning**: Optuna integration for automated tuning
8. **Model Merging**: Merge LoRA back into base model for deployment

## 9. Performance Optimizations

1. **Memory**:
   - LoRA reduces trainable params by 99.9%
   - Gradient checkpointing for activations
   - Batch size tuned to memory limit
   - Clear intermediate tensors aggressively

2. **Compute**:
   - MLX optimized for Metal GPU
   - Mixed precision (float16) by default
   - Efficient attention implementations
   - Lazy evaluation where possible

3. **I/O**:
   - Minimize checkpoint frequency
   - Efficient JSONL streaming
   - Async checkpoint writes
   - Compressed checkpoint storage

## 10. Quality Attributes

| Attribute | Implementation Strategy |
|-----------|------------------------|
| **Reliability** | Comprehensive error handling, checkpoint recovery, validation |
| **Maintainability** | Clear module boundaries, documentation, type hints |
| **Testability** | Dependency injection, mock-friendly interfaces, unit tests |
| **Performance** | Memory optimization, MLX native ops, batch efficiency |
| **Usability** | Clear logging, progress reporting, intuitive config |
| **Extensibility** | Plugin points, abstract interfaces, configuration-driven |
| **Security** | Input validation, safe file operations, no arbitrary code exec |

---

**Next Document**: `03_DATA_PIPELINE.md` - Detailed data structures and preprocessing
