# Configuration Management Schema

## 1. Overview

This document specifies the complete configuration schema for the MLX fine-tuning system. All configurable parameters are externalized in YAML files with validation and default values.

## 2. Configuration File Structure

### 2.1 Main Configuration File

**File**: `config/default_config.yaml`

```yaml
# MLX Fine-Tuning Configuration
# Mistral-7B-Instruct-v0.3 with LoRA

# ============================================================================
# Model Configuration
# ============================================================================
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"
  cache_dir: "~/.cache/huggingface"
  dtype: "float16"  # Use float16 for memory efficiency
  max_seq_length: 2048  # Mistral native context length
  trust_remote_code: true

# ============================================================================
# LoRA Configuration
# ============================================================================
lora:
  rank: 16  # LoRA rank (r) - controls adapter capacity
  alpha: 32  # LoRA alpha - scaling factor (typically 2*rank)
  dropout: 0.05  # Dropout probability for LoRA layers
  target_modules:  # Which modules to apply LoRA to
    - "q_proj"  # Query projection in attention
    - "v_proj"  # Value projection in attention
  bias: "none"  # Bias training: "none", "all", "lora_only"
  task_type: "CAUSAL_LM"  # Task type for model

# ============================================================================
# Data Configuration
# ============================================================================
data:
  dataset_path: "/Users/jrosenbaum/Documents/Code/Synthetic Conversations/syngen_toolset_v1.0.0_claude.jsonl"
  train_split: 0.9  # 90% training, 10% validation
  shuffle: true  # Shuffle training data
  seed: 42  # Random seed for reproducibility
  max_examples: null  # Limit dataset size (null = use all)
  validation_size: null  # Override validation size (null = use train_split)

# ============================================================================
# Training Configuration
# ============================================================================
training:
  # Batch and Accumulation
  batch_size: 2  # Batch size per step (limited by memory)
  gradient_accumulation_steps: 4  # Effective batch = 2 * 4 = 8

  # Epochs and Steps
  num_epochs: 3  # Number of training epochs
  max_steps: null  # Override max steps (null = use num_epochs)

  # Optimization
  learning_rate: 2.0e-4  # Initial learning rate
  weight_decay: 0.01  # Weight decay for AdamW
  adam_beta1: 0.9  # Adam beta1
  adam_beta2: 0.999  # Adam beta2
  adam_epsilon: 1.0e-8  # Adam epsilon
  max_grad_norm: 1.0  # Gradient clipping threshold

  # Learning Rate Schedule
  lr_scheduler_type: "cosine"  # "cosine", "linear", "constant"
  warmup_steps: 100  # Linear warmup steps
  warmup_ratio: null  # Alternative: warmup as ratio of total steps
  min_lr_ratio: 0.1  # Minimum LR as ratio of initial LR

  # Logging and Evaluation
  logging_steps: 10  # Log metrics every N steps
  eval_steps: 50  # Evaluate every N steps
  save_steps: 100  # Save checkpoint every N steps

  # Checkpoint Management
  save_total_limit: 3  # Keep last N checkpoints
  save_best_checkpoint: true  # Always save best model
  load_best_at_end: true  # Load best model after training

  # Stability
  label_smoothing: 0.0  # Label smoothing factor (0 = disabled)
  early_stopping_patience: null  # Early stopping patience (null = disabled)
  early_stopping_threshold: 1.0e-4  # Early stopping improvement threshold

# ============================================================================
# Output Configuration
# ============================================================================
output:
  output_dir: "./outputs"  # Base output directory
  checkpoint_dir: "./outputs/checkpoints"  # Checkpoint directory
  final_model_dir: "./outputs/final_model"  # Final model directory
  logs_dir: "./outputs/logs"  # Logs directory
  metrics_dir: "./outputs/metrics"  # Metrics directory
  overwrite_output_dir: false  # Overwrite existing outputs

# ============================================================================
# Logging Configuration
# ============================================================================
logging:
  level: "INFO"  # Logging level: DEBUG, INFO, WARNING, ERROR
  console: true  # Log to console
  file: true  # Log to file
  log_file: "./outputs/logs/training.log"  # Log file path
  json_logs: true  # Output structured JSON logs
  tensorboard: false  # Enable TensorBoard logging (if available)
  wandb: false  # Enable Weights & Biases logging (if available)

# ============================================================================
# System Configuration
# ============================================================================
system:
  seed: 42  # Global random seed
  num_workers: 0  # Data loading workers (MLX uses 0)
  device: "mps"  # Device: "mps" for Metal, "cpu" for CPU
  mixed_precision: true  # Use mixed precision training

# ============================================================================
# Monitoring Configuration
# ============================================================================
monitoring:
  track_memory: true  # Monitor memory usage
  memory_log_steps: 50  # Log memory every N steps
  track_gradients: true  # Track gradient statistics
  gradient_log_steps: 100  # Log gradients every N steps
  save_metrics_plot: true  # Save metrics plots
  plot_frequency: "epoch"  # Plot frequency: "step", "epoch", "end"

# ============================================================================
# Evaluation Configuration
# ============================================================================
evaluation:
  eval_on_start: false  # Evaluate before training
  eval_accumulation_steps: 10  # Accumulate eval batches
  generate_samples: true  # Generate text samples during eval
  num_samples: 5  # Number of samples to generate
  sample_max_length: 100  # Max tokens for generated samples
  sample_temperature: 0.7  # Temperature for sampling
  sample_top_p: 0.9  # Top-p for nucleus sampling

# ============================================================================
# Advanced Configuration
# ============================================================================
advanced:
  gradient_checkpointing: false  # Enable gradient checkpointing (saves memory)
  compile_model: false  # Compile model for faster execution (MLX feature)
  use_cache: true  # Use KV cache for generation
  pad_to_multiple_of: null  # Pad sequences to multiple (null = disabled)
  dataloader_pin_memory: false  # Pin memory for faster transfer

  # Debugging
  debug_mode: false  # Enable debug mode with extra checks
  profile_training: false  # Profile training performance
  detect_anomaly: false  # Detect NaN/Inf anomalies
```

## 3. Configuration Data Classes

### 3.1 Configuration Classes

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    cache_dir: str = "~/.cache/huggingface"
    dtype: str = "float16"
    max_seq_length: int = 2048
    trust_remote_code: bool = True

    def __post_init__(self):
        self.cache_dir = str(Path(self.cache_dir).expanduser())

@dataclass
class LoRAConfig:
    """LoRA configuration."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError(f"bias must be one of none/all/lora_only, got {self.bias}")

@dataclass
class DataConfig:
    """Data configuration."""
    dataset_path: str = ""
    train_split: float = 0.9
    shuffle: bool = True
    seed: int = 42
    max_examples: Optional[int] = None
    validation_size: Optional[int] = None

    def __post_init__(self):
        if not 0 < self.train_split < 1:
            raise ValueError(f"train_split must be in (0, 1), got {self.train_split}")
        if not Path(self.dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Batch and accumulation
    batch_size: int = 2
    gradient_accumulation_steps: int = 4

    # Epochs and steps
    num_epochs: int = 3
    max_steps: Optional[int] = None

    # Optimization
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    warmup_ratio: Optional[float] = None
    min_lr_ratio: float = 0.1

    # Logging and evaluation
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100

    # Checkpoint management
    save_total_limit: int = 3
    save_best_checkpoint: bool = True
    load_best_at_end: bool = True

    # Stability
    label_smoothing: float = 0.0
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 1e-4

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.lr_scheduler_type not in ["cosine", "linear", "constant"]:
            raise ValueError(f"Invalid lr_scheduler_type: {self.lr_scheduler_type}")

@dataclass
class OutputConfig:
    """Output configuration."""
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./outputs/checkpoints"
    final_model_dir: str = "./outputs/final_model"
    logs_dir: str = "./outputs/logs"
    metrics_dir: str = "./outputs/metrics"
    overwrite_output_dir: bool = False

    def __post_init__(self):
        # Create directories
        for dir_path in [
            self.output_dir,
            self.checkpoint_dir,
            self.final_model_dir,
            self.logs_dir,
            self.metrics_dir
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    console: bool = True
    file: bool = True
    log_file: str = "./outputs/logs/training.log"
    json_logs: bool = True
    tensorboard: bool = False
    wandb: bool = False

    def __post_init__(self):
        if self.level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError(f"Invalid logging level: {self.level}")
        if self.file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

@dataclass
class SystemConfig:
    """System configuration."""
    seed: int = 42
    num_workers: int = 0
    device: str = "mps"
    mixed_precision: bool = True

    def __post_init__(self):
        if self.device not in ["mps", "cpu"]:
            raise ValueError(f"Invalid device: {self.device}")

@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    track_memory: bool = True
    memory_log_steps: int = 50
    track_gradients: bool = True
    gradient_log_steps: int = 100
    save_metrics_plot: bool = True
    plot_frequency: str = "epoch"

    def __post_init__(self):
        if self.plot_frequency not in ["step", "epoch", "end"]:
            raise ValueError(f"Invalid plot_frequency: {self.plot_frequency}")

@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    eval_on_start: bool = False
    eval_accumulation_steps: int = 10
    generate_samples: bool = True
    num_samples: int = 5
    sample_max_length: int = 100
    sample_temperature: float = 0.7
    sample_top_p: float = 0.9

    def __post_init__(self):
        if not 0 < self.sample_temperature <= 2:
            raise ValueError(f"sample_temperature must be in (0, 2], got {self.sample_temperature}")
        if not 0 < self.sample_top_p <= 1:
            raise ValueError(f"sample_top_p must be in (0, 1], got {self.sample_top_p}")

@dataclass
class AdvancedConfig:
    """Advanced configuration."""
    gradient_checkpointing: bool = False
    compile_model: bool = False
    use_cache: bool = True
    pad_to_multiple_of: Optional[int] = None
    dataloader_pin_memory: bool = False
    debug_mode: bool = False
    profile_training: bool = False
    detect_anomaly: bool = False

@dataclass
class Config:
    """Main configuration container."""
    model: ModelConfig
    lora: LoRAConfig
    data: DataConfig
    training: TrainingConfig
    output: OutputConfig
    logging: LoggingConfig
    system: SystemConfig
    monitoring: MonitoringConfig
    evaluation: EvaluationConfig
    advanced: AdvancedConfig

    def validate(self) -> List[str]:
        """
        Validate configuration for consistency.

        Returns:
            List of validation warnings (errors raise exceptions)
        """
        warnings = []

        # Check batch size
        if self.training.batch_size > 4:
            warnings.append(
                f"batch_size={self.training.batch_size} may cause OOM on 24GB. "
                "Recommended: batch_size <= 2"
            )

        # Check sequence length
        if self.model.max_seq_length > 2048:
            warnings.append(
                f"max_seq_length={self.model.max_seq_length} exceeds Mistral's "
                "native context of 2048"
            )

        # Check LoRA rank
        if self.lora.rank > 64:
            warnings.append(
                f"lora_rank={self.lora.rank} is quite large. "
                "Typical values: 8-32"
            )

        # Check effective batch size
        effective_batch = (
            self.training.batch_size *
            self.training.gradient_accumulation_steps
        )
        if effective_batch < 4:
            warnings.append(
                f"Effective batch size={effective_batch} is small. "
                "May impact training stability."
            )

        # Check warmup steps
        total_steps = self._estimate_total_steps()
        if total_steps and self.training.warmup_steps > total_steps * 0.2:
            warnings.append(
                f"warmup_steps={self.training.warmup_steps} is >20% of total steps. "
                "Recommended: 5-10% of total steps"
            )

        return warnings

    def _estimate_total_steps(self) -> Optional[int]:
        """Estimate total training steps."""
        if self.training.max_steps:
            return self.training.max_steps

        # Rough estimate (needs dataset size)
        # Will be computed properly during data loading
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': self.model.__dict__,
            'lora': self.lora.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'output': self.output.__dict__,
            'logging': self.logging.__dict__,
            'system': self.system.__dict__,
            'monitoring': self.monitoring.__dict__,
            'evaluation': self.evaluation.__dict__,
            'advanced': self.advanced.__dict__,
        }

    def save(self, path: str):
        """Save configuration to YAML file."""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
```

## 4. Configuration Manager

```python
class ConfigurationManager:
    """Manages configuration loading and validation."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to YAML config file (optional)
        """
        self.config_path = config_path
        self.config: Optional[Config] = None

    def load(self, config_path: Optional[str] = None) -> Config:
        """
        Load configuration from YAML file.

        Args:
            config_path: Override config path

        Returns:
            Loaded and validated Config object
        """
        path = config_path or self.config_path

        if path is None:
            # Use default config
            self.config = self._create_default_config()
        else:
            # Load from file
            self.config = self._load_from_yaml(path)

        # Validate
        warnings = self.config.validate()
        if warnings:
            logger.warning("Configuration validation warnings:")
            for warning in warnings:
                logger.warning(f"  - {warning}")

        return self.config

    def _load_from_yaml(self, path: str) -> Config:
        """Load config from YAML file."""
        import yaml

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Parse into config objects
        config = Config(
            model=ModelConfig(**config_dict.get('model', {})),
            lora=LoRAConfig(**config_dict.get('lora', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            output=OutputConfig(**config_dict.get('output', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            system=SystemConfig(**config_dict.get('system', {})),
            monitoring=MonitoringConfig(**config_dict.get('monitoring', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            advanced=AdvancedConfig(**config_dict.get('advanced', {})),
        )

        return config

    def _create_default_config(self) -> Config:
        """Create default configuration."""
        return Config(
            model=ModelConfig(),
            lora=LoRAConfig(),
            data=DataConfig(),
            training=TrainingConfig(),
            output=OutputConfig(),
            logging=LoggingConfig(),
            system=SystemConfig(),
            monitoring=MonitoringConfig(),
            evaluation=EvaluationConfig(),
            advanced=AdvancedConfig(),
        )

    def override(self, **kwargs) -> Config:
        """
        Override configuration parameters.

        Args:
            **kwargs: Config overrides in dot notation
                Example: training.batch_size=4

        Returns:
            Updated config
        """
        if self.config is None:
            raise ValueError("Config not loaded. Call load() first.")

        for key, value in kwargs.items():
            parts = key.split('.')
            if len(parts) != 2:
                raise ValueError(f"Invalid override key: {key}. Use 'section.param' format.")

            section, param = parts
            if not hasattr(self.config, section):
                raise ValueError(f"Invalid config section: {section}")

            section_obj = getattr(self.config, section)
            if not hasattr(section_obj, param):
                raise ValueError(f"Invalid config parameter: {section}.{param}")

            setattr(section_obj, param, value)

        # Re-validate
        warnings = self.config.validate()
        if warnings:
            logger.warning("Configuration validation warnings after override:")
            for warning in warnings:
                logger.warning(f"  - {warning}")

        return self.config

    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        if self.config is None:
            raise ValueError("Config not loaded")
        return self.config.model

    def get_lora_config(self) -> LoRAConfig:
        """Get LoRA configuration."""
        if self.config is None:
            raise ValueError("Config not loaded")
        return self.config.lora

    def get_training_config(self) -> TrainingConfig:
        """Get training configuration."""
        if self.config is None:
            raise ValueError("Config not loaded")
        return self.config.training

    def get_data_config(self) -> DataConfig:
        """Get data configuration."""
        if self.config is None:
            raise ValueError("Config not loaded")
        return self.config.data
```

## 5. Configuration Inheritance and Experiments

### 5.1 Experiment Configurations

Create experiment-specific configs that inherit from default:

**File**: `config/experiments/high_rank_lora.yaml`

```yaml
# Experiment: Higher LoRA rank
lora:
  rank: 32
  alpha: 64

training:
  num_epochs: 5
  learning_rate: 1.5e-4
```

**File**: `config/experiments/larger_batch.yaml`

```yaml
# Experiment: Larger effective batch size
training:
  batch_size: 2
  gradient_accumulation_steps: 8  # Effective batch = 16
  learning_rate: 3e-4  # Scale LR with batch size
```

**File**: `config/experiments/quick_test.yaml`

```yaml
# Quick test configuration
data:
  max_examples: 100  # Use only 100 examples

training:
  num_epochs: 1
  eval_steps: 10
  save_steps: 20
  logging_steps: 5

logging:
  level: "DEBUG"

advanced:
  debug_mode: true
```

### 5.2 Loading Experiment Configs

```python
# Load with experiment override
config_manager = ConfigurationManager('config/default_config.yaml')
config = config_manager.load()

# Apply experiment overrides
experiment_config_manager = ConfigurationManager('config/experiments/high_rank_lora.yaml')
experiment_overrides = experiment_config_manager.load()

# Merge configurations
config.lora = experiment_overrides.lora
config.training.num_epochs = experiment_overrides.training.num_epochs
```

## 6. Environment Variables

Support environment variable overrides:

```python
import os

def load_config_with_env_overrides(config_path: str) -> Config:
    """Load config with environment variable overrides."""
    config_manager = ConfigurationManager(config_path)
    config = config_manager.load()

    # Environment variable overrides
    if 'BATCH_SIZE' in os.environ:
        config.training.batch_size = int(os.environ['BATCH_SIZE'])

    if 'LEARNING_RATE' in os.environ:
        config.training.learning_rate = float(os.environ['LEARNING_RATE'])

    if 'NUM_EPOCHS' in os.environ:
        config.training.num_epochs = int(os.environ['NUM_EPOCHS'])

    if 'LORA_RANK' in os.environ:
        config.lora.rank = int(os.environ['LORA_RANK'])

    if 'DATASET_PATH' in os.environ:
        config.data.dataset_path = os.environ['DATASET_PATH']

    if 'OUTPUT_DIR' in os.environ:
        config.output.output_dir = os.environ['OUTPUT_DIR']

    # Re-validate after overrides
    config.validate()

    return config
```

## 7. Command-Line Interface

```python
import argparse

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Mistral-7B with MLX and LoRA"
    )

    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='Path to config file'
    )

    # Overrides
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    parser.add_argument('--num-epochs', type=int, help='Override number of epochs')
    parser.add_argument('--lora-rank', type=int, help='Override LoRA rank')
    parser.add_argument('--dataset-path', type=str, help='Override dataset path')
    parser.add_argument('--output-dir', type=str, help='Override output directory')

    # Modes
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--eval-only', action='store_true', help='Run evaluation only')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    config_manager = ConfigurationManager(args.config)
    config = config_manager.load()

    # Apply CLI overrides
    overrides = {}
    if args.batch_size:
        overrides['training.batch_size'] = args.batch_size
    if args.learning_rate:
        overrides['training.learning_rate'] = args.learning_rate
    if args.num_epochs:
        overrides['training.num_epochs'] = args.num_epochs
    if args.lora_rank:
        overrides['lora.rank'] = args.lora_rank
    if args.dataset_path:
        overrides['data.dataset_path'] = args.dataset_path
    if args.output_dir:
        overrides['output.output_dir'] = args.output_dir
    if args.debug:
        overrides['logging.level'] = 'DEBUG'
        overrides['advanced.debug_mode'] = True

    if overrides:
        config = config_manager.override(**overrides)

    # Save effective config
    effective_config_path = os.path.join(
        config.output.output_dir, 'effective_config.yaml'
    )
    config.save(effective_config_path)

    # Run training or evaluation
    if args.eval_only:
        run_evaluation(config, args.resume)
    else:
        run_training(config, args.resume)
```

## 8. Configuration Best Practices

### 8.1 Parameter Selection Guidelines

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| `batch_size` | 1-2 | Limited by 24GB memory |
| `gradient_accumulation_steps` | 4-8 | Effective batch = 4-16 |
| `learning_rate` | 1e-4 to 5e-4 | Higher than full fine-tuning |
| `lora_rank` | 8-32 | Balance capacity vs efficiency |
| `lora_alpha` | 2 * rank | Standard scaling |
| `max_grad_norm` | 0.5-1.0 | Prevent instability |
| `warmup_steps` | 5-10% of total | Stabilize early training |
| `weight_decay` | 0.01-0.1 | Regularization |
| `num_epochs` | 3-5 | Avoid overfitting on 1000 examples |

### 8.2 Memory-Constrained Settings

For tighter memory budgets:

```yaml
model:
  max_seq_length: 1024  # Reduce from 2048

training:
  batch_size: 1  # Minimum batch size
  gradient_accumulation_steps: 8  # Maintain effective batch

lora:
  rank: 8  # Lower rank
  alpha: 16

advanced:
  gradient_checkpointing: true  # Trade compute for memory
```

### 8.3 Fast Iteration Settings

For quick experiments:

```yaml
data:
  max_examples: 100

training:
  num_epochs: 1
  eval_steps: 10
  save_steps: 50

logging:
  level: "DEBUG"
```

## 9. Configuration Validation

### 9.1 Validation Rules

```python
def validate_memory_budget(config: Config) -> bool:
    """Estimate if config will fit in memory."""
    # Rough memory estimation
    base_model_gb = 7.0  # Mistral-7B in float16
    lora_params_mb = config.lora.rank * 2 * 32 * 2  # Rough estimate
    optimizer_mb = lora_params_mb * 2  # Adam states

    batch_memory_gb = (
        config.training.batch_size *
        config.model.max_seq_length *
        4 * 4096  # Rough activation estimate
    ) / (1024 ** 3)

    total_gb = base_model_gb + (lora_params_mb + optimizer_mb) / 1024 + batch_memory_gb + 2

    if total_gb > 20:  # Leave 4GB headroom
        logger.warning(f"Estimated memory usage: {total_gb:.1f}GB (may OOM)")
        return False

    logger.info(f"Estimated memory usage: {total_gb:.1f}GB")
    return True


def validate_training_stability(config: Config) -> List[str]:
    """Check for potential stability issues."""
    issues = []

    # Learning rate too high
    if config.training.learning_rate > 5e-4:
        issues.append("Learning rate may be too high for stable training")

    # Very small effective batch
    eff_batch = config.training.batch_size * config.training.gradient_accumulation_steps
    if eff_batch < 4:
        issues.append("Effective batch size very small, may cause instability")

    # No warmup with high LR
    if config.training.warmup_steps == 0 and config.training.learning_rate > 1e-4:
        issues.append("Consider adding warmup steps for stability")

    return issues
```

---

**Next Document**: `06_ERROR_HANDLING.md` - Error management and monitoring strategies
