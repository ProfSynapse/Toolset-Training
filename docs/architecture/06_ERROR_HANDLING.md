# Error Handling and Monitoring

## 1. Overview

This document specifies the comprehensive error handling, monitoring, and quality assurance strategies for the MLX fine-tuning system. It covers error detection, recovery mechanisms, logging infrastructure, and observability.

## 2. Error Classification and Handling

### 2.1 Error Categories

| Category | Severity | Handling Strategy | Examples |
|----------|----------|------------------|----------|
| **Fatal Errors** | Critical | Stop execution, save state | OOM, file corruption, invalid config |
| **Recoverable Errors** | High | Retry, skip, or fallback | Network timeout, single bad data point |
| **Warnings** | Medium | Log and continue | Suboptimal config, data quality issue |
| **Info** | Low | Log only | Progress updates, metrics |

### 2.2 Error Handling Patterns

```python
from typing import Optional, Callable, TypeVar, Any
from functools import wraps
import traceback

T = TypeVar('T')

class MLXFineTuningError(Exception):
    """Base exception for MLX fine-tuning errors."""
    pass

class ConfigurationError(MLXFineTuningError):
    """Configuration validation or loading error."""
    pass

class DataError(MLXFineTuningError):
    """Data loading or validation error."""
    pass

class ModelError(MLXFineTuningError):
    """Model initialization or loading error."""
    pass

class TrainingError(MLXFineTuningError):
    """Training execution error."""
    pass

class CheckpointError(MLXFineTuningError):
    """Checkpoint save/load error."""
    pass

class MemoryError(MLXFineTuningError):
    """Memory allocation or OOM error."""
    pass


def with_error_handling(
    error_type: type = Exception,
    fallback: Optional[Callable] = None,
    log_error: bool = True,
) -> Callable:
    """
    Decorator for error handling with optional fallback.

    Args:
        error_type: Exception type to catch
        fallback: Fallback function to call on error
        log_error: Whether to log the error

    Example:
        @with_error_handling(DataError, fallback=lambda: [])
        def load_data():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except error_type as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}")
                    logger.debug(traceback.format_exc())

                if fallback:
                    logger.info(f"Using fallback for {func.__name__}")
                    return fallback()
                else:
                    raise

        return wrapper
    return decorator


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator for retrying on error with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exception types to catch

    Example:
        @retry_on_error(max_retries=3, delay=1.0)
        def download_model():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_delay = delay

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper
    return decorator
```

## 3. Component-Specific Error Handling

### 3.1 Configuration Errors

```python
class ConfigValidator:
    """Validates configuration and provides helpful error messages."""

    @staticmethod
    def validate_model_config(config: ModelConfig):
        """Validate model configuration."""
        # Check model name
        if not config.name:
            raise ConfigurationError("model.name is required")

        # Check dtype
        valid_dtypes = ["float16", "float32", "bfloat16"]
        if config.dtype not in valid_dtypes:
            raise ConfigurationError(
                f"Invalid dtype '{config.dtype}'. Must be one of {valid_dtypes}"
            )

        # Check cache directory
        cache_dir = Path(config.cache_dir).expanduser()
        if not cache_dir.parent.exists():
            raise ConfigurationError(
                f"Parent directory of cache_dir does not exist: {cache_dir.parent}"
            )

    @staticmethod
    def validate_data_config(config: DataConfig):
        """Validate data configuration."""
        # Check dataset path
        if not Path(config.dataset_path).exists():
            raise ConfigurationError(
                f"Dataset file not found: {config.dataset_path}\n"
                f"Please verify the path is correct."
            )

        # Check file is readable
        try:
            with open(config.dataset_path, 'r') as f:
                f.readline()
        except Exception as e:
            raise ConfigurationError(
                f"Cannot read dataset file: {e}"
            )

        # Check split ratio
        if not 0 < config.train_split < 1:
            raise ConfigurationError(
                f"train_split must be between 0 and 1, got {config.train_split}"
            )

    @staticmethod
    def validate_training_config(config: TrainingConfig):
        """Validate training configuration."""
        # Check batch size
        if config.batch_size < 1:
            raise ConfigurationError(
                f"batch_size must be positive, got {config.batch_size}"
            )

        if config.batch_size > 4:
            logger.warning(
                f"batch_size={config.batch_size} may cause OOM on 24GB. "
                "Recommended: batch_size <= 2"
            )

        # Check learning rate
        if config.learning_rate <= 0:
            raise ConfigurationError(
                f"learning_rate must be positive, got {config.learning_rate}"
            )

        if config.learning_rate > 1e-3:
            logger.warning(
                f"learning_rate={config.learning_rate} is quite high. "
                "Typical range: 1e-4 to 5e-4"
            )

        # Check scheduler type
        valid_schedulers = ["cosine", "linear", "constant"]
        if config.lr_scheduler_type not in valid_schedulers:
            raise ConfigurationError(
                f"Invalid lr_scheduler_type '{config.lr_scheduler_type}'. "
                f"Must be one of {valid_schedulers}"
            )
```

### 3.2 Data Loading Errors

```python
class DataValidator:
    """Validates dataset and individual examples."""

    def __init__(self, logger: Logger):
        self.logger = logger
        self.error_counts = {
            'schema': 0,
            'encoding': 0,
            'content': 0,
            'total': 0,
        }

    def validate_jsonl_line(self, line: str, line_num: int) -> Optional[Dict]:
        """
        Validate and parse a single JSONL line.

        Returns:
            Parsed dict if valid, None if invalid
        """
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            self.logger.warning(f"Line {line_num}: Invalid JSON - {e}")
            self.error_counts['schema'] += 1
            self.error_counts['total'] += 1
            return None

        # Validate schema
        try:
            self._validate_schema(data, line_num)
        except ValidationError as e:
            self.logger.warning(f"Line {line_num}: {e}")
            self.error_counts['schema'] += 1
            self.error_counts['total'] += 1
            return None

        # Validate content
        try:
            self._validate_content(data, line_num)
        except ValidationError as e:
            self.logger.warning(f"Line {line_num}: {e}")
            self.error_counts['content'] += 1
            self.error_counts['total'] += 1
            return None

        return data

    def _validate_schema(self, data: Dict, line_num: int):
        """Validate data schema."""
        # Required fields
        if 'conversations' not in data:
            raise ValidationError("Missing 'conversations' field")

        if 'label' not in data:
            raise ValidationError("Missing 'label' field")

        # Conversations must be list
        if not isinstance(data['conversations'], list):
            raise ValidationError("'conversations' must be a list")

        if not data['conversations']:
            raise ValidationError("'conversations' cannot be empty")

        # Each conversation must have 'from' and 'value'
        for i, msg in enumerate(data['conversations']):
            if not isinstance(msg, dict):
                raise ValidationError(f"Conversation {i} must be a dict")

            if 'from' not in msg or 'value' not in msg:
                raise ValidationError(
                    f"Conversation {i} missing 'from' or 'value'"
                )

    def _validate_content(self, data: Dict, line_num: int):
        """Validate data content."""
        # Validate label
        valid_labels = ['desirable', 'undesirable']
        if data['label'] not in valid_labels:
            raise ValidationError(
                f"Invalid label '{data['label']}'. Must be one of {valid_labels}"
            )

        # Validate conversation roles
        for i, msg in enumerate(data['conversations']):
            valid_roles = ['system', 'user', 'assistant']
            if msg['from'] not in valid_roles:
                raise ValidationError(
                    f"Conversation {i}: Invalid role '{msg['from']}'. "
                    f"Must be one of {valid_roles}"
                )

            # Check value is non-empty
            if not msg['value'] or not msg['value'].strip():
                raise ValidationError(f"Conversation {i}: Empty message value")

            # Check reasonable length
            if len(msg['value']) > 10000:
                logger.warning(
                    f"Line {line_num}, Conversation {i}: "
                    f"Very long message ({len(msg['value'])} chars)"
                )

    def check_error_threshold(self, total_lines: int):
        """Check if error rate exceeds threshold."""
        if total_lines == 0:
            raise DataError("No valid lines in dataset")

        error_rate = self.error_counts['total'] / total_lines

        if error_rate > 0.05:  # 5% threshold
            raise DataError(
                f"Error rate too high: {error_rate:.1%} "
                f"({self.error_counts['total']}/{total_lines} invalid)\n"
                f"Schema errors: {self.error_counts['schema']}\n"
                f"Content errors: {self.error_counts['content']}\n"
                f"Encoding errors: {self.error_counts['encoding']}"
            )

        if error_rate > 0.01:  # 1% warning threshold
            logger.warning(
                f"Data quality warning: {error_rate:.1%} invalid examples"
            )
```

### 3.3 Model Loading Errors

```python
class ModelLoader:
    """Handles model loading with error recovery."""

    @retry_on_error(max_retries=3, delay=5.0, exceptions=(ConnectionError, TimeoutError))
    def download_model(self, model_name: str, cache_dir: str):
        """
        Download model from Hugging Face with retry.

        Retries on network errors.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            logger.info(f"Downloading model: {model_name}")

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )

            return tokenizer, model

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise ModelError(f"Model download failed: {e}")

    def convert_to_mlx(self, model):
        """
        Convert model to MLX format.

        Handles conversion errors gracefully.
        """
        try:
            import mlx.core as mx
            from mlx.utils import tree_map

            logger.info("Converting model to MLX format...")

            # Convert weights
            mlx_model = tree_map(lambda x: mx.array(x.numpy()), model.state_dict())

            logger.info("Model conversion successful")
            return mlx_model

        except Exception as e:
            logger.error(f"Model conversion failed: {e}")
            raise ModelError(f"Failed to convert model to MLX: {e}")

    def verify_model_integrity(self, model):
        """Verify model loaded correctly."""
        try:
            # Check model has expected attributes
            if not hasattr(model, 'config'):
                raise ModelError("Model missing 'config' attribute")

            # Try a forward pass with dummy input
            import mlx.core as mx
            dummy_input = mx.ones((1, 10), dtype=mx.int32)

            output = model(dummy_input)

            if output is None:
                raise ModelError("Model forward pass returned None")

            logger.info("Model integrity check passed")

        except Exception as e:
            logger.error(f"Model integrity check failed: {e}")
            raise ModelError(f"Model integrity verification failed: {e}")
```

### 3.4 Training Errors

```python
class TrainingErrorHandler:
    """Handles training-time errors and anomalies."""

    def __init__(self, logger: Logger):
        self.logger = logger
        self.consecutive_nan_losses = 0
        self.nan_threshold = 3

    def check_loss_validity(self, loss: float, step: int):
        """Check if loss is valid (not NaN/Inf)."""
        if math.isnan(loss):
            self.consecutive_nan_losses += 1
            self.logger.error(f"Step {step}: NaN loss detected!")

            if self.consecutive_nan_losses >= self.nan_threshold:
                raise TrainingError(
                    f"Training unstable: {self.consecutive_nan_losses} "
                    f"consecutive NaN losses. Stopping training.\n"
                    "Possible causes:\n"
                    "  - Learning rate too high\n"
                    "  - Gradient explosion\n"
                    "  - Numerical instability in loss computation\n"
                    "Suggestions:\n"
                    "  - Reduce learning rate\n"
                    "  - Enable gradient clipping\n"
                    "  - Check for NaN in data"
                )
            return False

        if math.isinf(loss):
            self.logger.error(f"Step {step}: Infinite loss detected!")
            raise TrainingError(
                "Training produced infinite loss. "
                "This usually indicates severe numerical instability."
            )

        # Reset counter on valid loss
        self.consecutive_nan_losses = 0
        return True

    def check_gradient_validity(self, grads: Dict[str, mx.array], step: int):
        """Check gradients for NaN/Inf."""
        invalid_grads = []

        for name, grad in grads.items():
            if mx.any(mx.isnan(grad)):
                invalid_grads.append((name, 'NaN'))
            elif mx.any(mx.isinf(grad)):
                invalid_grads.append((name, 'Inf'))

        if invalid_grads:
            error_msg = f"Step {step}: Invalid gradients detected:\n"
            for name, issue in invalid_grads:
                error_msg += f"  - {name}: {issue}\n"

            self.logger.error(error_msg)
            raise TrainingError(error_msg)

    def check_memory_available(self, monitor: MemoryMonitor, required_gb: float = 2.0):
        """Check if enough memory available to continue."""
        stats = monitor.get_current_usage()

        if stats.available_gb < required_gb:
            self.logger.error(
                f"Low memory warning: {stats.available_gb:.2f}GB available, "
                f"{stats.used_gb:.2f}GB used ({stats.percent_used:.1f}%)"
            )
            raise MemoryError(
                f"Insufficient memory to continue training. "
                f"Only {stats.available_gb:.2f}GB available.\n"
                "Suggestions:\n"
                "  - Reduce batch size\n"
                "  - Reduce sequence length\n"
                "  - Enable gradient checkpointing\n"
                "  - Reduce LoRA rank"
            )
```

### 3.5 Checkpoint Errors

```python
class CheckpointErrorHandler:
    """Handles checkpoint save/load errors."""

    def __init__(self, logger: Logger):
        self.logger = logger

    def safe_save(self, checkpoint_data: Dict, path: str):
        """
        Save checkpoint with atomic write and verification.

        Ensures checkpoint is not corrupted.
        """
        # Save to temporary file first
        temp_path = path + '.tmp'

        try:
            import mlx.core as mx
            mx.savez(temp_path, **checkpoint_data)

            # Verify saved file
            self._verify_checkpoint(temp_path)

            # Atomic rename
            if os.path.exists(path):
                backup_path = path + '.backup'
                shutil.move(path, backup_path)
                shutil.move(temp_path, path)
                os.remove(backup_path)
            else:
                shutil.move(temp_path, path)

            self.logger.info(f"Checkpoint saved: {path}")

        except Exception as e:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            self.logger.error(f"Failed to save checkpoint: {e}")
            raise CheckpointError(f"Checkpoint save failed: {e}")

    def _verify_checkpoint(self, path: str):
        """Verify checkpoint can be loaded."""
        try:
            import mlx.core as mx
            checkpoint = mx.load(path)

            # Check required keys
            required_keys = ['model_state', 'optimizer_state', 'training_state']
            for key in required_keys:
                if key not in checkpoint:
                    raise CheckpointError(f"Checkpoint missing required key: {key}")

        except Exception as e:
            raise CheckpointError(f"Checkpoint verification failed: {e}")

    def safe_load(self, path: str) -> Dict:
        """
        Load checkpoint with error handling.

        Returns:
            Loaded checkpoint data
        """
        if not os.path.exists(path):
            raise CheckpointError(f"Checkpoint not found: {path}")

        try:
            import mlx.core as mx
            checkpoint = mx.load(path)

            # Verify integrity
            self._verify_checkpoint(path)

            self.logger.info(f"Checkpoint loaded: {path}")
            return checkpoint

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")

            # Try backup if exists
            backup_path = path + '.backup'
            if os.path.exists(backup_path):
                self.logger.info(f"Attempting to load backup: {backup_path}")
                try:
                    checkpoint = mx.load(backup_path)
                    self._verify_checkpoint(backup_path)
                    self.logger.info("Backup checkpoint loaded successfully")
                    return checkpoint
                except:
                    pass

            raise CheckpointError(f"Checkpoint load failed: {e}")
```

## 4. Logging Infrastructure

### 4.1 Logger Implementation

```python
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

class StructuredLogger:
    """
    Structured logger with JSON and console output.
    """

    def __init__(self, name: str, config: LoggingConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.level))

        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Console handler
        if config.console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self._create_console_formatter())
            self.logger.addHandler(console_handler)

        # File handler
        if config.file:
            Path(config.log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setFormatter(self._create_file_formatter())
            self.logger.addHandler(file_handler)

        # JSON log handler
        if config.json_logs:
            json_log_file = config.log_file.replace('.log', '_json.log')
            json_handler = logging.FileHandler(json_log_file)
            json_handler.setFormatter(self._create_json_formatter())
            self.logger.addHandler(json_handler)

    def _create_console_formatter(self) -> logging.Formatter:
        """Create human-readable console formatter."""
        return logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _create_file_formatter(self) -> logging.Formatter:
        """Create detailed file formatter."""
        return logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _create_json_formatter(self) -> logging.Formatter:
        """Create JSON formatter for structured logs."""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno,
                }

                # Add extra fields
                if hasattr(record, 'extra'):
                    log_data.update(record.extra)

                return json.dumps(log_data)

        return JSONFormatter()

    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        extra = {'extra': kwargs} if kwargs else {}
        self.logger.info(message, extra=extra)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        extra = {'extra': kwargs} if kwargs else {}
        self.logger.warning(message, extra=extra)

    def error(self, message: str, **kwargs):
        """Log error message."""
        extra = {'extra': kwargs} if kwargs else {}
        self.logger.error(message, extra=extra)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        extra = {'extra': kwargs} if kwargs else {}
        self.logger.debug(message, extra=extra)

    def metrics(self, step: int, metrics: Dict[str, float]):
        """Log metrics in structured format."""
        metrics_data = {
            'step': step,
            'metrics': metrics,
        }
        self.logger.info(f"Metrics @ step {step}", extra={'extra': metrics_data})
```

### 4.2 Progress Reporting

```python
from tqdm import tqdm
from typing import Iterator

class ProgressReporter:
    """Progress reporting for training."""

    def __init__(self, total_steps: int, logger: Logger):
        self.total_steps = total_steps
        self.logger = logger
        self.pbar: Optional[tqdm] = None

    def start_epoch(self, epoch: int, num_batches: int):
        """Start progress bar for epoch."""
        self.pbar = tqdm(
            total=num_batches,
            desc=f"Epoch {epoch + 1}",
            unit="batch",
            ncols=100,
        )

    def update(self, metrics: Dict[str, float]):
        """Update progress bar with metrics."""
        if self.pbar:
            self.pbar.set_postfix(metrics)
            self.pbar.update(1)

    def close(self):
        """Close progress bar."""
        if self.pbar:
            self.pbar.close()
            self.pbar = None

    def log_summary(self, summary: Dict[str, Any]):
        """Log epoch summary."""
        self.logger.info("="  * 60)
        for key, value in summary.items():
            if isinstance(value, float):
                self.logger.info(f"{key}: {value:.4f}")
            else:
                self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 60)
```

## 5. Monitoring and Observability

### 5.1 Metrics Tracking

```python
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

class MetricsTracker:
    """Track and visualize training metrics."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.metrics = defaultdict(list)
        self.steps = []

    def log(self, step: int, metrics: Dict[str, float]):
        """Log metrics for a step."""
        self.steps.append(step)

        for key, value in metrics.items():
            self.metrics[key].append(value)

    def get_history(self) -> pd.DataFrame:
        """Get metrics as DataFrame."""
        data = {'step': self.steps}
        data.update(self.metrics)
        return pd.DataFrame(data)

    def save_to_csv(self, filename: str = 'metrics.csv'):
        """Save metrics to CSV."""
        df = self.get_history()
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        logger.info(f"Metrics saved to: {path}")

    def plot(self, metrics: List[str], save_path: Optional[str] = None):
        """Plot specified metrics."""
        df = self.get_history()

        fig, axes = plt.subplots(
            len(metrics), 1,
            figsize=(10, 4 * len(metrics))
        )

        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            if metric in df.columns:
                ax.plot(df['step'], df[metric])
                ax.set_xlabel('Step')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} over time')
                ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to: {save_path}")
        else:
            plt.show()

    def plot_all(self):
        """Plot all tracked metrics."""
        metric_groups = {
            'Loss': [k for k in self.metrics.keys() if 'loss' in k.lower()],
            'Learning Rate': [k for k in self.metrics.keys() if 'lr' in k.lower()],
            'Gradients': [k for k in self.metrics.keys() if 'grad' in k.lower()],
            'Memory': [k for k in self.metrics.keys() if 'memory' in k.lower()],
        }

        for group_name, metrics in metric_groups.items():
            if metrics:
                save_path = self.output_dir / f'{group_name.lower().replace(" ", "_")}.png'
                self.plot(metrics, save_path=str(save_path))
```

### 5.2 System Monitoring

```python
import psutil
import mlx.core as mx

class SystemMonitor:
    """Monitor system resources."""

    def __init__(self, logger: Logger):
        self.logger = logger

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        # MLX memory
        mlx_used = mx.metal.get_active_memory() / (1024 ** 3)
        mlx_peak = mx.metal.get_peak_memory() / (1024 ** 3)

        # System memory
        vm = psutil.virtual_memory()
        system_used = vm.used / (1024 ** 3)
        system_total = vm.total / (1024 ** 3)

        return {
            'mlx_memory_gb': mlx_used,
            'mlx_peak_gb': mlx_peak,
            'system_memory_gb': system_used,
            'system_total_gb': system_total,
            'system_percent': vm.percent,
        }

    def get_cpu_stats(self) -> Dict[str, float]:
        """Get CPU statistics."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'cpu_count': psutil.cpu_count(),
        }

    def log_system_stats(self):
        """Log comprehensive system statistics."""
        mem_stats = self.get_memory_stats()
        cpu_stats = self.get_cpu_stats()

        self.logger.info("System Statistics:")
        self.logger.info(f"  MLX Memory: {mem_stats['mlx_memory_gb']:.2f}GB "
                        f"(peak: {mem_stats['mlx_peak_gb']:.2f}GB)")
        self.logger.info(f"  System Memory: {mem_stats['system_memory_gb']:.2f}GB / "
                        f"{mem_stats['system_total_gb']:.2f}GB "
                        f"({mem_stats['system_percent']:.1f}%)")
        self.logger.info(f"  CPU: {cpu_stats['cpu_percent']:.1f}%")
```

## 6. Health Checks

```python
class HealthChecker:
    """Perform health checks during training."""

    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger

    def check_all(self, training_state: TrainingState, metrics: StepMetrics) -> bool:
        """
        Run all health checks.

        Returns:
            True if all checks pass, False otherwise
        """
        checks = [
            self.check_loss_trend(training_state),
            self.check_memory_usage(metrics),
            self.check_gradient_norm(metrics),
            self.check_learning_rate(metrics),
        ]

        return all(checks)

    def check_loss_trend(self, state: TrainingState) -> bool:
        """Check if loss is trending downward."""
        if len(state.training_history) < 10:
            return True  # Not enough data

        recent_losses = [h['loss'] for h in state.training_history[-10:]]
        avg_recent = sum(recent_losses) / len(recent_losses)

        older_losses = [h['loss'] for h in state.training_history[-20:-10]]
        if not older_losses:
            return True

        avg_older = sum(older_losses) / len(older_losses)

        if avg_recent > avg_older * 1.5:
            self.logger.warning(
                f"Loss increasing: recent avg={avg_recent:.4f}, "
                f"older avg={avg_older:.4f}"
            )
            return False

        return True

    def check_memory_usage(self, metrics: StepMetrics) -> bool:
        """Check memory usage is within bounds."""
        if metrics.memory_used_gb > 20:  # 20GB threshold
            self.logger.warning(
                f"High memory usage: {metrics.memory_used_gb:.2f}GB"
            )
            return False

        return True

    def check_gradient_norm(self, metrics: StepMetrics) -> bool:
        """Check gradient norm is reasonable."""
        if metrics.gradient_norm > 10.0:
            self.logger.warning(
                f"Large gradient norm: {metrics.gradient_norm:.4f}"
            )
            return False

        if metrics.gradient_norm < 1e-6:
            self.logger.warning(
                f"Very small gradient norm: {metrics.gradient_norm:.4e}"
            )
            return False

        return True

    def check_learning_rate(self, metrics: StepMetrics) -> bool:
        """Check learning rate is in reasonable range."""
        if metrics.learning_rate > 1e-3:
            self.logger.warning(
                f"High learning rate: {metrics.learning_rate:.4e}"
            )
            return False

        if metrics.learning_rate < 1e-8:
            self.logger.warning(
                f"Very low learning rate: {metrics.learning_rate:.4e}"
            )
            return False

        return True
```

---

**Next Document**: `07_IMPLEMENTATION_ROADMAP.md` - Development sequence and testing strategy
