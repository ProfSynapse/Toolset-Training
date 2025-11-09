# Training Pipeline Architecture

## 1. Overview

The Training Pipeline orchestrates the complete training process, including forward/backward passes, optimization, checkpointing, and evaluation. It is designed for memory efficiency on Mac M4 (24GB) and integrates MLX-specific optimizations.

## 2. Training Loop Architecture

### 2.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Loop Architecture                   │
└─────────────────────────────────────────────────────────────────┘

Initialization
    ├─→ Load model with LoRA
    ├─→ Initialize optimizer (AdamW)
    ├─→ Initialize scheduler (Cosine with warmup)
    ├─→ Initialize training state
    └─→ Load checkpoint (if resuming)

Training Loop (epochs)
    │
    For each epoch:
    │
    ├─→ Set model to train mode
    │
    ├─→ For each batch in train_loader:
    │   │
    │   ├─→ Forward Pass
    │   │   ├─→ model(input_ids, attention_mask)
    │   │   └─→ compute_loss(logits, labels)
    │   │
    │   ├─→ Backward Pass
    │   │   ├─→ compute_gradients(loss)
    │   │   └─→ accumulate_gradients()
    │   │
    │   ├─→ If accumulation_step_reached:
    │   │   ├─→ clip_gradients(max_norm)
    │   │   ├─→ optimizer.step()
    │   │   ├─→ scheduler.step()
    │   │   ├─→ zero_gradients()
    │   │   └─→ global_step += 1
    │   │
    │   ├─→ Log metrics (every logging_steps)
    │   │
    │   ├─→ Evaluate (every eval_steps)
    │   │
    │   └─→ Save checkpoint (every save_steps)
    │
    ├─→ End of epoch evaluation
    │
    └─→ Save epoch checkpoint

Finalization
    ├─→ Save final model
    ├─→ Generate final metrics report
    └─→ Export LoRA adapters
```

### 2.2 Training State Management

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
import mlx.core as mx

@dataclass
class TrainingState:
    """Complete training state for checkpointing."""
    epoch: int
    global_step: int
    best_val_loss: float
    samples_seen: int
    optimizer_state: Dict[str, Any]
    scheduler_state: Dict[str, Any]
    rng_state: Any  # MLX random state
    training_history: List[Dict[str, float]]

@dataclass
class StepMetrics:
    """Metrics for a single training step."""
    loss: float
    learning_rate: float
    gradient_norm: float
    step_time: float
    memory_used_gb: float

@dataclass
class EpochMetrics:
    """Metrics for a complete epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    val_perplexity: float
    epoch_time: float
    samples_per_second: float

@dataclass
class TrainingResult:
    """Final training results."""
    total_epochs: int
    total_steps: int
    best_val_loss: float
    best_checkpoint_path: str
    final_model_path: str
    training_history: List[Dict[str, float]]
    total_time: float
```

## 3. Core Training Components

### 3.1 Loss Computation

**Loss Function**: Cross-Entropy Loss with label smoothing (optional)

```python
def compute_loss(
    logits: mx.array,
    labels: mx.array,
    attention_mask: mx.array,
    label_smoothing: float = 0.0,
    ignore_index: int = -100
) -> mx.array:
    """
    Compute cross-entropy loss for language modeling.

    Args:
        logits: Model predictions, shape (batch_size, seq_length, vocab_size)
        labels: Target token IDs, shape (batch_size, seq_length)
        attention_mask: Mask for valid positions, shape (batch_size, seq_length)
        label_smoothing: Optional label smoothing factor
        ignore_index: Index to ignore in loss (padding)

    Returns:
        Scalar loss value
    """
    # Reshape for loss computation
    batch_size, seq_length, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)

    # Create mask for valid positions (not padding)
    mask = (labels_flat != ignore_index).astype(mx.float32)

    # Compute cross-entropy
    if label_smoothing > 0:
        # Label smoothing: reduce confidence on true label
        loss = cross_entropy_with_smoothing(
            logits_flat, labels_flat, label_smoothing, mask
        )
    else:
        # Standard cross-entropy
        loss = mx.nn.losses.cross_entropy(
            logits_flat, labels_flat, reduction='none'
        )

    # Apply mask and average
    masked_loss = loss * mask
    total_loss = mx.sum(masked_loss)
    num_tokens = mx.sum(mask)

    return total_loss / num_tokens


def cross_entropy_with_smoothing(
    logits: mx.array,
    labels: mx.array,
    smoothing: float,
    mask: mx.array
) -> mx.array:
    """Cross-entropy with label smoothing."""
    vocab_size = logits.shape[-1]
    confidence = 1.0 - smoothing
    smoothing_value = smoothing / (vocab_size - 1)

    # Create smoothed label distribution
    log_probs = mx.log_softmax(logits, axis=-1)

    # True label component
    true_label_loss = -mx.take_along_axis(
        log_probs, labels[:, None], axis=1
    ).squeeze(1)

    # Smoothing component
    smooth_loss = -mx.mean(log_probs, axis=-1)

    return confidence * true_label_loss + smoothing * smooth_loss
```

### 3.2 Gradient Computation and Accumulation

**Gradient Accumulation**: Accumulate over 4 steps for effective batch size of 8

```python
class GradientAccumulator:
    """Manages gradient accumulation across multiple steps."""

    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.accumulated_grads = None
        self.current_step = 0

    def accumulate(self, grads: Dict[str, mx.array]):
        """Add gradients to accumulator."""
        if self.accumulated_grads is None:
            # First accumulation
            self.accumulated_grads = {
                k: v / self.accumulation_steps for k, v in grads.items()
            }
        else:
            # Add to existing
            for k in grads:
                self.accumulated_grads[k] += grads[k] / self.accumulation_steps

        self.current_step += 1

    def should_update(self) -> bool:
        """Check if optimizer should update."""
        return self.current_step >= self.accumulation_steps

    def get_gradients(self) -> Dict[str, mx.array]:
        """Get accumulated gradients and reset."""
        grads = self.accumulated_grads
        self.reset()
        return grads

    def reset(self):
        """Clear accumulated gradients."""
        self.accumulated_grads = None
        self.current_step = 0


def compute_gradients(
    model: mx.nn.Module,
    batch: Batch,
    loss_fn: Callable
) -> Tuple[mx.array, Dict[str, mx.array]]:
    """
    Compute loss and gradients for a batch.

    Args:
        model: The model to train
        batch: Input batch
        loss_fn: Loss computation function

    Returns:
        (loss, gradients_dict)
    """
    def forward_and_loss(params):
        """Forward pass and loss computation."""
        # Update model parameters
        model.update(params)

        # Forward pass
        logits = model(batch.input_ids, batch.attention_mask)

        # Compute loss
        loss = loss_fn(logits, batch.labels, batch.attention_mask)

        return loss

    # Get trainable parameters
    params = model.trainable_parameters()

    # Compute gradients using MLX's grad
    loss_and_grad_fn = mx.value_and_grad(forward_and_loss)
    loss, grads = loss_and_grad_fn(params)

    return loss, grads
```

### 3.3 Gradient Clipping

**Purpose**: Prevent exploding gradients and stabilize training

```python
def clip_gradients(
    grads: Dict[str, mx.array],
    max_norm: float = 1.0
) -> Tuple[Dict[str, mx.array], float]:
    """
    Clip gradients by global norm.

    Args:
        grads: Dictionary of gradients
        max_norm: Maximum allowed gradient norm

    Returns:
        (clipped_gradients, actual_norm)
    """
    # Compute global norm
    total_norm = 0.0
    for grad in grads.values():
        total_norm += mx.sum(grad ** 2)
    total_norm = mx.sqrt(total_norm)

    # Compute clip coefficient
    clip_coef = max_norm / (total_norm + 1e-6)

    # Clip if necessary
    if clip_coef < 1.0:
        clipped_grads = {
            k: v * clip_coef for k, v in grads.items()
        }
        return clipped_grads, float(total_norm)
    else:
        return grads, float(total_norm)
```

### 3.4 Optimizer Configuration

**Optimizer**: AdamW (Adam with weight decay)

```python
def create_optimizer(
    model: mx.nn.Module,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8
) -> mx.optimizers.AdamW:
    """
    Create AdamW optimizer for LoRA parameters.

    Args:
        model: Model with trainable parameters
        learning_rate: Initial learning rate
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters
        eps: Epsilon for numerical stability

    Returns:
        Configured AdamW optimizer
    """
    optimizer = mx.optimizers.AdamW(
        learning_rate=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    return optimizer
```

**Hyperparameter Rationale**:
- **Learning Rate (2e-4)**: Standard for LoRA fine-tuning, higher than full fine-tuning
- **Weight Decay (0.01)**: Moderate regularization for LoRA parameters
- **Betas (0.9, 0.999)**: Standard Adam values, work well for transformers
- **Epsilon (1e-8)**: Standard numerical stability constant

### 3.5 Learning Rate Scheduling

**Scheduler**: Cosine schedule with linear warmup

```python
class CosineWarmupScheduler:
    """
    Cosine learning rate schedule with linear warmup.

    Warmup: Linear increase from 0 to max_lr over warmup_steps
    Cosine: Cosine decay from max_lr to min_lr over remaining steps
    """

    def __init__(
        self,
        optimizer: mx.optimizers.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = optimizer.learning_rate
        self.min_lr = self.base_lr * min_lr_ratio
        self.current_step = 0

    def step(self):
        """Update learning rate for next step."""
        self.current_step += 1
        new_lr = self.get_lr()
        self.optimizer.learning_rate = new_lr

    def get_lr(self) -> float:
        """Compute learning rate for current step."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

    def state_dict(self) -> Dict:
        """Get scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
        }

    def load_state_dict(self, state: Dict):
        """Load scheduler state from checkpoint."""
        self.current_step = state['current_step']
```

**Schedule Visualization**:
```
Learning Rate
    │
max │     ╱‾‾‾╲
    │    ╱     ╲
    │   ╱       ╲___
    │  ╱            ╲___
min │ ╱                 ╲___
    └────────────────────────────→ Steps
      ↑        ↑               ↑
   warmup   peak           end
   (100)                 (total)
```

## 4. Training Engine Implementation

### 4.1 Main Training Engine Class

```python
class TrainingEngine:
    """
    Main training engine for MLX fine-tuning.
    """

    def __init__(
        self,
        model: mx.nn.Module,
        train_loader: MLXDataLoader,
        val_loader: MLXDataLoader,
        optimizer: mx.optimizers.Optimizer,
        scheduler: CosineWarmupScheduler,
        config: TrainingConfig,
        logger: Logger,
        checkpoint_manager: CheckpointManager,
        memory_monitor: MemoryMonitor,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.memory_monitor = memory_monitor

        # Initialize training state
        self.state = TrainingState(
            epoch=0,
            global_step=0,
            best_val_loss=float('inf'),
            samples_seen=0,
            optimizer_state={},
            scheduler_state={},
            rng_state=None,
            training_history=[],
        )

        # Gradient accumulator
        self.grad_accumulator = GradientAccumulator(
            config.gradient_accumulation_steps
        )

        # Metrics tracking
        self.metrics_tracker = MetricsTracker()

    def train(self) -> TrainingResult:
        """
        Execute complete training loop.

        Returns:
            TrainingResult with final metrics and paths
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Configuration: {self.config}")

        start_time = time.time()

        try:
            for epoch in range(self.state.epoch, self.config.num_epochs):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                self.logger.info(f"{'='*60}")

                # Train one epoch
                epoch_metrics = self.train_epoch(epoch)

                # Log epoch summary
                self.logger.info(f"Epoch {epoch + 1} complete:")
                self.logger.info(f"  Train Loss: {epoch_metrics.train_loss:.4f}")
                self.logger.info(f"  Val Loss: {epoch_metrics.val_loss:.4f}")
                self.logger.info(f"  Val Perplexity: {epoch_metrics.val_perplexity:.2f}")
                self.logger.info(f"  Time: {epoch_metrics.epoch_time:.2f}s")

                # Update best model
                if epoch_metrics.val_loss < self.state.best_val_loss:
                    self.state.best_val_loss = epoch_metrics.val_loss
                    self.save_checkpoint(is_best=True)
                    self.logger.info(f"  New best model! Val loss: {epoch_metrics.val_loss:.4f}")

                # Save epoch checkpoint
                self.save_checkpoint(is_best=False)

                self.state.epoch = epoch + 1

        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            self.save_checkpoint(is_best=False)

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

        total_time = time.time() - start_time

        # Save final model
        final_model_path = self.save_final_model()

        self.logger.info(f"\nTraining complete!")
        self.logger.info(f"Total time: {total_time / 3600:.2f} hours")
        self.logger.info(f"Best val loss: {self.state.best_val_loss:.4f}")
        self.logger.info(f"Final model saved to: {final_model_path}")

        return TrainingResult(
            total_epochs=self.state.epoch,
            total_steps=self.state.global_step,
            best_val_loss=self.state.best_val_loss,
            best_checkpoint_path=self.checkpoint_manager.get_best_checkpoint(),
            final_model_path=final_model_path,
            training_history=self.state.training_history,
            total_time=total_time,
        )

    def train_epoch(self, epoch: int) -> EpochMetrics:
        """Train for one epoch."""
        self.model.train()

        epoch_start = time.time()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Train step
            step_metrics = self.train_step(batch)

            total_loss += step_metrics.loss
            num_batches += 1

            # Logging
            if self.state.global_step % self.config.logging_steps == 0:
                self.log_step_metrics(step_metrics)

            # Evaluation
            if self.state.global_step % self.config.eval_steps == 0:
                eval_metrics = self.evaluate()
                self.log_eval_metrics(eval_metrics)

            # Checkpointing
            if self.state.global_step % self.config.save_steps == 0:
                self.save_checkpoint(is_best=False)

        # End of epoch evaluation
        eval_metrics = self.evaluate()

        epoch_time = time.time() - epoch_start
        avg_train_loss = total_loss / num_batches

        return EpochMetrics(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=eval_metrics.val_loss,
            val_perplexity=eval_metrics.perplexity,
            epoch_time=epoch_time,
            samples_per_second=len(self.train_loader.dataset) / epoch_time,
        )

    def train_step(self, batch: Batch) -> StepMetrics:
        """Execute single training step."""
        step_start = time.time()

        # Forward and backward
        loss, grads = compute_gradients(
            self.model, batch, compute_loss
        )

        # Accumulate gradients
        self.grad_accumulator.accumulate(grads)

        # Update if accumulation complete
        if self.grad_accumulator.should_update():
            # Get accumulated gradients
            accumulated_grads = self.grad_accumulator.get_gradients()

            # Clip gradients
            clipped_grads, grad_norm = clip_gradients(
                accumulated_grads, self.config.max_grad_norm
            )

            # Optimizer step
            self.optimizer.update(self.model, clipped_grads)

            # Scheduler step
            self.scheduler.step()

            # Increment global step
            self.state.global_step += 1
        else:
            grad_norm = 0.0  # Not updating yet

        # Track samples
        self.state.samples_seen += batch.metadata['batch_size']

        # Metrics
        step_time = time.time() - step_start
        memory_used = self.memory_monitor.get_current_usage()

        return StepMetrics(
            loss=float(loss),
            learning_rate=self.scheduler.get_lr(),
            gradient_norm=grad_norm,
            step_time=step_time,
            memory_used_gb=memory_used.used_gb,
        )

    def evaluate(self) -> EvalMetrics:
        """Evaluate on validation set."""
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        for batch in self.val_loader:
            # Forward pass only
            logits = self.model(batch.input_ids, batch.attention_mask)
            loss = compute_loss(logits, batch.labels, batch.attention_mask)

            # Accumulate
            num_tokens = mx.sum(batch.attention_mask)
            total_loss += float(loss) * float(num_tokens)
            total_tokens += float(num_tokens)

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        self.model.train()

        return EvalMetrics(
            val_loss=avg_loss,
            perplexity=perplexity,
            num_samples=len(self.val_loader.dataset),
            avg_sequence_length=total_tokens / len(self.val_loader.dataset),
            samples=[],  # Could add generated samples here
        )

    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint_data = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'training_state': self.state,
            'config': self.config,
        }

        self.checkpoint_manager.save(
            checkpoint_data,
            step=self.state.global_step,
            is_best=is_best,
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Resume from checkpoint."""
        checkpoint = self.checkpoint_manager.load(checkpoint_path)

        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.state = checkpoint['training_state']

        self.logger.info(f"Resumed from checkpoint: {checkpoint_path}")
        self.logger.info(f"  Epoch: {self.state.epoch}")
        self.logger.info(f"  Global step: {self.state.global_step}")
        self.logger.info(f"  Best val loss: {self.state.best_val_loss}")

    def save_final_model(self) -> str:
        """Save final LoRA adapters."""
        output_dir = os.path.join(self.config.output_dir, 'final_model')
        os.makedirs(output_dir, exist_ok=True)

        # Save LoRA adapters only
        adapters_path = os.path.join(output_dir, 'lora_adapters.npz')
        mx.savez(adapters_path, **self.model.trainable_parameters())

        # Save config
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)

        return output_dir

    def log_step_metrics(self, metrics: StepMetrics):
        """Log training step metrics."""
        self.logger.info(
            f"Step {self.state.global_step}: "
            f"loss={metrics.loss:.4f}, "
            f"lr={metrics.learning_rate:.2e}, "
            f"grad_norm={metrics.gradient_norm:.4f}, "
            f"memory={metrics.memory_used_gb:.2f}GB"
        )

        self.metrics_tracker.log(self.state.global_step, {
            'train/loss': metrics.loss,
            'train/learning_rate': metrics.learning_rate,
            'train/gradient_norm': metrics.gradient_norm,
            'system/memory_gb': metrics.memory_used_gb,
        })

    def log_eval_metrics(self, metrics: EvalMetrics):
        """Log evaluation metrics."""
        self.logger.info(
            f"Eval @ step {self.state.global_step}: "
            f"val_loss={metrics.val_loss:.4f}, "
            f"perplexity={metrics.perplexity:.2f}"
        )

        self.metrics_tracker.log(self.state.global_step, {
            'eval/loss': metrics.val_loss,
            'eval/perplexity': metrics.perplexity,
        })
```

## 5. Memory Optimization Strategies

### 5.1 Memory Budget Breakdown

**Total Available**: 24GB
**Target Peak Usage**: 14-16GB (67% utilization)

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| Base Model (frozen) | ~7GB | float16, 7B params |
| LoRA Parameters | ~32MB | rank=16, 2 modules/layer |
| Optimizer State | ~64MB | 2x LoRA params (Adam) |
| Activations (batch=2) | ~4-6GB | Forward pass activations |
| Gradients | ~32MB | LoRA gradients only |
| Dataset | ~24MB | 1000 examples tokenized |
| System/MLX Overhead | ~2-3GB | MLX runtime, cache |
| **Total** | **~14-16GB** | Within budget |

### 5.2 Memory Optimization Techniques

**1. LoRA for Parameter Efficiency**
```python
# Only ~8M trainable parameters instead of 7B
# Reduces optimizer memory by 99.9%
lora_config = LoRAConfig(
    rank=16,           # Low rank = less memory
    alpha=32,          # Scaling factor
    target_modules=["q_proj", "v_proj"],  # 2 modules/layer
)
```

**2. Small Batch Size**
```python
# batch_size=2 keeps activation memory low
# Use gradient accumulation for effective larger batch
batch_size = 2
gradient_accumulation_steps = 4  # Effective batch = 8
```

**3. Mixed Precision (float16)**
```python
# Model loaded in float16 by default
# Reduces memory by 2x compared to float32
dtype = mx.float16
```

**4. Gradient Checkpointing (if needed)**
```python
# Trade compute for memory
# Recompute activations during backward pass
# Not implemented initially, add if memory constrained
```

**5. Efficient Tensor Management**
```python
# Clear intermediate tensors
def clear_cache():
    """Clear MLX cache to free memory."""
    mx.metal.clear_cache()

# Call after checkpoint save or evaluation
```

### 5.3 Memory Monitoring

```python
class MemoryMonitor:
    """Monitor memory usage during training."""

    def __init__(self, logger: Logger):
        self.logger = logger
        self.peak_memory = 0.0

    def get_current_usage(self) -> MemoryStats:
        """Get current memory statistics."""
        # MLX-specific memory query
        used = mx.metal.get_active_memory() / (1024 ** 3)  # GB
        peak = mx.metal.get_peak_memory() / (1024 ** 3)

        self.peak_memory = max(self.peak_memory, peak)

        return MemoryStats(
            used_gb=used,
            peak_gb=peak,
            available_gb=24.0 - used,
            percent_used=(used / 24.0) * 100,
        )

    def log_memory(self, context: str):
        """Log memory at specific point."""
        stats = self.get_current_usage()
        self.logger.info(
            f"Memory [{context}]: "
            f"{stats.used_gb:.2f}GB used, "
            f"{stats.peak_gb:.2f}GB peak, "
            f"{stats.percent_used:.1f}% utilization"
        )

    def check_available(self, required_gb: float) -> bool:
        """Check if enough memory available."""
        stats = self.get_current_usage()
        return stats.available_gb >= required_gb
```

## 6. Stability and Quality Checks

### 6.1 Gradient Health Checks

```python
def check_gradients_health(grads: Dict[str, mx.array], logger: Logger):
    """Check for NaN/Inf in gradients."""
    for name, grad in grads.items():
        if mx.any(mx.isnan(grad)):
            logger.error(f"NaN detected in gradient: {name}")
            raise ValueError(f"NaN gradient: {name}")

        if mx.any(mx.isinf(grad)):
            logger.error(f"Inf detected in gradient: {name}")
            raise ValueError(f"Inf gradient: {name}")
```

### 6.2 Loss Monitoring

```python
class LossMonitor:
    """Monitor loss for anomalies."""

    def __init__(self, window_size: int = 100):
        self.losses = []
        self.window_size = window_size

    def add(self, loss: float):
        """Add new loss value."""
        if math.isnan(loss) or math.isinf(loss):
            raise ValueError(f"Invalid loss: {loss}")

        self.losses.append(loss)
        if len(self.losses) > self.window_size:
            self.losses.pop(0)

    def get_avg(self) -> float:
        """Get average over window."""
        return sum(self.losses) / len(self.losses) if self.losses else 0.0

    def is_diverging(self, threshold: float = 2.0) -> bool:
        """Check if loss is diverging."""
        if len(self.losses) < 10:
            return False

        recent_avg = sum(self.losses[-10:]) / 10
        overall_avg = self.get_avg()

        return recent_avg > overall_avg * threshold
```

### 6.3 Early Stopping

```python
class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Returns:
            True if should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
```

## 7. Checkpointing Strategy

### 7.1 Checkpoint Manager

```python
class CheckpointManager:
    """Manage training checkpoints."""

    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        checkpoint_data: Dict,
        step: int,
        is_best: bool = False
    ):
        """Save checkpoint to disk."""
        # Regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f'checkpoint_step_{step}.pt'
        )

        # Save atomically
        temp_path = checkpoint_path + '.tmp'
        mx.save(temp_path, checkpoint_data)
        os.rename(temp_path, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pt')
            shutil.copy(checkpoint_path, best_path)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

    def load(self, checkpoint_path: str) -> Dict:
        """Load checkpoint from disk."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = mx.load(checkpoint_path)
        return checkpoint

    def get_latest(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].replace('.pt', '')))
        return os.path.join(self.checkpoint_dir, checkpoints[-1])

    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pt')
        return best_path if os.path.exists(best_path) else None

    def list_checkpoints(self) -> List[str]:
        """List all checkpoints."""
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith('checkpoint_step_') and f.endswith('.pt')
        ]
        return checkpoints

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping last N."""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= self.keep_last_n:
            return

        # Sort by step
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].replace('.pt', '')))

        # Remove oldest
        to_remove = checkpoints[:-self.keep_last_n]
        for ckpt in to_remove:
            path = os.path.join(self.checkpoint_dir, ckpt)
            os.remove(path)
```

## 8. Performance Optimization

### 8.1 MLX-Specific Optimizations

```python
# 1. Lazy evaluation
# MLX uses lazy evaluation - operations queued until needed
# Force evaluation with mx.eval() when timing matters

# 2. Graph compilation
# MLX compiles computation graphs for efficiency
# First iteration slower, subsequent iterations faster

# 3. Metal GPU utilization
# MLX automatically uses Metal GPU
# No explicit device management needed

# 4. Unified memory
# CPU and GPU share memory on M4
# No expensive data transfers
```

### 8.2 Training Throughput

**Expected Performance**:
- Batch size: 2
- Sequence length: 2048
- Time per step: ~2-3 seconds
- Steps per epoch: 450 (900 examples / 2)
- Epoch time: ~15-20 minutes
- Total training: ~4-6 hours (3 epochs)

**Optimization Opportunities**:
1. **Increase batch size**: If memory allows, batch_size=4 would speed up training
2. **Reduce sequence length**: If conversations are shorter, reduce max_length
3. **Reduce validation frequency**: eval_steps=100 instead of 50
4. **Optimize data loading**: Cache tokenized data to disk

## 9. Testing Strategy

### 9.1 Unit Tests

```python
def test_loss_computation():
    """Test loss function."""
    logits = mx.random.normal((2, 10, 100))  # batch, seq, vocab
    labels = mx.random.randint(0, 100, (2, 10))
    attention_mask = mx.ones((2, 10))

    loss = compute_loss(logits, labels, attention_mask)
    assert not mx.isnan(loss)
    assert loss > 0

def test_gradient_clipping():
    """Test gradient clipping."""
    grads = {
        'layer1': mx.array([10.0, 20.0]),
        'layer2': mx.array([30.0, 40.0]),
    }

    clipped, norm = clip_gradients(grads, max_norm=1.0)
    # Check norm is <= 1.0
    assert all(mx.linalg.norm(g) <= 1.1 for g in clipped.values())

def test_learning_rate_schedule():
    """Test LR scheduler."""
    optimizer = create_optimizer(model, learning_rate=1e-3)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=10, total_steps=100)

    # Check warmup
    for i in range(10):
        lr = scheduler.get_lr()
        scheduler.step()
        assert lr <= 1e-3

    # Check decay
    for i in range(90):
        lr = scheduler.get_lr()
        scheduler.step()
    assert lr < 1e-3  # Should decay
```

### 9.2 Integration Tests

```python
def test_full_training_step():
    """Test complete training step."""
    # Setup
    model = create_test_model()
    optimizer = create_optimizer(model)
    batch = create_test_batch()

    # Training step
    loss, grads = compute_gradients(model, batch, compute_loss)
    clipped_grads, _ = clip_gradients(grads)
    optimizer.update(model, clipped_grads)

    # Verify
    assert not mx.isnan(loss)
    assert all(not mx.isnan(g).any() for g in grads.values())
```

---

**Next Document**: `05_CONFIGURATION_SCHEMA.md` - Complete configuration specifications
