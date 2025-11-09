# Data Pipeline Architecture

## 1. Overview

The Data Pipeline is responsible for transforming the raw JSONL dataset into tokenized, batched MLX arrays ready for training. It handles validation, preprocessing, and efficient data loading while ensuring data quality and reproducibility.

## 2. Input Data Specification

### 2.1 JSONL Schema

**File**: `syngen_toolset_v1.0.0_claude.jsonl`

**Expected Format**:
```json
{
  "conversations": [
    {
      "from": "system",
      "value": "You are a helpful assistant..."
    },
    {
      "from": "user",
      "value": "What is the capital of France?"
    },
    {
      "from": "assistant",
      "value": "The capital of France is Paris."
    }
  ],
  "label": "desirable"
}
```

**Field Specifications**:

| Field | Type | Required | Description | Valid Values |
|-------|------|----------|-------------|--------------|
| `conversations` | List[Dict] | Yes | Conversation turns | Non-empty list |
| `conversations[].from` | String | Yes | Speaker role | "system", "user", "assistant" |
| `conversations[].value` | String | Yes | Message content | Non-empty string |
| `label` | String | Yes | Quality label | "desirable", "undesirable" |

**Dataset Statistics**:
- Total examples: 1000
- Desirable: 746 (74.6%)
- Undesirable: 254 (25.4%)
- Average conversation length: ~3-5 turns

### 2.2 Validation Rules

The pipeline enforces these validation rules:

1. **Schema Validation**:
   - All required fields present
   - Correct data types
   - Non-empty conversations list
   - Valid role values

2. **Content Validation**:
   - Non-empty message values
   - Valid UTF-8 encoding
   - Reasonable length limits (< 10000 chars per message)
   - Label in allowed set

3. **Conversation Structure**:
   - Alternating user/assistant turns (after optional system message)
   - Ends with assistant response
   - System message only at start (if present)

4. **Quality Thresholds**:
   - Maximum 5% invalid entries allowed
   - Warn if class imbalance > 80/20
   - Error if insufficient data (< 100 examples)

## 3. Data Transformation Pipeline

### 3.1 Processing Stages

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Transformation Flow                     │
└─────────────────────────────────────────────────────────────────┘

Stage 1: JSONL Parsing
┌──────────────┐
│ Read JSONL   │ → Parse each line → JSON objects → List[Dict]
└──────────────┘
       ↓
Stage 2: Schema Validation
┌──────────────┐
│ Validate     │ → Check schema → Filter invalid → Valid examples
│ Each Entry   │                                   + error log
└──────────────┘
       ↓
Stage 3: Conversation Formatting
┌──────────────┐
│ Format for   │ → Apply template → Formatted text
│ Mistral      │    (Instruct)
└──────────────┘
       ↓
Stage 4: Tokenization
┌──────────────┐
│ Tokenize     │ → Mistral tokenizer → Token IDs
│ Text         │                        + attention masks
└──────────────┘
       ↓
Stage 5: Padding/Truncation
┌──────────────┐
│ Pad or Trim  │ → Max length 2048 → Uniform length
│ Sequences    │
└──────────────┘
       ↓
Stage 6: Label Creation
┌──────────────┐
│ Create       │ → Shift input IDs → Target labels
│ Training     │    Mask padding
│ Labels       │
└──────────────┘
       ↓
Stage 7: MLX Conversion
┌──────────────┐
│ Convert to   │ → mx.array() → MLX tensors
│ MLX Arrays   │
└──────────────┘
       ↓
Stage 8: Batching
┌──────────────┐
│ Create       │ → Batch size 2 → Batched arrays
│ Batches      │    + metadata
└──────────────┘
       ↓
Stage 9: Shuffling (if enabled)
┌──────────────┐
│ Shuffle      │ → Random order → Ready for training
│ Batches      │    (seeded)
└──────────────┘
```

### 3.2 Conversation Formatting

**Mistral Instruct Format**:
```
<s>[INST] {system_message}

{user_message_1} [/INST] {assistant_response_1}</s>[INST] {user_message_2} [/INST] {assistant_response_2}</s>
```

**Formatting Function**:
```python
def format_conversation_for_mistral(conversation: List[Dict]) -> str:
    """
    Format conversation according to Mistral Instruct template.

    Args:
        conversation: List of message dicts with 'from' and 'value'

    Returns:
        Formatted string ready for tokenization
    """
    formatted_parts = []
    system_msg = None

    # Extract system message if present
    if conversation[0]['from'] == 'system':
        system_msg = conversation[0]['value']
        conversation = conversation[1:]

    # Format turns
    current_turn = []
    for msg in conversation:
        if msg['from'] == 'user':
            user_content = msg['value']
            if system_msg and len(current_turn) == 0:
                current_turn.append(f"<s>[INST] {system_msg}\n\n{user_content} [/INST]")
            else:
                current_turn.append(f"<s>[INST] {user_content} [/INST]")
        elif msg['from'] == 'assistant':
            current_turn.append(f" {msg['value']}</s>")
            formatted_parts.append(''.join(current_turn))
            current_turn = []

    return ''.join(formatted_parts)
```

**Example Transformation**:

Input:
```json
{
  "conversations": [
    {"from": "system", "value": "You are helpful."},
    {"from": "user", "value": "Hello!"},
    {"from": "assistant", "value": "Hi there!"}
  ]
}
```

Output:
```
<s>[INST] You are helpful.

Hello! [/INST] Hi there!</s>
```

### 3.3 Tokenization Strategy

**Tokenizer**: Mistral-7B-Instruct tokenizer (from Hugging Face)

**Configuration**:
```python
tokenizer_config = {
    'max_length': 2048,          # Mistral's native context length
    'padding': 'max_length',     # Pad to max_length
    'truncation': True,          # Truncate if longer
    'return_attention_mask': True,
    'return_tensors': None,      # Return lists, convert to MLX later
}
```

**Tokenization Process**:
```python
def tokenize_example(formatted_text: str, tokenizer) -> TokenizedExample:
    """
    Tokenize formatted conversation text.

    Returns:
        TokenizedExample with input_ids, attention_mask, and metadata
    """
    encoding = tokenizer(
        formatted_text,
        max_length=2048,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Find actual sequence length (before padding)
    actual_length = sum(attention_mask)

    # Create labels (shifted input_ids, -100 for padding)
    labels = create_labels(input_ids, attention_mask)

    return TokenizedExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        actual_length=actual_length,
    )
```

**Label Creation**:
```python
def create_labels(input_ids: List[int], attention_mask: List[int]) -> List[int]:
    """
    Create training labels from input IDs.

    Labels are input_ids shifted by 1, with -100 for padding positions.
    -100 is the ignore_index for PyTorch/MLX cross-entropy loss.
    """
    labels = []
    for i, (token_id, mask) in enumerate(zip(input_ids, attention_mask)):
        if mask == 0:  # Padding position
            labels.append(-100)
        else:
            # Next token prediction: label is the next token
            if i + 1 < len(input_ids) and attention_mask[i + 1] == 1:
                labels.append(input_ids[i + 1])
            else:
                labels.append(-100)  # Last token or before padding

    return labels
```

### 3.4 Train/Validation Split

**Split Strategy**:
- 90% training, 10% validation (900 / 100 examples)
- Stratified split to maintain label distribution
- Seeded random split for reproducibility

**Implementation**:
```python
def create_train_val_split(
    examples: List[Dict],
    train_ratio: float = 0.9,
    seed: int = 42,
    stratify_by: str = 'label'
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset into train and validation sets.

    Args:
        examples: List of parsed JSONL examples
        train_ratio: Fraction for training (default 0.9)
        seed: Random seed for reproducibility
        stratify_by: Field to stratify split (maintains distribution)

    Returns:
        (train_examples, val_examples)
    """
    from sklearn.model_selection import train_test_split

    # Extract stratification labels
    labels = [ex[stratify_by] for ex in examples]

    train_examples, val_examples = train_test_split(
        examples,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )

    return train_examples, val_examples
```

**Expected Split**:
- Train: 900 examples (671 desirable, 229 undesirable)
- Val: 100 examples (75 desirable, 25 undesirable)

## 4. Data Structures

### 4.1 Internal Representations

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import mlx.core as mx

@dataclass
class RawExample:
    """Parsed JSONL entry before processing."""
    conversations: List[Dict[str, str]]
    label: str
    index: int  # Original position in dataset

@dataclass
class FormattedExample:
    """After conversation formatting."""
    text: str
    label: str
    index: int
    metadata: Dict[str, Any]  # Original conversations, etc.

@dataclass
class TokenizedExample:
    """After tokenization."""
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    actual_length: int
    label: str
    index: int

@dataclass
class MLXExample:
    """Converted to MLX arrays."""
    input_ids: mx.array  # Shape: (seq_length,)
    attention_mask: mx.array  # Shape: (seq_length,)
    labels: mx.array  # Shape: (seq_length,)
    metadata: Dict[str, Any]

@dataclass
class Batch:
    """Batched examples ready for training."""
    input_ids: mx.array  # Shape: (batch_size, seq_length)
    attention_mask: mx.array  # Shape: (batch_size, seq_length)
    labels: mx.array  # Shape: (batch_size, seq_length)
    metadata: Dict[str, Any]  # Batch-level metadata

    def __len__(self) -> int:
        return self.input_ids.shape[0]
```

### 4.2 Dataset Class

```python
class MLXDataset:
    """
    Dataset class for MLX fine-tuning.

    Stores tokenized examples and provides indexing.
    """

    def __init__(self, examples: List[TokenizedExample], tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> MLXExample:
        """Get a single example as MLX arrays."""
        example = self.examples[idx]

        return MLXExample(
            input_ids=mx.array(example.input_ids),
            attention_mask=mx.array(example.attention_mask),
            labels=mx.array(example.labels),
            metadata={
                'label': example.label,
                'index': example.index,
                'actual_length': example.actual_length,
            }
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        lengths = [ex.actual_length for ex in self.examples]
        labels = [ex.label for ex in self.examples]

        return {
            'num_examples': len(self.examples),
            'avg_length': sum(lengths) / len(lengths),
            'max_length': max(lengths),
            'min_length': min(lengths),
            'label_distribution': {
                'desirable': labels.count('desirable'),
                'undesirable': labels.count('undesirable'),
            }
        }
```

### 4.3 DataLoader Class

```python
class MLXDataLoader:
    """
    Iterable data loader for MLX training.

    Creates batches from MLXDataset with optional shuffling.
    """

    def __init__(
        self,
        dataset: MLXDataset,
        batch_size: int = 2,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

    def __len__(self) -> int:
        """Number of batches."""
        num_examples = len(self.dataset)
        if self.drop_last:
            return num_examples // self.batch_size
        else:
            return (num_examples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Batch]:
        """Iterate over batches."""
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            # Use epoch-dependent seed for different shuffles each epoch
            import random
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(indices)

        # Create batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]

            # Skip incomplete batch if drop_last
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            # Collect examples
            examples = [self.dataset[idx] for idx in batch_indices]

            # Stack into batch
            batch = self._collate(examples)
            yield batch

        self._epoch += 1

    def _collate(self, examples: List[MLXExample]) -> Batch:
        """Collate examples into a batch."""
        # Stack arrays
        input_ids = mx.stack([ex.input_ids for ex in examples])
        attention_mask = mx.stack([ex.attention_mask for ex in examples])
        labels = mx.stack([ex.labels for ex in examples])

        # Aggregate metadata
        metadata = {
            'batch_size': len(examples),
            'labels': [ex.metadata['label'] for ex in examples],
            'indices': [ex.metadata['index'] for ex in examples],
            'actual_lengths': [ex.metadata['actual_length'] for ex in examples],
        }

        return Batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            metadata=metadata,
        )
```

## 5. Data Pipeline Implementation

### 5.1 Main Pipeline Class

```python
class DataPipeline:
    """
    Complete data pipeline from JSONL to MLX batches.
    """

    def __init__(self, config: DataConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None

    def initialize(self):
        """Initialize tokenizer and validate paths."""
        from transformers import AutoTokenizer

        # Load tokenizer
        self.logger.info("Loading Mistral tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            trust_remote_code=True,
        )

        # Verify dataset path
        if not os.path.exists(self.config.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.config.dataset_path}")

        self.logger.info("Data pipeline initialized")

    def load_and_prepare(self) -> Tuple[MLXDataLoader, MLXDataLoader]:
        """
        Load dataset, validate, and create train/val loaders.

        Returns:
            (train_loader, val_loader)
        """
        # Stage 1: Load JSONL
        self.logger.info("Loading JSONL dataset...")
        raw_examples = self._load_jsonl(self.config.dataset_path)
        self.logger.info(f"Loaded {len(raw_examples)} examples")

        # Stage 2: Validate
        self.logger.info("Validating dataset schema...")
        valid_examples = self._validate_examples(raw_examples)
        self.logger.info(f"Validated {len(valid_examples)} examples")

        # Stage 3: Train/val split
        self.logger.info("Creating train/val split...")
        train_examples, val_examples = create_train_val_split(
            valid_examples,
            train_ratio=self.config.train_split,
            seed=self.config.seed,
        )
        self.logger.info(f"Split: {len(train_examples)} train, {len(val_examples)} val")

        # Stage 4-7: Process examples
        self.logger.info("Processing training examples...")
        train_tokenized = self._process_examples(train_examples)

        self.logger.info("Processing validation examples...")
        val_tokenized = self._process_examples(val_examples)

        # Create datasets
        self.train_dataset = MLXDataset(train_tokenized, self.tokenizer)
        self.val_dataset = MLXDataset(val_tokenized, self.tokenizer)

        # Log statistics
        train_stats = self.train_dataset.get_statistics()
        val_stats = self.val_dataset.get_statistics()
        self.logger.info(f"Train stats: {train_stats}")
        self.logger.info(f"Val stats: {val_stats}")

        # Create data loaders
        train_loader = MLXDataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            seed=self.config.seed,
        )

        val_loader = MLXDataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        self.logger.info(f"Created loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")

        return train_loader, val_loader

    def _load_jsonl(self, path: str) -> List[RawExample]:
        """Load and parse JSONL file."""
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    data = json.loads(line)
                    examples.append(RawExample(
                        conversations=data['conversations'],
                        label=data['label'],
                        index=idx,
                    ))
                except Exception as e:
                    self.logger.warning(f"Failed to parse line {idx}: {e}")
        return examples

    def _validate_examples(self, examples: List[RawExample]) -> List[RawExample]:
        """Validate and filter examples."""
        valid = []
        invalid_count = 0

        for ex in examples:
            try:
                self._validate_single_example(ex)
                valid.append(ex)
            except ValidationError as e:
                self.logger.warning(f"Invalid example {ex.index}: {e}")
                invalid_count += 1

        # Check quality threshold
        if invalid_count / len(examples) > 0.05:
            raise ValueError(f"Too many invalid examples: {invalid_count}/{len(examples)}")

        return valid

    def _validate_single_example(self, example: RawExample):
        """Validate a single example."""
        # Check conversations is non-empty
        if not example.conversations:
            raise ValidationError("Empty conversations")

        # Check label
        if example.label not in ['desirable', 'undesirable']:
            raise ValidationError(f"Invalid label: {example.label}")

        # Check conversation structure
        for msg in example.conversations:
            if 'from' not in msg or 'value' not in msg:
                raise ValidationError("Missing 'from' or 'value' in message")

            if msg['from'] not in ['system', 'user', 'assistant']:
                raise ValidationError(f"Invalid role: {msg['from']}")

            if not msg['value'] or not msg['value'].strip():
                raise ValidationError("Empty message value")

    def _process_examples(self, examples: List[RawExample]) -> List[TokenizedExample]:
        """Process examples through formatting and tokenization."""
        tokenized = []

        for ex in examples:
            # Format conversation
            formatted_text = format_conversation_for_mistral(ex.conversations)

            # Tokenize
            tokenized_ex = tokenize_example(formatted_text, self.tokenizer)
            tokenized_ex.label = ex.label
            tokenized_ex.index = ex.index

            tokenized.append(tokenized_ex)

        return tokenized
```

## 6. Memory Considerations

### 6.1 Memory Usage Estimation

**Per Example**:
- Input IDs: 2048 tokens × 4 bytes (int32) = 8 KB
- Attention mask: 2048 × 4 bytes = 8 KB
- Labels: 2048 × 4 bytes = 8 KB
- **Total per example**: ~24 KB

**Per Batch (batch_size=2)**:
- 2 examples × 24 KB = 48 KB (negligible)

**Entire Dataset in Memory**:
- 1000 examples × 24 KB = 24 MB (acceptable)

**Conclusion**: Entire dataset can be held in memory without issues.

### 6.2 Optimization Strategies

1. **Lazy Loading**: Not needed given small dataset size
2. **On-the-fly Tokenization**: Could save memory, but adds compute overhead (not recommended)
3. **Batch Prefetching**: Not needed for MLX (no multi-processing)
4. **Dynamic Batching**: Could group similar lengths, but adds complexity (not implemented)

## 7. Error Handling

### 7.1 Error Types and Responses

| Error Type | Cause | Response |
|------------|-------|----------|
| `FileNotFoundError` | Dataset path invalid | Raise error, stop execution |
| `ValidationError` | Invalid JSONL entry | Log warning, skip entry |
| `TooManyInvalidExamples` | >5% invalid | Raise error, stop execution |
| `TokenizationError` | Tokenizer fails | Log warning, skip example |
| `InsufficientData` | <100 valid examples | Raise error, stop execution |
| `ImbalancedData` | >80% one class | Log warning, continue |

### 7.2 Validation Error Details

```python
class ValidationError(Exception):
    """Raised when data validation fails."""
    pass

def validate_dataset_quality(valid_count: int, total_count: int):
    """Check overall dataset quality."""
    if total_count == 0:
        raise ValidationError("Empty dataset")

    if valid_count < 100:
        raise ValidationError(f"Insufficient valid examples: {valid_count}")

    invalid_rate = (total_count - valid_count) / total_count
    if invalid_rate > 0.05:
        raise ValidationError(f"Too many invalid examples: {invalid_rate:.1%}")
```

## 8. Testing Strategy

### 8.1 Unit Tests

```python
def test_conversation_formatting():
    """Test Mistral conversation formatting."""
    conversation = [
        {"from": "user", "value": "Hello"},
        {"from": "assistant", "value": "Hi"},
    ]
    formatted = format_conversation_for_mistral(conversation)
    assert "<s>[INST]" in formatted
    assert "[/INST]" in formatted
    assert "</s>" in formatted

def test_label_creation():
    """Test label creation from input IDs."""
    input_ids = [1, 2, 3, 0, 0]  # 0 is padding
    attention_mask = [1, 1, 1, 0, 0]
    labels = create_labels(input_ids, attention_mask)
    assert labels[-2:] == [-100, -100]  # Padding positions

def test_validation():
    """Test example validation."""
    valid_example = RawExample(
        conversations=[
            {"from": "user", "value": "Hello"},
            {"from": "assistant", "value": "Hi"},
        ],
        label="desirable",
        index=0,
    )
    # Should not raise
    pipeline._validate_single_example(valid_example)

    invalid_example = RawExample(
        conversations=[],
        label="invalid",
        index=1,
    )
    with pytest.raises(ValidationError):
        pipeline._validate_single_example(invalid_example)
```

### 8.2 Integration Tests

```python
def test_full_pipeline():
    """Test complete data pipeline."""
    config = DataConfig(
        dataset_path="test_data.jsonl",
        batch_size=2,
        train_split=0.8,
    )

    pipeline = DataPipeline(config, logger)
    pipeline.initialize()
    train_loader, val_loader = pipeline.load_and_prepare()

    # Check loaders
    assert len(train_loader) > 0
    assert len(val_loader) > 0

    # Check batch
    batch = next(iter(train_loader))
    assert batch.input_ids.shape[0] == 2  # batch_size
    assert batch.input_ids.shape[1] == 2048  # seq_length
```

## 9. Performance Metrics

### 9.1 Pipeline Benchmarks

Expected performance on Mac M4:

| Operation | Time | Notes |
|-----------|------|-------|
| Load JSONL | ~0.1s | 1000 examples |
| Validation | ~0.2s | Schema checks |
| Formatting | ~0.5s | Template application |
| Tokenization | ~5-10s | Mistral tokenizer |
| Batch creation | ~0.1s | Array operations |
| **Total** | ~6-11s | One-time cost |

### 9.2 Optimization Opportunities

1. **Parallel Tokenization**: Use multiprocessing (if bottleneck)
2. **Caching**: Cache tokenized data to disk
3. **Batch Tokenization**: Tokenize multiple examples at once

**Recommendation**: Current performance is acceptable for 1000 examples. Optimize only if scaling to much larger datasets.

## 10. Data Augmentation (Future)

Potential augmentation strategies for future iterations:

1. **Paraphrasing**: Rephrase user queries while preserving intent
2. **Back-translation**: Translate to another language and back
3. **Synthetic Variations**: Generate similar conversations with different phrasings
4. **Label Smoothing**: Soft labels instead of hard desirable/undesirable

**Not implemented in current architecture** but extension point exists in `_process_examples`.

---

**Next Document**: `04_TRAINING_PIPELINE.md` - Training loop and optimization strategies
