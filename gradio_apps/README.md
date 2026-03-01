---
title: English to French Transformer
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: apache-2.0
authors:
  - Jeremy Vachier
tags:
  - translation
  - transformer
  - neural-machine-translation
  - english
  - french
  - tensorflow
  - keras
  - from-scratch
  - nlp
  - seq2seq
  - attention-mechanism
  - encoder-decoder
  - deep-learning
  - machine-translation
  - multilingual
  - text-generation
  - custom-model
  - educational
---

# English to French Enhanced Transformer

A custom Transformer architecture for English-to-French neural machine translation, built entirely from scratch using Keras/TensorFlow. This model demonstrates the power of attention mechanisms without relying on pre-trained models.

## Model Description

This is an **enhanced Transformer model** implementing the encoder-decoder architecture with multi-head self-attention. The model was trained from scratch on English-French parallel text data and achieves competitive translation quality.

### Key Features

- **Built from Scratch**: No pre-trained models or transfer learning - pure Transformer implementation
- **Enhanced Architecture**: Improved version with optimized hyperparameters
- **Dual Decoding**: Supports both greedy decoding (fast) and beam search (higher quality)
- **Production Ready**: Clean code with logging, error handling, and optimized inference

## Architecture

### Model Specifications

| Component | Configuration |
|-----------|--------------|
| Encoder Layers | 2 |
| Decoder Layers | 2 |
| Embedding Dimension | 128 |
| Attention Heads | 4 per layer |
| Feed-Forward Dimension | 2048 |
| Total Parameters | 44,967,200 |
| Vocabulary Size | Variable (from training data) |
| Max Sequence Length | 20 tokens |

### Custom Components

1. **Positional Embedding**: Combines token embeddings with learned positional encodings
2. **Transformer Encoder**: Multi-head self-attention with feed-forward networks and residual connections
3. **Transformer Decoder**: Masked self-attention + encoder-decoder attention + feed-forward networks
4. **Custom Learning Rate Schedule**: Warmup-based schedule for stable training

## Usage

### Input

- **English text** (string): Any English sentence or phrase
- **Beam search toggle** (boolean): Enable for higher quality (slower) or disable for speed

### Output

- **French translation** (string): The translated text in French

### Examples

Try these example translations:

```
Input: "Hello, how are you?"
Output: "bonjour comment allez vous"

Input: "The weather is beautiful today"
Output: "le temps est beau aujourd hui"

Input: "I would like to order a coffee please"
Output: "je voudrais commander un café s il vous plaît"

Input: "Where is the nearest train station?"
Output: "où est la gare la plus proche"
```

## Training

### Dataset

- Source: English-French parallel corpus
- Training approach: Supervised learning with teacher forcing
- Preprocessing: Tokenization, lowercasing, special tokens ([start], [end])

### Hyperparameters

- **Optimizer**: Adam with custom learning rate schedule
- **Warmup Steps**: 4000
- **Batch Size**: 64
- **Loss Function**: Sparse categorical cross-entropy
- **Regularization**: Dropout in attention layers

### Training Process

1. Text preprocessing with custom standardization
2. Vocabulary building from training corpus
3. Sequence padding/truncation to fixed length
4. Teacher forcing during training
5. Validation on held-out test set

## Performance

- **Parameters**: 44,967,200
- **BLEU Score**: 0.58
- **Speed**: Real-time inference for typical sentences
- **Decoding**:
  - Greedy: Fast, good quality
  - Beam Search (k=3): Better quality, slightly slower

## Limitations

- **Training Data**: Limited to the domain and vocabulary of training corpus
- **Sequence Length**: Maximum 20 tokens per sequence (longer texts are truncated)
- **Informal Language**: May struggle with slang, idioms, or very informal expressions
- **Context**: No cross-sentence context (translates each input independently)
- **Named Entities**: May not preserve proper nouns correctly
- **From-Scratch Training**: Performance is limited compared to models pre-trained on massive corpora

## Technical Details

### Dependencies

- TensorFlow/Keras
- Gradio
- NumPy
- Python 3.8+

### Custom Objects

The model includes custom Keras layers that must be registered for loading:
- `PositionalEmbedding`
- `TransformerEncoder`
- `TransformerDecoder`
- `CustomSchedule` (learning rate schedule)

### Inference Methods

**Greedy Decoding**: Selects the most probable token at each step
```python
# Fast, deterministic, good quality
translate_text("Hello", use_beam_search=False)
```

**Beam Search**: Maintains multiple hypotheses for better translations
```python
# Higher quality, explores multiple paths
translate_text("Hello", use_beam_search=True)
```

## Project Links

- **Kaggle Notebook**: [Transformer NMT EN-FR](https://www.kaggle.com/code/jvachier/transformer-nmt-en-fr)
- **GitHub Repository**: [Sentiment_Analysis](https://github.com/jvachier/Sentiment_Analysis)

## License

Apache License 2.0 - See LICENSE file for details

## Citation

If you use this model in your research or project, please cite:

```bibtex
@misc{enhanced-transformer-en-fr,
  author = {Vachier, Jeremy},
  title = {Enhanced Transformer for English-French Translation},
  year = {2026},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/spaces/Jvachier/transformer-nmt-en-fr}}
}
```

## Contact

For questions, issues, or suggestions, please open an issue on the GitHub repository.

---

**Note**: This is an educational/research model built from scratch.
