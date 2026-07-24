# StyleNet — Neural Text Style Transfer

A neural sequence-to-sequence model for **controllable text style transfer**, built with GRU-based encoder-decoder architecture. Given an input sentence, StyleNet rewrites it in a target style (e.g., formal to informal, factual to stylistic) while preserving the core meaning.

## What is Text Style Transfer?

Text style transfer is the task of changing the *style* of a piece of text — its tone, formality, sentiment, or writing manner — while keeping the *content* the same. This is a challenging NLP problem that sits at the intersection of:

- **Natural Language Generation** — producing fluent, coherent output
- **Controllable Generation** — steering the output towards a target style
- **Sequence Modeling** — understanding and transforming language at the sentence level

## Architecture

```
Input Sentence
      |
  [Encoder]  — GRU-based bidirectional encoder
      |
  [Content Representation]
      +
  [Style Vector]  — learned style embedding
      |
  [Decoder]  — GRU-based autoregressive decoder
      |
 Output Sentence (in target style)
```

- **Encoder:** Bidirectional GRU that reads the input sentence and produces a fixed-size content representation.
- **Style Embedding:** A learned vector for each style class, concatenated with the content representation.
- **Decoder:** GRU decoder that generates the output token by token, conditioned on content + style.
- **Training:** Trained on parallel and non-parallel corpora using reconstruction loss and style classifier feedback.

## Features

- GRU-based sequence-to-sequence architecture
- Controllable style conditioning via learned embeddings
- Supports Bengali and English corpora
- Configurable style classes (formal/informal, factual/stylistic)
- Attention mechanism for better content preservation

## Requirements

```bash
pip install torch numpy pandas tqdm
```

## Usage

```python
# Train the model
python train.py --data_path data/ --style formal --epochs 50

# Generate styled text
python generate.py --input "The weather is bad today." --target_style formal
# Output: "Today's meteorological conditions are unfavorable."
```

## Results

The model is evaluated on:
- **BLEU Score** — measures content preservation
- **Style Classifier Accuracy** — measures how well the target style is achieved
- **Perplexity** — measures fluency of generated text

## Related Work

This implementation is inspired by:
- *Style Transfer from Non-Parallel Text by Cross-Alignment* (Shen et al., 2017)
- *Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer* (Li et al., 2018)
- The original StyleNet paper for factual text generation

## Author

**Nishat Tasnim Mahi** — AI/ML Researcher | NLP | Computer Vision | Multimodal Learning

[GitHub](https://github.com/nishatmahi)