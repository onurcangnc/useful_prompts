# TurniLab - AI Text Detection Research Framework

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Research-orange.svg" alt="Status">
  <img src="https://img.shields.io/badge/Purpose-Academic-purple.svg" alt="Purpose">
</p>

A comprehensive Python framework for studying AI text detection mechanisms, linguistic pattern analysis, and text stylometry. Designed for academic research into Natural Language Processing (NLP) classification systems.

## 📋 Overview

TurniLab provides tools to understand how AI text detectors work by analyzing:

- **Perplexity patterns** - Statistical predictability of token sequences
- **Burstiness metrics** - Variation in sentence length and structure
- **Stylometric features** - Part-of-speech distributions, syntactic complexity
- **Lexical diversity** - Type-token ratios, vocabulary richness

This framework is intended for researchers studying the strengths and limitations of AI detection systems.

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from model import TurniLab

# Initialize the framework
tl = TurniLab()

# Analyze text for linguistic patterns
result = tl.analyze("Your text here")
print(f"Linguistic Analysis: {result}")

# Train classifier on your dataset
human_texts = [...]  # Human-written samples
ai_texts = [...]     # AI-generated samples
metrics = tl.train_on_dataset(human_texts, ai_texts, save_path="model.pkl")
print(f"Model AUC: {metrics['test_auc']:.3f}")
```

### Command Line Interface

```bash
# Run demonstration
python model.py --mode demo

# Analyze specific text
python model.py --mode analyze --text "Your text here"

# Train on custom dataset
python model.py --mode train --human-file human.txt --ai-file ai.txt

# Transform and analyze
python model.py --mode transform --text "Text to analyze" --output results.json
```

## 📁 Project Structure

```
turnilab/
├── model.py              # Core analysis framework (TurniLab class)
├── evasion.py            # Text transformation utilities (Turkish)
├── requirements.txt      # Python dependencies
├── deepseek-author.md    # LLM prompt research for text generation
├── opus4_5-author.md     # Human authorship enhancement system prompt
└── model_guide.md        # Quick reference guide
```

## 📝 Prompt Engineering Research

The project includes research into LLM prompt design for studying text generation patterns:

### `deepseek-author.md`
Research prompt template exploring:
- **Perplexity Maximization** - Techniques for varied token selection
- **Burstiness Engineering** - Sentence length and rhythm variation
- **Temporal Anchoring** - Grounding text in contemporary context
- **Stylometric Diversity** - Part-of-speech and syntax variations

### `opus4_5-author.md`
Human Authorship Enhancement Engine prompt focusing on:
- **Natural Rhythm** - Variable sentence structures and deliberate breaks
- **Cognitive Markers** - Thought process indicators ("At first glance...", "What surprised me...")
- **Subjective Anchoring** - Personal observations and interpretive comments
- **Self-Critique Loop** - Iterative refinement for mechanical pattern reduction
- **Hybrid Authorship Model** - AI as brainstorming partner, human as final author

### `model_guide.md`
Quick reference documentation covering:
- Installation steps
- Basic API usage examples
- Training workflow with custom datasets
- Command line interface options

## 🔬 Core Components

### 1. LinguisticAnalyzer
Comprehensive text analysis including:
- Sentence statistics (length, variation, burstiness)
- Lexical diversity metrics (TTR, MSTTR, HD-D, VOCD)
- Syntactic complexity (clause depth, coordination ratios)
- Punctuation pattern analysis

### 2. PerplexityCalculator
Token probability analysis using transformer models:
- GPT-2 based perplexity scoring
- Sliding window analysis
- Cross-entropy calculations

### 3. TextTransformer
Rule-based text transformation for studying detector behavior:
- Synonym substitution
- Sentence structure variation
- Style normalization

### 4. TurniLab (Main Class)
Unified interface combining all components:
- Full feature extraction pipeline
- Trainable classifier (Logistic Regression, Random Forest, SVM)
- Model persistence and loading

## 📊 Feature Extraction

The framework extracts 50+ linguistic features:

| Category | Features |
|----------|----------|
| Sentence Stats | count, avg_length, std, cv, min, max |
| Lexical Diversity | TTR, MSTTR, HD-D, MATTR, VOCD |
| Syntactic | clauses_per_sentence, coordination_ratio |
| Punctuation | comma, period, question, semicolon frequency |
| Readability | Flesch-Kincaid, complexity scores |
| Perplexity | mean, std, max, min (if transformers available) |

## 🎯 Training Your Model

```python
from model import TurniLab

tl = TurniLab()

# Load your dataset
with open("human_texts.txt", "r") as f:
    human_texts = [line.strip() for line in f if line.strip()]

with open("ai_texts.txt", "r") as f:
    ai_texts = [line.strip() for line in f if line.strip()]

# Train and evaluate
metrics = tl.train_on_dataset(human_texts, ai_texts)

print(f"Cross-validation AUC: {metrics['cv_mean_auc']:.3f}")
print(f"Test Accuracy: {metrics['test_accuracy']:.3f}")
print(f"Test AUC: {metrics['test_auc']:.3f}")

# Save model
tl.save_system("my_model/")

# Load later
tl2 = TurniLab()
tl2.load_system("my_model/")
```

## 📈 Sample Output

```
Analysis Results:
├── Text Length: 1250 characters
├── Word Count: 215 words
├── Sentence Count: 12
├── Avg Sentence Length: 17.9 words
├── Burstiness (CV): 0.42
├── Lexical Diversity (TTR): 0.68
├── Perplexity (mean): 45.3
└── AI Score: 0.73
```

## ⚙️ Requirements

```
nltk>=3.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
pandas>=2.0.0
torch>=2.0.0
transformers>=4.30.0
lexicalrichness>=0.2.0
```

## 🔧 Configuration

### Optional: GPU Acceleration
For faster perplexity calculations with large texts:
```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### NLTK Data
The framework automatically downloads required NLTK data:
- `punkt` (sentence tokenizer)
- `averaged_perceptron_tagger` (POS tagging)
- `stopwords` (lexical analysis)

## 📚 Research Applications

- **Detection System Analysis**: Understanding classifier decision boundaries
- **Stylometry Research**: Quantifying writing style characteristics
- **NLP Education**: Teaching linguistic feature extraction
- **Corpus Linguistics**: Large-scale text pattern analysis

## ⚠️ Ethical Guidelines

This framework is designed for **academic research purposes only**:

1. **Transparency**: Always disclose AI involvement in content creation
2. **Academic Integrity**: Do not use for bypassing plagiarism detection
3. **Responsible Research**: Focus on improving detection systems, not evading them
4. **Institutional Compliance**: Follow your institution's policies on AI tool usage

## 🤝 Contributing

Contributions are welcome for:
- Additional linguistic feature extractors
- Improved perplexity calculation methods
- Multi-language support
- Documentation improvements

## 📄 License

MIT License - See LICENSE file for details.

## 📖 Citation

If you use this framework in academic research:

```bibtex
@software{turnilab2025,
  title={TurniLab: AI Text Detection Research Framework},
  year={2025},
  url={https://github.com/username/turnilab}
}
```

## 🔗 Related Resources

- [NLTK Documentation](https://www.nltk.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Scikit-learn](https://scikit-learn.org/)

---

<p align="center">
  <i>Built for understanding AI text detection mechanisms through rigorous academic research.</i>
</p>