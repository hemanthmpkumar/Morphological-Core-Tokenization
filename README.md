# Morphological-Core Tokenization (MCT)

**A Novel Approach to Preserve Semantic Integrity in Large Language Models**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

## 📄 Abstract

Subword tokenization techniques like Byte-Pair Encoding (BPE) often fragment words into semantically hollow units, forcing models to expend capacity on deciphering basic morphology. This repository contains the official implementation of **Morphological-Core Tokenization (MCT)**, a hybrid algorithm that preserves the morphological core of a word while applying constrained subword segmentation to affixes.

MCT includes a stochastic dropout mechanism to regularize dependency on morphological analyzers. Empirical results on WMT14 (De-En) and arXiv summarization show that MCT outperforms BPE and is computationally more efficient than token-free baselines like ByT5.

## 🚀 Key Features

* **Morphological Core Identification:** Identifies and preserves word stems using a linguistic database (e.g., WordNet).
* **Constrained Affix Segmentation:** Applies BPE-style merges only to prefixes and suffixes, ensuring the root remains atomic.
* **Stochastic Regularization:** Implements "Stem Dropout" ($p_{drop}$) to improve robustness against analyzer imperfections.

## 🛠️ Installation

1. Clone this repository:
   ```bash
   git clone git@github.com:hemanthmpkumar/Morphological-Core-Tokenization.git
   cd Morphological-Core-Tokenization
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   *Note: This project relies on `nltk` (for WordNet) and standard deep learning libraries (PyTorch/TensorFlow).*

3. Download NLTK data (if using the default analyzer):
   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

## 💻 Usage

### 1. Training the Tokenizer
MCT requires a pre-processing step to build the vocabulary by identifying stems in your corpus.

```python
from mct_tokenizer import MCTTokenizer

# Initialize MCT with a target vocabulary size and dropout rate
tokenizer = MCTTokenizer(vocab_size=32000, p_drop=0.05)

# Train on your corpus (e.g., C4, WMT14)
tokenizer.train(files=["path/to/corpus.txt"])

# Save the tokenizer
tokenizer.save_model("mct_vocab")
```

### 2. Tokenization Example
Comparison of standard segmentation vs. MCT:

```python
text = "unnecessarily running"
tokens = tokenizer.encode(text)

# Standard BPE might output: ["un", "necess", "arily", "run", "ning"]
# MCT outputs (preserving core): ["un", "necessary", "ly", "run", "ning"]
```

## 📊 Reproduction of Results

To reproduce the experiments described in the paper, follow the scripts in the `experiments/` folder.

### Machine Translation (WMT14 De-En)
We trained a 6-layer Transformer from scratch. The MCT-based model achieved a **29.0 BLEU** score, outperforming BPE (27.5) and ByT5-small (28.3).

| Model | BLEU Score |
| :--- | :--- |
| BPE (Baseline) | 27.5 |
| WordPiece | 27.8 |
| ByT5-small | 28.3 |
| **MCT (Ours)** | **29.0** |

### Abstractive Summarization (arXiv)
MCT achieves higher ROUGE scores by effectively handling specialized terminology.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :--- | :--- | :--- | :--- |
| BPE | 40.1 | 17.5 | 36.5 |
| **MCT (Ours)** | **41.0** | **18.2** | **37.4** |

## ⚙️ Hyperparameters

The default hyperparameters used for the main results:
* **Learning Rate:** 1e-4
* **Batch Size:** 256
* **Optimizer:** AdamW
* **Stem Dropout ($p_{drop}$):** 0.05

## 📚 Citation

If you use MCT in your research, please cite our paper:

```bibtex
@article{papachappa2025mct,
  title={Morphological-Core Tokenization: A Novel Approach to Preserve Semantic Integrity in Large Language Models},
  author={Papachappa, Hemanth Manchabale},
  journal={Proceedings of [Conference/Journal Name]},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Hemanth Manchabale Papachappa - [hemanthmpkumar123@gmail.com](mailto:hemanthmpkumar123@gmail.com)
