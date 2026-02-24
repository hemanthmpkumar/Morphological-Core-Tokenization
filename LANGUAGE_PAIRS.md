# Multi-Language NMT Training Guide

This guide explains how to configure the NMT training pipeline for different languages and tokenizer configurations.

## Quick Start

### Default Configuration (de-en, fi-en)
```bash
cd ~/mct_tokenization
source .venv/bin/activate
python3 scripts/train_nmt_models.py
```

Trains 20 configurations:
- Models: small, medium
- Tokenizers: BPE_32K, MCT_Full, MCT_NoDrop, MCT_NoBoundary, MCT_NoMorphology  
- Languages: de-en, fi-en

Expected time: **2-4 hours** on single T4 GPU

---

## Custom Language Pairs

### All 10 Languages (Longest Training)
```bash
export MCT_LANG_PAIRS="de-en,fr-en,ru-en,cs-en,tr-en,fi-en,zh-en,ja-en,ar-en,sw-en"
python3 scripts/train_nmt_models.py
```

**Note**: Arabic (ar), Swahili (sw), and some others may not be in public WMT. Script will skip unavailable pairs.

Estimated configurations: ~100 (10 languages × 5 tokenizers × 2 sizes)  
**Expected time: 20-30 hours** on single T4

---

### Reduced Tokenizers (Faster)
For rapid iteration, reduce tokenizers to just baseline + best variant:

```bash
export MCT_TOKENIZERS="BPE_32K,MCT_Full"
export MCT_LANG_PAIRS="de-en,fr-en,ru-en,fi-en"
python3 scripts/train_nmt_models.py
```

Configurations: 4 languages × 2 tokenizers × 2 sizes = **16**  
**Expected time: 2-3 hours**

---

### Small Models Only
For even faster experiments:

```bash
# Edit scripts/train_nmt_models.py, line ~427:
# Change: model_sizes = ['small', 'medium']
# To:     model_sizes = ['small']

export MCT_LANG_PAIRS="de-en,fr-en,ru-en,fi-en"
python3 scripts/train_nmt_models.py
```

Configurations: 4 languages × 5 tokenizers × 1 size = **20**  
**Expected time: 2-3 hours**

---

## Recommended Configurations

### For Paper (Comprehensive)
```bash
export MCT_LANG_PAIRS="de-en,fr-en,ru-en,fi-en,cs-en,zh-en,ja-en"
export MCT_TOKENIZERS="BPE_32K,MCT_Full,MCT_NoDrop"
python3 scripts/train_nmt_models.py
```
- 7 languages (Latin + non-Latin, morphologically diverse)
- 3 tokenizers (baseline + core variants)
- 2 model sizes
- **Configurations: 42, Time: 5-7 hours**

### For Quick Validation
```bash
export MCT_LANG_PAIRS="de-en,zh-en"
export MCT_TOKENIZERS="BPE_32K,MCT_Full"
python3 scripts/train_nmt_models.py
```
- 2 languages (morphologically different)
- 2 tokenizers
- 2 model sizes  
- **Configurations: 8, Time: 1 hour**

---

## Available Language Pairs

### Well-supported in WMT (recommended)
- **de-en**: German-English (WMT14, morphologically rich)
- **fr-en**: French-English (WMT14)
- **cs-en**: Czech-English (WMT14, morphologically very rich)
- **ru-en**: Russian-English (WMT14, Cyrillic)
- **fi-en**: Finnish-English (WMT16, agglutinative)
- **tr-en**: Turkish-English (WMT16, agglutinative)
- **zh-en**: Chinese-English (WMT17, logographic)
- **ja-en**: Japanese-English (WMT17, mixed scripts)
- **ro-en**: Romanian-English (WMT14)
- **et-en**: Estonian-English (WMT18)

### Limited/Unavailable Support
- **ar-en**: Arabic (not in standard WMT)
- **hi-en**: Hindi (not in standard WMT)
- **sw-en**: Swahili (not in standard WMT)

To use unsupported pairs, you'll need to provide custom datasets.

---

## Downloading Datasets Separately

To pre-download datasets before training:

### Download Default Pairs
```bash
python3 scripts/download_datasets.py
```

### Download Custom Pairs
```bash
export MCT_LANG_PAIRS="de-en,fr-en,ru-en,zh-en,ja-en,fi-en"
python3 scripts/download_datasets.py
```

Datasets are cached in `data/raw/`

---

## Environment Variables Reference

| Variable | Values | Default | Example |
|----------|--------|---------|---------|
| `MCT_LANG_PAIRS` | comma-separated codes | `de-en,fi-en` | `de-en,fr-en,zh-en` |
| `MCT_TOKENIZERS` | comma-separated names | All 5 variants | `BPE_32K,MCT_Full` |
| `MCT_DEVICE` | `cuda:0`, `cpu`, `mps` | CUDA if available | `export MCT_DEVICE=cuda:0` |

---

## Runtime Estimation

For single T4 GPU:

| Configuration | Configs | Time |
|---------------|---------|------|
| Default (2 langs, 5 tok, 2 models) | 20 | 2-4 hours |
| 4 languages, 3 tokenizers, 2 models | 24 | 3-4 hours |
| 5 languages, 5 tokenizers, 2 models | 50 | 8-12 hours |
| 10 languages, 5 tokenizers, 2 models | 100 | 20-30 hours |

**Note**: Actual time depends on:
- Dataset size
- Sequence length
- GPU memory utilization
- Model size (small = faster, medium = ~1.5x slower)

---

## Multi-GPU Training

For faster training with multiple GPUs, consider:
1. Running different language pairs on different GPUs in parallel
2. Using distributed training (would require script modifications)

---

## Results Location

After training completes:
- **Results**: `results/nmt_training_results.json`
- **Checkpoints**: `models/nmt_checkpoints/<size>_<tokenizer>_<lang_pair>/`
- **Analysis**: `results/analysis/`

---

## Example: Full Reproduction

```bash
# Setup
cd ~/mct_tokenization
source .venv/bin/activate

# Download all supported languages
export MCT_LANG_PAIRS="de-en,fr-en,ru-en,cs-en,fi-en,tr-en,zh-en,ja-en,ro-en"
python3 scripts/download_datasets.py

# Train with recommended config
export MCT_TOKENIZERS="BPE_32K,MCT_Full,MCT_NoBoundary"
python3 scripts/train_nmt_models.py

# Analyze results
python3 scripts/analyze_nmt_results.py

# View report
cat results/analysis/NMT_RESULTS_ANALYSIS.md
```

This trains 54 configurations (~7-8 hours), showing MCT's benefits across diverse typologically different languages.

