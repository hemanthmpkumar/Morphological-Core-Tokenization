# MCT Tokenization Scripts

This directory contains bash scripts for running various stages of the MCT (Morphological-Core Tokenization) experiment pipeline.

## Quick Start

```bash
# Run complete experiment pipeline (all 15 steps)
./scripts/run_complete_experiment.sh
```

## Available Scripts

### Main Pipeline

#### `run_complete_experiment.sh` ⭐ NEW
Complete end-to-end experiment execution with all 15 steps:

1. Verify project structure
2. Install dependencies
3. Download Stanza linguistic models
4. Download NLTK data
5. Run validation
6. Build vocabulary
7. Test tokenizer
8. Run unit tests
9. Validate configuration
10. Test morphological analyzer
11. Test constrained BPE
12. Run translation evaluation
13. Run summarization evaluation
14. Run morphology evaluation
15. Collect and document results

**Usage:**
```bash
./scripts/run_complete_experiment.sh
```

**Output:**
- Comprehensive log file: `results/experiment_YYYYMMDD_HHMMSS.log`
- Evaluation results: `results/*.json`
- Full report: `results/EVALUATION_REPORT.md`

---

### Legacy Scripts

#### `run_full_pipeline.sh`
Original full pipeline script for MCT model training and evaluation.

**Features:**
- Environment setup
- Data preparation
- Model training
- Evaluation on translation, summarization, and morphology tasks
- Results compilation

**Usage:**
```bash
./scripts/run_full_pipeline.sh
```

---

#### `setup_env.sh`
Sets up the Python environment with required dependencies.

**Features:**
- Creates/activates virtual environment
- Installs PyTorch, transformers, and other dependencies
- Downloads linguistic models

**Usage:**
```bash
./scripts/setup_env.sh
```

---

#### `build_vocab.sh`
Builds vocabulary from corpus using stem extraction.

**Features:**
- Extracts morphological stems
- Computes stem frequencies
- Saves vocabulary to `data/processed/stem_frequencies.json`

**Usage:**
```bash
./scripts/build_vocab.sh
```

---

#### `run_all_experiments.sh`
Runs all evaluation experiments independently.

**Includes:**
- Translation evaluation (de-en)
- Summarization evaluation
- Morphological analysis evaluation
- Comparison with baseline tokenizers

**Usage:**
```bash
./scripts/run_all_experiments.sh
```

---

#### `run_all_evals.sh`
Runs evaluation-only tasks without retraining.

**Features:**
- Quick evaluation on pre-trained models
- Results collection
- Metric computation

**Usage:**
```bash
./scripts/run_all_evals.sh
```

---

#### `train_baseline.sh`
Trains baseline BPE tokenizer for comparison.

**Features:**
- Standard BPE tokenization
- Evaluation on same tasks
- Baseline performance metrics

**Usage:**
```bash
./scripts/train_baseline.sh
```

---

#### `ablation_study.sh`
Runs ablation studies to analyze component importance.

**Tests:**
- Impact of morphological analysis
- Impact of stem dropout (p_drop)
- Impact of vocabulary size
- Constraint enforcement effects

**Usage:**
```bash
./scripts/ablation_study.sh
```

---

### Data Preparation

#### `DataPrepare/prepare_data.sh`
Prepares datasets for training and evaluation.

**Features:**
- Downloads datasets
- Preprocesses text
- Splits train/validation/test
- Creates vocabulary

**Usage:**
```bash
./scripts/DataPrepare/prepare_data.sh
```

---

#### `DataPrepare/prepare_data.py`
Python script for data preparation tasks.

**Features:**
- Custom corpus processing
- Tokenization
- Vocabulary building
- Format conversion

**Usage:**
```bash
python scripts/DataPrepare/prepare_data.py --corpus <path> --output <path>
```

---

## Pipeline Architecture

```
run_complete_experiment.sh
├── Environment Setup
│   ├── Check virtual environment
│   ├── Verify project structure
│   └── Install dependencies
├── Resource Downloading
│   ├── Download Stanza models (en, de, fi)
│   └── Download NLTK data (wordnet, omw-1.4)
├── Validation & Setup
│   ├── Run validation script
│   └── Build vocabulary
├── Component Testing
│   ├── Test tokenizer
│   ├── Test morphological analyzer
│   └── Test constrained BPE
├── Unit Tests
│   └── Run pytest (34 tests)
├── Configuration
│   └── Validate all configs
├── Evaluations
│   ├── Translation (de-en)
│   ├── Summarization (arXiv)
│   └── Morphology (MorphoBench)
└── Results Collection
    ├── Compile metrics
    ├── Generate report
    └── Save logs
```

## Configuration

### Environment Variables

- `PYTHONPATH`: Should include project root
- `CUDA_VISIBLE_DEVICES`: GPU selection (if applicable)
- `MCT_DEVICE`: Force a specific backend (e.g. `cuda:0`, `cpu`, `mps`).  By default the
  project prefers CUDA devices and will **not** fall back to Apple MPS unless
  this variable is set to `mps` explicitly.
- `MCT_VOCAB_SIZE`: Vocabulary size (default: 32000)
- `MCT_DROPOUT`: Stem dropout probability (default: 0.05)

### Output Directories

- `results/`: Evaluation results and reports
- `data/processed/`: Processed vocabularies and features
- `configs/`: Model configurations
- `logs/`: Experiment logs

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.3+
- Stanza 1.5+
- NLTK 3.8+
- YAML support
- Bash 4.0+

Install requirements with:
```bash
pip install -r requirements.txt
```

## Logging

All scripts generate detailed logs:

- Main log: `results/experiment_YYYYMMDD_HHMMSS.log`
- Component logs: Individual task outputs
- Error handling: Detailed error messages with context

View logs with:
```bash
tail -f results/experiment_*.log
```

## Results

Each evaluation generates:

- **JSON Results**: Machine-readable metrics
- **Evaluation Report**: Markdown summary with findings
- **Component Outputs**: Individual evaluation files

Generated files:
```
results/
├── EVALUATION_REPORT.md
├── comprehensive_results.json
├── translation_results.json
├── summarization_results.json
├── morphology_results.json
└── experiment_YYYYMMDD_HHMMSS.log
```

## Advanced Usage

### Run Specific Steps

To run individual steps, directly execute from scripts:

```bash
# Just setup environment
./scripts/setup_env.sh

# Just run evaluations
./scripts/run_all_evals.sh

# Just run tests
source .venv/bin/activate && pytest tests/ -v

# Just build vocabulary
./scripts/build_vocab.sh
```

### Parallel Execution

Some independent experiments can run in parallel:

```bash
./scripts/run_all_experiments.sh &
./scripts/ablation_study.sh &
wait
```

### Custom Configurations

Edit configuration files before running:

```bash
# Edit model config
vim configs/small_125m.yaml

# Edit tokenizer config
vim configs/tokenizer_config.json

# Then run pipeline
./scripts/run_complete_experiment.sh
```

## Retrieving Trained Checkpoints

After an experiment completes the serialized models are placed under
`models/nmt_checkpoints/` with a subdirectory for each configuration.
Here are a few convenient ways to download them:

- **Direct copy (local runs)**

  ```bash
  cp -r models/nmt_checkpoints ~/my_local_models/
  ```

- **From a Google Cloud VM**

  ```bash
  # pull the whole directory to your workstation
  gcloud compute scp --recurse mct-t4:~/mct_tokenization/models/nmt_checkpoints ./local_models

  # or archive & use Cloud Storage
  # (run on the VM)
  tar czf mct_models.tar.gz models/nmt_checkpoints
  gsutil cp mct_models.tar.gz gs://my-bucket/experiments/

  # on your laptop
  gsutil cp gs://my-bucket/experiments/mct_models.tar.gz .
  tar xzf mct_models.tar.gz
  ```

- **Upload to Hugging Face Hub**

  ```bash
  pip install huggingface_hub
  python - <<'PY'
  from huggingface_hub import Repository, upload_file
  repo = Repository("username/mct-small")
  repo.git_pull()
  upload_file(
      "models/nmt_checkpoints/small_MCT_Full_de-en/checkpoint.pt",
      path_in_repo="pytorch_model.bin",
      repo_id="username/mct-small"
  )
  PY
  ```

Once the files are in a shared location, colleagues can download them
with `hf_hub_download` or standard `scp`/`gsutil` commands.

## Troubleshooting

### Script Won't Run

```bash
# Make sure script is executable
chmod +x scripts/*.sh

# Check shebang line
head -1 scripts/run_complete_experiment.sh
```

### Virtual Environment Issues

```bash
# Recreate virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Missing Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Install additional evaluation packages
pip install pytest pytest-cov sacrebleu rouge_score
```

### GPU / MPS Device Selection

The scripts automatically choose a CUDA GPU when one is available.  On
Apple‑silicon machines the MPS backend is *not* selected unless you explicitly
force it using the `MCT_DEVICE` environment variable.

```bash
# Check what device PyTorch will pick
python -c "from src.utils.device import get_compute_device; print(get_compute_device())"

# Force CPU-only (ignore GPUs/MPS)
MCT_DEVICE=cpu ./scripts/run_complete_experiment.sh

# Force MPS for testing on Apple silicon
MCT_DEVICE=mps ./scripts/run_complete_experiment.sh
```

## Benchmarks

Expected execution times (on CPU):
- Environment Setup: ~2-3 minutes
- Download Models: ~5 minutes
- Vocabulary Building: ~3 minutes
- Unit Tests: ~5 seconds
- Component Tests: ~30 seconds
- Evaluations: ~2 minutes
- Results Collection: ~1 minute

**Total: ~15-20 minutes**

## Citation

If you use these scripts in your research, please cite:

```bibtex
@inproceedings{mct2024,
  title={MCT: Morphological-Core Tokenization for Efficient Language Models},
  author={...},
  year={2024}
}
```

## Support

For issues or questions:
1. Check the experiment log: `results/experiment_*.log`
2. Review error messages carefully
3. Verify all dependencies are installed
4. Check project structure is intact
5. See troubleshooting section above

## License

See LICENSE file in project root.
