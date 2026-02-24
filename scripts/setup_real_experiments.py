#!/usr/bin/env python3
"""
Setup real experimental datasets and baselines for MCT evaluation.
Designed for rigorous, large-scale experiments with strong scientific insights.

Resources: 32GB GPU, 600GB disk, ASAP timeline
Philosophy: Theory-driven analysis over metric chasing
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def setup_directories() -> Dict[str, Path]:
    """Create directory structure for experiments."""
    base_dir = Path(__file__).parent.parent
    dirs = {
        'data_raw': base_dir / 'data' / 'raw',
        'data_processed': base_dir / 'data' / 'processed',
        'data_tokenized': base_dir / 'data' / 'tokenized',
        'models': base_dir / 'models',
        'checkpoints': base_dir / 'models' / 'checkpoints',
        'results': base_dir / 'results',
        'analysis': base_dir / 'results' / 'analysis',
        'logs': base_dir / 'logs',
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified directory: {path}")
    
    return dirs


def verify_disk_space(required_gb: int = 300) -> bool:
    """Verify sufficient disk space for experiments."""
    import shutil
    stat = shutil.disk_usage('/')
    available_gb = stat.free / (1024**3)
    logger.info(f"Available disk space: {available_gb:.1f} GB")
    
    if available_gb < required_gb:
        logger.warning(f"Only {available_gb:.1f} GB available, need {required_gb} GB")
        return False
    return True


def verify_gpu_memory() -> bool:
    """Verify CUDA GPU availability and memory.

    This helper deliberately ignores Apple MPS devices; the expectation for
    the "real" experiments is that they run on a CUDA-enabled server or
    cluster.  If the default device is not CUDA we log an error and return
    ``False`` so upstream logic knows that only CPU fallback is possible.
    """
    try:
        from src.utils.device import get_compute_device
        import torch

        device = get_compute_device()
        if device.type != "cuda":
            logger.error(f"GPU verification failed: device = {device} (CUDA required)")
            return False

        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {gpu_memory_gb:.1f} GB")

        if gpu_memory_gb < 20:
            logger.warning(f"GPU memory {gpu_memory_gb:.1f} GB is less than recommended 32GB")
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking GPU: {e}")
        return False


def create_experiment_config() -> Dict:
    """Create configuration for rigorous experiments."""
    config = {
        "experiment_name": "MCT_Real_Experiments_Q1_2026",
        "philosophy": "Theory-driven science over metric chasing",
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        
        "datasets": {
            "wmt14_de_en": {
                "name": "WMT14 German-English",
                "train_pairs": 700000,
                "test_set": "newstest2014",
                "language_pair": "de-en",
                "morphology": "high",
                "importance": "primary"
            },
            "wmt16_fi_en": {
                "name": "WMT16 Finnish-English",
                "train_pairs": 2200000,
                "test_set": "newstest2016",
                "language_pair": "fi-en",
                "morphology": "very_high",
                "importance": "secondary"
            },
            "arxiv_sample": {
                "name": "arXiv abstracts (technical domain)",
                "num_documents": 50000,
                "purpose": "domain adaptation analysis",
                "morphology": "medium",
                "importance": "analysis"
            },
            "pubmed_sample": {
                "name": "PubMed abstracts (biomedical domain)",
                "num_documents": 50000,
                "purpose": "domain adaptation analysis",
                "morphology": "medium",
                "importance": "analysis"
            }
        },
        
        "tokenization_variants": [
            {
                "name": "BPE_32K",
                "type": "baseline",
                "vocab_size": 32000,
                "description": "Standard BPE baseline"
            },
            {
                "name": "WordPiece_32K",
                "type": "baseline",
                "vocab_size": 32000,
                "description": "WordPiece baseline from HuggingFace"
            },
            {
                "name": "MCT_Full",
                "type": "proposed",
                "vocab_size": 32000,
                "morpheme_dropout": 0.05,
                "boundary_constraints": True,
                "morphological_analysis": True,
                "description": "Full MCT with all components"
            },
            {
                "name": "MCT_NoDrop",
                "type": "ablation",
                "vocab_size": 32000,
                "morpheme_dropout": 0.0,
                "boundary_constraints": True,
                "morphological_analysis": True,
                "description": "Ablation: no stem dropout"
            },
            {
                "name": "MCT_NoBoundary",
                "type": "ablation",
                "vocab_size": 32000,
                "morpheme_dropout": 0.05,
                "boundary_constraints": False,
                "morphological_analysis": True,
                "description": "Ablation: no boundary constraints"
            },
            {
                "name": "MCT_NoMorphology",
                "type": "ablation",
                "vocab_size": 32000,
                "morpheme_dropout": 0.05,
                "boundary_constraints": True,
                "morphological_analysis": False,
                "description": "Ablation: no morphological analysis (pure BPE + boundaries)"
            }
        ],
        
        "evaluation_metrics": {
            "translation": {
                "primary": ["BLEU", "ChrF"],
                "secondary": ["TER", "METEOR"],
                "custom": ["OOV_rate", "segmentation_quality", "morpheme_hit_rate"]
            },
            "tokenization": {
                "vocab_coverage": "% of training vocab in 32K tokens",
                "oov_rate": "% of test tokens not in vocab",
                "subword_length": "average subword length",
                "morpheme_alignment": "% of morpheme boundaries correctly preserved",
                "stem_dropout_impact": "% of words affected by dropout regularization"
            }
        },
        
        "experimental_design": {
            "runs_per_tokenizer": 3,
            "train_seed": [42, 123, 456],
            "test_set_size": 3000,
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 3,
            "warmup_steps": 4000,
            "statistical_testing": "paired t-test with p<0.05",
            "model_size_variants": ["small", "medium", "large"],
            "size_notes": "small=~125M, medium=~350M, large=~1B (approx params)"
        },
        
        "analysis_plan": {
            "morphological_breakdown": {
                "by_inflection_type": ["tense", "case", "number", "gender"],
                "by_derivation_type": ["suffix", "prefix", "circumfix"],
                "by_frequency": ["high_freq_words", "medium_freq", "low_freq_rare"]
            },
            "error_analysis": {
                "segmentation_errors": "tokens where MCT segmentation differs from BPE",
                "morpheme_dropout_effects": "cases where dropout helps vs. hurts",
                "analyzer_coverage": "% words where morphological analyzer found analysis",
                "fallback_patterns": "when and why fallback to BPE is necessary"
            },
            "ablation_insights": {
                "dropout_contribution": "how much dropout regularization helps?",
                "boundary_contribution": "impact of morpheme boundary constraints",
                "morphology_contribution": "value of linguistic analysis vs. pure BPE"
            }
        },
        
        "resources": {
            "gpu_memory_gb": 32,
            "disk_space_gb": 600,
            "estimated_compute_hours": 280,
            "estimated_data_size_gb": 150
        },
        "model_sizes": {
            "small": {
                "approx_params": "~125M",
                "encoder_layers": 6,
                "decoder_layers": 6,
                "d_model": 512,
                "num_heads": 8,
                "notes": "fast for local experiments and debugging"
            },
            "medium": {
                "approx_params": "~350M",
                "encoder_layers": 12,
                "decoder_layers": 12,
                "d_model": 768,
                "num_heads": 12,
                "notes": "recommended for main ablations (balance speed and capacity)"
            },
            "large": {
                "approx_params": "~1B",
                "encoder_layers": 24,
                "decoder_layers": 24,
                "d_model": 1024,
                "num_heads": 16,
                "notes": "high-capacity reference model; use on multi-GPU or cloud"
            }
        },
        
        "timeline": {
            "phase_1_setup": "Week 1 - Data download and baseline training",
            "phase_2_mct": "Week 2 - MCT training with ablations",
            "phase_3_analysis": "Week 3 - Detailed morphological analysis",
            "phase_4_paper": "Week 4 - Paper restructuring with insights"
        }
    }
    
    return config


def main():
    """Main setup routine."""
    logger.info("=" * 80)
    logger.info("MCT RIGOROUS EXPERIMENTS SETUP")
    logger.info("=" * 80)
    
    # Verify resources
    logger.info("\n[1] Verifying resources...")
    if not verify_disk_space(300):
        logger.warning("Disk space warning - continuing anyway")
    if not verify_gpu_memory():
        logger.warning("GPU memory check failed - proceeding without CUDA (will run on CPU)")
        # continue on CPU; training will be slower but pipeline remains runnable
    
    # Setup directories
    logger.info("\n[2] Setting up directories...")
    dirs = setup_directories()
    
    # Create experiment config
    logger.info("\n[3] Creating experiment configuration...")
    config = create_experiment_config()
    
    config_path = dirs['logs'] / 'experiment_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_path}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT SETUP COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nPhilosophy: {config['philosophy']}")
    logger.info(f"\nPrimary datasets:")
    for name, ds in config['datasets'].items():
        if ds.get('importance') == 'primary':
            logger.info(f"  - {ds['name']}: {ds.get('train_pairs', ds.get('num_documents'))} samples")
    
    logger.info(f"\nTokenization variants: {len(config['tokenization_variants'])} (2 baselines + 4 ablations)")
    
    logger.info(f"\nNext steps:")
    logger.info("  1. Run: python3 scripts/download_datasets.py")
    logger.info("  2. Run: python3 scripts/train_baselines.py")
    logger.info("  3. Run: python3 scripts/train_mct_variants.py")
    logger.info("  4. Run: python3 scripts/evaluate_all.py")
    logger.info("  5. Run: python3 scripts/analyze_morphology.py")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
