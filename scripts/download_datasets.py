#!/usr/bin/env python3
"""
Download and prepare real datasets for rigorous MCT evaluation.
Handles: WMT14 (De-En), WMT16 (Fi-En), arXiv, PubMed
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def download_wmt14_de_en(output_dir: Path) -> Tuple[Path, Path, Path]:
    """Download WMT14 German-English dataset."""
    logger.info("Downloading WMT14 German-English...")
    
    try:
        from datasets import load_dataset
        
        # Load from HuggingFace datasets
        dataset = load_dataset('wmt14', 'de-en', cache_dir=str(output_dir))
        
        # Extract and save
        train_path = output_dir / 'wmt14_de_en.train.jsonl'
        test_path = output_dir / 'wmt14_de_en.newstest2014.jsonl'
        
        logger.info(f"WMT14 De-En: {len(dataset['train'])} train pairs")
        logger.info(f"WMT14 De-En: {len(dataset['test'])} test pairs")
        
        # Save in JSONL format for easy processing
        with open(train_path, 'w') as f:
            for example in dataset['train']:
                record = {
                    'de': example['translation']['de'],
                    'en': example['translation']['en']
                }
                f.write(json.dumps(record) + '\n')
        
        with open(test_path, 'w') as f:
            for example in dataset['test']:
                record = {
                    'de': example['translation']['de'],
                    'en': example['translation']['en']
                }
                f.write(json.dumps(record) + '\n')
        
        logger.info(f"Saved WMT14 to {output_dir}")
        return train_path, test_path, output_dir / 'wmt14_de_en.vocab'
        
    except Exception as e:
        logger.error(f"Error downloading WMT14: {e}")
        return None, None, None


def download_wmt16_fi_en(output_dir: Path) -> Tuple[Path, Path, Path]:
    """Download WMT16 Finnish-English dataset."""
    logger.info("Downloading WMT16 Finnish-English...")
    
    try:
        from datasets import load_dataset
        
        # Load from HuggingFace datasets
        dataset = load_dataset('wmt16', 'fi-en', cache_dir=str(output_dir))
        
        train_path = output_dir / 'wmt16_fi_en.train.jsonl'
        test_path = output_dir / 'wmt16_fi_en.newstest2016.jsonl'
        
        logger.info(f"WMT16 Fi-En: {len(dataset['train'])} train pairs")
        logger.info(f"WMT16 Fi-En: {len(dataset['test'])} test pairs")
        
        with open(train_path, 'w') as f:
            for example in dataset['train']:
                record = {
                    'fi': example['translation']['fi'],
                    'en': example['translation']['en']
                }
                f.write(json.dumps(record) + '\n')
        
        with open(test_path, 'w') as f:
            for example in dataset['test']:
                record = {
                    'fi': example['translation']['fi'],
                    'en': example['translation']['en']
                }
                f.write(json.dumps(record) + '\n')
        
        logger.info(f"Saved WMT16 to {output_dir}")
        return train_path, test_path, output_dir / 'wmt16_fi_en.vocab'
        
    except Exception as e:
        logger.error(f"Error downloading WMT16: {e}")
        return None, None, None


def download_arxiv_sample(output_dir: Path, num_samples: int = 50000) -> Path:
    """Download arXiv abstracts for domain adaptation analysis."""
    logger.info(f"Downloading arXiv sample ({num_samples} abstracts)...")
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset('arxiv', cache_dir=str(output_dir))
        
        arxiv_path = output_dir / 'arxiv_abstracts.jsonl'
        
        logger.info(f"Total arXiv records available: {len(dataset['train'])}")
        
        # Save first N abstracts
        with open(arxiv_path, 'w') as f:
            for i, example in enumerate(dataset['train']):
                if i >= num_samples:
                    break
                record = {
                    'title': example['title'],
                    'abstract': example['abstract'],
                    'categories': example['categories']
                }
                f.write(json.dumps(record) + '\n')
        
        logger.info(f"Saved {num_samples} arXiv abstracts to {arxiv_path}")
        return arxiv_path
        
    except Exception as e:
        logger.error(f"Error downloading arXiv: {e}")
        logger.info("Note: arXiv dataset is optional for core experiments")
        return None


def download_pubmed_sample(output_dir: Path, num_samples: int = 50000) -> Path:
    """Download PubMed abstracts for domain adaptation analysis."""
    logger.info(f"Downloading PubMed sample ({num_samples} abstracts)...")
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset('pubmed', cache_dir=str(output_dir))
        
        pubmed_path = output_dir / 'pubmed_abstracts.jsonl'
        
        logger.info(f"Total PubMed records available: {len(dataset['train'])}")
        
        # Save first N abstracts
        with open(pubmed_path, 'w') as f:
            for i, example in enumerate(dataset['train']):
                if i >= num_samples:
                    break
                record = {
                    'title': example['title'],
                    'abstract': example['abstract'],
                    'medline_id': example.get('medline_id', '')
                }
                f.write(json.dumps(record) + '\n')
        
        logger.info(f"Saved {num_samples} PubMed abstracts to {pubmed_path}")
        return pubmed_path
        
    except Exception as e:
        logger.error(f"Error downloading PubMed: {e}")
        logger.info("Note: PubMed dataset is optional for core experiments")
        return None


def verify_downloads(output_dir: Path) -> Dict[str, bool]:
    """Verify all datasets downloaded successfully."""
    files = {
        'wmt14_de_en.train.jsonl': output_dir / 'wmt14_de_en.train.jsonl',
        'wmt14_de_en.newstest2014.jsonl': output_dir / 'wmt14_de_en.newstest2014.jsonl',
        'wmt16_fi_en.train.jsonl': output_dir / 'wmt16_fi_en.train.jsonl',
        'wmt16_fi_en.newstest2016.jsonl': output_dir / 'wmt16_fi_en.newstest2016.jsonl',
    }
    
    results = {}
    for name, path in files.items():
        exists = path.exists()
        size_mb = (path.stat().st_size / (1024*1024)) if exists else 0
        results[name] = exists
        status = f"✓ ({size_mb:.1f} MB)" if exists else "✗ MISSING"
        logger.info(f"  {name}: {status}")
    
    return results


def create_dataset_manifest(output_dir: Path, stats: Dict) -> None:
    """Create manifest of downloaded datasets."""
    manifest = {
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'datasets': {
            'wmt14_de_en': {
                'train_path': str(output_dir / 'wmt14_de_en.train.jsonl'),
                'test_path': str(output_dir / 'wmt14_de_en.newstest2014.jsonl'),
                'language_pair': 'de-en',
                'status': 'ready' if stats.get('wmt14_de_en.train.jsonl') else 'missing'
            },
            'wmt16_fi_en': {
                'train_path': str(output_dir / 'wmt16_fi_en.train.jsonl'),
                'test_path': str(output_dir / 'wmt16_fi_en.newstest2016.jsonl'),
                'language_pair': 'fi-en',
                'status': 'ready' if stats.get('wmt16_fi_en.train.jsonl') else 'missing'
            },
            'arxiv': {
                'path': str(output_dir / 'arxiv_abstracts.jsonl'),
                'type': 'optional_analysis',
                'status': 'ready' if (output_dir / 'arxiv_abstracts.jsonl').exists() else 'skipped'
            },
            'pubmed': {
                'path': str(output_dir / 'pubmed_abstracts.jsonl'),
                'type': 'optional_analysis',
                'status': 'ready' if (output_dir / 'pubmed_abstracts.jsonl').exists() else 'skipped'
            }
        }
    }
    
    manifest_path = output_dir / 'dataset_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"\nDataset manifest saved to {manifest_path}")


def main():
    """Main download routine."""
    output_dir = Path(__file__).parent.parent / 'data' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("DOWNLOADING REAL DATASETS FOR MCT EXPERIMENTS")
    logger.info("=" * 80)
    logger.info(f"\nOutput directory: {output_dir}\n")
    
    # Download core datasets
    logger.info("[1] Core datasets (WMT14, WMT16)...")
    download_wmt14_de_en(output_dir)
    download_wmt16_fi_en(output_dir)
    
    # Download optional analysis datasets
    logger.info("\n[2] Optional analysis datasets (arXiv, PubMed)...")
    download_arxiv_sample(output_dir, num_samples=50000)
    download_pubmed_sample(output_dir, num_samples=50000)
    
    # Verify
    logger.info("\n[3] Verifying downloads...")
    stats = verify_downloads(output_dir)
    
    # Create manifest
    logger.info("\n[4] Creating dataset manifest...")
    create_dataset_manifest(output_dir, stats)
    
    # Summary
    core_ready = all([
        stats.get('wmt14_de_en.train.jsonl'),
        stats.get('wmt16_fi_en.train.jsonl')
    ])
    
    logger.info("\n" + "=" * 80)
    if core_ready:
        logger.info("✓ DATASETS READY FOR EXPERIMENTS")
        logger.info("\nNext: python3 scripts/train_baselines.py")
    else:
        logger.error("✗ CRITICAL DATASETS MISSING")
        logger.error("Cannot proceed without WMT14 and WMT16 data")
    logger.info("=" * 80)
    
    return core_ready


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
