#!/usr/bin/env python3
"""
Download and prepare real datasets for rigorous MCT evaluation.
Handles: WMT14-17 multiple language pairs, arXiv, PubMed

Supported language pairs (with -en):
- Morphologically rich: de, fi, ru, cs, tr, ar, hi
- Simpler morphology: fr
- Non-Latin scripts: zh, ja, ar, hi, sw
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# WMT dataset availability mapping: lang_code -> wmt_year
WMT_AVAILABILITY = {
    'de': [14, 15, 16, 17],  # German
    'fr': [14, 15, 16, 17],  # French
    'cs': [14, 15, 16, 17],  # Czech
    'ru': [14, 15, 17],      # Russian
    'fi': [15, 16],          # Finnish
    'tr': [16, 17],          # Turkish
    'zh': [17, 18, 19, 20],  # Chinese
    'ja': [17, 18, 19, 21],  # Japanese
    'ro': [15, 16, 17],      # Romanian
    'et': [18, 19],          # Estonian
    'cs': [14, 15, 16, 17],  # Czech
    'de': [14, 15, 16, 17],  # German (duplicate key, will overwrite - fix below)
}

# Better mapping
WMT_AVAILABILITY = {
    'de': 14,   # Primary year
    'fr': 14,
    'cs': 14,
    'ru': 14,
    'fi': 16,
    'tr': 16,
    'zh': 17,
    'ja': 17,
    'ro': 14,
}


def download_wmt_pair(lang_pair: str, output_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Download any WMT language pair dynamically.
    
    Args:
        lang_pair: Language pair string like 'de-en', 'fr-en', etc.
        output_dir: Directory to save files
    
    Returns:
        Tuple of (train_path, test_path) or (None, None) if unavailable
    """
    from datasets import load_dataset
    
    src_lang, tgt_lang = lang_pair.split('-')
    src_lang = src_lang.strip()
    tgt_lang = tgt_lang.strip()
    
    logger.info(f"Downloading WMT dataset for {lang_pair}...")
    
    # Determine which WMT year to use
    wmt_years = WMT_AVAILABILITY.get(src_lang, [14, 15, 16, 17])
    if isinstance(wmt_years, int):
        wmt_years = [wmt_years]
    
    train_path = None
    test_path = None
    
    # Try each WMT year
    for wmt_year in sorted(wmt_years, reverse=True):
        try:
            logger.info(f"  Trying WMT{wmt_year} {lang_pair}...")
            dataset = load_dataset(f'wmt{wmt_year}', lang_pair, cache_dir=str(output_dir))
            
            # Get dataset splits
            if 'train' not in dataset:
                logger.warning(f"    WMT{wmt_year} has no 'train' split")
                continue
            
            if 'test' not in dataset:
                logger.warning(f"    WMT{wmt_year} has no 'test' split, using validation")
                test_split = 'validation' if 'validation' in dataset else 'train'
            else:
                test_split = 'test'
            
            logger.info(f"  ✓ Found: {len(dataset['train'])} train, {len(dataset[test_split])} test")
            
            # Determine test file naming (newstest, newsdev, etc.)
            test_key = f'newstest{wmt_year}' if wmt_year < 18 else f'wmt{wmt_year}'
            train_filename = f'wmt{wmt_year}_{src_lang}_{tgt_lang}.train.jsonl'
            test_filename = f'wmt{wmt_year}_{src_lang}_{tgt_lang}.test.jsonl'
            
            train_path = output_dir / train_filename
            test_path = output_dir / test_filename
            
            # Save in JSONL format
            with open(train_path, 'w') as f:
                for example in dataset['train']:
                    record = {
                        src_lang: example['translation'][src_lang],
                        tgt_lang: example['translation'][tgt_lang]
                    }
                    f.write(json.dumps(record) + '\n')
            
            with open(test_path, 'w') as f:
                for example in dataset[test_split]:
                    record = {
                        src_lang: example['translation'][src_lang],
                        tgt_lang: example['translation'][tgt_lang]
                    }
                    f.write(json.dumps(record) + '\n')
            
            logger.info(f"  Saved WMT{wmt_year} {lang_pair} to {output_dir}")
            return train_path, test_path
            
        except Exception as e:
            logger.debug(f"  WMT{wmt_year} not available: {e}")
            continue
    
    # If no WMT dataset found
    logger.error(f"Could not find {lang_pair} in any WMT dataset")
    logger.warning(f"Skipping {lang_pair} - not available in public WMT benchmarks")
    return None, None


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


def verify_downloads(output_dir: Path, lang_pairs: list) -> Dict[str, bool]:
    """Verify all datasets downloaded successfully."""
    results = {}
    
    for lang_pair in lang_pairs:
        src, tgt = lang_pair.split('-')
        # Find any JSONL files matching this language pair
        train_files = list(output_dir.glob(f'wmt*_{src}_{tgt}.train.jsonl'))
        test_files = list(output_dir.glob(f'wmt*_{src}_{tgt}.test.jsonl'))
        
        if train_files:
            train_path = train_files[0]
            exists = train_path.exists()
            size_mb = (train_path.stat().st_size / (1024*1024)) if exists else 0
            results[lang_pair] = exists
            status = f"✓ ({size_mb:.1f} MB)" if exists else "✗ MISSING"
            logger.info(f"  {train_path.name}: {status}")
        else:
            results[lang_pair] = False
            logger.warning(f"  {lang_pair}: ✗ NO FILES FOUND")
    
    return results


def create_dataset_manifest(output_dir: Path, stats: Dict, lang_pairs: list) -> None:
    """Create manifest of downloaded datasets."""
    manifest = {
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'language_pairs': lang_pairs,
        'datasets': {},
        'optional_datasets': {}
    }
    
    # Add core language pair datasets
    for lang_pair in lang_pairs:
        src, tgt = lang_pair.split('-')
        train_files = list(output_dir.glob(f'wmt*_{src}_{tgt}.train.jsonl'))
        test_files = list(output_dir.glob(f'wmt*_{src}_{tgt}.test.jsonl'))
        
        if train_files and test_files:
            manifest['datasets'][lang_pair] = {
                'train_path': str(train_files[0]),
                'test_path': str(test_files[0]),
                'status': 'ready'
            }
        else:
            manifest['datasets'][lang_pair] = {
                'status': 'unavailable'
            }
    
    # Add optional datasets
    arxiv_path = output_dir / 'arxiv_abstracts.jsonl'
    pubmed_path = output_dir / 'pubmed_abstracts.jsonl'
    
    if arxiv_path.exists():
        manifest['optional_datasets']['arxiv'] = {
            'path': str(arxiv_path),
            'type': 'optional_analysis',
            'status': 'ready'
        }
    
    if pubmed_path.exists():
        manifest['optional_datasets']['pubmed'] = {
            'path': str(pubmed_path),
            'type': 'optional_analysis',
            'status': 'ready'
        }
    
    manifest_path = output_dir / 'dataset_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"\nDataset manifest saved to {manifest_path}")


def main():
    """Main download routine - supports any language pairs."""
    output_dir = Path(__file__).parent.parent / 'data' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("DOWNLOADING REAL DATASETS FOR MCT EXPERIMENTS")
    logger.info("=" * 80)
    logger.info(f"\nOutput directory: {output_dir}\n")
    
    # Get language pairs from environment or use defaults
    default_pairs = ['de-en', 'fi-en']
    lang_pairs = os.environ.get('MCT_LANG_PAIRS', ','.join(default_pairs)).split(',')
    lang_pairs = [p.strip() for p in lang_pairs if p.strip()]
    
    logger.info(f"Language pairs to download: {lang_pairs}\n")
    
    # Download all specified language pairs
    manifest = {
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'datasets': {},
        'language_pairs': lang_pairs
    }
    
    for lang_pair in lang_pairs:
        logger.info(f"\n[{lang_pairs.index(lang_pair) + 1}/{len(lang_pairs)}] Downloading {lang_pair}...")
        train_path, test_path = download_wmt_pair(lang_pair, output_dir)
        
        if train_path and test_path:
            manifest['datasets'][lang_pair] = {
                'train_path': str(train_path),
                'test_path': str(test_path),
                'status': 'ready'
            }
        else:
            manifest['datasets'][lang_pair] = {
                'status': 'unavailable (not in WMT benchmarks)'
            }
    
    # Download optional analysis datasets
    logger.info("\n[Optional] Optional analysis datasets (arXiv, PubMed)...")
    download_arxiv_sample(output_dir, num_samples=50000)
    download_pubmed_sample(output_dir, num_samples=50000)
    
    # Verify
    logger.info("\n[3] Verifying downloads...")
    stats = verify_downloads(output_dir, lang_pairs)
    
    # Create manifest
    logger.info("\n[4] Creating dataset manifest...")
    create_dataset_manifest(output_dir, stats, lang_pairs)
    
    # Summary
    ready_pairs = [pair for pair in lang_pairs if stats.get(pair, False)]
    total_pairs = len(lang_pairs)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"DOWNLOAD SUMMARY: {len(ready_pairs)}/{total_pairs} language pairs ready")
    
    if ready_pairs:
        logger.info("\nReady pairs:")
        for pair in ready_pairs:
            logger.info(f"  ✓ {pair}")
    
    missing_pairs = [pair for pair in lang_pairs if not stats.get(pair, False)]
    if missing_pairs:
        logger.warning("\nUnavailable pairs (not in public WMT):")
        for pair in missing_pairs:
            logger.warning(f"  ✗ {pair}")
    
    if ready_pairs:
        logger.info(f"\nNext: python3 scripts/train_nmt_models.py")
        logger.info(f"Configure with: MCT_LANG_PAIRS='{','.join(ready_pairs)}'")
        core_ready = True
    else:
        logger.error("✗ NO DATASETS DOWNLOADED")
        core_ready = False
    
    logger.info("=" * 80)
    
    return core_ready


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
