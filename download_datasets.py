# download_datasets.py

import os
import json
import tarfile
import gzip
from pathlib import Path
import requests
from tqdm import tqdm
import datasets
from datasets import load_dataset, DatasetDict
import pandas as pd
import shutil

class DatasetDownloader:
    def __init__(self, base_dir: str = "data"):
        """
        Initialize dataset downloader.
        
        Args:
            base_dir: Base directory to store all datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs and configurations
        self.dataset_configs = {
            "c4": {
                "hf_name": "allenai/c4",
                "config": "en",
                "splits": ["train", "validation"],
                "output_dir": self.base_dir / "c4",
                "max_samples": {
                    "train": 100000,  # Reduced for testing; use None for full dataset
                    "validation": 10000
                }
            },
            "wmt14": {
                "hf_name": "wmt/wmt14",
                "config": "de-en",
                "splits": ["train", "validation", "test"],
                "output_dir": self.base_dir / "wmt14",
                "max_samples": {
                    "train": 50000,  # Reduced for testing
                    "validation": 3000,
                    "test": 3000
                }
            },
            "arxiv": {
                "hf_name": "ccdv/arxiv-summarization",
                "config": None,
                "splits": ["train", "validation", "test"],
                "output_dir": self.base_dir / "arxiv",
                "max_samples": {
                    "train": 10000,  # Reduced for testing
                    "validation": 1000,
                    "test": 1000
                }
            }
        }
        
        # Create subdirectories
        for config in self.dataset_configs.values():
            config["output_dir"].mkdir(parents=True, exist_ok=True)
    
    def download_c4(self):
        """Download and prepare C4 dataset."""
        print("\n" + "="*60)
        print("Downloading C4 (Common Crawl) Dataset")
        print("="*60)
        
        config = self.dataset_configs["c4"]
        output_dir = config["output_dir"]
        
        try:
            # Load dataset from Hugging Face
            print("Loading C4 dataset from Hugging Face...")
            
            for split in config["splits"]:
                print(f"\nProcessing {split} split...")
                
                # Load dataset with streaming for memory efficiency
                dataset = load_dataset(
                    config["hf_name"],
                    name=config["config"],
                    split=split,
                    streaming=True  # Use streaming for large datasets
                )
                
                # Take subset if specified
                if config["max_samples"][split]:
                    dataset = dataset.take(config["max_samples"][split])
                
                # Prepare output file
                output_file = output_dir / f"{split}.txt"
                
                # Extract and save text
                with open(output_file, 'w', encoding='utf-8') as f:
                    total_samples = config["max_samples"][split] or 1000000  # Approximate for progress bar
                    
                    for i, example in tqdm(enumerate(dataset), total=total_samples, desc=f"Processing {split}"):
                        text = example.get('text', '').strip()
                        if text:
                            # Clean text and split into sentences
                            sentences = self._split_into_sentences(text)
                            for sentence in sentences:
                                if len(sentence) > 20:  # Minimum length
                                    f.write(sentence + '\n')
                        
                        if config["max_samples"][split] and i >= config["max_samples"][split] - 1:
                            break
                
                print(f"Saved {split} split to {output_file}")
                print(f"File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
            
            # Create a smaller sample for testing
            self._create_sample_files(output_dir, "c4_sample.txt", 1000)
            
            print("\n✓ C4 dataset download completed!")
            
        except Exception as e:
            print(f"Error downloading C4 dataset: {e}")
            raise
    
    def download_wmt14(self):
        """Download and prepare WMT14 German-English dataset."""
        print("\n" + "="*60)
        print("Downloading WMT14 German-English Dataset")
        print("="*60)
        
        config = self.dataset_configs["wmt14"]
        output_dir = config["output_dir"]
        
        try:
            print("Loading WMT14 dataset from Hugging Face...")
            
            for split in config["splits"]:
                print(f"\nProcessing {split} split...")
                
                # Load dataset
                dataset = load_dataset(
                    config["hf_name"],
                    name=config["config"],
                    split=split
                )
                
                # Prepare output files
                de_file = output_dir / f"{split}.de"
                en_file = output_dir / f"{split}.en"
                
                # Get maximum samples for this split
                max_samples = config["max_samples"][split]
                
                with open(de_file, 'w', encoding='utf-8') as f_de, \
                     open(en_file, 'w', encoding='utf-8') as f_en:
                    
                    total_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
                    
                    for i in tqdm(range(total_samples), desc=f"Processing {split}"):
                        example = dataset[i]
                        
                        # Extract German and English sentences
                        de_text = example.get('translation', {}).get('de', '').strip()
                        en_text = example.get('translation', {}).get('en', '').strip()
                        
                        if de_text and en_text:
                            # Clean and normalize
                            de_clean = self._clean_translation_text(de_text)
                            en_clean = self._clean_translation_text(en_text)
                            
                            f_de.write(de_clean + '\n')
                            f_en.write(en_clean + '\n')
                
                print(f"Saved German sentences to {de_file}")
                print(f"Saved English sentences to {en_file}")
            
            # Create a smaller sample for testing
            self._create_sample_files(output_dir, "wmt14_sample.de", 500, source_file="train.de")
            self._create_sample_files(output_dir, "wmt14_sample.en", 500, source_file="train.en")
            
            # Download original test sets if available
            self._download_wmt14_test_sets(output_dir)
            
            print("\n✓ WMT14 dataset download completed!")
            
        except Exception as e:
            print(f"Error downloading WMT14 dataset: {e}")
            raise
    
    def download_arxiv(self):
        """Download and prepare arXiv summarization dataset."""
        print("\n" + "="*60)
        print("Downloading arXiv Summarization Dataset")
        print("="*60)
        
        config = self.dataset_configs["arxiv"]
        output_dir = config["output_dir"]
        
        try:
            print("Loading arXiv dataset from Hugging Face...")
            
            for split in config["splits"]:
                print(f"\nProcessing {split} split...")
                
                # Load dataset
                dataset = load_dataset(
                    config["hf_name"],
                    split=split
                )
                
                # Prepare output file (JSONL format)
                output_file = output_dir / f"{split}.jsonl"
                
                # Get maximum samples for this split
                max_samples = config["max_samples"][split]
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    total_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
                    
                    for i in tqdm(range(total_samples), desc=f"Processing {split}"):
                        example = dataset[i]
                        
                        # Extract article and summary
                        article = example.get('article', '').strip()
                        abstract = example.get('abstract', '').strip()
                        
                        if article and abstract:
                            # Truncate very long articles (for memory efficiency)
                            if len(article) > 10000:
                                article = article[:10000] + "..."
                            
                            # Create JSON record
                            record = {
                                "id": example.get("id", f"{split}_{i}"),
                                "article": article,
                                "summary": abstract,
                                "title": example.get("title", ""),
                                "categories": example.get("categories", [])
                            }
                            
                            f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
                print(f"Saved {split} split to {output_file}")
                print(f"File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
            
            # Create a smaller sample for testing
            self._create_sample_files(output_dir, "arxiv_sample.jsonl", 200, source_file="train.jsonl")
            
            # Also create text-only version for tokenizer training
            self._create_text_version(output_dir)
            
            print("\n✓ arXiv dataset download completed!")
            
        except Exception as e:
            print(f"Error downloading arXiv dataset: {e}")
            raise
    
    def _split_into_sentences(self, text: str) -> list:
        """Simple sentence splitting."""
        # Basic sentence splitting - could be enhanced with NLTK
        sentences = []
        current = []
        
        for char in text:
            current.append(char)
            if char in '.!?':
                sentence = ''.join(current).strip()
                if len(sentence) > 5:  # Minimum sentence length
                    sentences.append(sentence)
                current = []
        
        # Add remaining text if any
        if current:
            sentence = ''.join(current).strip()
            if len(sentence) > 5:
                sentences.append(sentence)
        
        return sentences
    
    def _clean_translation_text(self, text: str) -> str:
        """Clean translation text."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Ensure proper ending punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def _create_sample_files(self, output_dir: Path, sample_name: str, num_samples: int, source_file: str = None):
        """Create smaller sample files for testing."""
        if source_file:
            # Create sample from specific source file
            source_path = output_dir / source_file
            sample_path = output_dir / sample_name
            
            if source_path.exists():
                with open(source_path, 'r', encoding='utf-8') as src, \
                     open(sample_path, 'w', encoding='utf-8') as dst:
                    
                    lines = []
                    for i, line in enumerate(src):
                        if i < num_samples:
                            dst.write(line)
                        else:
                            break
                
                print(f"Created sample file: {sample_path}")
        else:
            # Create sample by combining all files
            sample_path = output_dir / sample_name
            all_text = []
            
            for file in output_dir.glob("*.txt"):
                if file.name != sample_name:
                    with open(file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[:num_samples // 3]  # Take subset from each file
                        all_text.extend(lines)
            
            with open(sample_path, 'w', encoding='utf-8') as f:
                f.writelines(all_text[:num_samples])
            
            print(f"Created sample file: {sample_path}")
    
    def _create_text_version(self, output_dir: Path):
        """Create text-only version for tokenizer training."""
        for split in ["train", "validation", "test"]:
            jsonl_file = output_dir / f"{split}.jsonl"
            txt_file = output_dir / f"{split}_text.txt"
            
            if jsonl_file.exists():
                with open(jsonl_file, 'r', encoding='utf-8') as jf, \
                     open(txt_file, 'w', encoding='utf-8') as tf:
                    
                    for line in jf:
                        try:
                            record = json.loads(line)
                            # Write article and summary
                            tf.write(record.get("article", "") + "\n")
                            tf.write(record.get("summary", "") + "\n")
                        except json.JSONDecodeError:
                            continue
                
                print(f"Created text version: {txt_file}")
    
    def _download_wmt14_test_sets(self, output_dir: Path):
        """Download original WMT14 test sets if available."""
        print("\nAttempting to download original WMT14 test sets...")
        
        test_sets = {
            "newstest2014.de": "http://www.statmt.org/wmt14/test-full.tgz",
            # Add other test sets as needed
        }
        
        for filename, url in test_sets.items():
            try:
                print(f"Downloading {filename}...")
                response = requests.get(url, stream=True, timeout=30)
                
                if response.status_code == 200:
                    # Save tar file
                    tar_path = output_dir / "test-full.tgz"
                    with open(tar_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Extract if needed
                    if tar_path.suffix == '.tgz' or tar_path.suffix == '.tar.gz':
                        with tarfile.open(tar_path, 'r:gz') as tar:
                            tar.extractall(output_dir)
                        
                        print(f"Extracted test sets to {output_dir}")
                    
                    # Clean up tar file
                    tar_path.unlink()
                    
            except Exception as e:
                print(f"Could not download original test set {filename}: {e}")
                print("Using Hugging Face version instead.")
    
    def create_morphology_dataset(self):
        """Create synthetic morphology dataset for testing."""
        print("\n" + "="*60)
        print("Creating Morphological Awareness Dataset")
        print("="*60)
        
        output_dir = self.base_dir / "morphology"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample morphological patterns
        patterns = [
            # (prefix, stem, suffix, full_word)
            ("un", "happy", "", "unhappy"),
            ("re", "write", "en", "rewritten"),
            ("pre", "process", "ing", "preprocessing"),
            ("dis", "agree", "ment", "disagreement"),
            ("im", "possible", "", "impossible"),
            ("in", "correct", "ly", "incorrectly"),
            ("mis", "understand", "ing", "misunderstanding"),
            ("over", "achieve", "ment", "overachievement"),
            ("under", "estimate", "ed", "underestimated"),
            ("anti", "social", "", "antisocial"),
            ("non", "sense", "", "nonsense"),
            ("co", "operate", "ion", "cooperation"),
            ("counter", "act", "ive", "counteractive"),
            ("extra", "ordinary", "", "extraordinary"),
            ("hyper", "active", "", "hyperactive"),
            ("inter", "national", "", "international"),
            ("intra", "mural", "", "intramural"),
            ("macro", "economics", "", "macroeconomics"),
            ("micro", "scope", "", "microscope"),
            ("mono", "tone", "", "monotone"),
            ("multi", "cultural", "", "multicultural"),
            ("poly", "glot", "", "polyglot"),
            ("post", "modern", "ism", "postmodernism"),
            ("pre", "determine", "ed", "predetermined"),
            ("pro", "active", "", "proactive"),
            ("semi", "final", "", "semifinal"),
            ("sub", "marine", "", "submarine"),
            ("super", "human", "", "superhuman"),
            ("trans", "atlantic", "", "transatlantic"),
            ("ultra", "sound", "", "ultrasound"),
        ]
        
        # Create CSV file
        csv_file = output_dir / "morphology.csv"
        
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("word,root,prefix,suffix\n")
            for prefix, stem, suffix, word in patterns:
                f.write(f"{word},{stem},{prefix},{suffix}\n")
        
        # Create text corpus for training
        txt_file = output_dir / "morphology_corpus.txt"
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            # Create example sentences
            for prefix, stem, suffix, word in patterns:
                sentences = [
                    f"The word '{word}' can be analyzed as '{prefix}{stem}{suffix}'.",
                    f"In '{word}', the root is '{stem}' with prefix '{prefix}' and suffix '{suffix}'.",
                    f"Morphological analysis shows '{word}' = '{prefix}' + '{stem}' + '{suffix}'.",
                    f"The prefix '{prefix}' and suffix '{suffix}' modify the stem '{stem}' to form '{word}'.",
                ]
                for sentence in sentences:
                    f.write(sentence + "\n")
        
        print(f"Created morphology dataset at {output_dir}")
        print(f"  - CSV file: {csv_file}")
        print(f"  - Text corpus: {txt_file}")
    
    def create_dataset_info(self):
        """Create dataset information file."""
        info_file = self.base_dir / "dataset_info.json"
        
        info = {
            "datasets": {},
            "total_size": 0,
            "download_date": pd.Timestamp.now().isoformat()
        }
        
        # Calculate sizes
        total_size = 0
        
        for dataset_name, config in self.dataset_configs.items():
            output_dir = config["output_dir"]
            dataset_info = {
                "path": str(output_dir),
                "files": [],
                "total_size_mb": 0
            }
            
            if output_dir.exists():
                for file in output_dir.glob("*"):
                    if file.is_file():
                        file_info = {
                            "name": file.name,
                            "size_mb": file.stat().st_size / (1024 * 1024),
                            "modified": pd.Timestamp.fromtimestamp(file.stat().st_mtime).isoformat()
                        }
                        dataset_info["files"].append(file_info)
                        dataset_info["total_size_mb"] += file_info["size_mb"]
                        total_size += file_info["size_mb"]
            
            info["datasets"][dataset_name] = dataset_info
        
        info["total_size"] = total_size
        
        # Save info file
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"\nDataset information saved to {info_file}")
        print(f"Total dataset size: {total_size:.2f} MB")
    
    def download_all(self, include_morphology: bool = True):
        """Download all datasets."""
        print("="*80)
        print("DOWNLOADING ALL DATASETS")
        print("="*80)
        
        # Download main datasets
        self.download_c4()
        self.download_wmt14()
        self.download_arxiv()
        
        # Create morphology dataset
        if include_morphology:
            self.create_morphology_dataset()
        
        # Create dataset info
        self.create_dataset_info()
        
        print("\n" + "="*80)
        print("ALL DATASETS DOWNLOADED SUCCESSFULLY!")
        print("="*80)
        print(f"\nDatasets are available in: {self.base_dir}")
        print("\nDirectory structure:")
        print(self.base_dir)
        
        # Show directory tree
        self._print_directory_tree(self.base_dir, max_depth=3)
    
    def _print_directory_tree(self, path: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0):
        """Print directory tree structure."""
        if current_depth >= max_depth:
            return
        
        # Get items
        items = list(path.iterdir())
        items.sort()
        
        for i, item in enumerate(items):
            connector = "└── " if i == len(items) - 1 else "├── "
            print(f"{prefix}{connector}{item.name}")
            
            if item.is_dir():
                extension = "    " if i == len(items) - 1 else "│   "
                self._print_directory_tree(item, prefix + extension, max_depth, current_depth + 1)

# ==================== Command Line Interface ====================

def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download datasets for MCT experiments")
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Output directory for datasets (default: data)")
    parser.add_argument("--dataset", type=str, choices=["c4", "wmt14", "arxiv", "all", "morphology"],
                       default="all", help="Dataset to download (default: all)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per split (default: None for full dataset)")
    parser.add_argument("--skip-morphology", action="store_true",
                       help="Skip creation of morphology dataset")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download even if files exist")
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = DatasetDownloader(base_dir=args.output_dir)
    
    # Update max samples if specified
    if args.max_samples:
        for dataset in downloader.dataset_configs.values():
            for split in dataset["splits"]:
                dataset["max_samples"][split] = args.max_samples
    
    # Download selected dataset(s)
    if args.dataset == "all":
        downloader.download_all(include_morphology=not args.skip_morphology)
    
    elif args.dataset == "c4":
        downloader.download_c4()
        downloader.create_dataset_info()
    
    elif args.dataset == "wmt14":
        downloader.download_wmt14()
        downloader.create_dataset_info()
    
    elif args.dataset == "arxiv":
        downloader.download_arxiv()
        downloader.create_dataset_info()
    
    elif args.dataset == "morphology":
        downloader.create_morphology_dataset()
    
    print("\nDownload complete!")

if __name__ == "__main__":
    # Check if datasets library is installed
    try:
        import datasets
        main()
    except ImportError:
        print("Error: The 'datasets' library is required.")
        print("Please install it using: pip install datasets")
        
        # Alternative: provide manual download instructions
        print("\nManual download instructions:")
        print("1. C4 Dataset: https://huggingface.co/datasets/allenai/c4")
        print("2. WMT14 Dataset: https://huggingface.co/datasets/wmt/wmt14")
        print("3. arXiv Dataset: https://huggingface.co/datasets/ccdv/arxiv-summarization")
        print("\nPlace downloaded files in the 'data/' directory with appropriate structure.")
