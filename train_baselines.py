#!/usr/bin/env python3
"""
train_baselines.py

Train baseline tokenizers (BPE, WordPiece, SentencePiece) for comparison with MCT.
These baselines match what was used in the paper experiments.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# Tokenizer libraries
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit
from tokenizers.normalizers import NFKC, Sequence, Lowercase
from tokenizers.processors import TemplateProcessing
from tokenizers import decoders

# SentencePiece
try:
    import sentencepiece as spm
    SP_AVAILABLE = True
except ImportError:
    SP_AVAILABLE = False
    print("Warning: sentencepiece not installed. Install with: pip install sentencepiece")

# Hugging Face transformers tokenizers
try:
    from transformers import PreTrainedTokenizerFast, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ==================== Configuration ====================

@dataclass
class TokenizerConfig:
    """Configuration for tokenizer training."""
    tokenizer_type: str  # "bpe", "wordpiece", "sentencepiece", "bytelevel"
    vocab_size: int = 32000
    min_frequency: int = 2
    max_token_length: int = 100
    lowercase: bool = True
    special_tokens: List[str] = field(default_factory=lambda: [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
    ])
    coverage: float = 1.0  # For SentencePiece character coverage
    model_type: str = "bpe"  # For SentencePiece: "bpe", "unigram", "char", "word"
    
@dataclass
class TrainingConfig:
    """Configuration for dataset and training."""
    dataset_paths: List[str]
    output_dir: str = "models/baselines"
    max_samples: Optional[int] = None
    batch_size: int = 1000
    seed: int = 42

# ==================== Dataset Loading ====================

class CorpusIterator:
    """Iterator for streaming text corpus."""
    
    def __init__(self, file_paths: List[str], max_samples: Optional[int] = None):
        self.file_paths = file_paths
        self.max_samples = max_samples
        self.current_file_idx = 0
        self.current_line = 0
        self.total_samples = 0
        
    def __iter__(self):
        for file_path in self.file_paths:
            if self.max_samples and self.total_samples >= self.max_samples:
                break
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if self.max_samples and self.total_samples >= self.max_samples:
                            break
                        text = line.strip()
                        if text:  # Skip empty lines
                            yield text
                            self.total_samples += 1
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue

def load_c4_corpus(config: TrainingConfig) -> CorpusIterator:
    """Load C4 corpus for training."""
    print(f"Loading C4 corpus from {len(config.dataset_paths)} files...")
    
    # Collect all text files
    all_files = []
    for path in config.dataset_paths:
        if os.path.isdir(path):
            # If directory, collect all .txt files
            for file in Path(path).glob("*.txt"):
                all_files.append(str(file))
        else:
            all_files.append(path)
    
    return CorpusIterator(all_files, config.max_samples)

def load_wmt14_corpus(config: TrainingConfig) -> CorpusIterator:
    """Load WMT14 corpus (both German and English)."""
    print("Loading WMT14 corpus...")
    
    all_files = []
    for path in config.dataset_paths:
        if path.endswith('.de') or path.endswith('.en'):
            all_files.append(path)
    
    return CorpusIterator(all_files, config.max_samples)

def load_arxiv_corpus(config: TrainingConfig) -> CorpusIterator:
    """Load arXiv corpus from JSONL files."""
    print("Loading arXiv corpus...")
    
    def arxiv_iterator():
        total = 0
        for file_path in config.dataset_paths:
            if config.max_samples and total >= config.max_samples:
                break
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if config.max_samples and total >= config.max_samples:
                            break
                        try:
                            data = json.loads(line)
                            # Yield both article and summary
                            if 'article' in data:
                                yield data['article'].strip()
                            if 'summary' in data:
                                yield data['summary'].strip()
                            total += 1
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue
    
    # Wrap in a class to provide length estimate
    class ArxivIterator:
        def __iter__(self):
            return arxiv_iterator()
    
    return ArxivIterator()

# ==================== BPE Tokenizer ====================

def train_bpe_tokenizer(config: TokenizerConfig, train_config: TrainingConfig) -> Tokenizer:
    """Train a BPE tokenizer."""
    print("\n" + "="*60)
    print("Training BPE Tokenizer")
    print("="*60)
    
    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token=config.special_tokens[1]))
    
    # Set pre-tokenizer
    tokenizer.pre_tokenizer = Whitespace()
    
    # Set normalizer
    if config.lowercase:
        tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
    else:
        tokenizer.normalizer = NFKC()
    
    # Set decoder
    tokenizer.decoder = decoders.BPEDecoder()
    
    # Set post-processor
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.special_tokens[2]} $A {config.special_tokens[3]}",
        pair=f"{config.special_tokens[2]} $A {config.special_tokens[3]} $B:1 {config.special_tokens[3]}:1",
        special_tokens=[
            (config.special_tokens[2], 2),  # [CLS]
            (config.special_tokens[3], 3),  # [SEP]
        ]
    )
    
    # Initialize trainer
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=config.special_tokens,
        show_progress=True,
        initial_alphabet=[]
    )
    
    # Load corpus
    corpus_iterator = load_c4_corpus(train_config)
    
    # Train tokenizer
    print(f"Training BPE tokenizer with vocab_size={config.vocab_size}...")
    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer, length=train_config.max_samples)
    
    # Save tokenizer
    output_path = Path(train_config.output_dir) / f"bpe_vocab{config.vocab_size}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    tokenizer.save(str(output_path / "tokenizer.json"))
    
    # Save vocabulary
    vocab = tokenizer.get_vocab()
    with open(output_path / "vocab.txt", 'w', encoding='utf-8') as f:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\n")
    
    print(f"BPE tokenizer saved to {output_path}")
    print(f"Vocabulary size: {len(vocab)}")
    
    return tokenizer

# ==================== WordPiece Tokenizer ====================

def train_wordpiece_tokenizer(config: TokenizerConfig, train_config: TrainingConfig) -> Tokenizer:
    """Train a WordPiece tokenizer."""
    print("\n" + "="*60)
    print("Training WordPiece Tokenizer")
    print("="*60)
    
    # Initialize tokenizer
    tokenizer = Tokenizer(WordPiece(
        unk_token=config.special_tokens[1],
        max_input_chars_per_word=config.max_token_length
    ))
    
    # Set pre-tokenizer
    tokenizer.pre_tokenizer = Whitespace()
    
    # Set normalizer
    if config.lowercase:
        tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
    else:
        tokenizer.normalizer = NFKC()
    
    # Set decoder
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    
    # Initialize trainer
    trainer = WordPieceTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=config.special_tokens,
        show_progress=True,
        continuing_subword_prefix="##"
    )
    
    # Load corpus
    corpus_iterator = load_c4_corpus(train_config)
    
    # Train tokenizer
    print(f"Training WordPiece tokenizer with vocab_size={config.vocab_size}...")
    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer, length=train_config.max_samples)
    
    # Save tokenizer
    output_path = Path(train_config.output_dir) / f"wordpiece_vocab{config.vocab_size}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    tokenizer.save(str(output_path / "tokenizer.json"))
    
    # Save vocabulary
    vocab = tokenizer.get_vocab()
    with open(output_path / "vocab.txt", 'w', encoding='utf-8') as f:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\n")
    
    print(f"WordPiece tokenizer saved to {output_path}")
    print(f"Vocabulary size: {len(vocab)}")
    
    return tokenizer

# ==================== SentencePiece Tokenizer ====================

def train_sentencepiece_tokenizer(config: TokenizerConfig, train_config: TrainingConfig) -> Optional[spm.SentencePieceProcessor]:
    """Train a SentencePiece tokenizer."""
    if not SP_AVAILABLE:
        print("Error: sentencepiece not installed. Skipping SentencePiece training.")
        print("Install with: pip install sentencepiece")
        return None
    
    print("\n" + "="*60)
    print("Training SentencePiece Tokenizer")
    print("="*60)
    
    # Create input file for SentencePiece training
    temp_input_file = Path(train_config.output_dir) / "sp_input.txt"
    
    print("Creating input file for SentencePiece...")
    with open(temp_input_file, 'w', encoding='utf-8') as f:
        corpus_iterator = load_c4_corpus(train_config)
        for i, text in enumerate(tqdm(corpus_iterator, desc="Writing corpus", total=train_config.max_samples)):
            f.write(text + "\n")
            if train_config.max_samples and i >= train_config.max_samples - 1:
                break
    
    # Create output directory
    output_path = Path(train_config.output_dir) / f"sentencepiece_vocab{config.vocab_size}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build SentencePiece model name
    model_prefix = output_path / "sp_model"
    
    # Training arguments
    train_args = [
        f'--input={temp_input_file}',
        f'--model_prefix={model_prefix}',
        f'--vocab_size={config.vocab_size}',
        f'--character_coverage={config.coverage}',
        '--model_type=bpe',  # Use BPE for fair comparison
        '--pad_id=0',
        '--unk_id=1',
        '--bos_id=2',
        '--eos_id=3',
        '--pad_piece=[PAD]',
        '--unk_piece=[UNK]',
        '--bos_piece=[CLS]',
        '--eos_piece=[SEP]',
        '--user_defined_symbols=[MASK]',
        '--split_digits=true',
        '--remove_extra_whitespaces=true',
        f'--max_sentence_length={train_config.batch_size * 100}',
    ]
    
    if config.lowercase:
        train_args.append('--normalization_rule_name=nfkc_cf')  # NFKC + case folding
    else:
        train_args.append('--normalization_rule_name=nfkc')
    
    # Train SentencePiece model
    print(f"Training SentencePiece tokenizer with vocab_size={config.vocab_size}...")
    spm.SentencePieceTrainer.Train(' '.join(train_args))
    
    # Load trained model
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(f'{model_prefix}.model')
    
    # Save vocabulary
    with open(output_path / "vocab.txt", 'w', encoding='utf-8') as f:
        for i in range(sp_model.GetPieceSize()):
            piece = sp_model.IdToPiece(i)
            f.write(f"{piece}\n")
    
    # Convert to Hugging Face format if transformers is available
    if TRANSFORMERS_AVAILABLE:
        try:
            from transformers import BertTokenizerFast
            from tokenizers import Tokenizer as HF_Tokenizer
            from tokenizers.models import BPE as HF_BPE
            from tokenizers.decoders import BPEDecoder
            
            # Create Hugging Face tokenizer
            hf_tokenizer = BertTokenizerFast(
                vocab_file=str(model_prefix) + ".vocab",
                tokenizer_file=None
            )
            
            # Add special tokens
            hf_tokenizer.add_special_tokens({
                'pad_token': '[PAD]',
                'unk_token': '[UNK]',
                'cls_token': '[CLS]',
                'sep_token': '[SEP]',
                'mask_token': '[MASK]'
            })
            
            hf_tokenizer.save_pretrained(output_path / "huggingface")
            print(f"Hugging Face compatible tokenizer saved")
            
        except Exception as e:
            print(f"Could not create Hugging Face tokenizer: {e}")
    
    # Clean up temporary file
    if temp_input_file.exists():
        temp_input_file.unlink()
    
    print(f"SentencePiece tokenizer saved to {output_path}")
    print(f"Vocabulary size: {sp_model.GetPieceSize()}")
    
    return sp_model

# ==================== Byte-Level BPE Tokenizer ====================

def train_bytelevel_tokenizer(config: TokenizerConfig, train_config: TrainingConfig) -> Tokenizer:
    """Train a Byte-Level BPE tokenizer (like GPT-2)."""
    print("\n" + "="*60)
    print("Training Byte-Level BPE Tokenizer")
    print("="*60)
    
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    
    # Initialize tokenizer
    tokenizer = Tokenizer(BPE())
    
    # Set ByteLevel pre-tokenizer
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    # Set ByteLevel decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # Special tokens for ByteLevel
    bytelevel_special_tokens = [
        "<|endoftext|>",  # EOS token
        "<|unk|>",
        "<|pad|>",
    ]
    
    # Initialize trainer
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=bytelevel_special_tokens,
        show_progress=True,
        initial_alphabet=[]  # ByteLevel handles this automatically
    )
    
    # Load corpus
    corpus_iterator = load_c4_corpus(train_config)
    
    # Train tokenizer
    print(f"Training ByteLevel BPE tokenizer with vocab_size={config.vocab_size}...")
    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer, length=train_config.max_samples)
    
    # Save tokenizer
    output_path = Path(train_config.output_dir) / f"bytelevel_vocab{config.vocab_size}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    tokenizer.save(str(output_path / "tokenizer.json"))
    
    # Save vocabulary
    vocab = tokenizer.get_vocab()
    with open(output_path / "vocab.txt", 'w', encoding='utf-8') as f:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\n")
    
    print(f"ByteLevel BPE tokenizer saved to {output_path}")
    print(f"Vocabulary size: {len(vocab)}")
    
    return tokenizer

# ==================== Hugging Face Tokenizers (For ByT5 comparison) ====================

def train_hf_byt5_tokenizer(config: TokenizerConfig, train_config: TrainingConfig):
    """Save configuration for ByT5 tokenizer (byte-level token-free baseline)."""
    print("\n" + "="*60)
    print("Setting up ByT5 (Byte-level) Configuration")
    print("="*60)
    
    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers not installed. Skipping ByT5 setup.")
        print("Install with: pip install transformers")
        return None
    
    output_path = Path(train_config.output_dir) / "byt5_config"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ByT5 uses a fixed byte-level vocabulary of size 256 + special tokens
    # We just need to document the configuration
    
    config_dict = {
        "tokenizer_type": "byt5",
        "vocab_size": 32128,  # 256 + special tokens
        "description": "ByT5 uses byte-level tokenization with fixed vocabulary of 256 bytes + special tokens",
        "special_tokens": [
            "<pad>", "</s>", "<unk>", "<s>", "<mask>", "<2en>", "<2de>", "<2fr>",
            "<2ro>", "<2zh>", "<extra_id_0>", "<extra_id_1>", "<extra_id_2>",
            "<extra_id_3>", "<extra_id_4>", "<extra_id_5>", "<extra_id_6>",
            "<extra_id_7>", "<extra_id_8>", "<extra_id_9>", "<extra_id_10>",
            "<extra_id_11>", "<extra_id_12>", "<extra_id_13>", "<extra_id_14>",
            "<extra_id_15>", "<extra_id_16>", "<extra_id_17>", "<extra_id_18>",
            "<extra_id_19>", "<extra_id_20>", "<extra_id_21>", "<extra_id_22>",
            "<extra_id_23>", "<extra_id_24>", "<extra_id_25>", "<extra_id_26>",
            "<extra_id_27>", "<extra_id_28>", "<extra_id_29>", "<extra_id_30>",
            "<extra_id_31>", "<extra_id_32>", "<extra_id_33>", "<extra_id_34>",
            "<extra_id_35>", "<extra_id_36>", "<extra_id_37>", "<extra_id_38>",
            "<extra_id_39>", "<extra_id_40>", "<extra_id_41>", "<extra_id_42>",
            "<extra_id_43>", "<extra_id_44>", "<extra_id_45>", "<extra_id_46>",
            "<extra_id_47>", "<extra_id_48>", "<extra_id_49>", "<extra_id_50>",
            "<extra_id_51>", "<extra_id_52>", "<extra_id_53>", "<extra_id_54>",
            "<extra_id_55>", "<extra_id_56>", "<extra_id_57>", "<extra_id_58>",
            "<extra_id_59>", "<extra_id_60>", "<extra_id_61>", "<extra_id_62>",
            "<extra_id_63>", "<extra_id_64>", "<extra_id_65>", "<extra_id_66>",
            "<extra_id_67>", "<extra_id_68>", "<extra_id_69>", "<extra_id_70>",
            "<extra_id_71>", "<extra_id_72>", "<extra_id_73>", "<extra_id_74>",
            "<extra_id_75>", "<extra_id_76>", "<extra_id_77>", "<extra_id_78>",
            "<extra_id_79>", "<extra_id_80>", "<extra_id_81>", "<extra_id_82>",
            "<extra_id_83>", "<extra_id_84>", "<extra_id_85>", "<extra_id_86>",
            "<extra_id_87>", "<extra_id_88>", "<extra_id_89>", "<extra_id_90>",
            "<extra_id_91>", "<extra_id_92>", "<extra_id_93>", "<extra_id_94>",
            "<extra_id_95>", "<extra_id_96>", "<extra_id_97>", "<extra_id_98>",
            "<extra_id_99>"
        ],
        "note": "ByT5 is a token-free model that operates directly on bytes. This is just a configuration file.",
        "paper_reference": "Xue et al., 'ByT5: Towards a token-free future with pre-trained byte-to-byte models', 2022"
    }
    
    with open(output_path / "config.json", 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"ByT5 configuration saved to {output_path}")
    print("Note: ByT5 uses fixed byte-level tokenization, not trainable from corpus")
    
    return config_dict

# ==================== Evaluation and Testing ====================

def test_tokenizer(tokenizer, tokenizer_type: str, test_samples: List[str]):
    """Test tokenizer on sample texts."""
    print(f"\n{'='*60}")
    print(f"Testing {tokenizer_type.upper()} Tokenizer")
    print(f"{'='*60}")
    
    for i, text in enumerate(test_samples):
        print(f"\nSample {i+1}: {text}")
        
        if tokenizer_type == "sentencepiece" and SP_AVAILABLE:
            # SentencePiece tokenization
            tokens = tokenizer.EncodeAsPieces(text)
            token_ids = tokenizer.EncodeAsIds(text)
            print(f"Tokens: {tokens}")
            print(f"Token IDs: {token_ids}")
            print(f"Token count: {len(tokens)}")
            
        elif hasattr(tokenizer, 'encode'):
            # Hugging Face tokenizers
            if tokenizer_type == "byt5":
                # For ByT5, we'd need the actual tokenizer
                print("ByT5 tokenization would convert text to bytes")
                byte_repr = text.encode('utf-8')
                print(f"UTF-8 bytes ({len(byte_repr)}): {list(byte_repr)}")
            else:
                # Standard tokenizer
                encoding = tokenizer.encode(text)
                tokens = tokenizer.tokenize(text)
                print(f"Tokens: {tokens}")
                print(f"Token IDs: {encoding.ids}")
                print(f"Token count: {len(tokens)}")
        
        print("-" * 40)

def analyze_tokenizer_stats(tokenizer, tokenizer_type: str, test_corpus: List[str],
                          output_path: Path):
    """Analyze tokenizer statistics."""
    print(f"\nAnalyzing {tokenizer_type.upper()} tokenizer statistics...")
    
    stats = {
        'tokenizer_type': tokenizer_type,
        'total_samples': len(test_corpus),
        'token_counts': [],
        'char_counts': [],
        'compression_ratios': [],
        'unk_counts': []
    }
    
    for text in tqdm(test_corpus, desc="Analyzing"):
        char_count = len(text)
        stats['char_counts'].append(char_count)
        
        if tokenizer_type == "sentencepiece" and SP_AVAILABLE:
            tokens = tokenizer.EncodeAsPieces(text)
            token_count = len(tokens)
            # Count UNK tokens
            unk_count = sum(1 for token in tokens if token == "▁<unk>" or token == "<unk>")
        elif hasattr(tokenizer, 'encode'):
            if tokenizer_type == "byt5":
                # ByT5: bytes
                token_count = len(text.encode('utf-8'))
                unk_count = 0
            else:
                encoding = tokenizer.encode(text)
                token_count = len(encoding.ids)
                # Count UNK tokens
                if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token:
                    unk_id = tokenizer.token_to_id(tokenizer.unk_token)
                    unk_count = encoding.ids.count(unk_id)
                else:
                    unk_count = 0
        else:
            continue
        
        stats['token_counts'].append(token_count)
        stats['unk_counts'].append(unk_count)
        
        if token_count > 0:
            compression_ratio = char_count / token_count
            stats['compression_ratios'].append(compression_ratio)
    
    # Calculate statistics
    if stats['token_counts']:
        stats['avg_tokens_per_sample'] = np.mean(stats['token_counts'])
        stats['std_tokens_per_sample'] = np.std(stats['token_counts'])
        stats['avg_chars_per_token'] = np.mean(stats['compression_ratios'])
        stats['avg_unk_rate'] = np.mean([unk/len(tokens) if tokens > 0 else 0
                                        for unk, tokens in zip(stats['unk_counts'], stats['token_counts'])])
        stats['vocab_size'] = tokenizer.get_vocab_size() if hasattr(tokenizer, 'get_vocab_size') else 'N/A'
    else:
        stats['avg_tokens_per_sample'] = 0
        stats['std_tokens_per_sample'] = 0
        stats['avg_chars_per_token'] = 0
        stats['avg_unk_rate'] = 0
        stats['vocab_size'] = 'N/A'
    
    # Save statistics
    stats_file = output_path / f"{tokenizer_type}_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        # Convert numpy types to Python types for JSON serialization
        serializable_stats = {}
        for key, value in stats.items():
            if isinstance(value, (np.integer, np.floating)):
                serializable_stats[key] = value.item()
            elif isinstance(value, np.ndarray):
                serializable_stats[key] = value.tolist()
            else:
                serializable_stats[key] = value
        
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n{tokenizer_type.upper()} Statistics:")
    print(f"  Average tokens per sample: {stats['avg_tokens_per_sample']:.2f}")
    print(f"  Average chars per token: {stats['avg_chars_per_token']:.2f}")
    print(f"  UNK rate: {stats['avg_unk_rate']:.4f}")
    print(f"  Vocabulary size: {stats['vocab_size']}")
    
    return stats

# ==================== Main Function ====================

def main():
    parser = argparse.ArgumentParser(description="Train baseline tokenizers for MCT comparison")
    
    # Tokenizer selection
    parser.add_argument("--tokenizer", type=str,
                       choices=["bpe", "wordpiece", "sentencepiece", "bytelevel", "byt5", "all"],
                       default="all",
                       help="Tokenizer type to train")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str,
                       choices=["c4", "wmt14", "arxiv", "all"],
                       default="c4",
                       help="Dataset to train on")
    
    # Training parameters
    parser.add_argument("--vocab-size", type=int, default=32000,
                       help="Vocabulary size")
    parser.add_argument("--max-samples", type=int, default=100000,
                       help="Maximum number of samples to use for training")
    parser.add_argument("--output-dir", type=str, default="models/baselines",
                       help="Output directory for trained tokenizers")
    
    # Additional options
    parser.add_argument("--test-only", action="store_true",
                       help="Only test existing tokenizers, don't train")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze tokenizer statistics")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which tokenizers to train
    tokenizers_to_train = []
    if args.tokenizer == "all":
        tokenizers_to_train = ["bpe", "wordpiece", "sentencepiece", "bytelevel", "byt5"]
    else:
        tokenizers_to_train = [args.tokenizer]
    
    # Skip SentencePiece if not available
    if "sentencepiece" in tokenizers_to_train and not SP_AVAILABLE:
        print("Warning: sentencepiece not available, skipping SentencePiece training")
        tokenizers_to_train.remove("sentencepiece")
    
    # Determine dataset paths
    dataset_paths = []
    data_dir = Path("data")
    
    if args.dataset in ["c4", "all"]:
        c4_dir = data_dir / "c4"
        if c4_dir.exists():
            dataset_paths.extend(list(c4_dir.glob("train*.txt")))
            dataset_paths.extend(list(c4_dir.glob("validation*.txt")))
    
    if args.dataset in ["wmt14", "all"]:
        wmt_dir = data_dir / "wmt14"
        if wmt_dir.exists():
            dataset_paths.extend(list(wmt_dir.glob("*.de")))
            dataset_paths.extend(list(wmt_dir.glob("*.en")))
    
    if args.dataset in ["arxiv", "all"]:
        arxiv_dir = data_dir / "arxiv"
        if arxiv_dir.exists():
            dataset_paths.extend(list(arxiv_dir.glob("*.jsonl")))
    
    if not dataset_paths:
        print(f"Warning: No dataset files found for {args.dataset}")
        print("Make sure you've downloaded the datasets first.")
        print("Run: python download_datasets.py --dataset {args.dataset}")
        return
    
    print(f"Found {len(dataset_paths)} dataset files")
    
    # Training configuration
    train_config = TrainingConfig(
        dataset_paths=[str(p) for p in dataset_paths],
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed
    )
    
    # Test samples
    test_samples = [
        "unnecessarily running through the morphological analysis",
        "Die deutsche Sprache hat eine komplexe Morphologie.",
        "The tokenizer should preserve semantic integrity.",
        "This is a test sentence for evaluating tokenization quality."
    ]
    
    # Train and test tokenizers
    trained_tokenizers = {}
    
    for tokenizer_type in tokenizers_to_train:
        print(f"\n{'='*80}")
        print(f"Processing {tokenizer_type.upper()}")
        print(f"{'='*80}")
        
        # Tokenizer configuration
        tokenizer_config = TokenizerConfig(
            tokenizer_type=tokenizer_type,
            vocab_size=args.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        )
        
        try:
            if args.test_only:
                print("Skipping training (test-only mode)")
                # Load existing tokenizer if available
                tokenizer_path = output_dir / f"{tokenizer_type}_vocab{args.vocab_size}"
                if tokenizer_path.exists():
                    if tokenizer_type == "sentencepiece":
                        if SP_AVAILABLE:
                            tokenizer = spm.SentencePieceProcessor()
                            tokenizer.Load(str(tokenizer_path / "sp_model.model"))
                        else:
                            continue
                    else:
                        tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
                    trained_tokenizers[tokenizer_type] = tokenizer
                else:
                    print(f"No existing tokenizer found at {tokenizer_path}")
                    continue
            else:
                # Train tokenizer
                if tokenizer_type == "bpe":
                    tokenizer = train_bpe_tokenizer(tokenizer_config, train_config)
                elif tokenizer_type == "wordpiece":
                    tokenizer = train_wordpiece_tokenizer(tokenizer_config, train_config)
                elif tokenizer_type == "sentencepiece":
                    tokenizer = train_sentencepiece_tokenizer(tokenizer_config, train_config)
                elif tokenizer_type == "bytelevel":
                    tokenizer = train_bytelevel_tokenizer(tokenizer_config, train_config)
                elif tokenizer_type == "byt5":
                    tokenizer = train_hf_byt5_tokenizer(tokenizer_config, train_config)
                else:
                    print(f"Unknown tokenizer type: {tokenizer_type}")
                    continue
                
                if tokenizer:
                    trained_tokenizers[tokenizer_type] = tokenizer
            
            # Test tokenizer
            if tokenizer_type in trained_tokenizers:
                test_tokenizer(trained_tokenizers[tokenizer_type], tokenizer_type, test_samples)
                
                # Analyze statistics if requested
                if args.analyze:
                    # Use a small subset for analysis
                    analysis_corpus = test_samples * 10  # Just reuse test samples
                    analyze_tokenizer_stats(
                        trained_tokenizers[tokenizer_type],
                        tokenizer_type,
                        analysis_corpus,
                        output_dir
                    )
            
        except Exception as e:
            print(f"Error training {tokenizer_type} tokenizer: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary report
    if trained_tokenizers:
        print(f"\n{'='*80}")
        print("TRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"Successfully trained {len(trained_tokenizers)} tokenizer(s):")
        for tokenizer_type in trained_tokenizers:
            print(f"  - {tokenizer_type.upper()}")
        
        print(f"\nTokenizers saved to: {output_dir}")
        print("\nTo use these tokenizers in experiments:")
        print("1. Reference them in run_experiment.py")
        print("2. Or load them directly:")
        print("   from tokenizers import Tokenizer")
        print(f'   tokenizer = Tokenizer.from_file("{output_dir}/bpe_vocab32000/tokenizer.json")')
    
    # Save training configuration
    config_summary = {
        "args": vars(args),
        "tokenizers_trained": list(trained_tokenizers.keys()),
        "dataset_files": [str(p) for p in dataset_paths],
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    with open(output_dir / "training_config.json", 'w', encoding='utf-8') as f:
        json.dump(config_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nConfiguration saved to: {output_dir / 'training_config.json'}")

if __name__ == "__main__":
    main()
