# test_tokenizer.py
from mct_tokenizer import MCTTokenizer
import os

def test_mct_tokenizer():
    print("Testing MCT Tokenizer...")
    
    # Initialize tokenizer
    tokenizer = MCTTokenizer(vocab_size=32000, p_drop=0.05)
    
    # Test training (mock)
    print("Testing training...")
    
    # Create sample training files
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample text files
        samples = [
            "This is a test sentence for tokenizer training.",
            "Morphological analysis is important for language models.",
            "The tokenizer should preserve word stems.",
            "Affixes should be segmented properly."
        ]
        
        file_paths = []
        for i, text in enumerate(samples):
            file_path = os.path.join(tmpdir, f"sample_{i}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            file_paths.append(file_path)
        
        # Train tokenizer
        tokenizer.train(file_paths)
    
    # Test encoding
    print("\nTesting encoding...")
    test_texts = [
        "unnecessarily running",
        "This is a simple test.",
        "The tokenization should work properly."
    ]
    
    for text in test_texts:
        tokens = tokenizer.encode(text)
        print(f"Text: {text}")
        print(f"Tokens: {tokens}")
        print(f"Number of tokens: {len(tokens)}")
        print()
    
    # Test save/load (if implemented)
    print("Testing save/load...")
    tokenizer.save_model("test_tokenizer")
    
    print("Tokenizer test completed!")

if __name__ == "__main__":
    test_mct_tokenizer()



# ==================== TOKENIZER TRAINING ====================
# Step 9: Train MCT tokenizer on C4 dataset
python train_tokenizer.py --dataset c4 --vocab-size 32000

# train_tokenizer.py
import argparse
import sys
from pathlib import Path

def train_mct_tokenizer(args):
    """Train MCT tokenizer on specified dataset."""
    from mct_tokenizer import MCTTokenizer
    from mct_datamodel import CorpusDataset
    
    print(f"Training MCT tokenizer on {args.dataset} dataset...")
    
    # Initialize tokenizer
    tokenizer = MCTTokenizer(
        vocab_size=args.vocab_size,
        p_drop=args.p_drop
    )
    
    # Load dataset
    if args.dataset == "c4":
        dataset_dir = Path("data/c4")
        train_files = list(dataset_dir.glob("train*.txt"))
        if args.sample:
            train_files = train_files[:1]  # Use only first file for testing
    elif args.dataset == "arxiv":
        dataset_dir = Path("data/arxiv")
        train_files = [dataset_dir / "train_text.txt"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"Using {len(train_files)} training files")
    
    # Train tokenizer
    tokenizer.train([str(f) for f in train_files])
    
    # Save tokenizer
    output_dir = Path("models/tokenizers")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(output_dir / f"mct_{args.dataset}_{args.vocab_size}"))
    
    # Test the tokenizer
    print("\nTesting trained tokenizer...")
    test_texts = [
        "unnecessarily running through the park",
        "The preprocessing of data is important for machine learning.",
        "Deutsche Sprache hat komplexe Morphologie."
    ]
    
    for text in test_texts:
        tokens = tokenizer.encode(text)
        print(f"\nText: {text}")
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")
    
    print(f"\nTokenizer saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MCT Tokenizer")
    parser.add_argument("--dataset", type=str, choices=["c4", "arxiv", "wmt14"],
                       default="c4", help="Dataset to train on")
    parser.add_argument("--vocab-size", type=int, default=32000,
                       help="Vocabulary size")
    parser.add_argument("--p-drop", type=float, default=0.05,
                       help="Stem dropout probability")
    parser.add_argument("--sample", action="store_true",
                       help="Use sample data for quick testing")
    
    args = parser.parse_args()
    train_mct_tokenizer(args)

# ==================== BASELINE TOKENIZER TRAINING ====================
# Step 10: Train baseline tokenizers for comparison
python train_baselines.py

# train_baselines.py
from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer
import argparse

def train_bpe_tokenizer(args):
    """Train BPE tokenizer for comparison."""
    from datasets import load_dataset
    
    print("Training BPE tokenizer...")
    
    # Initialize tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # Load dataset
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    # Prepare training data
    def batch_iterator(batch_size=1000):
        batch = []
        for example in dataset:
            batch.append(example["text"])
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    
    # Train tokenizer
    tokenizer.train_from_iterator(
        batch_iterator(),
        vocab_size=args.vocab_size,
        min_frequency=2,
        show_progress=True
    )
    
    # Save tokenizer
    tokenizer.save(f"models/tokenizers/bpe_{args.vocab_size}.json")
    print(f"BPE tokenizer saved")

def train_wordpiece_tokenizer(args):
    """Train WordPiece tokenizer for comparison."""
    print("Training WordPiece tokenizer...")
    
    tokenizer = BertWordPieceTokenizer(
        vocab_size=args.vocab_size,
        lowercase=True,
        strip_accents=True
    )
    
    # Similar training logic as BPE
    # ...
    
    tokenizer.save(f"models/tokenizers/wordpiece_{args.vocab_size}")
    print(f"WordPiece tokenizer saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline tokenizers")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--tokenizer", type=str, choices=["bpe", "wordpiece", "all"],
                       default="all")
    
    args = parser.parse_args()
    
    if args.tokenizer in ["bpe", "all"]:
        train_bpe_tokenizer(args)
    
    if args.tokenizer in ["wordpiece", "all"]:
        train_wordpiece_tokenizer(args)

# ==================== EXPERIMENT 1: MORPHOLOGICAL AWARENESS ====================
# Step 11: Run morphological awareness experiment
python run_experiment.py --task morphology --tokenizer mct

# run_experiment.py
import argparse
from mct_datamodel import *

def run_morphology_experiment(tokenizer_type):
    """Run morphological awareness benchmark."""
    
    config = ExperimentConfig(
        tokenizer_type=tokenizer_type,
        tokenizer_config=TokenizerConfig(vocab_size=10000, p_drop=0.05),
        training_config=TrainingConfig(
            batch_size=16,
            num_epochs=5,
            learning_rate=1e-4
        ),
        task=TaskType.MORPHOLOGICAL_AWARENESS,
        language=Language.ENGLISH,
        dataset_paths={
            'data': 'data/morphology/morphology.csv'
        },
        output_dir=f'results/morphology/{tokenizer_type.value}',
        seed=42
    )
    
    manager = ExperimentManager(config)
    result = manager.run_experiment()
    
    return result

def run_translation_experiment(tokenizer_type):
    """Run machine translation experiment."""
    
    config = ExperimentConfig(
        tokenizer_type=tokenizer_type,
        tokenizer_config=TokenizerConfig(vocab_size=32000, p_drop=0.05),
        training_config=TrainingConfig(
            batch_size=32,
            num_epochs=10,
            learning_rate=5e-5
        ),
        task=TaskType.MACHINE_TRANSLATION,
        language=Language.GERMAN,
        dataset_paths={
            'train_src': 'data/wmt14/train.de',
            'train_tgt': 'data/wmt14/train.en',
            'val_src': 'data/wmt14/validation.de',
            'val_tgt': 'data/wmt14/validation.en',
            'test_src': 'data/wmt14/test.de',
            'test_tgt': 'data/wmt14/test.en'
        },
        output_dir=f'results/translation/{tokenizer_type.value}',
        seed=42
    )
    
    manager = ExperimentManager(config)
    result = manager.run_experiment()
    
    return result

def run_summarization_experiment(tokenizer_type):
    """Run summarization experiment."""
    
    config = ExperimentConfig(
        tokenizer_type=tokenizer_type,
        tokenizer_config=TokenizerConfig(vocab_size=32000, p_drop=0.05),
        training_config=TrainingConfig(
            batch_size=8,  # Smaller batch for longer sequences
            num_epochs=5,
            learning_rate=3e-5,
            max_seq_length=1024
        ),
        task=TaskType.SUMMARIZATION,
        language=Language.ENGLISH,
        dataset_paths={
            'train': 'data/arxiv/train.jsonl',
            'val': 'data/arxiv/validation.jsonl',
            'test': 'data/arxiv/test.jsonl'
        },
        output_dir=f'results/summarization/{tokenizer_type.value}',
        seed=42
    )
    
    manager = ExperimentManager(config)
    result = manager.run_experiment()
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCT Experiments")
    parser.add_argument("--task", type=str,
                       choices=["morphology", "translation", "summarization", "all"],
                       default="morphology")
    parser.add_argument("--tokenizer", type=str,
                       choices=["mct", "bpe", "wordpiece", "bytelevel", "all"],
                       default="mct")
    
    args = parser.parse_args()
    
    # Map tokenizer types
    tokenizer_types = []
    if args.tokenizer == "all":
        tokenizer_types = [TokenizerType.MCT, TokenizerType.BPE,
                          TokenizerType.WORDPIECE, TokenizerType.BYTELEVEL]
    else:
        tokenizer_types = [TokenizerType(args.tokenizer)]
    
    # Run experiments
    results = []
    
    for tokenizer_type in tokenizer_types:
        print(f"\n{'='*60}")
        print(f"Running {args.task} experiment with {tokenizer_type.value}")
        print(f"{'='*60}")
        
        if args.task == "morphology":
            result = run_morphology_experiment(tokenizer_type)
        elif args.task == "translation":
            result = run_translation_experiment(tokenizer_type)
        elif args.task == "summarization":
            result = run_summarization_experiment(tokenizer_type)
        elif args.task == "all":
            # Run all tasks
            result1 = run_morphology_experiment(tokenizer_type)
            result2 = run_translation_experiment(tokenizer_type)
            result3 = run_summarization_experiment(tokenizer_type)
            results.extend([result1, result2, result3])
            continue
        
        results.append(result)
    
    print(f"\nCompleted {len(results)} experiments")
