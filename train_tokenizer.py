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
