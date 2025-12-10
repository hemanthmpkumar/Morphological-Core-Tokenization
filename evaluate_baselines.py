# evaluate_baselines.py

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import tokenizers
try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("Warning: tokenizers library not available")

# Try to import sentencepiece
try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    print("Warning: sentencepiece library not available")

def load_baseline_tokenizers(baselines_dir: str = "models/baselines") -> Dict:
    """Load all trained baseline tokenizers."""
    baselines_dir = Path(baselines_dir)
    tokenizers = {}
    
    # Load BPE
    bpe_path = baselines_dir / "bpe_vocab32000" / "tokenizer.json"
    if bpe_path.exists() and TOKENIZERS_AVAILABLE:
        try:
            tokenizers["bpe"] = Tokenizer.from_file(str(bpe_path))
            print("✓ Loaded BPE tokenizer")
        except Exception as e:
            print(f"✗ Failed to load BPE tokenizer: {e}")
    
    # Load WordPiece
    wp_path = baselines_dir / "wordpiece_vocab32000" / "tokenizer.json"
    if wp_path.exists() and TOKENIZERS_AVAILABLE:
        try:
            tokenizers["wordpiece"] = Tokenizer.from_file(str(wp_path))
            print("✓ Loaded WordPiece tokenizer")
        except Exception as e:
            print(f"✗ Failed to load WordPiece tokenizer: {e}")
    
    # Load SentencePiece
    sp_path = baselines_dir / "sentencepiece_vocab32000" / "sp_model.model"
    if sp_path.exists() and SENTENCEPIECE_AVAILABLE:
        try:
            sp = spm.SentencePieceProcessor()
            sp.Load(str(sp_path))
            tokenizers["sentencepiece"] = sp
            print("✓ Loaded SentencePiece tokenizer")
        except Exception as e:
            print(f"✗ Failed to load SentencePiece tokenizer: {e}")
    
    # Load ByteLevel
    bl_path = baselines_dir / "bytelevel_vocab32000" / "tokenizer.json"
    if bl_path.exists() and TOKENIZERS_AVAILABLE:
        try:
            tokenizers["bytelevel"] = Tokenizer.from_file(str(bl_path))
            print("✓ Loaded ByteLevel tokenizer")
        except Exception as e:
            print(f"✗ Failed to load ByteLevel tokenizer: {e}")
    
    # Load ByT5 config
    byt5_path = baselines_dir / "byt5_config" / "config.json"
    if byt5_path.exists():
        try:
            with open(byt5_path, 'r', encoding='utf-8') as f:
                byt5_config = json.load(f)
            tokenizers["byt5"] = byt5_config
            print("✓ Loaded ByT5 configuration")
        except Exception as e:
            print(f"✗ Failed to load ByT5 config: {e}")
    
    return tokenizers

def compare_tokenization(tokenizers: Dict, text: str) -> pd.DataFrame:
    """Compare tokenization of the same text by different tokenizers."""
    results = []
    
    for name, tokenizer in tokenizers.items():
        if name == "sentencepiece" and SENTENCEPIECE_AVAILABLE:
            try:
                tokens = tokenizer.EncodeAsPieces(text)
                token_ids = tokenizer.EncodeAsIds(text)
                token_count = len(tokens)
                compression_ratio = len(text) / token_count if token_count > 0 else 0
            except Exception as e:
                print(f"Error with SentencePiece tokenization: {e}")
                continue
                
        elif name == "byt5":
            # ByT5 is byte-level, simulate tokenization
            bytes_repr = text.encode('utf-8')
            tokens = [f"byte_{b}" for b in bytes_repr[:20]]  # Show first 20 bytes
            if len(bytes_repr) > 20:
                tokens.append("...")
            token_ids = list(bytes_repr)
            token_count = len(bytes_repr)
            compression_ratio = 1.0  # Bytes to tokens is 1:1
            tokens = tokens  # Store as list for display
            
        elif TOKENIZERS_AVAILABLE and isinstance(tokenizer, Tokenizer):
            try:
                encoding = tokenizer.encode(text)
                # Get tokens from encoding
                tokens = encoding.tokens
                token_ids = encoding.ids
                token_count = len(tokens)
                compression_ratio = len(text) / token_count if token_count > 0 else 0
            except Exception as e:
                print(f"Error with {name} tokenization: {e}")
                continue
        else:
            # Skip if tokenizer is not properly loaded
            continue
        
        results.append({
            "tokenizer": name.upper(),
            "tokens": tokens,
            "token_ids": token_ids[:10] if isinstance(token_ids, list) else "N/A",  # First 10 IDs for display
            "token_count": token_count,
            "char_count": len(text),
            "compression_ratio": compression_ratio
        })
    
    return pd.DataFrame(results)

def benchmark_tokenizers(tokenizers: Dict, corpus: List[str]) -> pd.DataFrame:
    """Benchmark tokenizers on a corpus."""
    benchmark_results = []
    
    for name, tokenizer in tokenizers.items():
        print(f"Benchmarking {name}...")
        
        total_tokens = 0
        total_chars = 0
        total_time = 0
        total_unks = 0
        valid_samples = 0
        
        import time
        
        for text in corpus:
            if not text.strip():
                continue
                
            chars = len(text)
            total_chars += chars
            
            try:
                start = time.perf_counter()
                
                if name == "sentencepiece" and SENTENCEPIECE_AVAILABLE:
                    tokens = tokenizer.EncodeAsPieces(text)
                    # Count UNK tokens
                    unk_count = sum(1 for token in tokens if token == "▁<unk>" or token == "<unk>" or "unk" in token.lower())
                    
                elif name == "byt5":
                    # ByT5: byte-level
                    bytes_repr = text.encode('utf-8')
                    tokens = bytes_repr
                    unk_count = 0  # No UNK in byte-level
                    
                elif TOKENIZERS_AVAILABLE and isinstance(tokenizer, Tokenizer):
                    encoding = tokenizer.encode(text)
                    tokens = encoding.tokens
                    # Count UNK tokens if possible
                    try:
                        unk_token = tokenizer.token_to_id("[UNK]") if hasattr(tokenizer, 'token_to_id') else None
                        if unk_token is not None:
                            unk_count = encoding.ids.count(unk_token)
                        else:
                            unk_count = 0
                    except:
                        unk_count = 0
                else:
                    continue
                    
                end = time.perf_counter()
                
                total_tokens += len(tokens)
                total_time += (end - start)
                total_unks += unk_count
                valid_samples += 1
                
            except Exception as e:
                print(f"  Warning: Failed to tokenize sample with {name}: {e}")
                continue
        
        if valid_samples == 0:
            print(f"  No valid samples for {name}")
            continue
            
        avg_tokens = total_tokens / valid_samples
        avg_chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
        avg_time_per_sample = total_time / valid_samples * 1000  # ms
        unk_rate = total_unks / total_tokens if total_tokens > 0 else 0
        
        benchmark_results.append({
            "tokenizer": name.upper(),
            "avg_tokens_per_sample": avg_tokens,
            "avg_chars_per_token": avg_chars_per_token,
            "avg_time_ms": avg_time_per_sample,
            "unk_rate": unk_rate,
            "valid_samples": valid_samples,
            "total_tokens": total_tokens
        })
    
    return pd.DataFrame(benchmark_results)

def create_comparison_plots(benchmark_df: pd.DataFrame, output_dir: Path = Path("results")):
    """Create comparison plots from benchmark results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Plot 1: Average tokens per sample
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=benchmark_df, x="tokenizer", y="avg_tokens_per_sample")
    ax.set_title('Average Tokens per Sample by Tokenizer')
    ax.set_xlabel('Tokenizer')
    ax.set_ylabel('Tokens per Sample')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "tokens_per_sample.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Average characters per token (compression ratio)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=benchmark_df, x="tokenizer", y="avg_chars_per_token")
    ax.set_title('Compression Ratio (Chars per Token) by Tokenizer')
    ax.set_xlabel('Tokenizer')
    ax.set_ylabel('Characters per Token')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "chars_per_token.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Tokenization speed
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=benchmark_df, x="tokenizer", y="avg_time_ms")
    ax.set_title('Tokenization Speed by Tokenizer')
    ax.set_xlabel('Tokenizer')
    ax.set_ylabel('Time per Sample (ms)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "tokenization_speed.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: UNK rate
    if "unk_rate" in benchmark_df.columns:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=benchmark_df, x="tokenizer", y="unk_rate")
        ax.set_title('UNK Rate by Tokenizer')
        ax.set_xlabel('Tokenizer')
        ax.set_ylabel('UNK Rate')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / "unk_rate.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Combined plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Top-left: Tokens per sample
    sns.barplot(data=benchmark_df, x="tokenizer", y="avg_tokens_per_sample", ax=axes[0, 0])
    axes[0, 0].set_title('Tokens per Sample')
    axes[0, 0].set_xlabel('')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Top-right: Characters per token
    sns.barplot(data=benchmark_df, x="tokenizer", y="avg_chars_per_token", ax=axes[0, 1])
    axes[0, 1].set_title('Characters per Token')
    axes[0, 1].set_xlabel('')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Bottom-left: Tokenization speed
    sns.barplot(data=benchmark_df, x="tokenizer", y="avg_time_ms", ax=axes[1, 0])
    axes[1, 0].set_title('Tokenization Speed (ms)')
    axes[1, 0].set_xlabel('')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Bottom-right: UNK rate
    if "unk_rate" in benchmark_df.columns:
        sns.barplot(data=benchmark_df, x="tokenizer", y="unk_rate", ax=axes[1, 1])
        axes[1, 1].set_title('UNK Rate')
        axes[1, 1].set_xlabel('')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].axis('off')
    
    plt.suptitle('Tokenizer Comparison Benchmark', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "combined_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")

def save_detailed_comparison(tokenizers: Dict, corpus: List[str], output_dir: Path):
    """Save detailed tokenization comparison to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tokenization examples
    examples_file = output_dir / "tokenization_examples.txt"
    with open(examples_file, 'w', encoding='utf-8') as f:
        f.write("TOKENIZATION EXAMPLES\n")
        f.write("=" * 80 + "\n\n")
        
        test_texts = [
            "The morphological-core tokenization preserves word stems.",
            "unnecessarily running through complex linguistic structures",
            "Die Sprachverarbeitung erfordert präzise Tokenisierung.",
            "This is a simple test sentence for comparison.",
        ]
        
        for text in test_texts:
            f.write(f"\nText: {text}\n")
            f.write("-" * 80 + "\n")
            
            df = compare_tokenization(tokenizers, text)
            for _, row in df.iterrows():
                f.write(f"\n{row['tokenizer']}:\n")
                f.write(f"  Tokens: {row['tokens'][:15]}")
                if len(row['tokens']) > 15:
                    f.write("...")
                f.write(f"\n  Token count: {row['token_count']}")
                f.write(f"\n  Compression ratio: {row['compression_ratio']:.2f} chars/token\n")
    
    print(f"Detailed examples saved to {examples_file}")

if __name__ == "__main__":
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Load tokenizers
    print("Loading baseline tokenizers...")
    tokenizers = load_baseline_tokenizers()
    print(f"\n✓ Loaded {len(tokenizers)} tokenizer(s)")
    
    if not tokenizers:
        print("No tokenizers loaded. Make sure to train them first:")
        print("  python train_baselines.py --tokenizer bpe --vocab-size 32000")
        exit(1)
    
    # Test samples
    test_texts = [
        "The morphological-core tokenization preserves word stems.",
        "unnecessarily running through complex linguistic structures",
        "Die Sprachverarbeitung erfordert präzise Tokenisierung.",
        "This is a simple test sentence for comparison.",
    ]
    
    # Compare tokenization
    print("\n" + "="*80)
    print("TOKENIZATION COMPARISON")
    print("="*80)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nExample {i}:")
        print(f"Text: {text}")
        print("-" * 60)
        
        df = compare_tokenization(tokenizers, text)
        if not df.empty:
            # Display summary
            print(df[["tokenizer", "token_count", "compression_ratio"]].to_string(index=False))
            
            # Display first few tokens for each
            print("\nFirst few tokens:")
            for _, row in df.iterrows():
                tokens_preview = str(row['tokens'][:8])
                if len(row['tokens']) > 8:
                    tokens_preview = tokens_preview[:-1] + ", ...]" if tokens_preview.endswith(']') else tokens_preview + "..."
                print(f"  {row['tokenizer']}: {tokens_preview}")
        else:
            print("  No tokenizers could process this text")
        print()
    
    # Benchmark on larger corpus
    print("\n" + "="*80)
    print("BENCHMARKING TOKENIZERS")
    print("="*80)
    
    # Load test corpus
    test_corpus = []
    c4_sample = Path("data/c4/c4_sample.txt")
    if c4_sample.exists():
        with open(c4_sample, 'r', encoding='utf-8') as f:
            test_corpus = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(test_corpus)} samples from {c4_sample}")
    else:
        # Use test texts multiple times as fallback
        test_corpus = test_texts * 25  # 100 samples
        print(f"Using {len(test_corpus)} generated test samples")
    
    if test_corpus:
        benchmark_df = benchmark_tokenizers(tokenizers, test_corpus[:100])  # Limit to 100 samples
        
        if not benchmark_df.empty:
            print("\n" + "-"*80)
            print("BENCHMARK RESULTS")
            print("-"*80)
            print(benchmark_df.to_string(index=False))
            
            # Save results
            benchmark_file = output_dir / "benchmark_results.csv"
            benchmark_df.to_csv(benchmark_file, index=False)
            print(f"\n✓ Benchmark results saved to {benchmark_file}")
            
            # Create visualizations
            print("\nCreating visualizations...")
            create_comparison_plots(benchmark_df, output_dir)
            
            # Save detailed comparison
            save_detailed_comparison(tokenizers, test_corpus[:50], output_dir)
            
            # Print summary statistics
            print("\n" + "="*80)
            print("SUMMARY STATISTICS")
            print("="*80)
            
            # Find best performers
            if "avg_chars_per_token" in benchmark_df.columns:
                best_compression = benchmark_df.loc[benchmark_df['avg_chars_per_token'].idxmax()]
                print(f"\nBest compression (most chars/token):")
                print(f"  Tokenizer: {best_compression['tokenizer']}")
                print(f"  Characters per token: {best_compression['avg_chars_per_token']:.2f}")
            
            if "avg_time_ms" in benchmark_df.columns:
                fastest = benchmark_df.loc[benchmark_df['avg_time_ms'].idxmin()]
                print(f"\nFastest tokenizer:")
                print(f"  Tokenizer: {fastest['tokenizer']}")
                print(f"  Time per sample: {fastest['avg_time_ms']:.2f} ms")
            
            if "unk_rate" in benchmark_df.columns:
                lowest_unk = benchmark_df.loc[benchmark_df['unk_rate'].idxmin()]
                print(f"\nLowest UNK rate:")
                print(f"  Tokenizer: {lowest_unk['tokenizer']}")
                print(f"  UNK rate: {lowest_unk['unk_rate']:.4f}")
            
            # Recommendations
            print("\n" + "-"*80)
            print("RECOMMENDATIONS")
            print("-"*80)
            print("Based on the benchmark results:")
            
            # Check if we have BPE and WordPiece results
            bpe_result = benchmark_df[benchmark_df['tokenizer'] == 'BPE']
            wp_result = benchmark_df[benchmark_df['tokenizer'] == 'WORDPIECE']
            
            if not bpe_result.empty and not wp_result.empty:
                bpe_tokens = bpe_result.iloc[0]['avg_tokens_per_sample']
                wp_tokens = wp_result.iloc[0]['avg_tokens_per_sample']
                
                if bpe_tokens < wp_tokens:
                    print("✓ BPE produces fewer tokens per sample than WordPiece")
                else:
                    print("✓ WordPiece produces fewer tokens per sample than BPE")
            
            sp_result = benchmark_df[benchmark_df['tokenizer'] == 'SENTENCEPIECE']
            if not sp_result.empty:
                sp_unk = sp_result.iloc[0].get('unk_rate', 0)
                if sp_unk < 0.01:
                    print("✓ SentencePiece has very low UNK rate")
            
            print("\nFor comparison with MCT tokenizer:")
            print("1. Use these benchmarks as baseline metrics")
            print("2. Run MCT tokenizer on the same test corpus")
            print("3. Compare compression ratio, speed, and UNK rates")
            print("4. Refer to paper for BLEU score comparisons")
        else:
            print("No benchmark results generated")
    else:
        print("No test corpus available for benchmarking")
    
    print(f"\n✓ Evaluation complete! Check '{output_dir}/' for results.")
