import json
from pathlib import Path
from typing import Tuple, Optional
from .mct_transformer import MCTTransformer
from .configuration_mct import MCTConfig
from src.tokenizer.mct_tokenizer import MCTTokenizer
from src.tokenizer.mct_analyzer import MCTAnalyzer
from src.tokenizer.constrained_bpe import ConstrainedBPETrainer


def get_mct_model_and_tokenizer(
    size: str = "small",
    vocab_path: Optional[str] = None,
    p_drop: float = 0.05,
    config_path: Optional[str] = None,
    analyzer_cache_size: int = 10000
) -> Tuple[MCTTransformer, MCTTokenizer]:
    """
    Creates MCT model and tokenizer as specified in paper [cite: 157-160, 92, 112]
    
    Args:
        size: Model size - "small" (125M), "medium" (350M), or "large" (1B)
        vocab_path: Path to vocabulary JSON file (if None, uses config default)
        p_drop: Stochastic stem dropout probability (default 0.05 from paper)
        config_path: Optional path to custom config YAML
        analyzer_cache_size: Size of morphological analyzer cache
    
    Returns:
        Tuple of (MCTTransformer model, MCTTokenizer instance)
    """
    
    # =====================================================
    # STEP 1: Load or Create Configuration
    # =====================================================
    if config_path:
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = MCTConfig(**config_dict)
    else:
        # Use predefined configs from paper
        if size == "small":
            config = MCTConfig.small()
        elif size == "medium":
            config = MCTConfig.medium()
        elif size == "large":
            config = MCTConfig.large()
        else:
            raise ValueError(f"Unknown model size: {size}. Choose: small, medium, large")
    
    # =====================================================
    # STEP 2: Create Model
    # =====================================================
    model = MCTTransformer(
        vocab_size=config.vocab_size or 32000,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_dropout_prob,
        max_position_embeddings=config.max_seq_length,
        initializer_range=config.initializer_range,
    )
    
    # =====================================================
    # STEP 3: Load Vocabulary
    # =====================================================
    if vocab_path and Path(vocab_path).exists():
        with open(vocab_path) as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
        print(f"✓ Loaded vocabulary from {vocab_path} ({vocab_size} tokens)")
    else:
        # Create placeholder vocabulary
        vocab = {f"token_{i}": i for i in range(config.vocab_size or 32000)}
        print(f"⚠ Using placeholder vocabulary ({len(vocab)} tokens)")
    
    # =====================================================
    # STEP 4: Initialize Tokenizer Components
    # =====================================================
    
    # 4a: Initialize Morphological Analyzer [cite: 79, 112]
    analyzer = MCTAnalyzer(cache_size=analyzer_cache_size)
    print(f"✓ Initialized MCTAnalyzer with cache_size={analyzer_cache_size}")
    
    # 4b: Initialize Constrained BPE Trainer [cite: 114-117]
    bpe_trainer = ConstrainedBPETrainer(vocab_size=int(0.7 * config.vocab_size))
    
    # If vocabulary contains affix tokens, try to load BPE model
    affix_tokens = [t for t in vocab.keys() if t.startswith("##")]
    if affix_tokens:
        print(f"✓ Found {len(affix_tokens)} affix tokens in vocabulary")
    
    # =====================================================
    # STEP 5: Create MCT Tokenizer
    # =====================================================
    tokenizer = MCTTokenizer(
        analyzer=analyzer,
        bpe_trainer=bpe_trainer,
        vocab=vocab,
        vocab_size=len(vocab),
        p_drop=p_drop  # Stochastic stem dropout [cite: 92]
    )
    
    print(f"✓ Created MCTTokenizer with p_drop={p_drop}")
    print(f"")
    print(f"Model Summary:")
    print(f"  - Size: {size} ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters)")
    print(f"  - Vocab Size: {len(vocab)}")
    print(f"  - Config: {config}")
    
    return model, tokenizer


def load_mct_model(checkpoint_path: str, size: str = "small") -> MCTTransformer:
    """
    Load a pretrained MCT model from checkpoint
    
    Args:
        checkpoint_path: Path to saved model checkpoint
        size: Model size (for config)
    
    Returns:
        MCTTransformer instance
    """
    import torch
    
    # Get config for model size
    if size == "small":
        config = MCTConfig.small()
    elif size == "medium":
        config = MCTConfig.medium()
    elif size == "large":
        config = MCTConfig.large()
    else:
        raise ValueError(f"Unknown model size: {size}")
    
    # Create model
    model = MCTTransformer(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_heads,
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from {checkpoint_path}")
    
    return model


def load_mct_tokenizer(vocab_path: str, p_drop: float = 0.05) -> MCTTokenizer:
    """
    Load a pre-built MCT tokenizer from vocabulary file
    
    Args:
        vocab_path: Path to vocabulary JSON
        p_drop: Dropout probability
    
    Returns:
        MCTTokenizer instance
    """
    with open(vocab_path) as f:
        vocab = json.load(f)
    
    analyzer = MCTAnalyzer()
    bpe_trainer = ConstrainedBPETrainer(vocab_size=len(vocab))
    
    tokenizer = MCTTokenizer(
        analyzer=analyzer,
        bpe_trainer=bpe_trainer,
        vocab=vocab,
        vocab_size=len(vocab),
        p_drop=p_drop
    )
    
    print(f"✓ Loaded tokenizer from {vocab_path}")
    return tokenizer

