"""
Configuration Validator for MCT Models

Validates that model/tokenizer configurations match paper specifications.
References: [cite: 92, 157-160, 173-175, 280-283]
"""

from typing import Dict, Tuple, Optional, List
import json
from pathlib import Path


class ConfigValidator:
    """Validates MCT configurations against paper specifications"""
    
    # Paper-specified hyperparameters [cite: 92, 157-160]
    PAPER_SPECS = {
        "small": {
            "model_size_params": (125, 140),  # 125-140M
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "vocab_size": 32000,
        },
        "medium": {
            "model_size_params": (350, 370),  # 350-370M
            "hidden_size": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "vocab_size": 32000,
        },
        "large": {
            "model_size_params": (1000, 1200),  # ~1B
            "hidden_size": 1280,
            "num_layers": 36,
            "num_heads": 20,
            "vocab_size": 32000,
        },
    }
    
    # Tokenizer specifications [cite: 92, 112]
    TOKENIZER_SPECS = {
        "p_drop": 0.05,  # Stochastic stem dropout probability
        "min_stem_length": 2,  # Minimum stem length
        "special_tokens": [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "[STEM]", "[PREFIX]", "[SUFFIX]"
        ],
        "vocab_size": 32000,
        "stem_vocab_ratio": 0.3,  # 30% stems, 70% affixes
    }
    
    # Training specifications [cite: 173-175]
    TRAINING_SPECS = {
        "optimizer": "adamw",
        "learning_rate": 6e-4,
        "beta1": 0.9,
        "beta2": 0.98,
        "weight_decay": 0.01,
        "warmup_steps": 4000,
        "batch_size": 128,
        "max_seq_length": 512,
    }
    
    # Evaluation specifications [cite: 280-283]
    EVALUATION_SPECS = {
        "translation_bleu_threshold": 30.0,  # Expected BLEU > 30
        "summarization_rouge_threshold": 30.0,  # Expected ROUGE-1 > 30
        "morphology_accuracy_threshold": 85.0,  # Expected accuracy > 85%
    }
    
    @classmethod
    def validate_model_config(cls, config: Dict, size: str = "small") -> Tuple[bool, List[str]]:
        """
        Validate model configuration against paper specs
        
        Args:
            config: Configuration dictionary
            size: Model size (small, medium, large)
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        if size not in cls.PAPER_SPECS:
            return False, [f"Unknown model size: {size}"]
        
        specs = cls.PAPER_SPECS[size]
        
        # Check hidden size
        if config.get("hidden_size") != specs["hidden_size"]:
            warnings.append(
                f"hidden_size={config.get('hidden_size')} differs from spec={specs['hidden_size']}"
            )
        
        # Check number of layers
        if config.get("num_layers") != specs["num_layers"]:
            warnings.append(
                f"num_layers={config.get('num_layers')} differs from spec={specs['num_layers']}"
            )
        
        # Check number of heads
        if config.get("num_heads") != specs["num_heads"]:
            warnings.append(
                f"num_heads={config.get('num_heads')} differs from spec={specs['num_heads']}"
            )
        
        # Check vocabulary size
        vocab_size = config.get("vocab_size", 32000)
        if vocab_size < 20000 or vocab_size > 50000:
            warnings.append(
                f"vocab_size={vocab_size} is outside recommended range [20000, 50000]"
            )
        
        return len(warnings) == 0, warnings
    
    @classmethod
    def validate_tokenizer_config(cls, config: Dict) -> Tuple[bool, List[str]]:
        """
        Validate tokenizer configuration
        
        Args:
            config: Tokenizer configuration
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        # Check p_drop
        p_drop = config.get("p_drop", 0.05)
        if abs(p_drop - cls.TOKENIZER_SPECS["p_drop"]) > 0.01:
            warnings.append(
                f"p_drop={p_drop} differs from paper spec={cls.TOKENIZER_SPECS['p_drop']}"
            )
        
        # Check vocab size
        vocab_size = config.get("vocab_size", 32000)
        if vocab_size != cls.TOKENIZER_SPECS["vocab_size"]:
            warnings.append(
                f"vocab_size={vocab_size} differs from spec={cls.TOKENIZER_SPECS['vocab_size']}"
            )
        
        # Check special tokens
        special_tokens = config.get("special_tokens", [])
        missing_tokens = set(cls.TOKENIZER_SPECS["special_tokens"]) - set(special_tokens)
        if missing_tokens:
            warnings.append(f"Missing special tokens: {missing_tokens}")
        
        # Check stem vocab ratio
        stem_ratio = config.get("stem_vocab_ratio", 0.3)
        if abs(stem_ratio - cls.TOKENIZER_SPECS["stem_vocab_ratio"]) > 0.05:
            warnings.append(
                f"stem_vocab_ratio={stem_ratio} differs from spec={cls.TOKENIZER_SPECS['stem_vocab_ratio']}"
            )
        
        return len(warnings) == 0, warnings
    
    @classmethod
    def validate_training_config(cls, config: Dict) -> Tuple[bool, List[str]]:
        """
        Validate training configuration
        
        Args:
            config: Training configuration
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        specs = cls.TRAINING_SPECS
        
        # Check optimizer
        optimizer = config.get("optimizer", "adamw").lower()
        if optimizer != specs["optimizer"]:
            warnings.append(
                f"optimizer={optimizer} differs from spec={specs['optimizer']}"
            )
        
        # Check learning rate
        lr = config.get("learning_rate", 6e-4)
        if abs(lr - specs["learning_rate"]) / specs["learning_rate"] > 0.1:
            warnings.append(
                f"learning_rate={lr:.2e} differs from spec={specs['learning_rate']:.2e}"
            )
        
        # Check beta parameters
        beta1 = config.get("beta1", 0.9)
        if beta1 != specs["beta1"]:
            warnings.append(
                f"beta1={beta1} differs from spec={specs['beta1']}"
            )
        
        beta2 = config.get("beta2", 0.98)
        if beta2 != specs["beta2"]:
            warnings.append(
                f"beta2={beta2} differs from spec={specs['beta2']}"
            )
        
        # Check warmup steps
        warmup = config.get("warmup_steps", 4000)
        if warmup != specs["warmup_steps"]:
            warnings.append(
                f"warmup_steps={warmup} differs from spec={specs['warmup_steps']}"
            )
        
        # Check batch size
        batch_size = config.get("batch_size", 128)
        if batch_size < 64 or batch_size > 512:
            warnings.append(
                f"batch_size={batch_size} is outside recommended range [64, 512]"
            )
        
        # Check max sequence length
        max_seq = config.get("max_seq_length", 512)
        if max_seq < 128 or max_seq > 1024:
            warnings.append(
                f"max_seq_length={max_seq} is outside recommended range [128, 1024]"
            )
        
        return len(warnings) == 0, warnings
    
    @classmethod
    def validate_vocabulary(cls, vocab: Dict) -> Tuple[bool, List[str]]:
        """
        Validate vocabulary structure
        
        Args:
            vocab: Vocabulary dictionary {token: id}
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        # Check vocabulary size
        vocab_size = len(vocab)
        if vocab_size < 20000 or vocab_size > 50000:
            warnings.append(
                f"Vocabulary size {vocab_size} is outside [20000, 50000]"
            )
        
        # Check for special tokens
        special_tokens = cls.TOKENIZER_SPECS["special_tokens"]
        missing_special = [t for t in special_tokens if t not in vocab]
        if missing_special:
            warnings.append(f"Missing special tokens: {missing_special}")
        
        # Check for stem vs affix tokens
        stem_tokens = [t for t in vocab.keys() if not t.startswith("##")]
        affix_tokens = [t for t in vocab.keys() if t.startswith("##")]
        
        if len(stem_tokens) == 0:
            warnings.append("No stem tokens found in vocabulary")
        if len(affix_tokens) == 0:
            warnings.append("No affix tokens (##-prefixed) found in vocabulary")
        
        # Check stem/affix ratio
        if len(vocab) > 0:
            stem_ratio = len(stem_tokens) / len(vocab)
            expected_ratio = cls.TOKENIZER_SPECS["stem_vocab_ratio"]
            if abs(stem_ratio - expected_ratio) > 0.1:
                warnings.append(
                    f"Stem ratio {stem_ratio:.1%} differs from spec {expected_ratio:.1%}"
                )
        
        return len(warnings) == 0, warnings
    
    @classmethod
    def validate_results(
        cls,
        results: Dict[str, float],
        task: str = "translation"
    ) -> Tuple[bool, List[str]]:
        """
        Validate evaluation results against paper benchmarks
        
        Args:
            results: Results dictionary with metrics
            task: Task type (translation, summarization, morphology)
        
        Returns:
            Tuple of (meets_benchmark, list_of_messages)
        """
        messages = []
        specs = cls.EVALUATION_SPECS
        
        if task == "translation":
            bleu = results.get("bleu", 0)
            threshold = specs["translation_bleu_threshold"]
            if bleu >= threshold:
                messages.append(f"✓ Translation BLEU={bleu:.1f} meets threshold {threshold}")
            else:
                messages.append(f"✗ Translation BLEU={bleu:.1f} below threshold {threshold}")
            return bleu >= threshold, messages
        
        elif task == "summarization":
            rouge_1 = results.get("rouge1", 0)
            threshold = specs["summarization_rouge_threshold"]
            if rouge_1 >= threshold:
                messages.append(f"✓ Summarization ROUGE-1={rouge_1:.1f} meets threshold {threshold}")
            else:
                messages.append(f"✗ Summarization ROUGE-1={rouge_1:.1f} below threshold {threshold}")
            return rouge_1 >= threshold, messages
        
        elif task == "morphology":
            accuracy = results.get("stem_accuracy", 0) * 100
            threshold = specs["morphology_accuracy_threshold"]
            if accuracy >= threshold:
                messages.append(f"✓ Morphology accuracy={accuracy:.1f}% meets threshold {threshold}%")
            else:
                messages.append(f"✗ Morphology accuracy={accuracy:.1f}% below threshold {threshold}%")
            return accuracy >= threshold, messages
        
        return False, ["Unknown task type"]
    
    @classmethod
    def print_validation_report(
        cls,
        config: Dict,
        size: str = "small",
        verbose: bool = True
    ) -> None:
        """
        Print a comprehensive validation report
        
        Args:
            config: Configuration to validate
            size: Model size
            verbose: Print detailed warnings
        """
        print("\n" + "=" * 60)
        print("MCT Configuration Validation Report")
        print("=" * 60)
        
        # Validate model config
        is_valid, warnings = cls.validate_model_config(config, size)
        print(f"\n[Model Config - {size.upper()}]")
        print(f"Status: {'✓ PASS' if is_valid else '⚠ WARNINGS'}")
        if verbose and warnings:
            for w in warnings:
                print(f"  - {w}")
        
        # Validate tokenizer config
        is_valid, warnings = cls.validate_tokenizer_config(config)
        print(f"\n[Tokenizer Config]")
        print(f"Status: {'✓ PASS' if is_valid else '⚠ WARNINGS'}")
        if verbose and warnings:
            for w in warnings:
                print(f"  - {w}")
        
        # Validate training config
        is_valid, warnings = cls.validate_training_config(config)
        print(f"\n[Training Config]")
        print(f"Status: {'✓ PASS' if is_valid else '⚠ WARNINGS'}")
        if verbose and warnings:
            for w in warnings:
                print(f"  - {w}")
        
        print("\n" + "=" * 60 + "\n")


def validate_config_file(config_path: str, size: str = "small") -> bool:
    """
    Validate a configuration file
    
    Args:
        config_path: Path to config file (YAML or JSON)
        size: Model size
    
    Returns:
        Whether configuration is valid
    """
    import yaml
    
    with open(config_path) as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    validator = ConfigValidator()
    validator.print_validation_report(config, size, verbose=True)
    
    # Return overall validity
    is_valid_model, _ = ConfigValidator.validate_model_config(config, size)
    is_valid_tokenizer, _ = ConfigValidator.validate_tokenizer_config(config)
    is_valid_training, _ = ConfigValidator.validate_training_config(config)
    
    return is_valid_model and is_valid_tokenizer and is_valid_training


if __name__ == "__main__":
    # Example usage
    example_config = {
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "vocab_size": 32000,
        "p_drop": 0.05,
        "learning_rate": 6e-4,
        "beta1": 0.9,
        "beta2": 0.98,
        "warmup_steps": 4000,
    }
    
    ConfigValidator.print_validation_report(example_config, "small")
