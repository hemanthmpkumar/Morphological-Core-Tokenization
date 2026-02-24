import numpy as np
import logging
from typing import Dict, List, Tuple
from datasets import load_dataset
import evaluate
from src.utils.device import get_compute_device

logger = logging.getLogger(__name__)


def evaluate_morphology(model, tokenizer, test_data="MorphoBench", device=None):
    """
    Evaluates the model on its ability to recover morphological cores.
    Metric: Stem Identification Accuracy [cite: 280-283].
    
    Args:
        model: MCT transformer model
        tokenizer: MCT tokenizer
        test_data: Dataset name ("MorphoBench", "SIGMORPHON", etc.)
        device: Device to run evaluation on.  If ``None`` the project-wide
            device helper will pick CUDA when available (MPS is not used
            unless explicitly requested via ``MCT_DEVICE``).
    
    Returns:
        Dictionary with morphology evaluation metrics
    """
    if device is None:
        device = get_compute_device()
    logger.info(f"Evaluating morphological awareness on {test_data}")
    
    results = {
        'stem_identification_accuracy': 0.0,
        'lemmatization_accuracy': 0.0,
        'morpheme_boundary_preservation': 0.0,
        'total_samples': 0
    }
    
    try:
        if test_data == "MorphoBench":
            return _evaluate_morphobench(model, tokenizer, device)
        elif test_data == "SIGMORPHON":
            return _evaluate_sigmorphon(model, tokenizer, device)
        else:
            logger.warning(f"Unknown test data: {test_data}")
            return results
    
    except Exception as e:
        logger.error(f"Error during morphology evaluation: {e}")
        return results


def _evaluate_morphobench(model, tokenizer, device="cpu") -> Dict:
    """Evaluate on MorphoBench dataset."""
    try:
        dataset = load_dataset("OpenDCAI/MorphoBench")
        test_split = dataset.get("test", dataset.get("validation"))
        
        if not test_split:
            logger.warning("Could not load MorphoBench dataset")
            return {}
        
        correct_stems = 0
        total = 0
        
        logger.info(f"Evaluating on {len(test_split)} MorphoBench samples")
        
        for example in test_split:
            word = example.get("word", "")
            gold_stem = example.get("lemma", "")
            
            if not word or not gold_stem:
                continue
            
            # Tokenize word with MCT
            tokens = tokenizer.tokenize(word)
            
            # Check if stem is preserved as atomic unit
            if gold_stem in tokens:
                correct_stems += 1
            
            total += 1
        
        accuracy = correct_stems / total if total > 0 else 0.0
        
        return {
            'stem_identification_accuracy': accuracy,
            'total_samples': total,
            'correct_stems': correct_stems,
            'dataset': 'MorphoBench'
        }
    
    except Exception as e:
        logger.error(f"MorphoBench evaluation failed: {e}")
        return {}


def _evaluate_sigmorphon(model, tokenizer, device="cpu") -> Dict:
    """Evaluate on SIGMORPHON dataset."""
    try:
        # Load SIGMORPHON inflection generation data
        dataset = load_dataset("sigmorphon2022", "inflection", trust_remote_code=True)
        test_split = dataset.get("test")
        
        if not test_split:
            logger.warning("Could not load SIGMORPHON dataset")
            return {}
        
        correct = 0
        total = 0
        
        logger.info(f"Evaluating on {len(test_split)} SIGMORPHON samples")
        
        for example in test_split:
            # SIGMORPHON format: lemma -> inflected form
            lemma = example.get("lemma", "")
            inflected = example.get("inflected", "")
            
            if not lemma or not inflected:
                continue
            
            tokens = tokenizer.tokenize(inflected)
            
            # Check if lemma (stem) is preserved
            if lemma in tokens:
                correct += 1
            
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'stem_identification_accuracy': accuracy,
            'total_samples': total,
            'correct_stems': correct,
            'dataset': 'SIGMORPHON'
        }
    
    except Exception as e:
        logger.error(f"SIGMORPHON evaluation failed: {e}")
        return {}
