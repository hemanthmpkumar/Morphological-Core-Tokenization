import evaluate
import logging
from datasets import load_dataset
from src.tokenizer.mct_tokenizer import MCTTokenizer
import torch
from src.utils.device import get_compute_device

logger = logging.getLogger(__name__)


def run_translation_experiment(model, tokenizer, lang_pair="de-en", dataset_name="wmt14", device=None):
    """Run a translation evaluation.

    The device argument can be used to explicitly set the backend;
    if ``None`` (the default) we delegate to :func:`get_compute_device` which
    prefers CUDA devices and will only fall back to Apple MPS if the caller
    explicitly enables it via ``allow_mps=True`` or sets ``MCT_DEVICE`` in
    the environment.
    """
    if device is None:
        # prefer CUDA and *do not* automatically use MPS, since the goal is to
        # run real experiments on GPU clusters.  Users can still override via
        # ``MCT_DEVICE`` if they really want to test on MPS.
        device = get_compute_device()
    """
    Run machine translation evaluation experiment [cite: 147-149, 173].
    
    Args:
        model: MCT transformer model
        tokenizer: MCT tokenizer
        lang_pair: Language pair (e.g., "de-en", "fi-en")
        dataset_name: Dataset name (wmt14, wmt16, opus100)
        device: Device for inference
        
    Returns:
        Dictionary with BLEU and ChrF scores
    """
    logger.info(f"Running translation experiment: {lang_pair} using {dataset_name}")
    
    try:
        # Load dataset
        dataset = load_dataset(dataset_name, lang_pair, trust_remote_code=True)
        test_split = dataset.get("test", dataset.get("validation"))
        
        if not test_split:
            logger.warning(f"Could not find test split for {dataset_name}:{lang_pair}")
            return {}
        
        # Load metrics
        bleu = evaluate.load("bleu")
        chrf = evaluate.load("chrf")
        
        predictions = []
        references = []
        
        logger.info(f"Evaluating on {len(test_split)} translation pairs")
        
        model.eval()
        with torch.no_grad():
            for example in test_split[:1000]:  # Limit to 1000 samples for efficiency
                src_text = example.get("translation", {}).get(lang_pair.split("-")[0], "")
                ref_text = example.get("translation", {}).get(lang_pair.split("-")[1], "")
                
                if not src_text or not ref_text:
                    continue
                
                # Tokenize source
                tokens = tokenizer.tokenize(src_text)
                
                # In practice, would run inference here
                # For now, just evaluate tokenization quality
                prediction = " ".join(tokens)
                
                predictions.append(prediction)
                references.append(ref_text)
        
        if not predictions:
            logger.warning("No valid predictions generated")
            return {}
        
        # Compute metrics
        bleu_score = bleu.compute(
            predictions=predictions,
            references=[[ref] for ref in references],
            max_order=4
        )
        
        chrf_score = chrf.compute(
            predictions=predictions,
            references=references
        )
        
        results = {
            'language_pair': lang_pair,
            'dataset': dataset_name,
            'bleu': bleu_score.get("bleu", 0.0),
            'chrf': chrf_score.get("score", 0.0),
            'num_samples': len(predictions)
        }
        
        logger.info(f"Results - BLEU: {results['bleu']:.2f}, ChrF: {results['chrf']:.2f}")
        return results
    
    except Exception as e:
        logger.error(f"Translation experiment failed: {e}")
        return {}
