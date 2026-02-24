from datasets import load_dataset
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def load_morphology_dataset(name="MorphoBench"):
    """
    Loads specific benchmarks for morphological awareness.
    
    Args:
        name: Dataset name ("MorphoBench", "SIGMORPHON", "UniMorph")
        
    Returns:
        Loaded dataset
    """
    logger.info(f"Loading morphology dataset: {name}")
    
    if name == "MorphoBench":
        try:
            return load_dataset("OpenDCAI/MorphoBench")
        except Exception as e:
            logger.error(f"Failed to load MorphoBench: {e}")
            logger.info("Attempting to load from local file...")
            try:
                return load_dataset("json", data_files="data/raw/morphology/morphobench.json")
            except:
                return None
    
    elif name == "SIGMORPHON":
        try:
            return load_dataset("sigmorphon2022", "inflection", trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load SIGMORPHON: {e}")
            return None
    
    elif name == "UniMorph":
        logger.info("Loading UniMorph dataset (local)")
        return load_dataset("json", data_files="data/raw/morphology/unimorph.json")
    
    else:
        logger.warning(f"Unknown morphology dataset: {name}")
        return None


def load_translation_dataset(lang_pair="de-en", dataset_name="wmt14"):
    """
    Load translation dataset.
    
    Args:
        lang_pair: Language pair (e.g., "de-en", "fi-en")
        dataset_name: Dataset name (wmt14, wmt16, opus100)
        
    Returns:
        Loaded dataset
    """
    logger.info(f"Loading {dataset_name}:{lang_pair} translation dataset")
    
    try:
        return load_dataset(dataset_name, lang_pair, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load {dataset_name}:{lang_pair}: {e}")
        return None


def load_summarization_dataset(dataset_name="arxiv"):
    """
    Load summarization dataset.
    
    Args:
        dataset_name: Dataset name ("arxiv", "pubmed")
        
    Returns:
        Loaded dataset
    """
    logger.info(f"Loading {dataset_name} summarization dataset")
    
    try:
        return load_dataset("scientific_papers", dataset_name, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load {dataset_name}: {e}")
        return None


def save_corpus_to_file(corpus: List[str], filepath: str):
    """
    Save text corpus to file (one item per line).
    
    Args:
        corpus: List of text items
        filepath: Output file path
    """
    logger.info(f"Saving {len(corpus)} items to {filepath}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in corpus:
            f.write(item + '\n')


def load_corpus_from_file(filepath: str) -> List[str]:
    """
    Load text corpus from file.
    
    Args:
        filepath: Input file path
        
    Returns:
        List of text items
    """
    logger.info(f"Loading corpus from {filepath}")
    
    corpus = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(line)
    
    logger.info(f"Loaded {len(corpus)} items")
    return corpus
