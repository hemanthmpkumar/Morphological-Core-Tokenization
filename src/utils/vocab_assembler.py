"""
Vocabulary assembly pipeline for MCT tokenization.
Combines stems and affix-BPE vocabulary into final MCT vocabulary.
"""

import json
import logging
from typing import Dict, List, Set, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class VocabularyAssembler:
    """
    Assembles MCT vocabulary from stems and affix-BPE tokens.
    Stages 5-6 of the vocabulary construction pipeline.
    """
    
    def __init__(self, vocab_size: int = 32000, stem_reserved_ratio: float = 0.3):
        """
        Args:
            vocab_size: Final vocabulary size
            stem_reserved_ratio: Fraction of vocab reserved for stems (default: 0.3)
        """
        self.vocab_size = vocab_size
        self.stem_reserved_ratio = stem_reserved_ratio
        self.num_stem_slots = int(vocab_size * stem_reserved_ratio)
        self.num_affix_slots = vocab_size - self.num_stem_slots
        
        self.final_vocab = {}
        self.stem_vocab = {}
        self.affix_vocab = {}
        self.special_tokens = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "[STEM]", "[PREFIX]", "[SUFFIX]"
        ]
    
    def assemble(self, 
                 stem_freq_map: Dict[str, int],
                 affix_tokens: List[str],
                 affix_frequencies: Dict[str, int] = None) -> Dict[str, int]:
        """
        Assemble final vocabulary from stems and affix tokens.
        
        Args:
            stem_freq_map: Dictionary of stem -> frequency
            affix_tokens: List of BPE affix tokens
            affix_frequencies: Optional frequencies for affix tokens
            
        Returns:
            Final vocabulary (token -> id mapping)
        """
        logger.info("Starting vocabulary assembly")
        
        # Reserve special tokens
        token_id = 0
        for special_token in self.special_tokens:
            self.final_vocab[special_token] = token_id
            token_id += 1
        
        # Stage 5: Select top stems [cite: 114-116]
        logger.info(f"Selecting top {self.num_stem_slots} stems")
        top_stems = self._select_top_stems(stem_freq_map, self.num_stem_slots)
        
        for stem in top_stems:
            self.final_vocab[stem] = token_id
            self.stem_vocab[stem] = token_id
            token_id += 1
        
        # Stage 6: Select top affix BPE tokens [cite: 117-119]
        logger.info(f"Selecting top {self.num_affix_slots} affix tokens")
        top_affixes = self._select_top_affixes(
            affix_tokens, 
            affix_frequencies, 
            self.num_affix_slots
        )
        
        for affix_token in top_affixes:
            if affix_token not in self.final_vocab:  # Avoid duplicates
                self.final_vocab[affix_token] = token_id
                self.affix_vocab[affix_token] = token_id
                token_id += 1
        
        logger.info(f"Final vocabulary assembled: {len(self.final_vocab)} tokens")
        logger.info(f"  - Special tokens: {len(self.special_tokens)}")
        logger.info(f"  - Stems: {len(self.stem_vocab)}")
        logger.info(f"  - Affix tokens: {len(self.affix_vocab)}")
        
        return self.final_vocab
    
    def _select_top_stems(self, stem_freq_map: Dict[str, int], k: int) -> List[str]:
        """Select top k stems by frequency."""
        sorted_stems = sorted(
            stem_freq_map.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [stem for stem, _ in sorted_stems[:k]]
    
    def _select_top_affixes(self, 
                           affix_tokens: List[str],
                           affix_frequencies: Dict[str, int] = None,
                           k: int = None) -> List[str]:
        """
        Select top k affix tokens by frequency.
        
        Args:
            affix_tokens: List of affix tokens
            affix_frequencies: Optional pre-computed frequencies
            k: Number of tokens to select
            
        Returns:
            Top k affix tokens
        """
        if k is None:
            k = self.num_affix_slots
        
        if affix_frequencies:
            # Use provided frequencies
            sorted_affixes = sorted(
                affix_frequencies.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return [affix for affix, _ in sorted_affixes[:k]]
        else:
            # Compute frequencies from token list
            freq = Counter(affix_tokens)
            return [affix for affix, _ in freq.most_common(k)]
    
    def save_vocabulary(self, filepath: str):
        """Save vocabulary to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.final_vocab, f, ensure_ascii=False, indent=2)
        logger.info(f"Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str):
        """Load vocabulary from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.final_vocab = json.load(f)
        
        # Reconstruct stem/affix vocabs
        self.stem_vocab = {
            token: idx for token, idx in self.final_vocab.items()
            if not token.startswith("[") and not token.startswith("##")
        }
        self.affix_vocab = {
            token: idx for token, idx in self.final_vocab.items()
            if token.startswith("##") or (token not in self.special_tokens and token not in self.stem_vocab)
        }
        
        logger.info(f"Loaded vocabulary with {len(self.final_vocab)} tokens")
    
    def get_vocabulary_stats(self) -> Dict:
        """Return vocabulary statistics."""
        return {
            'total_size': len(self.final_vocab),
            'num_stems': len(self.stem_vocab),
            'num_affixes': len(self.affix_vocab),
            'num_special_tokens': len(self.special_tokens),
            'stem_ratio': len(self.stem_vocab) / len(self.final_vocab) if self.final_vocab else 0,
            'affix_ratio': len(self.affix_vocab) / len(self.final_vocab) if self.final_vocab else 0
        }
    
    def is_stem(self, token: str) -> bool:
        """Check if token is a stem in vocabulary."""
        return token in self.stem_vocab
    
    def is_affix(self, token: str) -> bool:
        """Check if token is an affix in vocabulary."""
        return token in self.affix_vocab
    
    def is_special_token(self, token: str) -> bool:
        """Check if token is a special token."""
        return token in self.special_tokens
