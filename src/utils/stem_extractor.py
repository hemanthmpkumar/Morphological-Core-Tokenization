"""
Utility for extracting stem frequencies from corpus for MCT vocabulary building.
"""

import json
import logging
from typing import Dict, List, Tuple
from collections import Counter
from tqdm import tqdm

logger = logging.getLogger(__name__)


class StemFrequencyExtractor:
    """Extracts stems and their frequencies from a corpus using morphological analysis."""
    
    def __init__(self, analyzer, min_stem_length: int = 2):
        """
        Args:
            analyzer: Morphological analyzer (must have analyze() method)
            min_stem_length: Minimum stem length to consider
        """
        self.analyzer = analyzer
        self.min_stem_length = min_stem_length
        self.stem_frequencies = Counter()
        self.affix_list = []
        self.analysis_cache = {}
    
    def extract_from_corpus(self, corpus: List[str], sample_size: int = None) -> Dict[str, int]:
        """
        Extract stem frequencies from a corpus.
        
        Args:
            corpus: List of sentences or words
            sample_size: If set, only process first sample_size items
            
        Returns:
            Dictionary mapping stems to their frequencies
        """
        items = corpus[:sample_size] if sample_size else corpus
        
        logger.info(f"Extracting stems from {len(items)} corpus items")
        
        for item in tqdm(items, desc="Extracting stems"):
            words = item.split() if isinstance(item, str) else [item]
            
            for word in words:
                if not word or len(word) < 2:
                    continue
                
                # Get morphological analysis
                analyses = self.analyzer.get_morphological_variants(word)
                
                if analyses:
                    # Take the first (most likely) analysis
                    stem, prefixes, suffixes = analyses[0]
                    
                    if len(stem) >= self.min_stem_length:
                        self.stem_frequencies[stem] += 1
                        
                        # Collect affixes
                        for prefix in prefixes:
                            if prefix:
                                self.affix_list.append(prefix)
                        for suffix in suffixes:
                            if suffix:
                                self.affix_list.append(suffix)
        
        logger.info(f"Extracted {len(self.stem_frequencies)} unique stems")
        return dict(self.stem_frequencies)
    
    def get_stem_freq_map(self, min_freq: int = 1) -> Dict[str, int]:
        """Get stem frequency map, filtering by minimum frequency."""
        return {
            stem: freq for stem, freq in self.stem_frequencies.items()
            if freq >= min_freq
        }
    
    def get_affixes(self) -> List[str]:
        """Get list of all extracted affixes."""
        return self.affix_list
    
    def save_stem_frequencies(self, filepath: str):
        """Save stem frequencies to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(dict(self.stem_frequencies), f, indent=2)
        logger.info(f"Stem frequencies saved to {filepath}")
    
    def load_stem_frequencies(self, filepath: str):
        """Load stem frequencies from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.stem_frequencies = Counter(data)
        logger.info(f"Loaded {len(self.stem_frequencies)} stems from {filepath}")
    
    def get_top_stems(self, k: int = 100) -> List[Tuple[str, int]]:
        """Get top k most frequent stems."""
        return self.stem_frequencies.most_common(k)
    
    def get_statistics(self) -> Dict:
        """Return extraction statistics."""
        freqs = list(self.stem_frequencies.values())
        return {
            'total_unique_stems': len(self.stem_frequencies),
            'total_stem_occurrences': sum(freqs),
            'mean_frequency': sum(freqs) / len(freqs) if freqs else 0,
            'max_frequency': max(freqs) if freqs else 0,
            'min_frequency': min(freqs) if freqs else 0,
            'total_affixes': len(self.affix_list),
            'unique_affixes': len(set(self.affix_list))
        }
