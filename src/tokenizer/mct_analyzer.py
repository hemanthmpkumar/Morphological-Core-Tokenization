import abc
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BaseMorphologicalAnalyzer(abc.ABC):
    @abc.abstractmethod
    def analyze(self, word: str) -> List[Tuple[str, List[str], List[str]]]:
        """Returns a list of possible (stem, prefixes, suffixes) tuples."""
        pass

class MCTAnalyzer:
    def __init__(self, analyzer: BaseMorphologicalAnalyzer, stem_freq_map: Dict[str, int], 
                 min_stem_length: int = 2, min_freq: int = 1):
        """
        Args:
            analyzer: A wrapper for a linguistic database like WordNet or Stanza[cite: 29].
            stem_freq_map: Frequency of stems in the training corpus for ambiguity resolution[cite: 82, 85].
            min_stem_length: Minimum stem length to consider valid (prevents over-segmentation).
            min_freq: Minimum frequency threshold for stems.
        """
        self.analyzer = analyzer
        self.stem_freq_map = stem_freq_map
        self.min_stem_length = min_stem_length
        self.min_freq = min_freq
        self.cache = {}  # Cache morphological analyses

    def get_best_analysis(self, word: str) -> Optional[Tuple[str, List[str], List[str]]]:
        """
        Implements the formula: (s*, P*, S*) = arg max Freq(s) * Confidence(s, w)[cite: 84].
        Returns the most confident morphological analysis based on stem frequency.
        """
        # Check cache first
        if word in self.cache:
            return self.cache[word]
        
        # Get all possible analyses [cite: 79-81]
        # Use get_morphological_variants method from LinguisticDB
        if hasattr(self.analyzer, 'get_morphological_variants'):
            analyses = self.analyzer.get_morphological_variants(word)
        else:
            analyses = self.analyzer.analyze(word)
        
        if not analyses:
            self.cache[word] = None
            return None

        # Filter valid analyses (stem must be above minimum length)
        valid_analyses = [
            (stem, prefixes, suffixes) 
            for stem, prefixes, suffixes in analyses
            if len(stem) >= self.min_stem_length
        ]
        
        if not valid_analyses:
            self.cache[word] = None
            return None

        # Ambiguity Resolution: Pick stem with highest frequency [cite: 82, 85]
        best_analysis = max(
            valid_analyses,
            key=lambda x: self.stem_freq_map.get(x[0], self.min_freq)
        )
        
        # Cache result
        self.cache[word] = best_analysis
        return best_analysis
    
    def clear_cache(self):
        """Clear morphological analysis cache."""
        self.cache.clear()
