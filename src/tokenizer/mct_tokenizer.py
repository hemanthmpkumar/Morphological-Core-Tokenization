import random
from typing import List
from .mct_analyzer import MCTAnalyzer
from .constrained_bpe import ConstrainedBPETrainer
import logging

logger = logging.getLogger(__name__)

class MCTTokenizer:
    def __init__(self, analyzer: MCTAnalyzer, bpe_model: ConstrainedBPETrainer, p_drop: float = 0.05,
                 stem_token_prefix: str = "[STEM]", affix_token_prefix: str = ""):
        """
        Args:
            analyzer: The MCTAnalyzer instance[cite: 29].
            bpe_model: The trained ConstrainedBPETrainer.
            p_drop: Probability of bypassing morphological analysis (Default: 0.05)[cite: 97].
            stem_token_prefix: Prefix for stem tokens (optional)
            affix_token_prefix: Prefix for affix tokens (optional)
        """
        self.analyzer = analyzer
        self.bpe = bpe_model
        self.p_drop = p_drop
        self.stem_token_prefix = stem_token_prefix
        self.affix_token_prefix = affix_token_prefix
        self.stats = {
            'total_words': 0,
            'words_using_morphology': 0,
            'words_using_bpe_fallback': 0,
            'analysis_failures': 0
        }

    def tokenize(self, text: str, track_stats: bool = False) -> List[str]:
        """Implements Algorithm 1: Morphological-Core Tokenization[cite: 121]."""
        final_tokens = []
        words = text.split()  # Simplified word segmentation [cite: 125]

        for w in words:
            if not w:  # Skip empty strings
                continue
            
            if track_stats:
                self.stats['total_words'] += 1
            
            # Step 1: Stochastic Stem Dropout [cite: 31, 92, 123]
            if random.random() < self.p_drop:
                # Fallback to pure BPE tokenization
                final_tokens.extend(self.bpe.tokenize_affix(w))
                if track_stats:
                    self.stats['words_using_bpe_fallback'] += 1
                continue

            # Step 2: Morphological Analysis [cite: 104, 123]
            analysis = self.analyzer.get_best_analysis(w)
            if not analysis:
                # Fallback to BPE if analysis fails [cite: 106, 123]
                final_tokens.extend(self.bpe.tokenize_affix(w))
                if track_stats:
                    self.stats['analysis_failures'] += 1
                continue

            stem, prefixes, suffixes = analysis  # [cite: 123]
            
            if track_stats:
                self.stats['words_using_morphology'] += 1

            # Step 3: Segment Prefixes [cite: 110, 123]
            for p in prefixes:
                prefix_tokens = self.bpe.tokenize_affix(p)
                if self.affix_token_prefix:
                    prefix_tokens = [f"{self.affix_token_prefix}{t}" for t in prefix_tokens]
                final_tokens.extend(prefix_tokens)

            # Step 4: Atomic Stem Preservation [cite: 74, 91, 123]
            stem_token = stem
            if self.stem_token_prefix:
                stem_token = f"{self.stem_token_prefix}{stem}"
            final_tokens.append(stem_token)

            # Step 5: Segment Suffixes [cite: 110, 123]
            for s in suffixes:
                suffix_tokens = self.bpe.tokenize_affix(s)
                if self.affix_token_prefix:
                    suffix_tokens = [f"{self.affix_token_prefix}{t}" for t in suffix_tokens]
                final_tokens.extend(suffix_tokens)

        return final_tokens
    
    def tokenize_batch(self, texts: List[str], track_stats: bool = False) -> List[List[str]]:
        """Tokenize a batch of texts."""
        return [self.tokenize(text, track_stats=track_stats) for text in texts]
    
    def get_stats(self) -> dict:
        """Return tokenization statistics."""
        total = self.stats['total_words']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'morphology_rate': self.stats['words_using_morphology'] / total,
            'bpe_fallback_rate': self.stats['words_using_bpe_fallback'] / total,
            'failure_rate': self.stats['analysis_failures'] / total
        }
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            'total_words': 0,
            'words_using_morphology': 0,
            'words_using_bpe_fallback': 0,
            'analysis_failures': 0
        }
