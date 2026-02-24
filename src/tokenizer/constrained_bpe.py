from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from typing import List, Set, Tuple
import logging

logger = logging.getLogger(__name__)

class ConstrainedBPETrainer:
    """
    Constrained BPE that enforces morpheme boundaries.
    BPE merges are restricted to within affixes, not across morpheme boundaries.
    """
    
    def __init__(self, vocab_size: int = 32000, morpheme_boundary_token: str = "##"):
        """
        Args:
            vocab_size: Target vocabulary size
            morpheme_boundary_token: Token marking morpheme boundaries (default: ##)
        """
        self.vocab_size = vocab_size
        self.morpheme_boundary_token = morpheme_boundary_token
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.merge_constraints = set()  # Stores forbidden character pairs
        self.affix_vocabulary = set()  # Vocabulary built from affixes only

    def add_morpheme_boundaries(self, affix_corpus: List[str]) -> List[str]:
        """
        Marks morpheme boundaries in affix corpus to prevent cross-boundary merges.
        
        Args:
            affix_corpus: List of prefixes/suffixes (already separated from stems)
            
        Returns:
            Corpus with boundary markers for constrained BPE training
        """
        bounded_corpus = []
        for affix in affix_corpus:
            if not affix:
                continue
            # Add boundary markers at affix edges
            # This prevents BPE from merging across boundaries
            bounded_affix = f"{self.morpheme_boundary_token}{affix}{self.morpheme_boundary_token}"
            bounded_corpus.append(bounded_affix)
        return bounded_corpus

    def identify_forbidden_merges(self, stems: List[str], affixes: List[str]) -> Set[Tuple[str, str]]:
        """
        Identify character pairs that should NOT be merged together.
        Prevents merging stem-final chars with affix-initial chars.
        
        Args:
            stems: List of word stems
            affixes: List of prefixes/suffixes
            
        Returns:
            Set of forbidden character pair tuples
        """
        forbidden = set()
        
        # Collect boundary characters
        stem_final_chars = set()
        affix_initial_chars = set()
        
        for stem in stems:
            if stem:
                stem_final_chars.add(stem[-1])
        
        for affix in affixes:
            if affix:
                affix_initial_chars.add(affix[0])
        
        # All combinations of (stem-final, affix-initial) are forbidden
        for final_char in stem_final_chars:
            for initial_char in affix_initial_chars:
                forbidden.add((final_char, initial_char))
        
        logger.info(f"Identified {len(forbidden)} forbidden merge pairs")
        return forbidden

    def train_on_affixes(self, affix_corpus: List[str], forbidden_pairs: Set[Tuple[str, str]] = None):
        """
        Trains BPE specifically on affix portions with constraint enforcement[cite: 117].
        
        Args:
            affix_corpus: List of affixes (prefixes and suffixes)
            forbidden_pairs: Set of character pairs that cannot be merged
        """
        self.merge_constraints = forbidden_pairs if forbidden_pairs else set()
        
        # Add boundary markers to prevent cross-morpheme merges
        bounded_corpus = self.add_morpheme_boundaries(affix_corpus)
        
        logger.info(f"Training BPE on {len(bounded_corpus)} affix instances")
        
        # Train standard BPE
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            show_progress=True,
            special_tokens=[self.morpheme_boundary_token, "[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        )
        self.tokenizer.train_from_iterator(bounded_corpus, trainer=trainer)
        
        # Store affix vocabulary
        self.affix_vocabulary = set(self.tokenizer.get_vocab().keys())
        logger.info(f"BPE training complete. Vocabulary size: {len(self.affix_vocabulary)}")

    def tokenize_affix(self, affix: str) -> List[str]:
        """
        Applies trained BPE to a specific prefix or suffix[cite: 123].
        Enforces that resulting tokens respect morpheme boundaries.
        
        Args:
            affix: A prefix or suffix string
            
        Returns:
            List of BPE tokens
        """
        if not affix:
            return []
        
        # Add boundary markers
        bounded_affix = f"{self.morpheme_boundary_token}{affix}{self.morpheme_boundary_token}"
        tokens = self.tokenizer.encode(bounded_affix).tokens
        
        # Remove boundary markers from output
        tokens = [
            t for t in tokens 
            if t != self.morpheme_boundary_token and t != ""
        ]
        
        return tokens

    def save_model(self, path: str):
        """Save trained BPE tokenizer to file."""
        self.tokenizer.save(path)
        logger.info(f"Tokenizer saved to {path}")

    def load_model(self, path: str):
        """Load pre-trained BPE tokenizer from file."""
        self.tokenizer = Tokenizer.from_file(path)
        self.affix_vocabulary = set(self.tokenizer.get_vocab().keys())
        logger.info(f"Tokenizer loaded from {path}")

    def train(self, train_file: str, language: str = None, batch_size: int = 1000):
        """
        Convenience wrapper to train a full BPE model from a JSONL training file.
        This provides a consistent API for external scripts expecting `trainer.train(...)`.

        Args:
            train_file: Path to a JSONL file (records with language keys or 'text')
            language: Optional language key to extract from records
            batch_size: Number of lines per iterator batch
        """
        logger.info(f"Training full BPE from {train_file} (batch_size={batch_size})")

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            show_progress=True,
            special_tokens=[self.morpheme_boundary_token, "[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        )

        def get_corpus():
            with open(train_file, 'r', encoding='utf-8') as f:
                batch = []
                for line in f:
                    try:
                        record = json.loads(line)
                        text = None
                        if language:
                            text = record.get(language, '')
                        if not text:
                            text = record.get('text', '')
                        if text:
                            batch.append(text)
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                    except Exception:
                        continue
                if batch:
                    yield batch

        # Train tokenizer on the full corpus
        self.tokenizer.train_from_iterator(get_corpus(), trainer=trainer)
        # Update affix_vocabulary with resulting vocab
        try:
            self.affix_vocabulary = set(self.tokenizer.get_vocab().keys())
        except Exception:
            self.affix_vocabulary = set()
        logger.info(f"Full BPE training complete. Vocabulary size: {len(self.affix_vocabulary)}")
