import stanza
import nltk
from nltk.corpus import wordnet as wn
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class LinguisticDB:
    """
    Linguistic database wrapper supporting Stanza and WordNet for morphological analysis.
    """
    
    def __init__(self, lang='en'):
        """
        Initialize linguistic database.
        
        Args:
            lang: Language code ('en', 'de', 'fi', etc.)
        """
        self.lang = lang
        
        try:
            # Initialize Stanza for lemmatization and morphological feature extraction
            logger.info(f"Loading Stanza pipeline for language: {lang}")
            self.nlp = stanza.Pipeline(lang=lang, processors='tokenize,mwt,pos,lemma,depparse')
        except Exception as e:
            logger.warning(f"Failed to load Stanza for {lang}: {e}")
            self.nlp = None
        
        # Download WordNet for English
        if lang == 'en':
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                logger.info("Downloading WordNet...")
                nltk.download('wordnet', quiet=True)

    def get_morphological_variants(self, word: str) -> List[Tuple[str, List[str], List[str]]]:
        """
        Returns possible analyses for a word.
        Format: [(stem, [prefixes], [suffixes]), ...]
        
        Args:
            word: Word to analyze
            
        Returns:
            List of (stem, prefixes, suffixes) tuples
        """
        results = []
        
        # Try Stanza analysis first
        if self.nlp:
            try:
                doc = self.nlp(word)
                
                for sent in doc.sentences:
                    for word_obj in sent.words:
                        lemma = word_obj.lemma if word_obj.lemma else word_obj.text
                        
                        # Extract affixes by comparing word and lemma
                        prefixes, suffixes = self._derive_affixes(word, lemma)
                        
                        # Avoid adding invalid analyses
                        if len(lemma) >= 2:  # Minimum stem length
                            results.append((lemma, prefixes, suffixes))
            except Exception as e:
                logger.warning(f"Stanza analysis failed for '{word}': {e}")
        
        # Try WordNet for English
        if self.lang == 'en' and not results:
            try:
                wordnet_results = self._wordnet_analysis(word)
                results.extend(wordnet_results)
            except Exception as e:
                logger.warning(f"WordNet analysis failed for '{word}': {e}")
        
        # Fallback: treat whole word as stem with no affixes
        if not results:
            results.append((word, [], []))
        
        return results

    def _derive_affixes(self, word: str, lemma: str) -> Tuple[List[str], List[str]]:
        """
        Helper to isolate affixes by comparing word and lemma.
        
        Args:
            word: Original word form
            lemma: Lemmatized form
            
        Returns:
            (prefixes, suffixes) tuple
        """
        prefixes = []
        suffixes = []
        
        # Handle suffix stripping (most common case)
        if word.endswith(lemma) and len(lemma) > 0:
            suffix = word[len(lemma):]
            if suffix:
                suffixes.append(suffix)
        
        # Handle prefix stripping
        elif word.startswith(lemma) and len(lemma) > 0:
            prefix = word[:-len(lemma)]
            if prefix:
                prefixes.append(prefix)
        
        # Handle both prefix and suffix
        elif len(lemma) > 0:
            # Try to find lemma as substring
            idx = word.find(lemma)
            if idx > 0:
                prefix = word[:idx]
                if prefix:
                    prefixes.append(prefix)
            if idx >= 0 and idx + len(lemma) < len(word):
                suffix = word[idx + len(lemma):]
                if suffix:
                    suffixes.append(suffix)
        
        return prefixes, suffixes

    def _wordnet_analysis(self, word: str) -> List[Tuple[str, List[str], List[str]]]:
        """
        Perform morphological analysis using WordNet.
        
        Args:
            word: Word to analyze
            
        Returns:
            List of (stem, prefixes, suffixes) tuples
        """
        results = []
        
        # Get WordNet lemmas
        lemmas = wn.lemmas(word)
        
        if not lemmas:
            return results
        
        seen = set()
        for lemma in lemmas:
            lemma_name = lemma.name()
            
            if lemma_name in seen:
                continue
            seen.add(lemma_name)
            
            # Extract stem from lemma
            stem = lemma_name.replace('_', '')  # Handle multi-word lemmas
            
            if len(stem) >= 2:
                prefixes, suffixes = self._derive_affixes(word, stem)
                results.append((stem, prefixes, suffixes))
        
        return results
    
    def get_pos_tag(self, word: str) -> Optional[str]:
        """Get part-of-speech tag for a word."""
        if not self.nlp:
            return None
        
        try:
            doc = self.nlp(word)
            for sent in doc.sentences:
                for word_obj in sent.words:
                    return word_obj.pos
        except:
            pass
        
        return None
    
    def is_known_word(self, word: str) -> bool:
        """Check if word is in linguistic database."""
        if self.lang == 'en':
            return len(wn.synsets(word)) > 0
        
        # For other languages, check if Stanza can analyze it
        if self.nlp:
            try:
                doc = self.nlp(word)
                return len(doc.sentences) > 0
            except:
                return False
        
        return True
