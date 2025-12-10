import nltk
from typing import List, Dict, Optional, Tuple
import random
import re
from collections import defaultdict, Counter
import heapq
from tqdm import tqdm

class MCTTokenizer:
    """
    Morphological-Core Tokenization (MCT) Algorithm.

    A novel hybrid tokenizer that preserves morphological cores and applies
    constrained BPE to affixes, enhanced with a stochastic dropout mechanism.
    """
    def __init__(self, vocab_size: int = 32000, p_drop: float = 0.05):
        """
        Initializes the Morphological-Core Tokenizer.
        
        Args:
            vocab_size (int): Target vocabulary size.
            [cite_start]p_drop (float): Probability of stem dropout (stochastic regularization). [cite: 44, 174]
        """
        self.vocab_size = vocab_size
        self.p_drop = p_drop
        self.vocab: Dict[str, int] = {}
        self.wordnet_analyzer: Optional[nltk.stem.WordNetLemmatizer] = None
        self.stems: Dict[str, str] = {}  # Cache for word->stem mapping
        self.bpe_merges: Dict[Tuple[str, str], int] = {}  # BPE merge rules for affixes
        self.affixes_vocab: Dict[str, int] = {}  # Vocabulary for affixes
        self.fallback_tokenizer = None  # Placeholder for fallback BPE tokenizer
        
        # [cite_start]Ensure NLTK data is available for Stage 1: Core Identification [cite: 37, 164]
        try:
            nltk.data.find('corpora/wordnet')
            self.wordnet_analyzer = nltk.stem.WordNetLemmatizer()
        except LookupError:
            print("Downloading NLTK WordNet data for morphological analysis...")
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            self.wordnet_analyzer = nltk.stem.WordNetLemmatizer()

    def _get_stem(self, word: str) -> str:
        """
        Identify morphological stem using WordNet lemmatizer.
        
        Args:
            word (str): Input word.
            
        Returns:
            str: Identified stem.
        """
        if word in self.stems:
            return self.stems[word]
        
        # Try different POS tags to find the best stem
        stem = self.wordnet_analyzer.lemmatize(word, 'n')  # Noun
        if stem == word:
            stem = self.wordnet_analyzer.lemmatize(word, 'v')  # Verb
        if stem == word:
            stem = self.wordnet_analyzer.lemmatize(word, 'a')  # Adjective
        if stem == word:
            stem = self.wordnet_analyzer.lemmatize(word, 'r')  # Adverb
            
        self.stems[word] = stem
        return stem

    def _extract_affixes(self, word: str, stem: str) -> Tuple[str, str]:
        """
        Extract prefixes and suffixes from a word given its stem.
        
        Args:
            word (str): Full word.
            stem (str): Identified stem.
            
        Returns:
            Tuple[str, str]: (prefix, suffix) pair.
        """
        if stem == word:
            return "", ""
            
        # Find stem position in word
        idx = word.find(stem)
        if idx == -1:
            # Stem not found directly (possible due to spelling variations)
            # Use heuristic: take longest common substring
            # This is a simplification; in practice, more sophisticated alignment might be needed
            for i in range(len(word)):
                for j in range(i+1, len(word)+1):
                    if word[i:j] in stem or stem in word[i:j]:
                        stem = word[i:j]
                        idx = i
                        break
                if idx != -1:
                    break
                    
        if idx == -1:
            return "", word  # Treat entire word as suffix if stem not found
            
        prefix = word[:idx]
        suffix = word[idx + len(stem):]
        return prefix, suffix

    def _train_bpe_on_affixes(self, affixes: List[str], target_size: int):
        """
        Train BPE on affixes with the constraint that merges don't cross stem boundaries.
        
        Args:
            affixes (List[str]): List of affixes (prefixes and suffixes).
            target_size (int): Target vocabulary size for affixes.
        """
        # Convert affixes to character sequences with end-of-affix marker
        vocab = Counter([' '.join(list(affix)) + ' </a>' for affix in affixes])
        
        # Initialize with individual characters
        bpe_vocab = set()
        for word in vocab:
            for char in word.split():
                bpe_vocab.add(char)
        
        # BPE training
        merges = {}
        num_merges = target_size - len(bpe_vocab)
        
        for i in range(num_merges):
            pairs = defaultdict(int)
            
            # Count frequency of pairs
            for word, freq in vocab.items():
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pairs[(symbols[j], symbols[j+1])] += freq
            
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Add merge to vocabulary
            merged = best_pair[0] + best_pair[1]
            merges[best_pair] = i
            bpe_vocab.add(merged)
            
            # Update vocabulary
            new_vocab = {}
            for word, freq in vocab.items():
                new_word = word
                symbols = word.split()
                j = 0
                while j < len(symbols) - 1:
                    if (symbols[j], symbols[j+1]) == best_pair:
                        symbols[j] = merged
                        symbols.pop(j+1)
                    else:
                        j += 1
                new_vocab[' '.join(symbols)] = freq
            vocab = new_vocab
            
        self.bpe_merges = merges
        self.affixes_vocab = {token: idx for idx, token in enumerate(bpe_vocab)}

    def train(self, files: List[str]):
        """
        Trains the MCT vocabulary in two stages:
        1. Morphological Core Identification (Stage 1)
        2. Constrained Affix Segmentation (Stage 2)
        
        Args:
            files (List[str]): List of corpus file paths to train on.
        """
        print(f"[INFO] Starting training of MCT tokenizer on {len(files)} files...")
        
        # [cite_start]Stage 1: Morphological Core Identification [cite: 34, 37, 161, 164]
        print("[INFO] Stage 1: Identifying Morphological Cores (WordNet Lemmatization)...")
        
        stems_set = set()
        affixes_list = []
        word_freq = Counter()
        
        # Process corpus files
        for file_path in tqdm(files, desc="Processing files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                # Tokenize into words
                words = re.findall(r'\b\w+\b', text.lower())
                word_freq.update(words)
                
                # Process each word
                for word in words:
                    # Get stem
                    stem = self._get_stem(word)
                    stems_set.add(stem)
                    
                    # Extract affixes
                    prefix, suffix = self._extract_affixes(word, stem)
                    if prefix:
                        affixes_list.append(prefix)
                    if suffix:
                        affixes_list.append(suffix)
                        
            except Exception as e:
                print(f"Warning: Could not process file {file_path}: {e}")
                continue
        
        # Add stems to vocabulary
        stems_list = sorted(list(stems_set))
        num_stems = min(len(stems_list), self.vocab_size // 2)  # Reserve half for stems
        for i, stem in enumerate(stems_list[:num_stems]):
            self.vocab[stem] = i
        
        # [cite_start]Stage 2: Constrained Affix Segmentation [cite: 34, 39, 161, 168]
        print("[INFO] Stage 2: Learning Affix BPE merges with boundary constraint...")
        
        # Train BPE on affixes
        affixes_target_size = self.vocab_size - num_stems - 2  # Reserve space for special tokens
        self._train_bpe_on_affixes(affixes_list, affixes_target_size)
        
        # Add affixes to vocabulary
        start_idx = num_stems
        for affix, idx in self.affixes_vocab.items():
            if affix not in ['</a>', ' ']:
                clean_affix = affix.replace('</a>', '').replace(' ', '')
                if clean_affix:  # Skip empty strings
                    self.vocab[clean_affix] = start_idx + idx
        
        # Add special tokens
        self.vocab['<pad>'] = len(self.vocab)
        self.vocab['<unk>'] = len(self.vocab)
        self.vocab['<s>'] = len(self.vocab)
        self.vocab['</s>'] = len(self.vocab)
        
        print(f"[INFO] Training complete. Vocabulary size: {len(self.vocab)}")
        print(f"[INFO] Stems: {num_stems}, Affixes: {len(self.affixes_vocab)}")

    def _apply_bpe_to_affix(self, affix: str) -> List[str]:
        """
        Apply BPE to an affix using trained merge rules.
        
        Args:
            affix (str): Input affix.
            
        Returns:
            List[str]: BPE-tokenized affix.
        """
        if not affix:
            return []
            
        # Convert to character sequence with end marker
        chars = list(affix) + ['</a>']
        
        # Apply BPE merges
        while len(chars) > 1:
            pairs = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
            
            # Find merge with highest priority (lowest merge index)
            best_pair = None
            best_idx = float('inf')
            
            for pair in pairs:
                if pair in self.bpe_merges and self.bpe_merges[pair] < best_idx:
                    best_idx = self.bpe_merges[pair]
                    best_pair = pair
            
            if best_pair is None:
                break
                
            # Apply the merge
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i+1]) == best_pair:
                    new_chars.append(chars[i] + chars[i+1])
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        
        # Remove end marker and return
        tokens = []
        for token in chars:
            if token != '</a>':
                tokens.append(token)
        return tokens

    def encode(self, text: str) -> List[str]:
        """
        Tokenizes text using the trained MCT vocabulary.
        
        Applies Stem Dropout for stochastic regularization if p_drop > 0.
        
        Args:
            text (str): The input text to tokenize.
            
        Returns:
            List[str]: The list of tokens.
        """
        # [cite_start]1. Apply Stem Dropout (Stochastic Regularization) [cite: 43, 47, 172, 177]
        if random.random() < self.p_drop:
            # Fallback to word-level tokenization (simulating BPE fragmentation)
            words = re.findall(r'\b\w+\b', text.lower())
            fragmented_tokens = []
            for word in words:
                # Simulate BPE fragmentation by splitting into characters
                fragmented_tokens.extend(list(word))
            return fragmented_tokens

        # 2. Main MCT Tokenization Logic
        tokens = []
        words = re.findall(r'\b\w+\b', text.lower())
        
        for word in words:
            # Get stem
            stem = self._get_stem(word)
            
            # Extract affixes
            prefix, suffix = self._extract_affixes(word, stem)
            
            # Tokenize affixes using BPE
            prefix_tokens = self._apply_bpe_to_affix(prefix) if prefix else []
            suffix_tokens = self._apply_bpe_to_affix(suffix) if suffix else []
            
            # Combine tokens: [prefix_tokens..., stem, suffix_tokens...]
            if stem in self.vocab:
                tokens.extend(prefix_tokens)
                tokens.append(stem)
                tokens.extend(suffix_tokens)
            else:
                # If stem not in vocabulary, use <unk> or fallback
                if word in self.vocab:
                    tokens.append(word)
                else:
                    # Try to segment the word completely with BPE
                    word_tokens = self._apply_bpe_to_affix(word)
                    if word_tokens:
                        tokens.extend(word_tokens)
                    else:
                        tokens.append('<unk>')
        
        return tokens

    def save_model(self, path: str):
        """
        Saves the vocabulary and learned BPE merges to disk for deployment.
        """
        print(f"[INFO] Saving MCT model to {path}...")
        
        import json
        import pickle
        
        # Save vocabulary
        with open(f"{path}_vocab.json", 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # Save BPE merges
        with open(f"{path}_bpe_merges.pkl", 'wb') as f:
            pickle.dump(self.bpe_merges, f)
        
        # Save stems cache
        with open(f"{path}_stems.json", 'w', encoding='utf-8') as f:
            json.dump(self.stems, f, ensure_ascii=False, indent=2)
        
        # Save configuration
        config = {
            'vocab_size': self.vocab_size,
            'p_drop': self.p_drop,
        }
        with open(f"{path}_config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] Model saved successfully to {path}_*")

    def load_model(self, path: str):
        """
        Loads a saved MCT model from disk.
        """
        import json
        import pickle
        
        print(f"[INFO] Loading MCT model from {path}...")
        
        # Load configuration
        with open(f"{path}_config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.vocab_size = config['vocab_size']
            self.p_drop = config['p_drop']
        
        # Load vocabulary
        with open(f"{path}_vocab.json", 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        # Load BPE merges
        with open(f"{path}_bpe_merges.pkl", 'rb') as f:
            self.bpe_merges = pickle.load(f)
        
        # Load stems cache
        with open(f"{path}_stems.json", 'r', encoding='utf-8') as f:
            self.stems = json.load(f)
        
        # Reconstruct affixes vocabulary from BPE merges
        self.affixes_vocab = {}
        affix_tokens = set()
        for pair in self.bpe_merges:
            affix_tokens.add(pair[0])
            affix_tokens.add(pair[1])
            affix_tokens.add(pair[0] + pair[1])
        
        for idx, token in enumerate(sorted(affix_tokens)):
            self.affixes_vocab[token] = idx
        
        print(f"[INFO] Model loaded successfully. Vocabulary size: {len(self.vocab)}")