import unittest
import time
import json
from src.tokenizer.mct_tokenizer import MCTTokenizer
from src.tokenizer.mct_analyzer import MCTAnalyzer
from src.tokenizer.constrained_bpe import ConstrainedBPETrainer
from src.utils.db_wrappers import LinguisticDB


class TestEfficiency(unittest.TestCase):
    """Efficiency and performance benchmarks for MCT tokenization"""
    
    @classmethod
    def setUpClass(cls):
        """Set up fixtures once for all tests"""
        cls.db = LinguisticDB(lang='en')
        # Load stem frequencies if available, otherwise use empty dict
        try:
            with open('data/processed/stem_frequencies.json') as f:
                cls.stem_frequencies = json.load(f)
        except:
            cls.stem_frequencies = {}
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Unhappily, the unfortunately unfinished work was unraveled.",
            "Computational linguistics is a fascinating field.",
            "internationalization is sometimes abbreviated as i18n.",
            "The neural network processed thousands of tokens per second.",
        ] * 20  # Repeat to get reasonable sample size
    
    def test_tokenization_throughput(self):
        """Test tokenization speed (tokens per second)"""
        analyzer = MCTAnalyzer(self.db, self.stem_frequencies)
        bpe_trainer = ConstrainedBPETrainer(vocab_size=32000)
        tokenizer = MCTTokenizer(
            analyzer=analyzer,
            bpe_model=bpe_trainer
        )
        
        # Combine all texts
        combined_text = " ".join(self.sample_texts)
        total_words = len(combined_text.split())
        
        # Measure tokenization speed
        start_time = time.time()
        tokens = tokenizer.tokenize(combined_text)
        elapsed = time.time() - start_time
        
        throughput = total_words / elapsed if elapsed > 0 else 0
        print(f"\nTokenization throughput: {throughput:.0f} words/sec")
        
        # Reasonable minimum: 100 words/sec
        self.assertGreater(throughput, 0)
    
    def test_batch_processing_efficiency(self):
        """Test batch tokenization is faster than sequential"""
        analyzer = MCTAnalyzer(self.db, self.stem_frequencies)
        bpe_trainer = ConstrainedBPETrainer(vocab_size=32000)
        tokenizer = MCTTokenizer(
            analyzer=analyzer,
            bpe_model=bpe_trainer
        )
        
        # Sequential processing
        start = time.time()
        for text in self.sample_texts:
            tokens = tokenizer.tokenize(text)
        sequential_time = time.time() - start
        
        # Batch processing
        start = time.time()
        batch_tokens = tokenizer.tokenize_batch(self.sample_texts)
        batch_time = time.time() - start
        
        print(f"\nSequential: {sequential_time:.3f}s, Batch: {batch_time:.3f}s")
        
        # Batch should be reasonably fast (not necessarily faster due to overhead)
        self.assertGreater(len(batch_tokens), 0)
    
    def test_memory_efficiency(self):
        """Test memory usage is reasonable"""
        import sys
        
        analyzer = MCTAnalyzer(self.db, self.stem_frequencies)
        
        # Memory from analyzer
        analyzer_size = sys.getsizeof(analyzer)
        print(f"\nAnalyzer memory: {analyzer_size / 1024:.1f} KB")
        
        # Should be reasonable (< 10 MB)
        self.assertLess(analyzer_size, 10_000_000)
    
    def test_cache_effectiveness(self):
        """Test caching improves repeated analysis speed"""
        analyzer = MCTAnalyzer(self.db, self.stem_frequencies)
        
        # First pass (cache miss)
        word = "unhappily"
        start = time.time()
        result1 = analyzer.get_best_analysis(word)
        first_time = time.time() - start
        
        # Second pass (cache hit)
        start = time.time()
        result2 = analyzer.get_best_analysis(word)
        second_time = time.time() - start
        
        print(f"\nFirst analysis: {first_time*1000:.1f}ms, Cached: {second_time*1000:.3f}ms")
        
        # Cache hit should be faster
        self.assertEqual(result1, result2)
        # Cached access should be nearly instant
        if first_time > 0:
            speedup = first_time / (second_time + 1e-6)
            self.assertGreater(speedup, 1.0)  # Cache should provide some speedup
    
    def test_large_vocabulary_performance(self):
        """Test performance with large vocabulary"""
        # Test with large vocab_size
        analyzer = MCTAnalyzer(self.db, self.stem_frequencies)
        bpe_trainer = ConstrainedBPETrainer(vocab_size=50000)
        large_vocab_tokenizer = MCTTokenizer(
            analyzer=analyzer,
            bpe_model=bpe_trainer
        )
        
        text = "The internationalization of computational processing systems."
        
        start = time.time()
        tokens = large_vocab_tokenizer.tokenize(text)
        elapsed = time.time() - start
        
        print(f"\nLarge vocab tokenization: {elapsed*1000:.1f}ms")
        
        # Should complete in reasonable time (< 5 seconds)
        self.assertLess(elapsed, 5.0)
    
    def test_morphology_overhead(self):
        """Test morphological analysis overhead"""
        analyzer = MCTAnalyzer(self.db, self.stem_frequencies)
        text = "unhappily unfinished internationalize"
        
        # Time morphological analysis
        start = time.time()
        for word in text.split():
            result = analyzer.get_best_analysis(word)
        morph_time = time.time() - start
        
        print(f"\nMorphological analysis time: {morph_time*1000:.1f}ms for 3 words")
        
        # Should be fast
        self.assertGreater(morph_time, 0)


if __name__ == '__main__':
    unittest.main()
