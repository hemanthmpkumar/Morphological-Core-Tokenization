import unittest
from unittest.mock import Mock, MagicMock
from src.tokenizer.mct_tokenizer import MCTTokenizer
from src.tokenizer.mct_analyzer import MCTAnalyzer
from src.tokenizer.constrained_bpe import ConstrainedBPETrainer


class TestMCTTokenizer(unittest.TestCase):
    """Test suite for MCTTokenizer"""
    
    def setUp(self):
        """Set up mock components for testing"""
        # Create mock analyzer
        self.mock_analyzer = Mock(spec=MCTAnalyzer)
        
        # Create mock BPE trainer
        self.mock_bpe = Mock(spec=ConstrainedBPETrainer)
        self.mock_bpe.tokenize_affix = Mock(return_value=['test', 'ing'])
    
    def test_stem_dropout_logic(self):
        """Test that stem dropout fallback works correctly"""
        # Set p_drop to 1.0 (Always fallback to BPE)
        tokenizer_fallback = MCTTokenizer(
            analyzer=self.mock_analyzer,
            bpe_model=self.mock_bpe,
            p_drop=1.0
        )
        
        # Set p_drop to 0.0 (Always use morphological analysis)
        self.mock_analyzer.get_best_analysis = Mock(
            return_value=("test", [], ["ing"])
        )
        tokenizer_mct = MCTTokenizer(
            analyzer=self.mock_analyzer,
            bpe_model=self.mock_bpe,
            p_drop=0.0
        )
        
        # Test fallback mode
        tokens_fallback = tokenizer_fallback.tokenize("testing")
        self.assertTrue(len(tokens_fallback) > 0)
        
        # Test morphology mode
        tokens_mct = tokenizer_mct.tokenize("testing")
        self.assertIn("test", tokens_mct)  # Stem should be preserved
    
    def test_atomic_stem_preservation(self):
        """Ensure the stem is not fragmented by BPE"""
        self.mock_analyzer.get_best_analysis = Mock(
            return_value=("happy", ["un"], [])
        )
        self.mock_bpe.tokenize_affix = Mock(side_effect=lambda x: [x])
        
        tokenizer = MCTTokenizer(
            analyzer=self.mock_analyzer,
            bpe_model=self.mock_bpe,
            p_drop=0.0
        )
        
        tokens = tokenizer.tokenize("unhappy")
        # Stem is prefixed with [STEM] by default, and prefixes are atomic
        self.assertIn("[STEM]happy", tokens)  # Stem must remain atomic
        self.assertTrue(any("happy" in t for t in tokens))
    
    def test_tokenization_with_statistics(self):
        """Test tokenization with statistics tracking"""
        self.mock_analyzer.get_best_analysis = Mock(
            return_value=("run", [], ["ing"])
        )
        self.mock_bpe.tokenize_affix = Mock(return_value=['ing'])
        
        tokenizer = MCTTokenizer(
            analyzer=self.mock_analyzer,
            bpe_model=self.mock_bpe,
            p_drop=0.0
        )
        
        tokens = tokenizer.tokenize("running", track_stats=True)
        stats = tokenizer.get_stats()
        
        self.assertEqual(stats['total_words'], 1)
        self.assertEqual(stats['words_using_morphology'], 1)
        self.assertEqual(stats['morphology_rate'], 1.0)
    
    def test_batch_tokenization(self):
        """Test batch tokenization"""
        self.mock_analyzer.get_best_analysis = Mock(
            return_value=("test", [], [])
        )
        self.mock_bpe.tokenize_affix = Mock(return_value=['test'])
        
        tokenizer = MCTTokenizer(
            analyzer=self.mock_analyzer,
            bpe_model=self.mock_bpe,
            p_drop=0.0
        )
        
        texts = ["testing one two", "testing three four"]
        batch_tokens = tokenizer.tokenize_batch(texts)
        
        self.assertEqual(len(batch_tokens), 2)
        self.assertTrue(all(isinstance(t, list) for t in batch_tokens))


if __name__ == '__main__':
    unittest.main()
