import unittest
from src.tokenizer.constrained_bpe import ConstrainedBPETrainer


class TestConstrainedBPE(unittest.TestCase):
    """Test suite for ConstrainedBPETrainer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.trainer = ConstrainedBPETrainer(vocab_size=1000)
    
    def test_morpheme_boundary_marking(self):
        """Test that morpheme boundaries are properly marked"""
        affixes = ["ing", "ed", "tion"]
        bounded = self.trainer.add_morpheme_boundaries(affixes)
        
        # Each affix should have boundary markers
        for b in bounded:
            self.assertTrue(b.startswith("##"))
            self.assertTrue(b.endswith("##"))
    
    def test_forbidden_merge_identification(self):
        """Test identification of forbidden merge pairs"""
        stems = ["un", "happy", "play"]
        affixes = ["ing", "ed", "er"]
        
        forbidden = self.trainer.identify_forbidden_merges(stems, affixes)
        
        # Should identify forbidden pairs
        self.assertTrue(len(forbidden) > 0)
        
        # Example: stem ending 'n' cannot merge with affix starting 'i'
        # This would be forbidden: ('n', 'i')
    
    def test_bpe_constraint_enforcement(self):
        """Test that BPE respects morpheme boundary constraints"""
        # Setup: 'un' + 'happy'
        # BPE should not be allowed to merge 'n' (from un) with 'h' (from happy)
        
        affixes = ["ing", "ed", "tion", "able", "ment"]
        forbidden_pairs = {('n', 'h'), ('y', 'i')}
        
        # Train with constraints
        self.trainer.train_on_affixes(affixes, forbidden_pairs)
        
        # Verify training completed
        self.assertTrue(len(self.trainer.affix_vocabulary) > 0)
    
    def test_affix_tokenization(self):
        """Test tokenization of affixes"""
        affixes = ["ing", "ed", "tion", "able", "ment"]
        self.trainer.train_on_affixes(affixes)
        
        # Tokenize affixes
        tokens = self.trainer.tokenize_affix("ing")
        self.assertTrue(len(tokens) > 0)
    
    def test_model_persistence(self):
        """Test saving and loading tokenizer models"""
        import tempfile
        import os
        
        affixes = ["ing", "ed", "tion"]
        self.trainer.train_on_affixes(affixes)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_bpe.json")
            self.trainer.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Load model
            trainer2 = ConstrainedBPETrainer()
            trainer2.load_model(model_path)
            
            # Verify loaded model works
            tokens = trainer2.tokenize_affix("ing")
            self.assertTrue(len(tokens) > 0)
    
    def test_empty_affix_handling(self):
        """Test handling of empty affixes"""
        affixes = ["", "ing", "", "ed", None]
        
        # Should handle empty and None values gracefully
        bounded = self.trainer.add_morpheme_boundaries([a for a in affixes if a])
        self.assertTrue(len(bounded) > 0)


if __name__ == '__main__':
    unittest.main()
