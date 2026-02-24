import unittest
from src.utils.metrics import (
    compute_bleu,
    compute_chrf,
    compute_rouge,
    compute_stem_accuracy,
)


class TestMetrics(unittest.TestCase):
    """Test suite for evaluation metrics"""
    
    def test_bleu_score_basic(self):
        """Test BLEU score computation"""
        hypothesis = "the cat is on the mat"
        reference = "the cat is on the mat"
        
        # Perfect match should give high BLEU
        bleu = compute_bleu(hypothesis, [reference])
        self.assertGreater(bleu, 0.9)
    
    def test_bleu_score_partial_match(self):
        """Test BLEU with partial match"""
        hypothesis = "the cat is on the"
        reference = "the cat is on the mat"
        
        bleu = compute_bleu(hypothesis, [reference])
        # Partial match should have lower BLEU than perfect
        self.assertGreater(bleu, 0.0)
        self.assertLess(bleu, 1.0)
    
    def test_bleu_no_match(self):
        """Test BLEU with completely different strings"""
        hypothesis = "dog bird fish"
        reference = "cat elephant mouse"
        
        bleu = compute_bleu(hypothesis, [reference])
        # Should be very low
        self.assertLess(bleu, 0.5)
    
    def test_chrf_score(self):
        """Test ChrF (character F-score) computation"""
        hypothesis = "the cat"
        reference = "the cat"
        
        chrf = compute_chrf(hypothesis, reference)
        # Perfect match
        self.assertGreater(chrf, 0.9)
    
    def test_chrf_character_level(self):
        """Test ChrF is character-aware"""
        hypothesis = "teh cat"  # Transposition
        reference = "the cat"
        
        chrf = compute_chrf(hypothesis, reference)
        # Should penalize character differences
        self.assertGreater(chrf, 0.3)
        self.assertLess(chrf, 1.0)
    
    def test_rouge_score(self):
        """Test ROUGE (Recall-Oriented Understudy for Gisting Evaluation)"""
        hypothesis = "the quick brown fox"
        reference = "the quick brown fox jumps"
        
        rouge_1, rouge_2, rouge_l = compute_rouge(hypothesis, reference)
        
        # All should be > 0
        self.assertGreater(rouge_1, 0.0)
        self.assertGreater(rouge_2, 0.0)
        self.assertGreater(rouge_l, 0.0)
    
    def test_rouge_variants(self):
        """Test different ROUGE variants"""
        hyp = "the cat is sleeping"
        ref = "the cat sleeps"
        
        rouge_1, rouge_2, rouge_l = compute_rouge(hyp, ref)
        
        # ROUGE-1 (unigrams) should be higher than ROUGE-2 (bigrams)
        self.assertGreater(rouge_1, rouge_2)
    
    def test_stem_accuracy(self):
        """Test stem identification accuracy"""
        # Each item: (tokenized_output, expected_stems)
        test_cases = [
            (["un##happy", "ly"], ["un", "happy", "ly"]),
            (["play", "ing"], ["play", "ing"]),
            (["compute", "##tion"], ["compute", "tion"]),
        ]
        
        accuracy = compute_stem_accuracy(test_cases)
        
        # Should measure % of correctly identified morphemes
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_metrics_normalization(self):
        """Test that metrics are normalized to [0, 1]"""
        test_cases = [
            ("", "test"),
            ("test", ""),
            ("", ""),
        ]
        
        for hyp, ref in test_cases:
            bleu = compute_bleu(hyp, [ref])
            chrf = compute_chrf(hyp, ref)
            
            self.assertGreaterEqual(bleu, 0.0)
            self.assertLessEqual(bleu, 1.0)
            self.assertGreaterEqual(chrf, 0.0)
            self.assertLessEqual(chrf, 1.0)
    
    def test_multiple_references(self):
        """Test BLEU with multiple references"""
        hypothesis = "the cat is sleeping"
        references = [
            "the cat sleeps",
            "the cat is asleep",
            "the cat takes a nap"
        ]
        
        bleu = compute_bleu(hypothesis, references)
        
        # Should handle multiple references
        self.assertGreater(bleu, 0.0)
        self.assertLess(bleu, 1.0)
    
    def test_case_insensitivity(self):
        """Test that metrics handle case variations"""
        hyp_lower = "the quick brown fox"
        hyp_upper = "THE QUICK BROWN FOX"
        ref = "the quick brown fox"
        
        chrf_lower = compute_chrf(hyp_lower, ref)
        chrf_upper = compute_chrf(hyp_upper, ref)
        
        # Should be similar (metrics may be case-aware, but close)
        self.assertGreater(chrf_lower, 0.8)
        self.assertGreater(chrf_upper, 0.5)


if __name__ == '__main__':
    unittest.main()
