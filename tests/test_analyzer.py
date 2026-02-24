import unittest
from unittest.mock import Mock
from src.tokenizer.mct_analyzer import MCTAnalyzer, BaseMorphologicalAnalyzer


class MockMorphologicalAnalyzer(BaseMorphologicalAnalyzer):
    """Mock analyzer for testing"""
    
    def __init__(self, analyses_map):
        """
        Args:
            analyses_map: Dict mapping words to their analyses
        """
        self.analyses_map = analyses_map
    
    def analyze(self, word: str):
        return self.analyses_map.get(word, [])


class TestMCTAnalyzer(unittest.TestCase):
    """Test suite for MCTAnalyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock stem frequency map
        self.freq_map = {
            "play": 100,
            "player": 50,
            "run": 80,
            "happy": 120
        }
        
        # Mock morphological analyses
        self.analyses_map = {
            "playing": [("play", [], ["ing"])],
            "players": [("player", [], ["s"]), ("play", [], ["ers"])],
            "unhappy": [("happy", ["un"], [])],
            "running": [("run", [], ["ning"])]
        }
        
        # Create mock analyzer
        self.analyzer = MockMorphologicalAnalyzer(self.analyses_map)
        
        # Create MCTAnalyzer
        self.mct_analyzer = MCTAnalyzer(self.analyzer, self.freq_map)

    def test_stem_identification(self):
        """Test basic stem identification"""
        res = self.mct_analyzer.get_best_analysis("playing")
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "play")  # Stem
        self.assertIn("ing", res[2])  # Suffix

    def test_ambiguity_resolution(self):
        """Test disambiguation based on stem frequency"""
        res = self.mct_analyzer.get_best_analysis("players")
        self.assertIsNotNone(res)
        # Should select "play" as stem (freq=100) over "player" (freq=50)
        self.assertEqual(res[0], "play")

    def test_prefix_extraction(self):
        """Test prefix extraction"""
        res = self.mct_analyzer.get_best_analysis("unhappy")
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "happy")  # Stem
        self.assertIn("un", res[1])  # Prefix

    def test_cache_functionality(self):
        """Test caching of morphological analyses"""
        word = "playing"
        
        # First call
        res1 = self.mct_analyzer.get_best_analysis(word)
        
        # Modify the underlying analyzer to verify caching
        self.mct_analyzer.analyzer.analyses_map[word] = [("other", [], [])]
        
        # Second call should return cached result
        res2 = self.mct_analyzer.get_best_analysis(word)
        
        self.assertEqual(res1, res2)

    def test_cache_clear(self):
        """Test cache clearing"""
        word = "playing"
        self.mct_analyzer.get_best_analysis(word)
        self.assertIn(word, self.mct_analyzer.cache)
        
        self.mct_analyzer.clear_cache()
        self.assertNotIn(word, self.mct_analyzer.cache)

    def test_minimum_stem_length(self):
        """Test minimum stem length filtering"""
        # Add analysis with very short stem
        self.analyses_map["xa"] = [("a", ["x"], [])]
        self.analyzer.analyses_map = self.analyses_map
        
        # With default min_stem_length=2, "a" should be filtered
        mct_analyzer = MCTAnalyzer(
            self.analyzer,
            self.freq_map,
            min_stem_length=2
        )
        
        res = mct_analyzer.get_best_analysis("xa")
        # Should return None since "a" is below minimum length
        self.assertIsNone(res)

    def test_no_analysis_fallback(self):
        """Test fallback when no analysis is available"""
        res = self.mct_analyzer.get_best_analysis("unknownword123")
        self.assertIsNone(res)


if __name__ == '__main__':
    unittest.main()
