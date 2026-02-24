import time
import torch
import evaluate
import numpy as np
from typing import List, Dict, Tuple, Union
from collections import Counter


class MCTEvaluator:
    """Main evaluator for MCT models"""
    def __init__(self):
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.chrf = evaluate.load("chrf")

    def compute_nlp_metrics(self, predictions, references, task="translation"):
        """Compute NLP metrics for translation or summarization"""
        if task == "translation":
            return self.bleu.compute(predictions=predictions, references=references)
        elif task == "summarization":
            return self.rouge.compute(predictions=predictions, references=references)

    def measure_efficiency(self, model, input_ids):
        """
        Measures Throughput and Peak Memory.
        Critical for the comparison against ByT5.

        The implementation is GPU-aware: if CUDA is available it will reset
        and read the CUDA peak memory statistics.  On other backends the memory
        measurement is skipped but latency/throughput are still computed.
        """
        device = torch.device("cuda") if torch.cuda.is_available() else None
        if device is not None:
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        with torch.no_grad():
            _ = model(input_ids)
        latency = time.time() - start_time

        peak_mem = None
        if device is not None:
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        throughput = input_ids.size(0) * input_ids.size(1) / latency  # tokens/sec

        result = {
            "latency_ms": latency * 1000,
            "throughput_tps": throughput,
        }
        if peak_mem is not None:
            result["peak_mem_mb"] = peak_mem
        return result


def compute_bleu(hypothesis: str, references: List[str], max_n: int = 4, weights: Tuple = None) -> float:
    """
    Compute BLEU score (Bilingual Evaluation Understudy)
    
    Args:
        hypothesis: Generated text
        references: List of reference texts
        max_n: Maximum n-gram (default 4 for BLEU-4)
        weights: Weights for n-grams (default uniform)
    
    Returns:
        BLEU score in [0, 1]
    """
    if not hypothesis or not references:
        return 0.0
    
    if weights is None:
        weights = tuple([1.0 / max_n] * max_n)
    
    # Tokenize
    hyp_tokens = hypothesis.lower().split()
    ref_tokens_list = [ref.lower().split() for ref in references]
    
    # Compute n-gram matches
    score = 0.0
    for n in range(1, max_n + 1):
        hyp_ngrams = _get_ngrams(hyp_tokens, n)
        
        # Find max matches across references
        max_matches = 0
        for ref_tokens in ref_tokens_list:
            ref_ngrams = _get_ngrams(ref_tokens, n)
            matches = sum((hyp_ngrams & ref_ngrams).values())
            max_matches = max(max_matches, matches)
        
        # Precision for this n-gram
        total_hyp_ngrams = max(len(hyp_tokens) - n + 1, 0)
        if total_hyp_ngrams > 0:
            precision = max_matches / total_hyp_ngrams
        else:
            precision = 0.0
        
        score += weights[n - 1] * precision
    
    # Brevity penalty
    hyp_len = len(hyp_tokens)
    ref_len = min([len(ref) for ref in ref_tokens_list], 
                   key=lambda x: abs(x - hyp_len))
    
    if hyp_len == 0:
        bp = 0.0
    elif hyp_len < ref_len:
        bp = np.exp(1 - ref_len / hyp_len)
    else:
        bp = 1.0
    
    return max(0.0, min(1.0, bp * score))


def compute_chrf(hypothesis: str, reference: str, order: int = 6, beta: float = 3.0) -> float:
    """
    Compute ChrF (character F-score)
    
    Args:
        hypothesis: Generated text
        reference: Reference text
        order: n-gram order (default 6)
        beta: Weighting factor (default 3 favors recall)
    
    Returns:
        ChrF score in [0, 1]
    """
    if not hypothesis and not reference:
        return 1.0
    if not hypothesis or not reference:
        return 0.0
    
    # Get character n-grams
    hyp_text = hypothesis.lower()
    ref_text = reference.lower()
    
    chrf_score = 0.0
    for n in range(1, order + 1):
        hyp_ngrams = _get_char_ngrams(hyp_text, n)
        ref_ngrams = _get_char_ngrams(ref_text, n)
        
        # Compute precision and recall
        matches = sum((hyp_ngrams & ref_ngrams).values())
        
        hyp_total = sum(hyp_ngrams.values())
        ref_total = sum(ref_ngrams.values())
        
        precision = matches / hyp_total if hyp_total > 0 else 0.0
        recall = matches / ref_total if ref_total > 0 else 0.0
        
        # F-score with beta weighting
        if precision + recall == 0:
            f_score = 0.0
        else:
            f_score = ((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall)
        
        chrf_score += f_score / order
    
    return max(0.0, min(1.0, chrf_score))


def compute_rouge(hypothesis: str, reference: str) -> Tuple[float, float, float]:
    """
    Compute ROUGE scores (Recall-Oriented Understudy for Gisting Evaluation)
    
    Args:
        hypothesis: Generated text
        reference: Reference text
    
    Returns:
        Tuple of (ROUGE-1, ROUGE-2, ROUGE-L) scores in [0, 1]
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    
    # ROUGE-1: Unigram recall
    hyp_unigrams = Counter(hyp_tokens)
    ref_unigrams = Counter(ref_tokens)
    matches_1 = sum((hyp_unigrams & ref_unigrams).values())
    rouge_1 = matches_1 / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    
    # ROUGE-2: Bigram recall
    hyp_bigrams = _get_ngrams(hyp_tokens, 2)
    ref_bigrams = _get_ngrams(ref_tokens, 2)
    matches_2 = sum((hyp_bigrams & ref_bigrams).values())
    ref_bigrams_count = max(len(ref_tokens) - 1, 0)
    rouge_2 = matches_2 / ref_bigrams_count if ref_bigrams_count > 0 else 0.0
    
    # ROUGE-L: Longest common subsequence
    lcs_len = _longest_common_subsequence(hyp_tokens, ref_tokens)
    rouge_l = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    
    return (
        max(0.0, min(1.0, rouge_1)),
        max(0.0, min(1.0, rouge_2)),
        max(0.0, min(1.0, rouge_l))
    )


def compute_stem_accuracy(tokenized_outputs: List[Tuple[List[str], List[str]]]) -> float:
    """
    Compute stem identification accuracy
    
    Args:
        tokenized_outputs: List of (generated_tokens, expected_stems)
    
    Returns:
        Accuracy in [0, 1]
    """
    if not tokenized_outputs:
        return 0.0
    
    correct = 0
    total = 0
    
    for generated_tokens, expected_stems in tokenized_outputs:
        # Extract stems from generated tokens (those marked with ##)
        found_stems = []
        for token in generated_tokens:
            if token.startswith("##"):
                found_stems.append(token[2:])
            else:
                found_stems.append(token)
        
        # Check matches
        for expected_stem in expected_stems:
            total += 1
            if expected_stem in found_stems:
                correct += 1
    
    return correct / total if total > 0 else 0.0


# Helper functions

def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams as a Counter"""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return Counter(ngrams)


def _get_char_ngrams(text: str, n: int) -> Counter:
    """Extract character n-grams as a Counter"""
    ngrams = []
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        ngrams.append(ngram)
    return Counter(ngrams)


def _longest_common_subsequence(seq1: List[str], seq2: List[str]) -> int:
    """Compute length of longest common subsequence"""
    m, n = len(seq1), len(seq2)
    
    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

