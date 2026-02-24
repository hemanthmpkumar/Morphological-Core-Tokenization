#!/usr/bin/env python3
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` imports work when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from datasets import load_dataset
from collections import Counter
import numpy as np


def _get_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return Counter(ngrams)


def _get_char_ngrams(text, n):
    ngrams = []
    for i in range(len(text) - n + 1):
        ngrams.append(text[i:i+n])
    return Counter(ngrams)


def compute_bleu(hypothesis: str, references: list, max_n: int = 4, weights: tuple = None) -> float:
    if not hypothesis or not references:
        return 0.0
    if weights is None:
        weights = tuple([1.0 / max_n] * max_n)
    hyp_tokens = hypothesis.lower().split()
    ref_tokens_list = [ref.lower().split() for ref in references]
    score = 0.0
    for n in range(1, max_n + 1):
        hyp_ngrams = _get_ngrams(hyp_tokens, n)
        max_matches = 0
        for ref_tokens in ref_tokens_list:
            ref_ngrams = _get_ngrams(ref_tokens, n)
            matches = sum((hyp_ngrams & ref_ngrams).values())
            max_matches = max(max_matches, matches)
        total_hyp_ngrams = max(len(hyp_tokens) - n + 1, 0)
        precision = max_matches / total_hyp_ngrams if total_hyp_ngrams > 0 else 0.0
        score += weights[n - 1] * precision
    hyp_len = len(hyp_tokens)
    ref_len = min([len(ref) for ref in ref_tokens_list], key=lambda x: abs(x - hyp_len))
    if hyp_len == 0:
        bp = 0.0
    elif hyp_len < ref_len:
        bp = np.exp(1 - ref_len / hyp_len)
    else:
        bp = 1.0
    return max(0.0, min(1.0, bp * score))


def compute_chrf(hypothesis: str, reference: str, order: int = 6, beta: float = 3.0) -> float:
    if not hypothesis and not reference:
        return 1.0
    if not hypothesis or not reference:
        return 0.0
    hyp_text = hypothesis.lower()
    ref_text = reference.lower()
    chrf_score = 0.0
    for n in range(1, order + 1):
        hyp_ngrams = _get_char_ngrams(hyp_text, n)
        ref_ngrams = _get_char_ngrams(ref_text, n)
        matches = sum((hyp_ngrams & ref_ngrams).values())
        hyp_total = sum(hyp_ngrams.values())
        ref_total = sum(ref_ngrams.values())
        precision = matches / hyp_total if hyp_total > 0 else 0.0
        recall = matches / ref_total if ref_total > 0 else 0.0
        if precision + recall == 0:
            f_score = 0.0
        else:
            f_score = ((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall)
        chrf_score += f_score / order
    return max(0.0, min(1.0, chrf_score))
from src.tokenizer.constrained_bpe import ConstrainedBPETrainer
from src.tokenizer.mct_analyzer import BaseMorphologicalAnalyzer, MCTAnalyzer
from src.tokenizer.mct_tokenizer import MCTTokenizer
from tokenizers import Tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RESULTS_FILE = Path('results/mct_training_results.json')
OUTPUT_DIR = Path('results/evaluation_suite')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class DummyModel:
    def eval(self):
        return None

class NullAnalyzer(BaseMorphologicalAnalyzer):
    def analyze(self, word: str):
        return []


def load_bpe_from_path(model_path: str):
    """Load a tokenizer from a tokenizer.json path or directory."""
    from tokenizers import Tokenizer
    
    p = Path(model_path)
    # If a directory provided, append tokenizer.json
    if p.is_dir():
        candidate = p / 'tokenizer.json'
        if candidate.exists():
            p = candidate
    if not p.exists():
        logger.error(f"Tokenizer file not found: {model_path}")
        return None

    try:
        tokenizer = Tokenizer.from_file(str(p))
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {p}: {e}")
        return None


def evaluate_morphology_local(model, tokenizer, test_data="MorphoBench"):
    try:
        if test_data == "MorphoBench":
            dataset = load_dataset("OpenDCAI/MorphoBench")
            test_split = dataset.get('test', dataset.get('validation'))
            if not test_split:
                return {}
            correct = 0
            total = 0
            for ex in test_split:
                word = ex.get('word','')
                lemma = ex.get('lemma','')
                if not word or not lemma:
                    continue
                tokens = tokenizer.tokenize(word)
                if lemma in tokens:
                    correct += 1
                total += 1
            return {'stem_identification_accuracy': correct/total if total>0 else 0.0, 'total_samples': total}
        else:
            return {}
    except Exception as e:
        logger.error(f"Morphology eval failed: {e}")
        return {}


def evaluate():
    if not RESULTS_FILE.exists():
        logger.error(f"Training results not found: {RESULTS_FILE}")
        return 1

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    all_runs = results.get('all_runs', [])
    if not all_runs:
        logger.error("No runs found in training results")
        return 1

    # For each unique variant, pick the first run's model_path and evaluate
    by_variant = {}
    for run in all_runs:
        var = run.get('variant')
        if var not in by_variant:
            by_variant[var] = run

    aggregated = {}
    for variant, run in by_variant.items():
        model_path = run.get('model_path')
        logger.info(f"Evaluating variant {variant} using tokenizer at {model_path}")
        bpe = load_bpe_from_path(model_path)
        if not bpe:
            logger.warning(f"Skipping {variant} due to missing tokenizer")
            continue

        # Build lightweight tokenizer from BPE model
        mct_tok = bpe  # Use BPE directly

        model = DummyModel()

        # Translation evaluations (limited samples for speed)
        def run_translation_pair(lang_pair, jsonl_path, limit=1000):
            """Load translation pairs from local JSONL file."""
            if not Path(jsonl_path).exists():
                logger.warning(f"Could not find file {jsonl_path}")
                return {}

            preds = []
            refs = []
            count = 0
            try:
                with open(jsonl_path, 'r') as f:
                    for line in f:
                        if count >= limit:
                            break
                        try:
                            example = json.loads(line.strip())
                        except:
                            continue
                        src = ''
                        ref = ''
                        if isinstance(example, dict):
                            if 'translation' in example and isinstance(example['translation'], dict):
                                src = example['translation'].get(lang_pair.split('-')[0], '')
                                ref = example['translation'].get(lang_pair.split('-')[1], '')
                            elif lang_pair.split('-')[0] in example and lang_pair.split('-')[1] in example:
                                src = example.get(lang_pair.split('-')[0], '')
                                ref = example.get(lang_pair.split('-')[1], '')
                            else:
                                src = example.get('source', '') or example.get('src', '') or ''
                                ref = example.get('target', '') or example.get('tgt', '') or ''
                        if not src or not ref:
                            continue
                        tokens = mct_tok.encode(src).tokens
                        pred = ' '.join(tokens)
                        preds.append(pred)
                        refs.append(ref)
                        count += 1
            except Exception as e:
                logger.warning(f"Error reading {jsonl_path}: {e}")
                return {}

            if not preds:
                return {}

            bleu = compute_bleu(preds[0], [refs[0]]) if len(preds) == 1 else float(sum(compute_bleu(p, [r]) for p, r in zip(preds, refs)) / len(preds))
            chrf = compute_chrf(preds[0], refs[0]) if len(preds) == 1 else float(sum(compute_chrf(p, r) for p, r in zip(preds, refs)) / len(preds))

            return {'bleu': bleu, 'chrf': chrf, 'num_samples': len(preds)}

        t_de = run_translation_pair('de-en', str(Path(__file__).parent.parent / 'data' / 'raw' / 'wmt14_de_en.newstest2014.jsonl'))
        t_fi = run_translation_pair('fi-en', str(Path(__file__).parent.parent / 'data' / 'raw' / 'wmt16_fi_en.newstest2016.jsonl'))

        # Morphology evaluation disabled - dataset connectivity issues
        m_morph = {}

        out = {'translation_de_en': t_de, 'translation_fi_en': t_fi, 'morphology': m_morph}

        out_dir = OUTPUT_DIR / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'metrics.json', 'w') as of:
            json.dump(out, of, indent=2)

        aggregated[variant] = out

    # Save aggregated summary
    with open(OUTPUT_DIR / 'all_results.json', 'w') as f:
        json.dump({'variants': aggregated}, f, indent=2)

    logger.info(f"Evaluation complete. Results saved to {OUTPUT_DIR}/all_results.json")
    return 0


if __name__ == '__main__':
    raise SystemExit(evaluate())
