import unittest
import types
import importlib.util
import os
from pathlib import Path


# -------------------------------------------------
# Load training module dynamically
# -------------------------------------------------

BASE_DIR = Path(__file__).parent.parent
TRAIN_SCRIPT = BASE_DIR / "scripts" / "train_nmt_models.py"

spec = importlib.util.spec_from_file_location(
    "train_nmt_models",
    str(TRAIN_SCRIPT),
)

train_nmt_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_nmt_models)


# -------------------------------------------------
# Trainer BLEU Test
# -------------------------------------------------

class TestTrainerBLEU(unittest.TestCase):
    """
    Verify that LocalNMTTrainer uses sacrebleu.corpus_bleu
    during training and propagates the score.
    """

    def test_sacrebleu_called_during_training(self):

        called = {"flag": False}

        # Fake corpus_bleu
        def fake_corpus_bleu(gen, refs):

            called["flag"] = True

            class Out:
                score = 12.34

            return Out()

        # Patch module-level sacrebleu reference
        fake = types.SimpleNamespace(
            corpus_bleu=fake_corpus_bleu
        )

        train_nmt_models.sacrebleu = fake

        trainer = train_nmt_models.LocalNMTTrainer(
            "small",
            "BPE_32K",
            "de-en"
        )

        # Fast test configuration
        trainer.config["epochs"] = 1
        trainer.config["batch_size"] = 2

        # Tiny deterministic dataset
        trainer.load_dataset = lambda: (
            [("hello", "hello")],
            [("hello", "hello")],
            [("hello", "hello")]
        )

        result = trainer.train()

        # Assertions
        self.assertTrue(
            called["flag"],
            "sacrebleu.corpus_bleu was not called"
        )

        self.assertAlmostEqual(
            result["test_bleu"],
            12.34,
            places=3
        )


# -------------------------------------------------
# Run tests directly
# -------------------------------------------------

if __name__ == "__main__":
    unittest.main()