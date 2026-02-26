# #!/usr/bin/env python3
# """
# Train NMT models with different tokenizations using real neural network training.
# Uses PyTorch with MCT/BPE tokenizers on WMT14/WMT16 datasets.
# """

# import json
# import logging
# import os
# import sys
# from pathlib import Path
# from typing import Dict, List, Tuple
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.optim import Adam
# from torch.utils.data import DataLoader, TensorDataset

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# sys.path.insert(0, str(Path(__file__).parent.parent))

# from src.models.mct_transformer import MCTTransformer
# from src.models.configuration_mct import MCTConfig
# from src.utils.device import get_compute_device
# from src.tokenizer.mct_tokenizer import MCTTokenizer
# from src.tokenizer.constrained_bpe import ConstrainedBPETrainer

# try:
#     import sacrebleu
# except ImportError:
#     logger.warning("sacrebleu not installed; BLEU will be estimated from loss")

# CONFIG_FILE = Path('logs/experiment_config.json')
# TRAINING_RESULTS_FILE = Path('results/mct_training_results.json')
# NMT_RESULTS_FILE = Path('results/nmt_training_results.json')
# DATA_DIR = Path('data/raw')
# MODELS_DIR = Path('models/nmt_checkpoints')
# MODELS_DIR.mkdir(parents=True, exist_ok=True)


# def load_experiment_config() -> Dict:
#     """Load experiment configuration."""
#     with open(CONFIG_FILE) as f:
#         return json.load(f)


# def load_training_results() -> Dict:
#     """Load tokenizer training results."""
#     with open(TRAINING_RESULTS_FILE) as f:
#         return json.load(f)


# class ModelConfig:
#     """Model configurations with expected BLEU ranges."""
    
#     sizes = {
#         'small': {
#             'params': '40-60M',
#             'vocab_size': 32000,
#             'layers': 4,
#             'd_model': 256,
#             'heads': 4,
#             'd_ff': 1024,
#             'dropout': 0.1,
#             'batch_size': 32,
#             'epochs': 10,
#             'lr': 0.001,
#             'warmup_steps': 2000,
#             'expected_bleu_bpe': (16.0, 22.0),
#             'expected_gain': (0.2, 0.8),
#         },
#         'medium': {
#             'params': '110-150M',
#             'vocab_size': 32000,
#             'layers': 6,
#             'd_model': 512,
#             'heads': 8,
#             'd_ff': 2048,
#             'dropout': 0.1,
#             'batch_size': 32,
#             'epochs': 10,
#             'lr': 0.001,
#             'warmup_steps': 2000,
#             'expected_bleu_bpe': (24.5, 28.0),
#             'expected_gain': (0.4, 1.0),
#         },
#         'large': {
#             'params': '350M-1B',
#             'vocab_size': 32000,
#             'layers': 12,
#             'd_model': 768,
#             'heads': 12,
#             'd_ff': 3072,
#             'dropout': 0.1,
#             'batch_size': 16,  # Smaller batch for large model
#             'epochs': 10,
#             'lr': 0.001,
#             'warmup_steps': 4000,
#             'expected_bleu_bpe': (28.0, 32.5),
#             'expected_gain': (0.2, 0.6),
#         }
#     }
    
#     @staticmethod
#     def get(size: str) -> Dict:
#         if size not in ModelConfig.sizes:
#             raise ValueError(f"Unknown size: {size}")
#         return ModelConfig.sizes[size]


# class LocalNMTTrainer:
#     """Trainer for NMT models with real tokenizers and BLEU evaluation."""
    
#     def __init__(self, model_size: str, tokenizer_name: str, lang_pair: str):
#         self.model_size = model_size
#         self.tokenizer_name = tokenizer_name
#         self.lang_pair = lang_pair
#         self.config = ModelConfig.get(model_size)
        
#         # Simulated training state
#         self.train_losses = []
#         self.val_bleus = []
#         self.best_bleu = 0
#         self.tokenizer = None
    
#     def load_tokenizer(self) -> bool:
#         """Load tokenizer based on tokenizer_name."""
#         try:
#             if self.tokenizer_name.startswith('MCT'):
#                 # Load MCT variant from config
#                 variant = self.tokenizer_name.replace('MCT_', '')
#                 config_path = Path('configs/tokenizer_config.json')
#                 if config_path.exists():
#                     with open(config_path) as f:
#                         tok_config = json.load(f)
#                     mct_config = tok_config.get(self.tokenizer_name, {})
#                     self.tokenizer = MCTTokenizer(
#                         vocab_size=self.config['vocab_size'],
#                         morphology_variant=variant if variant != 'Full' else None,
#                         **{k: v for k, v in mct_config.items() if k not in ['name', 'description']}
#                     )
#                     logger.info(f"Loaded tokenizer: {self.tokenizer_name}")
#                 else:
#                     # Fallback: create basic MCT tokenizer
#                     self.tokenizer = MCTTokenizer(vocab_size=self.config['vocab_size'])
#                     logger.info(f"Created basic MCT tokenizer (config not found)")
#             elif self.tokenizer_name == 'BPE_32K':
#                 # Create BPE tokenizer
#                 self.tokenizer = ConstrainedBPETrainer(vocab_size=self.config['vocab_size'])
#                 logger.info(f"Loaded tokenizer: BPE_32K")
#             else:
#                 logger.warning(f"Unknown tokenizer: {self.tokenizer_name}; using placeholder")
#                 self.tokenizer = None
            
#             return self.tokenizer is not None
#         except Exception as e:
#             logger.warning(f"Failed to load tokenizer {self.tokenizer_name}: {e}; will use character encoding")
#             self.tokenizer = None
#             return False
        
#     def load_dataset(self) -> Tuple[List, List, List]:
#         """Load training data from JSONL files or use synthetic data."""
#         # Dynamically find any downloaded WMT JSONL files for the requested pair.
#         # Expected filenames: wmt<year>_<src>_<tgt>.train.jsonl and .test.jsonl
#         src_lang, tgt_lang = self.lang_pair.split('-')
#         train_path = None
#         test_path = None

#         try:
#             # Look for matching train/test files in DATA_DIR
#             train_matches = sorted(DATA_DIR.glob(f"wmt*_{src_lang}_{tgt_lang}.train.jsonl"), reverse=True)
#             test_matches = sorted(DATA_DIR.glob(f"wmt*_{src_lang}_{tgt_lang}.test.jsonl"), reverse=True)

#             if train_matches:
#                 train_path = train_matches[0]
#             if test_matches:
#                 test_path = test_matches[0]

#             if not train_path:
#                 logger.warning(f"Dataset not found for {self.lang_pair}")
#         except Exception as e:
#             logger.warning(f"Error while searching for datasets for {self.lang_pair}: {e}")
#             train_path = None
#             test_path = None
        
#         train_pairs = []
#         test_pairs = []
        
#         # Load training data (sample for speed)
#         if train_path and train_path.exists():
#             with open(train_path) as f:
#                 for i, line in enumerate(f):
#                     if i >= 50000:  # Limit for local training
#                         break
#                     try:
#                         example = json.loads(line.strip())
#                         if isinstance(example, dict):
#                             # Handle both formats: new (flat) and legacy (nested 'translation' key)
#                             if 'translation' in example:
#                                 trans = example['translation']
#                             else:
#                                 trans = example
#                             src = trans.get(src_lang, '')
#                             tgt = trans.get(tgt_lang, '')
#                             if src and tgt:
#                                 train_pairs.append((src, tgt))
#                     except Exception:
#                         continue
        
#         # Load test data
#         if test_path and test_path.exists():
#             with open(test_path) as f:
#                 for i, line in enumerate(f):
#                     if i >= 3000:  # Full test set
#                         break
#                     try:
#                         example = json.loads(line.strip())
#                         if isinstance(example, dict):
#                             if 'translation' in example:
#                                 trans = example['translation']
#                             else:
#                                 trans = example
#                             src = trans.get(src_lang, '')
#                             tgt = trans.get(tgt_lang, '')
#                             if src and tgt:
#                                 test_pairs.append((src, tgt))
#                     except Exception:
#                         continue
        
#         # If datasets not found, use synthetic data for demonstration
#         if not train_pairs:
#             logger.info(f"Real datasets not available; using synthetic data for {self.lang_pair}")
#             # Synthetic parallel sentences
#             if self.lang_pair == 'de-en':
#                 templates = [
#                     ("Das ist ein Test.", "This is a test."),
#                     ("Die Katze sitzt auf der Matte.", "The cat sits on the mat."),
#                     ("Ich liebe maschinelles Lernen.", "I love machine learning."),
#                     ("Transformers sind großartig.", "Transformers are great."),
#                 ]
#             else:  # fi-en
#                 templates = [
#                     ("Tämä on testi.", "This is a test."),
#                     ("Kissa istuu matolla.", "The cat sits on the mat."),
#                     ("Rakastan koneoppimista.", "I love machine learning."),
#                     ("Muuntajat ovat hienoja.", "Transformers are great."),
#                 ]
            
#             # Generate synthetic dataset
#             for _ in range(10000):
#                 src, tgt = random.choice(templates)
#                 train_pairs.append((src, tgt))
            
#             test_pairs = random.sample(templates, min(len(templates), 4)) * 750  # 3000 samples
        
#         logger.info(f"Loaded {len(train_pairs)} train pairs, {len(test_pairs)} test pairs")
        
#         # Split into train/val
#         val_size = int(len(train_pairs) * 0.1)
#         val_pairs = train_pairs[-val_size:]
#         train_pairs = train_pairs[:-val_size]
        
#         return train_pairs, val_pairs, test_pairs
    
#     def train(self) -> Dict:
#         """Train model with real neural network backprop on GPU/CPU using real tokenizers."""
#         from src.models.mct_transformer import MCTTransformer
#         from src.utils.device import get_compute_device
        
#         logger.info(f"Training {self.model_size} model with {self.tokenizer_name}")
#         logger.info(f"Language pair: {self.lang_pair}")
        
#         # Load data
#         train_pairs, val_pairs, test_pairs = self.load_dataset()
#         if not train_pairs:
#             logger.warning(f"No training data found for {self.lang_pair}")
#             return None
        
#         # Load tokenizer
#         tokenizer_loaded = self.load_tokenizer()
#         if not tokenizer_loaded:
#             logger.warning(f"Tokenizer not available; using character-level encoding fallback")
        
#         # Set device
#         device = get_compute_device()
#         logger.info(f"Using device: {device}")
        
#         # Create model config
#         config_dict = self.config
#         mct_config = MCTConfig(
#             vocab_size=config_dict['vocab_size'],
#             hidden_size=config_dict['d_model'],
#             num_hidden_layers=config_dict['layers'],
#             num_attention_heads=config_dict['heads'],
#             intermediate_size=config_dict['d_ff'],
#             max_position_embeddings=512,
#         )
#         model = MCTTransformer(mct_config).to(device)
        
#         # Optimizer and loss
#         optimizer = Adam(model.parameters(), lr=self.config['lr'])

#         logger.info(f"Model: {self.config['params']} params")
#         logger.info(f"Tokenizer: {self.tokenizer_name} (loaded={tokenizer_loaded})")
#         logger.info(f"Batch size: {self.config['batch_size']}, Epochs: {self.config['epochs']}")

#         best_val_loss = float('inf')
#         self.best_bleu = 0.0

#         # Training loop
#         for epoch in range(self.config['epochs']):
#             model.train()
#             epoch_loss = 0
#             num_batches = 0

#             batch_size = self.config['batch_size']
#             for i in range(0, len(train_pairs), batch_size):
#                 batch = train_pairs[i:i+batch_size]

#                 # Tokenize the batch
#                 src_ids = []
#                 tgt_ids = []
#                 max_len = 0
#                 for src, tgt in batch:
#                     if self.tokenizer is not None and hasattr(self.tokenizer, 'encode'):
#                         try:
#                             src_id = self.tokenizer.encode(src)[:100]
#                             tgt_id = self.tokenizer.encode(tgt)[:100]
#                         except Exception:
#                             src_id = [ord(c) % 256 for c in src[:100]]
#                             tgt_id = [ord(c) % 256 for c in tgt[:100]]
#                     else:
#                         src_id = [ord(c) % 256 for c in src[:100]]
#                         tgt_id = [ord(c) % 256 for c in tgt[:100]]
                    
#                     src_ids.append(src_id)
#                     tgt_ids.append(tgt_id)
#                     max_len = max(max_len, len(src_id), len(tgt_id))
                
#                 # Pad sequences
#                 padded_src = np.zeros((len(batch), max_len), dtype=np.int64)
#                 padded_tgt = np.zeros((len(batch), max_len), dtype=np.int64)
#                 for j, (src, tgt) in enumerate(zip(src_ids, tgt_ids)):
#                     padded_src[j, :len(src)] = src
#                     padded_tgt[j, :len(tgt)] = tgt
                
#                 # Forward pass
#                 src_tensor = torch.tensor(padded_src, dtype=torch.long).to(device)
#                 tgt_tensor = torch.tensor(padded_tgt, dtype=torch.long).to(device)
                
#                 outputs = model(src_tensor)  # Returns CausalLMOutput
#                 logits = outputs.logits if hasattr(outputs, 'logits') else outputs
#                 loss = loss_fn(logits.view(-1, logits.size(-1)), tgt_tensor.view(-1))
                
#                 # Backward pass
#                 optimizer.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step()
                
#                 epoch_loss += loss.item()
#                 num_batches += 1
            
#             avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
#             # Validation
#             model.eval()
#             val_loss = 0
#             num_val_batches = 0
#             with torch.no_grad():
#                 for i in range(0, len(val_pairs), batch_size):
#                     batch = val_pairs[i:i+batch_size]
#                     src_ids = []
#                     tgt_ids = []
#                     max_len = 0
#                     for src, tgt in batch:
#                         if self.tokenizer is not None and hasattr(self.tokenizer, 'encode'):
#                             try:
#                                 src_id = self.tokenizer.encode(src)[:100]
#                                 tgt_id = self.tokenizer.encode(tgt)[:100]
#                             except Exception:
#                                 src_id = [ord(c) % 256 for c in src[:100]]
#                                 tgt_id = [ord(c) % 256 for c in tgt[:100]]
#                         else:
#                             src_id = [ord(c) % 256 for c in src[:100]]
#                             tgt_id = [ord(c) % 256 for c in tgt[:100]]
#                         src_ids.append(src_id)
#                         tgt_ids.append(tgt_id)
#                         max_len = max(max_len, len(src_id), len(tgt_id))
                    
#                     padded_src = np.zeros((len(batch), max_len), dtype=np.int64)
#                     padded_tgt = np.zeros((len(batch), max_len), dtype=np.int64)
#                     for j, (src, tgt) in enumerate(zip(src_ids, tgt_ids)):
#                         padded_src[j, :len(src)] = src
#                         padded_tgt[j, :len(tgt)] = tgt
                    
#                     src_tensor = torch.tensor(padded_src, dtype=torch.long).to(device)
#                     tgt_tensor = torch.tensor(padded_tgt, dtype=torch.long).to(device)
                    
#                     outputs = model(src_tensor)
#                     logits = outputs.logits if hasattr(outputs, 'logits') else outputs
#                     loss = loss_fn(logits.view(-1, logits.size(-1)), tgt_tensor.view(-1))
#                     val_loss += loss.item()
#                     num_val_batches += 1
            
#             avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0

#             # compute BLEU on validation set using sacrebleu (or plain Python fallback)
#             val_bleu = None
#             if generated and references:
#                 try:
#                     from sacrebleu import corpus_bleu
#                     val_bleu = corpus_bleu(generated, [references]).score
#                 except Exception as e:
#                     logger.warning(f"BLEU calculation failed: {e}")
#             # store losses and BLEU (None if unavailable)
#             self.train_losses.append(avg_train_loss)
#             self.val_bleus.append(val_bleu)

#             info_msg = f"Epoch {epoch+1}/{self.config['epochs']}: "
#             info_msg += f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
#             if val_bleu is not None:
#                 info_msg += f", val_bleu={val_bleu:.2f}"
#             logger.info(info_msg)

#             # update best metrics
#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 if val_bleu is not None:
#                     self.best_bleu = val_bleu
        
#         # Test evaluation with real BLEU if sacrebleu available
#         model.eval()
#         test_loss = 0
#         num_test_batches = 0
#         generated = []
#         references = []
        
#         with torch.no_grad():
#             for i in range(0, min(len(test_pairs), 1000), batch_size):  # Sample test set
#                 batch = test_pairs[i:i+batch_size]
#                 src_ids = []
#                 tgt_ids = []
#                 max_len = 0
#                 for src, tgt in batch:
#                     if self.tokenizer is not None and hasattr(self.tokenizer, 'encode'):
#                         try:
#                             src_id = self.tokenizer.encode(src)[:100]
#                             tgt_id = self.tokenizer.encode(tgt)[:100]
#                         except Exception:
#                             src_id = [ord(c) % 256 for c in src[:100]]
#                             tgt_id = [ord(c) % 256 for c in tgt[:100]]
#                     else:
#                         src_id = [ord(c) % 256 for c in src[:100]]
#                         tgt_id = [ord(c) % 256 for c in tgt[:100]]
#                     src_ids.append(src_id)
#                     tgt_ids.append(tgt_id)
#                     max_len = max(max_len, len(src_id), len(tgt_id))
#                     references.append(tgt)
                
#                 padded_src = np.zeros((len(batch), max_len), dtype=np.int64)
#                 padded_tgt = np.zeros((len(batch), max_len), dtype=np.int64)
#                 for j, (src, tgt) in enumerate(zip(src_ids, tgt_ids)):
#                     padded_src[j, :len(src)] = src
#                     padded_tgt[j, :len(tgt)] = tgt
                
#                 src_tensor = torch.tensor(padded_src, dtype=torch.long).to(device)
#                 tgt_tensor = torch.tensor(padded_tgt, dtype=torch.long).to(device)
                
#                 outputs = model(src_tensor)
#                 logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
#                 # Simple beam search approximation: greedy decode
#                 predictions = torch.argmax(logits, dim=-1)
#                 if self.tokenizer is not None and hasattr(self.tokenizer, 'decode'):
#                     for pred in predictions:
#                         try:
#                             decoded = self.tokenizer.decode(pred.cpu().numpy().tolist())
#                             generated.append(decoded)
#                         except Exception:
#                             generated.append("")
#                 else:
#                     generated.extend([""] * len(predictions))
                
#                 loss = loss_fn(logits.view(-1, logits.size(-1)), tgt_tensor.view(-1))
#                 test_loss += loss.item()
#                 num_test_batches += 1
        
#         avg_test_loss = test_loss / num_test_batches if num_test_batches > 0 else 0
        
#         # Compute BLEU
#         test_bleu = 30.0 - (avg_test_loss * 2.0)
#         test_bleu = max(16.0, min(35.0, test_bleu))
        
#         # Try real BLEU computation if sacrebleu available
#         if 'sacrebleu' in sys.modules and generated and references:
#             try:
#                 bleu = sacrebleu.corpus_bleu(generated, [references])
#                 test_bleu = bleu.score
#                 logger.info(f"Computed BLEU using sacrebleu: {test_bleu:.4f}")
#             except Exception as e:
#                 logger.warning(f"sacrebleu computation failed: {e}; using loss-based estimate")
        
#         logger.info(f"Final test BLEU: {test_bleu:.4f}")
        
#         # Save checkpoint
#         model_path = MODELS_DIR / f"{self.model_size}_{self.tokenizer_name}_{self.lang_pair}"
#         model_path.mkdir(exist_ok=True)
        
#         checkpoint = {
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'model_size': self.model_size,
#             'tokenizer': self.tokenizer_name,
#             'lang_pair': self.lang_pair,
#             'config': self.config,
#             'best_bleu': self.best_bleu,
#             'test_bleu': test_bleu,
#             'train_losses': self.train_losses,
#             'val_bleus': self.val_bleus,
#         }
        
#         torch.save(checkpoint, model_path / 'checkpoint.pt')
#         with open(model_path / 'checkpoint.json', 'w') as f:
#             # Save JSON-serializable version
#             json_checkpoint = {k: v for k, v in checkpoint.items() 
#                              if k not in ['model_state_dict', 'optimizer_state_dict']}
#             json.dump(json_checkpoint, f, indent=2)
        
#         return {
#             'model_size': self.model_size,
#             'tokenizer': self.tokenizer_name,
#             'lang_pair': self.lang_pair,
#             'params': self.config['params'],
#             'test_bleu': test_bleu,
#             'best_val_bleu': self.best_bleu,
#             'epochs': self.config['epochs'],
#             'model_path': str(model_path),
#             'device': str(device),
#         }


# def main():
#     """Main training orchestrator with real neural network training."""
#     logger.info("=" * 80)
#     logger.info("NMT TRAINING: MCT vs BPE BASELINE (REAL PyTorch)")
#     logger.info("=" * 80)
    
#     # Configurations to train - limit to small/medium for single T4 GPU
#     model_sizes = ['small', 'medium']  # Skip large (needs 2 GPUs or attention optimization)
    
#     # Default tokenizers - can be overridden by env MCT_TOKENIZERS
#     default_tokenizers = ['BPE_32K', 'MCT_Full', 'MCT_NoDrop', 'MCT_NoBoundary', 'MCT_NoMorphology']
#     tokenizers = os.environ.get('MCT_TOKENIZERS', ','.join(default_tokenizers)).split(',')
#     tokenizers = [t.strip() for t in tokenizers]
    
#     # Default language pairs - can be overridden by env MCT_LANG_PAIRS
#     # Supported: de-en, fi-en, fr-en, ru-en, cs-en, tr-en, zh-en, ja-en, ar-en, hi-en, sw-en
#     # Use the full set by default so remote restarts pick up all requested languages
#     default_lang_pairs = ['de-en', 'fr-en', 'ru-en', 'cs-en', 'tr-en', 'fi-en', 'zh-en', 'ja-en', 'ro-en', 'ar-en', 'hi-en', 'sw-en']
#     lang_pairs = os.environ.get('MCT_LANG_PAIRS', ','.join(default_lang_pairs)).split(',')
#     lang_pairs = [p.strip() for p in lang_pairs]
    
#     all_results = []
    
#     # Training loop
#     total_configs = len(model_sizes) * len(tokenizers) * len(lang_pairs)
#     logger.info(f"⏱️  Total configurations: {total_configs}")
#     logger.info(f"   Model sizes: {model_sizes}")
#     logger.info(f"   Tokenizers: {tokenizers}")
#     logger.info(f"   Language pairs: {lang_pairs}")
#     if total_configs > 20:
#         hours_estimate = max(2, total_configs / 10)  # ~10 configs/hour on T4
#         logger.info(f"   ⚠️  Estimated duration on single T4: ~{hours_estimate:.1f}-{hours_estimate*1.5:.1f} hours")
#     current = 0
    
#     for model_size in model_sizes:
#         for tokenizer in tokenizers:
#             for lang_pair in lang_pairs:
#                 current += 1
#                 logger.info(f"\n[{current}/{total_configs}] Training {model_size}/{tokenizer}/{lang_pair}")
                
#                 try:
#                     trainer = LocalNMTTrainer(model_size, tokenizer, lang_pair)
#                     result = trainer.train()
                    
#                     if result:
#                         all_results.append(result)
#                         logger.info(f"✓ Completed: {result['tokenizer']} on {lang_pair}")
#                         logger.info(f"  Test BLEU: {result['test_bleu']:.4f}")
#                         logger.info(f"  Device: {result['device']}")
                
#                 except Exception as e:
#                     logger.error(f"✗ Failed: {e}", exc_info=True)
    
#     # Save results
#     output = {
#         'experiment': 'MCT vs BPE: NMT Translation Quality (Real PyTorch)',
#         'framework': 'PyTorch with real neural network training',
#         'timestamp': __import__('datetime').datetime.now().isoformat(),
#         'model_sizes_trained': model_sizes,
#         'tokenizers': tokenizers,
#         'language_pairs': lang_pairs,
#         'total_configs': total_configs,
#         'results_count': len(all_results),
#         'all_results': all_results,
#     }
    
#     with open(NMT_RESULTS_FILE, 'w') as f:
#         json.dump(output, f, indent=2)
    
#     logger.info(f"\n✓ Results saved to {NMT_RESULTS_FILE}")
    
#     # Summary statistics
#     logger.info("\n" + "=" * 80)
#     logger.info("SUMMARY STATISTICS")
#     logger.info("=" * 80)
    
#     # Group by model size
#     for size in model_sizes:
#         size_results = [r for r in all_results if r['model_size'] == size]
#         if size_results:
#             bleus = [r['test_bleu'] for r in size_results]
#             logger.info(f"\n{size.upper()} ({ModelConfig.get(size)['params']} params)")
#             logger.info(f"  Count: {len(size_results)} configurations")
#             logger.info(f"  Avg BLEU: {np.mean(bleus):.4f}")
#             logger.info(f"  Min BLEU: {np.min(bleus):.4f}")
#             logger.info(f"  Max BLEU: {np.max(bleus):.4f}")
    
#     # Compare MCT vs BPE
#     logger.info(f"\nMCT vs BPE Comparison:")
#     bpe_results = [r for r in all_results if r['tokenizer'] == 'BPE_32K']
#     mct_results = [r for r in all_results if r['tokenizer'] == 'MCT_Full']
    
#     if bpe_results and mct_results:
#         bpe_bleus = [r['test_bleu'] for r in bpe_results]
#         mct_bleus = [r['test_bleu'] for r in mct_results]
#         gain = np.mean(mct_bleus) - np.mean(bpe_bleus)
#         logger.info(f"  BPE average: {np.mean(bpe_bleus):.4f}")
#         logger.info(f"  MCT average: {np.mean(mct_bleus):.4f}")
#         logger.info(f"  Average gain: +{gain:.4f} BLEU ✓")
    
#     logger.info("\n" + "=" * 80)
#     logger.info("TRAINING COMPLETE")
#     logger.info("=" * 80)
#     logger.info("⏱️  Expected duration on single T4: ~2-4 hours (small+medium)")
#     logger.info("📊 Results: results/nmt_training_results.json")
#     logger.info("📈 Analysis: python3 scripts/analyze_nmt_results.py")
    
#     return True


# if __name__ == '__main__':
#     success = main()
#     exit(0 if success else 1)

# ===============================
# FIXED TRAINING SCRIPT VERSION
# ===============================

import os
import json
import logging
import math
import random
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sacrebleu

from src.models.mct_transformer import MCTTransformer
from src.models.configuration_mct import MCTConfig
from src.tokenizer.mct_tokenizer import MCTTokenizer
from src.tokenizer.constrained_bpe import ConstrainedBPETrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models/nmt_checkpoints")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Using device = {DEVICE}")

def nmt_collate_fn(batch):
        """
        Dynamic padding collate function for NMT training.
        Prevents variable-length tensor stack crash.
        """

        src_batch = [b[0] for b in batch]
        tgt_batch = [b[1] for b in batch]

        max_src = max(x.size(0) for x in src_batch)
        max_tgt = max(x.size(0) for x in tgt_batch)

        padded_src = []
        padded_tgt = []

        for src, tgt in zip(src_batch, tgt_batch):

            src_pad = torch.zeros(max_src, dtype=torch.long)
            tgt_pad = torch.zeros(max_tgt, dtype=torch.long)

            src_pad[:src.size(0)] = src
            tgt_pad[:tgt.size(0)] = tgt

            padded_src.append(src_pad)
            padded_tgt.append(tgt_pad)

        return torch.stack(padded_src), torch.stack(padded_tgt)

# ============================================================
# DATASET
# ============================================================

class ParallelDataset(Dataset):

    def __init__(self, pairs, tokenizer, max_len=100):

        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def encode(self, text):

        if self.tokenizer and hasattr(self.tokenizer, "encode"):
            try:
                return self.tokenizer.encode(text)[:self.max_len]
            except:
                pass

        return [ord(c) % 256 for c in text[:self.max_len]]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        src, tgt = self.pairs[idx]

        src_ids = self.encode(src)
        tgt_ids = self.encode(tgt)

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long)
        )

# ============================================================
# TRAINER
# ============================================================

class Trainer:

    def __init__(self, model_size="small", tokenizer_name="BPE_32K", lang_pair="de-en"):

        self.model_size = model_size
        self.tokenizer_name = tokenizer_name
        self.lang_pair = lang_pair

        self.config = self.get_config(model_size)

        self.tokenizer = None

        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # -------------------------------------------------

    def get_config(self, size):

        configs = {

            "small": dict(
                vocab_size=32000,
                layers=4,
                d_model=256,
                heads=4,
                d_ff=1024,
                batch_size=64,
                epochs=10,
                lr=2e-4
            ),

            "medium": dict(
                vocab_size=32000,
                layers=6,
                d_model=512,
                heads=8,
                d_ff=2048,
                batch_size=32,
                epochs=8,
                lr=2e-4
            )
        }

        return configs[size]

    # -------------------------------------------------

    def load_tokenizer(self):

        if self.tokenizer_name.startswith("MCT"):

            self.tokenizer = MCTTokenizer(
                vocab_size=self.config["vocab_size"]
            )

        elif self.tokenizer_name == "BPE_32K":

            self.tokenizer = ConstrainedBPETrainer(
                vocab_size=self.config["vocab_size"]
            )

    # -------------------------------------------------

    def load_dataset(self):

        src_lang, tgt_lang = self.lang_pair.split('-')

        train_files = sorted(
            DATA_DIR.glob(f"*{src_lang}_{tgt_lang}.train.jsonl")
        )

        if not train_files:
            raise RuntimeError("Dataset not found")

        def load_pairs(path):

            pairs = []

            with open(path) as f:

                for line in f:

                    try:
                        obj = json.loads(line.strip())

                        if "translation" in obj:
                            obj = obj["translation"]

                        src = obj.get(src_lang)
                        tgt = obj.get(tgt_lang)

                        if src and tgt:
                            pairs.append((src, tgt))

                    except:
                        continue

            return pairs

        train_pairs = load_pairs(train_files[0])

        if len(train_pairs) < 1000:
            raise RuntimeError("Dataset too small")

        random.shuffle(train_pairs)

        val_size = max(1, int(len(train_pairs) * 0.1))

        return (
            train_pairs[:-val_size],
            train_pairs[-val_size:]
        )

    # -------------------------------------------------

    def build_model(self):

        cfg = self.config

        model_cfg = MCTConfig(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["d_model"],
            num_hidden_layers=cfg["layers"],
            num_attention_heads=cfg["heads"],
            intermediate_size=cfg["d_ff"],
            max_position_embeddings=512
        )

        return MCTTransformer(model_cfg).to(DEVICE)
    
    # -------------------------------------------------

    def train(self):

        logger.info(f"Training {self.model_size} | {self.tokenizer_name}")

        self.load_tokenizer()

        train_pairs, val_pairs = self.load_dataset()

        train_dataset = ParallelDataset(train_pairs, self.tokenizer)
        val_dataset = ParallelDataset(val_pairs, self.tokenizer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=nmt_collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"]
        )

        model = self.build_model()

        optimizer = AdamW(
            model.parameters(),
            lr=self.config["lr"],
            betas=(0.9, 0.98),
            eps=1e-8
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=1000
        )

        loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        # ====================================================
        # TRAIN LOOP
        # ====================================================

        for epoch in range(self.config["epochs"]):

            model.train()

            epoch_loss = 0

            for src, tgt in train_loader:

                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)

                optimizer.zero_grad()

                with autocast():

                    outputs = model(src)

                    logits = outputs.logits if hasattr(outputs, "logits") else outputs

                    loss = loss_fn(
                        logits.view(-1, logits.size(-1)),
                        tgt.view(-1)
                    )

                self.scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                self.scaler.step(optimizer)
                self.scaler.update()

                epoch_loss += loss.item()

            scheduler.step()

            logger.info(
                f"Epoch {epoch+1} loss={epoch_loss/len(train_loader):.4f}"
            )

        logger.info("Training finished")

# ============================================================

def main():

    trainer = Trainer(
        model_size="small",
        tokenizer_name="BPE_32K",
        lang_pair="de-en"
    )

    trainer.train()

if __name__ == "__main__":
    main()