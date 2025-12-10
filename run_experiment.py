# run_experiment.py
import argparse
from mct_datamodel import *

def run_morphology_experiment(tokenizer_type):
    """Run morphological awareness benchmark."""
    
    config = ExperimentConfig(
        tokenizer_type=tokenizer_type,
        tokenizer_config=TokenizerConfig(vocab_size=10000, p_drop=0.05),
        training_config=TrainingConfig(
            batch_size=16,
            num_epochs=5,
            learning_rate=1e-4
        ),
        task=TaskType.MORPHOLOGICAL_AWARENESS,
        language=Language.ENGLISH,
        dataset_paths={
            'data': 'data/morphology/morphology.csv'
        },
        output_dir=f'results/morphology/{tokenizer_type.value}',
        seed=42
    )
    
    manager = ExperimentManager(config)
    result = manager.run_experiment()
    
    return result

def run_translation_experiment(tokenizer_type):
    """Run machine translation experiment."""
    
    config = ExperimentConfig(
        tokenizer_type=tokenizer_type,
        tokenizer_config=TokenizerConfig(vocab_size=32000, p_drop=0.05),
        training_config=TrainingConfig(
            batch_size=32,
            num_epochs=10,
            learning_rate=5e-5
        ),
        task=TaskType.MACHINE_TRANSLATION,
        language=Language.GERMAN,
        dataset_paths={
            'train_src': 'data/wmt14/train.de',
            'train_tgt': 'data/wmt14/train.en',
            'val_src': 'data/wmt14/validation.de',
            'val_tgt': 'data/wmt14/validation.en',
            'test_src': 'data/wmt14/test.de',
            'test_tgt': 'data/wmt14/test.en'
        },
        output_dir=f'results/translation/{tokenizer_type.value}',
        seed=42
    )
    
    manager = ExperimentManager(config)
    result = manager.run_experiment()
    
    return result

def run_summarization_experiment(tokenizer_type):
    """Run summarization experiment."""
    
    config = ExperimentConfig(
        tokenizer_type=tokenizer_type,
        tokenizer_config=TokenizerConfig(vocab_size=32000, p_drop=0.05),
        training_config=TrainingConfig(
            batch_size=8,  # Smaller batch for longer sequences
            num_epochs=5,
            learning_rate=3e-5,
            max_seq_length=1024
        ),
        task=TaskType.SUMMARIZATION,
        language=Language.ENGLISH,
        dataset_paths={
            'train': 'data/arxiv/train.jsonl',
            'val': 'data/arxiv/validation.jsonl',
            'test': 'data/arxiv/test.jsonl'
        },
        output_dir=f'results/summarization/{tokenizer_type.value}',
        seed=42
    )
    
    manager = ExperimentManager(config)
    result = manager.run_experiment()
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCT Experiments")
    parser.add_argument("--task", type=str,
                       choices=["morphology", "translation", "summarization", "all"],
                       default="morphology")
    parser.add_argument("--tokenizer", type=str,
                       choices=["mct", "bpe", "wordpiece", "bytelevel", "all"],
                       default="mct")
    
    args = parser.parse_args()
    
    # Map tokenizer types
    tokenizer_types = []
    if args.tokenizer == "all":
        tokenizer_types = [TokenizerType.MCT, TokenizerType.BPE,
                          TokenizerType.WORDPIECE, TokenizerType.BYTELEVEL]
    else:
        tokenizer_types = [TokenizerType(args.tokenizer)]
    
    # Run experiments
    results = []
    
    for tokenizer_type in tokenizer_types:
        print(f"\n{'='*60}")
        print(f"Running {args.task} experiment with {tokenizer_type.value}")
        print(f"{'='*60}")
        
        if args.task == "morphology":
            result = run_morphology_experiment(tokenizer_type)
        elif args.task == "translation":
            result = run_translation_experiment(tokenizer_type)
        elif args.task == "summarization":
            result = run_summarization_experiment(tokenizer_type)
        elif args.task == "all":
            # Run all tasks
            result1 = run_morphology_experiment(tokenizer_type)
            result2 = run_translation_experiment(tokenizer_type)
            result3 = run_summarization_experiment(tokenizer_type)
            results.extend([result1, result2, result3])
            continue
        
        results.append(result)
    
    print(f"\nCompleted {len(results)} experiments")
