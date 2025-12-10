
# test_datamodel.py
import sys
import os
sys.path.append('.')

from mct_datamodel import (
    TokenizerType, TaskType, Language,
    TokenizerConfig, TrainingConfig, ExperimentConfig,
    CorpusDataset, ExperimentManager
)

def test_datamodel():
    print("Testing Data Model...")
    
    # Test configurations
    print("1. Testing configurations...")
    tokenizer_config = TokenizerConfig(vocab_size=1000, p_drop=0.05)
    training_config = TrainingConfig(batch_size=8, num_epochs=2)
    
    experiment_config = ExperimentConfig(
        tokenizer_type=TokenizerType.MCT,
        tokenizer_config=tokenizer_config,
        training_config=training_config,
        task=TaskType.MORPHOLOGICAL_AWARENESS,
        language=Language.ENGLISH,
        dataset_paths={
            'data': 'data/morphology/morphology.csv'
        },
        output_dir='results/test',
        seed=42
    )
    
    print(f"Tokenizer Config: {tokenizer_config}")
    print(f"Training Config: {training_config}")
    print(f"Experiment Config created successfully!")
    
    # Test datasets
    print("\n2. Testing datasets...")
    
    # Create a simple test dataset
    test_data = ["This is test sentence 1.", "This is test sentence 2."]
    with open('test_corpus.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_data))
    
    dataset = CorpusDataset(['test_corpus.txt'])
    print(f"Dataset created with {len(dataset)} samples")
    
    # Test experiment manager
    print("\n3. Testing experiment manager...")
    manager = ExperimentManager(experiment_config)
    print("Experiment manager initialized successfully!")
    
    # Clean up
    if os.path.exists('test_corpus.txt'):
        os.remove('test_corpus.txt')
    
    print("\nData model test completed!")

if __name__ == "__main__":
    test_datamodel()
