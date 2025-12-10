import json
from mct_datamodel import *

def ablation_study_analyzer_quality():
    """Study impact of analyzer quality (simulating imperfect WordNet)."""
    
    print("Running ablation study: Analyzer Quality")
    
    analyzer_qualities = [1.0, 0.75, 0.5, 0.25]  # 100%, 75%, 50%, 25% of WordNet
    
    results = {}
    
    for quality in analyzer_qualities:
        print(f"\nTesting with analyzer quality: {quality}")
        
        config = ExperimentConfig(
            tokenizer_type=TokenizerType.MCT,
            tokenizer_config=TokenizerConfig(
                vocab_size=10000,
                p_drop=0.05,
                analyzer_quality=quality
            ),
            training_config=TrainingConfig(
                batch_size=16,
                num_epochs=3
            ),
            task=TaskType.MORPHOLOGICAL_AWARENESS,
            language=Language.ENGLISH,
            dataset_paths={'data': 'data/morphology/morphology.csv'},
            output_dir=f'results/ablation/quality_{quality}',
            seed=42
        )
        
        manager = ExperimentManager(config)
        result = manager.run_experiment()
        
        results[f"quality_{quality}"] = result.model_metrics.to_dict()
    
    # Save results
    with open('results/ablation/analyzer_quality.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nAblation study completed!")

def ablation_study_dropout_probability():
    """Study impact of stem dropout probability."""
    
    print("Running ablation study: Dropout Probability")
    
    dropout_probs = [0.0, 0.05, 0.1, 0.2, 0.5]
    
    results = {}
    
    for p_drop in dropout_probs:
        print(f"\nTesting with dropout probability: {p_drop}")
        
        config = ExperimentConfig(
            tokenizer_type=TokenizerType.MCT,
            tokenizer_config=TokenizerConfig(
                vocab_size=10000,
                p_drop=p_drop
            ),
            training_config=TrainingConfig(
                batch_size=16,
                num_epochs=3
            ),
            task=TaskType.MORPHOLOGICAL_AWARENESS,
            language=Language.ENGLISH,
            dataset_paths={'data': 'data/morphology/morphology.csv'},
            output_dir=f'results/ablation/dropout_{p_drop}',
            seed=42
        )
        
        manager = ExperimentManager(config)
        result = manager.run_experiment()
        
        results[f"dropout_{p_drop}"] = result.model_metrics.to_dict()
    
    # Save results
    with open('results/ablation/dropout_probability.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nDropout ablation study completed!")

if __name__ == "__main__":
    ablation_study_analyzer_quality()
    ablation_study_dropout_probability()
