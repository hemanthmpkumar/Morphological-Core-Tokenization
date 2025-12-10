# analyze_results.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_all_results(results_dir="results"):
    """Load all experiment results."""
    all_results = []
    
    for json_file in Path(results_dir).rglob("*.json"):
        if "ablation" not in str(json_file):  # Skip ablation files for now
            with open(json_file, 'r') as f:
                try:
                    data = json.load(f)
                    data['file_path'] = str(json_file)
                    all_results.append(data)
                except json.JSONDecodeError:
                    continue
    
    return pd.DataFrame(all_results)

def create_comparison_tables(df):
    """Create comparison tables from results."""
    
    # Extract key information
    comparisons = []
    
    for _, row in df.iterrows():
        comparison = {
            'tokenizer': row['config']['tokenizer_type'],
            'task': row['config']['task'],
            'vocab_size': row.get('tokenizer_vocab_size', 0),
            'accuracy': row.get('metrics', {}).get('accuracy'),
            'bleu_score': row.get('metrics', {}).get('bleu_score'),
            'inference_speed': row.get('metrics', {}).get('inference_speed'),
            'avg_tokens_per_sample': row.get('tokenization_stats', {}).get('avg_tokens_per_sample')
        }
        
        # Add ROUGE scores if available
        rouge_scores = row.get('metrics', {}).get('rouge_scores', {})
        for rouge_type, score in rouge_scores.items():
            comparison[f'{rouge_type}'] = score
        
        comparisons.append(comparison)
    
    comparison_df = pd.DataFrame(comparisons)
    
    # Save to CSV
    comparison_df.to_csv('results/comparison_table.csv', index=False)
    
    # Create LaTeX table for paper
    latex_table = comparison_df.to_latex(index=False, float_format="%.3f")
    with open('results/comparison_table.tex', 'w') as f:
        f.write(latex_table)
    
    print("Comparison tables created!")
    return comparison_df

def create_visualizations(df):
    """Create visualizations from results."""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. BLEU scores comparison
    bleu_data = df[df['bleu_score'].notna()]
    if not bleu_data.empty:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=bleu_data, x='tokenizer', y='bleu_score')
        ax.set_title('BLEU Scores by Tokenizer (WMT14 De-En)')
        ax.set_xlabel('Tokenizer')
        ax.set_ylabel('BLEU Score')
        plt.tight_layout()
        plt.savefig('results/visualizations/bleu_scores.png', dpi=300)
        plt.close()
    
    # 2. Accuracy comparison
    acc_data = df[df['accuracy'].notna()]
    if not acc_data.empty:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=acc_data, x='tokenizer', y='accuracy')
        ax.set_title('Morphological Awareness Accuracy by Tokenizer')
        ax.set_xlabel('Tokenizer')
        ax.set_ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig('results/visualizations/accuracy_scores.png', dpi=300)
        plt.close()
    
    # 3. Token efficiency
    token_data = df[df['avg_tokens_per_sample'].notna()]
    if not token_data.empty:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=token_data, x='tokenizer', y='avg_tokens_per_sample')
        ax.set_title('Token Efficiency by Tokenizer')
        ax.set_xlabel('Tokenizer')
        ax.set_ylabel('Average Tokens per Sample')
        plt.tight_layout()
        plt.savefig('results/visualizations/token_efficiency.png', dpi=300)
        plt.close()
    
    # 4. Inference speed comparison
    speed_data = df[df['inference_speed'].notna()]
    if not speed_data.empty:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=speed_data, x='tokenizer', y='inference_speed')
        ax.set_title('Inference Speed by Tokenizer')
        ax.set_xlabel('Tokenizer')
        ax.set_ylabel('Tokens per Second')
        plt.tight_layout()
        plt.savefig('results/visualizations/inference_speed.png', dpi=300)
        plt.close()
    
    print("Visualizations created!")

def analyze_ablation_studies():
    """Analyze ablation study results."""
    
    # Analyzer quality ablation
    with open('results/ablation/analyzer_quality.json', 'r') as f:
        quality_data = json.load(f)
    
    # Create plot
    qualities = []
    accuracies = []
    
    for key, metrics in quality_data.items():
        quality = float(key.split('_')[1])
        qualities.append(quality)
        accuracies.append(metrics.get('accuracy', 0))
    
    plt.figure(figsize=(10, 6))
    plt.plot(qualities, accuracies, 'o-', linewidth=2, markersize=10)
    plt.title('Impact of Analyzer Quality on Morphological Awareness Accuracy')
    plt.xlabel('WordNet Coverage (%)')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/visualizations/analyzer_quality_impact.png', dpi=300)
    plt.close()
    
    # Dropout probability ablation
    with open('results/ablation/dropout_probability.json', 'r') as f:
        dropout_data = json.load(f)
    
    dropouts = []
    accuracies = []
    
    for key, metrics in dropout_data.items():
        dropout = float(key.split('_')[1])
        dropouts.append(dropout)
        accuracies.append(metrics.get('accuracy', 0))
    
    plt.figure(figsize=(10, 6))
    plt.plot(dropouts, accuracies, 's-', linewidth=2, markersize=10)
    plt.title('Impact of Stem Dropout Probability on Accuracy')
    plt.xlabel('Dropout Probability')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/visualizations/dropout_impact.png', dpi=300)
    plt.close()
    
    print("Ablation study visualizations created!")

if __name__ == "__main__":
    # Create directories
    Path("results/visualizations").mkdir(parents=True, exist_ok=True)
    
    # Load and analyze results
    print("Loading results...")
    df = load_all_results()
    
    print(f"Loaded {len(df)} experiment results")
    
    # Create comparison tables
    print("\nCreating comparison tables...")
    comparison_df = create_comparison_tables(df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(comparison_df)
    
    # Analyze ablation studies
    print("\nAnalyzing ablation studies...")
    if Path("results/ablation").exists():
        analyze_ablation_studies()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for task in comparison_df['task'].unique():
        task_data = comparison_df[comparison_df['task'] == task]
        print(f"\nTask: {task.upper()}")
        print("-"*40)
        
        for tokenizer in task_data['tokenizer'].unique():
            tokenizer_data = task_data[task_data['tokenizer'] == tokenizer]
            print(f"\n{tokenizer.upper()}:")
            
            if 'accuracy' in tokenizer_data.columns:
                acc = tokenizer_data['accuracy'].mean()
                print(f"  Accuracy: {acc:.3f}")
            
            if 'bleu_score' in tokenizer_data.columns:
                bleu = tokenizer_data['bleu_score'].mean()
                print(f"  BLEU: {bleu:.2f}")
            
            if 'avg_tokens_per_sample' in tokenizer_data.columns:
                tokens = tokenizer_data['avg_tokens_per_sample'].mean()
                print(f"  Avg tokens/sample: {tokens:.1f}")
    
    print("\nAnalysis complete! Check the 'results/' directory for outputs.")
