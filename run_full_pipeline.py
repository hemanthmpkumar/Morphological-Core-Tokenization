# run_full_pipeline.py
import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command with description."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"Output:\n{result.stdout}")
        if result.stderr:
            print(f"Warnings/Errors:\n{result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Error output:\n{e.stderr}")
        return False

def run_full_pipeline(args):
    """Run the complete MCT pipeline."""
    
    steps = []
    
    if args.mode in ["complete", "setup"]:
        steps.extend([
            ("pip install -r requirements_minimal.txt", "Install dependencies"),
            ("python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\"", "Download NLTK data"),
            ("python verify_installation.py", "Verify installation"),
        ])
    
    if args.mode in ["complete", "data"]:
        steps.extend([
            ("python quick_download.py", "Download datasets"),
        ])
    
    if args.mode in ["complete", "train"]:
        steps.extend([
            ("python train_tokenizer.py --dataset c4 --vocab-size 10000 --sample", "Train MCT tokenizer"),
            ("python train_baselines.py --vocab-size 10000 --tokenizer bpe", "Train BPE baseline"),
        ])
    
    if args.mode in ["complete", "experiments"]:
        steps.extend([
            ("python run_experiment.py --task morphology --tokenizer mct", "Run morphology experiment"),
            ("python run_experiment.py --task morphology --tokenizer bpe", "Run BPE baseline"),
            ("python run_experiment.py --task translation --tokenizer mct", "Run translation experiment"),
        ])
    
    if args.mode in ["complete", "analysis"]:
        steps.extend([
            ("python analyze_results.py", "Analyze results"),
            ("python ablation_study.py", "Run ablation studies"),
        ])
    
    # Create necessary directories
    Path("results").mkdir(exist_ok=True)
    Path("models/tokenizers").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # Run all steps
    successful_steps = 0
    total_steps = len(steps)
    
    for i, (cmd, description) in enumerate(steps, 1):
        print(f"\n\nStep {i}/{total_steps}")
        if run_command(cmd, description):
            successful_steps += 1
        else:
            print(f"Step {i} failed. Continue? (y/n)")
            if input().lower() != 'y':
                print("Pipeline stopped by user.")
                break
    
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETED")
    print(f"{'='*60}")
    print(f"Successful steps: {successful_steps}/{total_steps}")
    
    if successful_steps == total_steps:
        print("✓ All steps completed successfully!")
    else:
        print("⚠ Some steps failed. Check the logs above.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full MCT pipeline")
    parser.add_argument("--mode", type=str,
                       choices=["complete", "setup", "data", "train", "experiments", "analysis"],
                       default="complete",
                       help="Which parts of the pipeline to run")
    
    args = parser.parse_args()
    run_full_pipeline(args)
