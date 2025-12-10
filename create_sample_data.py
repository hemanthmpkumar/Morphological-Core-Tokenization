# create_sample_data.py

import json
import os
from pathlib import Path

def create_sample_translation_data(output_dir: str = "sample_data"):
    """Create sample WMT14-style translation data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample German-English sentences
    de_samples = [
        "Das ist ein Beispielsatz.",
        "Die Maschine lernt von Daten.",
        "Tokenisierung ist wichtig für Sprachmodelle.",
        "Morphologie betrifft die Wortstruktur.",
        "Deutsche Sprache hat komplexe Morphologie.",
        "Der Algorithmus funktioniert gut.",
        "Wir benötigen mehr Trainingsdaten.",
        "Die Ergebnisse sind vielversprechend.",
        "Forschung in NLP schreitet schnell voran.",
        "Künstliche Intelligenz verändert die Welt."
    ]
    
    en_samples = [
        "This is a sample sentence.",
        "The machine learns from data.",
        "Tokenization is important for language models.",
        "Morphology concerns word structure.",
        "German language has complex morphology.",
        "The algorithm works well.",
        "We need more training data.",
        "The results are promising.",
        "Research in NLP is advancing quickly.",
        "Artificial intelligence is changing the world."
    ]
    
    # Create train/val/test splits
    splits = {
        'train': (de_samples[:7], en_samples[:7]),
        'val': (de_samples[7:8], en_samples[7:8]),
        'test': (de_samples[8:], en_samples[8:])
    }
    
    for split, (de_data, en_data) in splits.items():
        with open(os.path.join(output_dir, f"{split}.de"), 'w', encoding='utf-8') as f:
            f.write("\n".join(de_data))
        with open(os.path.join(output_dir, f"{split}.en"), 'w', encoding='utf-8') as f:
            f.write("\n".join(en_data))
    
    print(f"Sample translation data created in {output_dir}")

def create_sample_morphology_data(output_dir: str = "sample_data"):
    """Create sample morphological awareness data."""
    samples = [
        "unnecessarily,necessary,un,ly",
        "running,run,,ing",
        "happiness,happy,ness,",
        "unbelievable,believe,un,able",
        "preprocessing,process,pre,ing",
        "disagreement,agree,dis,ment",
        "beautifully,beautiful,,ly",
        "rewritten,write,re,en",
        "impossible,possible,im,",
        "nationality,nation,al,ity"
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "morphology.csv"), 'w', encoding='utf-8') as f:
        f.write("word,root,prefix,suffix\n")
        f.write("\n".join(samples))
    
    print(f"Sample morphology data created in {output_dir}")

if __name__ == "__main__":
    create_sample_translation_data()
    create_sample_morphology_data()
