# MCT Experiment Results Report

This directory contains automatically generated reports and visualizations for the Morphological-Core Tokenization (MCT) experiments.

## Generated Files

### Analysis Files (CSV)
- `results_summary.csv`: Complete results from all experiments
- `comparison_table.csv`: Aggregated comparison by task and tokenizer

### Visualization Files (PNG/PDF)
- `performance_comparison.png/pdf`: BLEU scores and accuracy comparison
- `token_efficiency.png/pdf`: Tokenization compactness and efficiency
- `comprehensive_comparison.png/pdf`: Combined metrics visualization

### Paper Submission Files
- `paper/comparison_table.tex`: LaTeX table ready for paper submission
- `paper/*.pdf`: High-quality PDF figures for paper submission

### Summary Reports
- `executive_summary.md`: Summary of key findings and insights
- `README.md`: This file

## How to Use

### For Paper Submission
1. Use `paper/comparison_table.tex` in your LaTeX document
2. Include PDF figures from `paper/` directory
3. Reference the figures in your text

### For Presentations
1. Use PNG images from `report/` directory
2. Refer to `executive_summary.md` for key talking points
3. Use CSV files for detailed analysis if needed

### For Further Analysis
1. Load `results_summary.csv` in Pandas or Excel
2. Filter by task or tokenizer for specific analyses
3. Create additional visualizations as needed

## Regenerating the Report

To regenerate this report with new experiment results:
```bash
python generate_report.py
```

## Requirements

```bash
pip install pandas matplotlib seaborn
```
