# Assignment 2 Report

This directory contains a comprehensive LaTeX report for Assignment 2 of CS324: Deep Learning.

## Files

- `Assignment2_Report.tex` - LaTeX source file
- `Assignment2_Report.pdf` - Compiled PDF report (pre-generated)

## Report Contents

The report provides a detailed description of:

1. **Part I: Multi-Layer Perceptron**
   - Architecture and implementation details
   - Experiments on make_moons and CIFAR10 datasets
   - Results and analysis

2. **Part II: Convolutional Neural Network**
   - VGG-style architecture description
   - Training configuration and data augmentation
   - Expected performance and analysis
   - Dimensionality analysis through the network

3. **Part III: Vanilla RNN**
   - Mathematical formulation and implementation
   - Palindrome prediction task
   - Variable-length sequence experiments
   - Analysis of vanishing gradient problem

4. **Jupyter Notebooks**
   - Description of all four notebooks
   - Key features and capabilities

5. **Conclusions**
   - Key findings and lessons learned
   - Future directions

## Compiling the Report

If you need to recompile the LaTeX source:

### Prerequisites

Install LaTeX on your system:

**Ubuntu/Debian:**
```bash
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

**macOS (with Homebrew):**
```bash
brew install --cask mactex
```

**Windows:**
Download and install [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)

### Compilation

Run the following commands in this directory:

```bash
pdflatex Assignment2_Report.tex
pdflatex Assignment2_Report.tex  # Run twice to resolve references
```

This will generate `Assignment2_Report.pdf`.

### Cleaning Up

To remove auxiliary files after compilation:

```bash
rm -f *.aux *.log *.out *.toc *.lof *.lot
```

## Report Structure

The report includes:
- Abstract
- Introduction and objectives
- Detailed implementation for all three parts
- Mathematical formulations
- Experimental results and analysis
- Code listings
- Tables summarizing architectures and results
- Comprehensive bibliography

Total length: 10 pages

## Language

The report is written in English as requested.
