# An Approach for the Detection of Code Clones in Software Testing

MS Thesis code, Software Engineering
**Author:** Sakha (CIIT/SP21-RSE-024/ISB)
**Supervisor:** Dr. Iftikhar Azim Niaz, Department of Computer Science
**University:** COMSATS University Islamabad, Islamabad, Pakistan (Fall 2023)

## Overview
Developers often reuse code to meet deadlines, which creates Type 1 to Type 4 clones
that make testing harder. This work detects **syntactic code clones** using
Convolutional Neural Networks (CNN). The thesis describes a grid-based
representation of tokenized code (rows and columns) fed to a CNN with one or two
convolution and max-pooling layers (ReLU and sigmoid activations).

## What is in this repository
| File | Description |
|---|---|
| `bcb_clone_pipeline.py` | Clone detection pipeline on **BigCloneBench**: lexical analysis, tokenization, AST parsing, fragment extraction, threshold-based similarity analysis, a 1D-CNN (siamese), and a comparison with conventional ML models (Logistic Regression, Naive Bayes, KNN, Decision Tree, Random Forest, SVM). |

> **Note:** the thesis reports experiments on the CodeSearchNet corpus, while this
> script evaluates on BigCloneBench. Add your CodeSearchNet / grid-based CNN scripts
> here too if you want the repository to cover every experiment in the thesis.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset
The dataset is not included. Download the BigCloneBench files from
[CodeXGLUE Clone-detection-BigCloneBench](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench)
and place `data.jsonl`, `train.txt`, `valid.txt` and `test.txt` in a folder named `dataset/`.

## Run
```bash
python bcb_clone_pipeline.py --data_dir dataset --seed 42
```
Quick test: `python bcb_clone_pipeline.py --data_dir dataset --n_train 10000 --n_test 3000 --epochs 4`

Output: a table of accuracy, precision, recall and F1 for every model, saved as `results.csv`.

## Results
Add the table produced by your own run of the script here.

Results reported in the thesis (CodeSearchNet, 1-layer CNN, 10-fold cross-validation):
see Chapter 6, Tables 6.3 to 6.6.

## Citation
```
Sakha. "An Approach for the Detection of Code Clones in Software Testing."
MS Thesis, COMSATS University Islamabad, 2023.
```

## Acknowledgements
BigCloneBench (Svajlenko and Roy) and CodeXGLUE (Microsoft) for the dataset format;
CodeSearchNet for the corpus used in the thesis experiments.
