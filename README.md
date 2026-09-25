# adverse-drug-reactions

Predicting which of 27 adverse-drug-reaction (ADR) categories a drug causes, from its
structure alone, on the [SIDER](http://sideeffects.embl.de/) dataset (1,427 drugs, via
[DeepChem](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz)).

Started in 2023 as a NAIL107 (Machine learning in Bioinformatics, MFF UK) term project by
Julian Krumm and Juraj Trappl. Revisited in 2026: the original evaluation leaked test data,
so this repo now has an honest baseline pipeline (Phase 1).

## Phase 1 results

Mean ROC-AUC over the 27 ADRs, 5-fold CV, 3 seeds. Full tables: [`results/REPORT.md`](results/REPORT.md).

| Model | Scaffold split | Random split |
| --- | --- | --- |
| Coin flip (predict ADR frequency) | 0.500 | 0.500 |
| Molecule size only (2 numbers) | 0.532 | 0.550 |
| 5 nearest neighbours by Tanimoto | 0.604 | 0.637 |
| **Morgan fingerprint + RDKit descriptors, random forest** | **0.648** | **0.685** |

Published scaffold-split SIDER results with large pretrained models sit around 0.62-0.71,
so a fingerprint forest is a fair baseline to beat. The scaffold split (no shared
Murcko scaffold between train and test) is the number to quote: it measures
performance on structurally new drugs.

## What changed vs. the 2023 version

The 2023 notebook reported 66-100% accuracy. That came from three problems:

1. Minority-class rows were duplicated **before** the train/test split, so ~39% of test
   rows were copies of training rows (decision trees memorised them).
2. The autoencoder regressed token IDs with MSE, so its 50-dim "word embedding" mostly
   encodes molecule size; it scores the same as the size-only baseline above.
3. Accuracy on imbalanced labels: always predicting the majority class already gives 74.9%.

Phase 1 fixes the evaluation: folds are fixed first, class imbalance is handled with class
weights inside each model (no resampling), metrics are ROC-AUC and PR-AUC per ADR, and
`tests/` checks that no scaffold or molecule appears on both sides of a split.

## Repository layout

```
adr/                    Phase 1 package
  data.py               load SIDER, strip counter-ions, Murcko scaffolds, duplicate groups
  features.py           Morgan fingerprint, RDKit descriptors, size baseline
  splits.py             scaffold and random k-fold (identical molecules kept together)
  models.py             baselines, random forest, LightGBM, logistic regression
  evaluate.py           cross-validation with per-ADR ROC-AUC / PR-AUC
scripts/
  run_phase1.py         run every model x split x seed (resumable)
  make_report.py        results/*.csv -> results/REPORT.md
tests/test_phase1.py    leakage and sanity checks
data/
  sider.csv             original DeepChem SIDER file
  sider_clean.csv       cleaned SMILES, scaffold and group per drug (generated)
  splits/               fold assignment per split and seed, reuse these to compare models
results/                Phase 1 outputs

sider.ipynb, we_network.py, models/, data/smiles_embedding.csv, data/pubchem_fetch.csv,
heatmap_tanimoto.png    2023 autoencoder pipeline, kept for reference (see issues above)
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q tests
python scripts/run_phase1.py      # ~25 min on 4 cores; rerun to resume if interrupted
python scripts/make_report.py
```

## Data cleaning notes

- 182 SMILES contain several fragments. Counter-ions and solvents are removed (Na+, Cl-,
  water, acetate, citrate...): the largest fragment is kept, plus any other fragment with
  at least 15 heavy atoms, so combination drugs stay intact. Known casualty: small co-drugs
  such as levodopa in carbidopa/levodopa.
- After cleaning, 28 molecules appear more than once (e.g. sodium and calcium acetate),
  and their labels agree only 79% of the time. Such duplicates are always put in the same fold.

## Next (Phase 2)

- Multi-task neural net on fingerprints; Chemprop (D-MPNN) with the saved scaffold splits
- Frozen pretrained embeddings (ChemBERTa, MoLFormer, Mol2vec) + the same classifiers
- Pharmacology features (ATC class, protein targets), which structure alone can't capture
