# adverse-drug-reactions

Predicting which of 27 adverse-drug-reaction (ADR) categories a drug causes, from its
structure alone, on the [SIDER](http://sideeffects.embl.de/) dataset (1,427 drugs, via
[DeepChem](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz)).

Started in 2023 as a NAIL107 (Machine learning in Bioinformatics, MFF UK) term project by
Julian Krumm and Juraj Trappl. Revisited in 2026: the original evaluation leaked test data,
so this repo now has an honest evaluation pipeline: classic baselines (Phase 1) and
neural / pretrained models (Phase 2), all scored on the same scaffold folds.

## Results

Mean ROC-AUC over the 27 ADRs, scaffold split (no shared Murcko scaffold between train and
test), 5-fold CV, 3 seeds, same folds for every model. Full tables:
[`results/REPORT.md`](results/REPORT.md).

| Model | Phase | Scaffold split |
| --- | --- | --- |
| Coin flip (predict ADR frequency) | 1 | 0.500 |
| Molecule size only (2 numbers) | 1 | 0.532 |
| Chemprop D-MPNN (graph neural net) | 2 | 0.597 |
| 5 nearest neighbours by Tanimoto | 1 | 0.604 |
| Chemprop D-MPNN + RDKit descriptors | 2 | 0.613 |
| Multi-task MLP on fingerprint + descriptors | 2 | 0.624 |
| Pretrained Mol2vec embedding + random forest | 2 | 0.628 |
| Multi-task MLP + random forest, averaged | 2 | 0.647 |
| **Morgan fingerprint + RDKit descriptors, random forest** | 1 | **0.648** |

No Phase 2 model beats the Phase 1 random forest. With 1,427 drugs, the graph network and
the MLPs don't have enough data to learn better features than a fingerprint, and the
pretrained Mol2vec embedding loses a little information compared with the raw bits.
Published scaffold-split SIDER results with large pretrained models sit around 0.62-0.71,
so 0.65 is in the expected range; the ceiling seems to come from the labels, not the
model (near-identical drugs share only ~2/3 of their ADR labels). Phase 3 therefore
targets the data: label structure and pharmacology features.

## What changed vs. the 2023 version

The 2023 notebook reported 66-100% accuracy. That came from three problems:

1. Minority-class rows were duplicated **before** the train/test split, so ~39% of test
   rows were copies of training rows (decision trees memorised them).
2. The autoencoder regressed token IDs with MSE, so its 50-dim "word embedding" mostly
   encodes molecule size; it scores the same as the size-only baseline above.
3. Accuracy on imbalanced labels: always predicting the majority class already gives 74.9%.

The new pipeline fixes the evaluation: folds are fixed first, class imbalance is handled with class
weights inside each model (no resampling), metrics are ROC-AUC and PR-AUC per ADR, and
`tests/` checks that no scaffold or molecule appears on both sides of a split.

## Repository layout

```
adr/                    pipeline package
  data.py               load SIDER, strip counter-ions, Murcko scaffolds, duplicate groups
  features.py           Morgan fingerprint, RDKit descriptors, Mol2vec, size baseline
  splits.py             scaffold and random k-fold (identical molecules kept together)
  models.py             baselines, random forest, LightGBM, logistic regression, ensembles
  nn.py                 multi-task MLP and Chemprop D-MPNN (Phase 2, needs torch)
  evaluate.py           cross-validation with per-ADR ROC-AUC / PR-AUC, per-fold caching
scripts/
  run_benchmark.py      run every model x split x seed of a set (resumable)
  make_report.py        results/*.csv -> results/REPORT.md
tests/                  leakage, fold and model sanity checks
data/
  sider.csv             original DeepChem SIDER file
  sider_clean.csv       cleaned SMILES, scaffold and group per drug (generated)
  splits/               fold assignment per split and seed, reuse these to compare models
results/                per-run and per-ADR CSVs + REPORT.md

sider.ipynb, we_network.py, models/, data/smiles_embedding.csv, data/pubchem_fetch.csv,
heatmap_tanimoto.png    2023 autoencoder pipeline, kept for reference (see issues above)
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q tests
python scripts/run_benchmark.py --set phase1   # ~25 min on 4 cores
python scripts/run_benchmark.py --set phase2   # ~1.5 h on 2 cores, mostly Chemprop
python scripts/make_report.py                   # rerun either benchmark to resume if interrupted
```

## Data cleaning notes

- 182 SMILES contain several fragments. Counter-ions and solvents are removed (Na+, Cl-,
  water, acetate, citrate...): the largest fragment is kept, plus any other fragment with
  at least 15 heavy atoms, so combination drugs stay intact. Known casualty: small co-drugs
  such as levodopa in carbidopa/levodopa.
- After cleaning, 28 molecules appear more than once (e.g. sodium and calcium acetate),
  and their labels agree only 79% of the time. Such duplicates are always put in the same fold.

## Next (Phase 3)

- Label structure: PCA of the 27 labels, predict the drug's ADR count first
- Pharmacology features (ATC class, protein targets), which structure alone can't capture
- Error analysis on near-identical drugs with different labels; learning curves
- ChemBERTa / MoLFormer embeddings (Hugging Face was blocked in the environment used for Phase 2)
