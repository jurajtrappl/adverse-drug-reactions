# Results

Mean over 27 ADRs of the per-ADR ROC-AUC, averaged over 5 folds and 3 seeds.
PR-AUC lift = PR-AUC minus the ADR's prevalence (0 = no better than guessing).
Scaffold split = no Murcko scaffold is shared between train and test (the honest
'new drug' setting). All models use the same folds (`data/splits/`).

Regenerate: `python scripts/run_benchmark.py --set phase1`, `python scripts/run_benchmark.py --set phase2`, `python scripts/make_report.py`.

## Scaffold-split leaderboard (Phase 1 + 2)

Δ vs RF = difference to `morgan+desc+rf` on the same seed, mean [min, max] over the 3 seeds.

| Model | Phase | What it is | ROC-AUC | Δ vs RF | PR-AUC lift |
| --- | --- | --- | --- | --- | --- |
| **morgan+desc+rf** | 1 | Morgan + descriptors, random forest per ADR | 0.648 ± 0.005 | — | +0.106 |
| mlp + rf ensemble | 2 | Average of the multi-task MLP and the random forest | 0.647 ± 0.005 | -0.000 [-0.001, +0.000] | +0.106 |
| morgan+rf | 1 | Morgan fingerprint, random forest per ADR | 0.645 ± 0.004 | -0.003 [-0.006, -0.001] | +0.106 |
| desc+rf | 1 | RDKit descriptors, random forest per ADR | 0.642 ± 0.006 | -0.006 [-0.008, -0.004] | +0.103 |
| morgan+desc+lgbm | 1 | Morgan + descriptors, LightGBM per ADR | 0.636 ± 0.008 | -0.012 [-0.015, -0.009] | +0.099 |
| morgan+desc+rf-multi | 1 | Morgan + descriptors, one forest for all 27 ADRs | 0.634 ± 0.005 | -0.014 [-0.018, -0.011] | +0.090 |
| mol2vec+rf | 2 | Pretrained Mol2vec embedding, random forest per ADR | 0.628 ± 0.006 | -0.020 [-0.021, -0.018] | +0.093 |
| morgan+desc+mlp | 2 | Morgan + descriptors, multi-task MLP (27 outputs) | 0.624 ± 0.006 | -0.024 [-0.025, -0.023] | +0.090 |
| mol2vec+logreg | 2 | Pretrained Mol2vec embedding, logistic regression | 0.616 ± 0.003 | -0.032 [-0.035, -0.029] | +0.077 |
| chemprop+desc | 2 | Chemprop D-MPNN + RDKit descriptors, multi-task | 0.613 ± 0.001 | -0.035 [-0.040, -0.030] | +0.071 |
| mol2vec+mlp | 2 | Pretrained Mol2vec embedding, multi-task MLP | 0.612 ± 0.013 | -0.036 [-0.042, -0.025] | +0.073 |
| knn5 tanimoto | 1 | Copy labels of the 5 most similar training drugs (no training) | 0.604 ± 0.002 | -0.043 [-0.046, -0.040] | +0.059 |
| chemprop | 2 | Chemprop D-MPNN graph network, multi-task | 0.597 ± 0.012 | -0.051 [-0.056, -0.041] | +0.059 |
| morgan+logreg | 1 | Morgan fingerprint, logistic regression | 0.590 ± 0.006 | -0.058 [-0.061, -0.054] | +0.066 |
| size+logreg | 1 | Heavy-atom count + SMILES length, logistic regression | 0.532 ± 0.003 | -0.115 [-0.120, -0.108] | +0.019 |
| prior | 1 | Predict each ADR's training frequency (coin flip by construction) | 0.500 ± 0.000 | -0.148 [-0.152, -0.143] | +0.000 |

## Scaffold vs random split (Phase 1)

The random split lets near-identical drugs sit on both sides, so it flatters every model.

| Model | ROC-AUC scaffold | ROC-AUC random |
| --- | --- | --- |
| prior | 0.500 | 0.500 |
| size+logreg | 0.532 | 0.550 |
| knn5 tanimoto | 0.604 | 0.637 |
| morgan+logreg | 0.590 | 0.632 |
| morgan+rf | 0.645 | 0.684 |
| desc+rf | 0.642 | 0.679 |
| morgan+desc+rf | 0.648 | 0.685 |
| morgan+desc+rf-multi | 0.634 | 0.665 |
| morgan+desc+lgbm | 0.636 | 0.674 |

## Per ADR, scaffold split

| ADR | Positive drugs | ROC-AUC `morgan+desc+rf` | PR-AUC `morgan+desc+rf` |
| --- | --- | --- | --- |
| Gastrointestinal disorders | 91% | 0.770 | 0.965 |
| Nervous system disorders | 91% | 0.731 | 0.967 |
| Blood and lymphatic system disorders | 62% | 0.723 | 0.813 |
| Hepatobiliary disorders | 52% | 0.707 | 0.704 |
| Endocrine disorders | 23% | 0.703 | 0.445 |
| Neoplasms benign, malignant and unspecified (incl cysts and polyps) | 26% | 0.690 | 0.482 |
| Cardiac disorders | 69% | 0.680 | 0.820 |
| Reproductive system and breast disorders | 51% | 0.679 | 0.705 |
| Eye disorders | 61% | 0.671 | 0.750 |
| Renal and urinary disorders | 64% | 0.665 | 0.766 |
| Investigations | 81% | 0.661 | 0.889 |
| Psychiatric disorders | 71% | 0.656 | 0.813 |
| Respiratory, thoracic and mediastinal disorders | 74% | 0.647 | 0.824 |
| Infections and infestations | 70% | 0.644 | 0.808 |
| Musculoskeletal and connective tissue disorders | 70% | 0.643 | 0.805 |
| Social circumstances | 18% | 0.633 | 0.280 |
| Immune system disorders | 72% | 0.630 | 0.816 |
| Ear and labyrinth disorders | 46% | 0.627 | 0.605 |
| Vascular disorders | 78% | 0.623 | 0.843 |
| Injury, poisoning and procedural complications | 66% | 0.617 | 0.756 |
| Metabolism and nutrition disorders | 70% | 0.607 | 0.772 |
| Congenital, familial and genetic disorders | 18% | 0.606 | 0.287 |
| Pregnancy, puerperium and perinatal conditions | 9% | 0.599 | 0.166 |
| Product issues | 2% | 0.592 | 0.034 |
| Surgical and medical procedures | 15% | 0.588 | 0.214 |
| General disorders and administration site conditions | 91% | 0.553 | 0.923 |
| Skin and subcutaneous tissue disorders | 92% | 0.545 | 0.937 |
