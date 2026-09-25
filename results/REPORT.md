# Results

Mean over 27 ADRs of the per-ADR ROC-AUC, averaged over 5 folds and 3 seeds.
PR-AUC lift = PR-AUC minus the ADR's prevalence (0 = no better than guessing).
Scaffold split = no Murcko scaffold is shared between train and test (the honest
'new drug' setting). All models use the same folds (`data/splits/`).
Input 'pharma' = ATC class and/or SIDER indications: known only for marketed drugs,
so those rows explain ADRs rather than predict them for a new molecule.

Regenerate: `python scripts/run_benchmark.py --set phase1`, `--set phase2`, `--set phase3`, then `python scripts/make_report.py`.

## Scaffold-split leaderboard (all phases)

Δ vs RF = difference to `morgan+desc+rf` on the same seed, mean [min, max] over the 3 seeds.

| Model | Phase | Input | What it is | ROC-AUC | Δ vs RF | PR-AUC lift |
| --- | --- | --- | --- | --- | --- | --- |
| **structure rf + pharma rf** | 3 | structure + pharma | Average of a structure forest and a pharmacology forest | 0.728 ± 0.003 | +0.080 [+0.078, +0.081] | +0.171 |
| atc+ind+rf | 3 | pharma | ATC + indications, random forest | 0.710 ± 0.000 | +0.062 [+0.058, +0.067] | +0.157 |
| ind+rf | 3 | pharma | SIDER indications (what the drug treats), random forest | 0.683 ± 0.001 | +0.035 [+0.030, +0.042] | +0.140 |
| morgan+desc+atc+ind+rf | 3 | structure + pharma | Structure + ATC + indications in one random forest | 0.664 ± 0.006 | +0.016 [+0.013, +0.019] | +0.120 |
| count-stacked rf | 3 | structure | Morgan + descriptors + predicted ADR count, random forest | 0.648 ± 0.005 | +0.001 [-0.001, +0.002] | +0.107 |
| morgan+desc+rf | 1 | structure | Morgan + descriptors, random forest per ADR | 0.648 ± 0.005 | — | +0.106 |
| mlp + rf ensemble | 2 | structure | Average of the multi-task MLP and the random forest | 0.647 ± 0.005 | -0.000 [-0.001, +0.000] | +0.106 |
| morgan+rf | 1 | structure | Morgan fingerprint, random forest per ADR | 0.645 ± 0.004 | -0.003 [-0.006, -0.001] | +0.106 |
| desc+rf | 1 | structure | RDKit descriptors, random forest per ADR | 0.642 ± 0.006 | -0.006 [-0.008, -0.004] | +0.103 |
| morgan+desc+lgbm | 1 | structure | Morgan + descriptors, LightGBM per ADR | 0.636 ± 0.008 | -0.012 [-0.015, -0.009] | +0.099 |
| morgan+desc+rf-multi | 1 | structure | Morgan + descriptors, one forest for all 27 ADRs | 0.634 ± 0.005 | -0.014 [-0.018, -0.011] | +0.090 |
| mol2vec+rf | 2 | structure | Pretrained Mol2vec embedding, random forest per ADR | 0.628 ± 0.006 | -0.020 [-0.021, -0.018] | +0.093 |
| morgan+desc+mlp | 2 | structure | Morgan + descriptors, multi-task MLP (27 outputs) | 0.624 ± 0.006 | -0.024 [-0.025, -0.023] | +0.090 |
| mol2vec+logreg | 2 | structure | Pretrained Mol2vec embedding, logistic regression | 0.616 ± 0.003 | -0.032 [-0.035, -0.029] | +0.077 |
| count-only | 3 | structure | Predicted ADR count only (from structure), logistic regression per ADR | 0.615 ± 0.004 | -0.033 [-0.034, -0.032] | +0.076 |
| chemprop+desc | 2 | structure | Chemprop D-MPNN + RDKit descriptors, multi-task | 0.613 ± 0.001 | -0.035 [-0.040, -0.030] | +0.071 |
| atc+rf | 3 | pharma | WHO ATC class (level 1+2, 45% of drugs matched), random forest | 0.612 ± 0.001 | -0.036 [-0.039, -0.032] | +0.077 |
| mol2vec+mlp | 2 | structure | Pretrained Mol2vec embedding, multi-task MLP | 0.612 ± 0.013 | -0.036 [-0.042, -0.025] | +0.073 |
| knn5 tanimoto | 1 | structure | Copy labels of the 5 most similar training drugs (no training) | 0.604 ± 0.002 | -0.043 [-0.046, -0.040] | +0.059 |
| chemprop | 2 | structure | Chemprop D-MPNN graph network, multi-task | 0.597 ± 0.012 | -0.051 [-0.056, -0.041] | +0.059 |
| morgan+logreg | 1 | structure | Morgan fingerprint, logistic regression | 0.590 ± 0.006 | -0.058 [-0.061, -0.054] | +0.066 |
| size+logreg | 1 | structure | Heavy-atom count + SMILES length, logistic regression | 0.532 ± 0.003 | -0.115 [-0.120, -0.108] | +0.019 |
| prior | 1 | — | Predict each ADR's training frequency (coin flip by construction) | 0.500 ± 0.000 | -0.148 [-0.152, -0.143] | +0.000 |

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

| ADR | Positive drugs | ROC-AUC `structure rf + pharma rf` | ROC-AUC `morgan+desc+rf` | PR-AUC `structure rf + pharma rf` |
| --- | --- | --- | --- | --- |
| Gastrointestinal disorders | 91% | 0.858 | 0.770 | 0.979 |
| Blood and lymphatic system disorders | 62% | 0.794 | 0.723 | 0.868 |
| Neoplasms benign, malignant and unspecified (incl cysts and polyps) | 26% | 0.774 | 0.690 | 0.609 |
| Nervous system disorders | 91% | 0.772 | 0.731 | 0.972 |
| Hepatobiliary disorders | 52% | 0.772 | 0.707 | 0.775 |
| Cardiac disorders | 69% | 0.764 | 0.680 | 0.882 |
| Endocrine disorders | 23% | 0.755 | 0.703 | 0.539 |
| Psychiatric disorders | 71% | 0.755 | 0.656 | 0.886 |
| Investigations | 81% | 0.754 | 0.661 | 0.923 |
| Reproductive system and breast disorders | 51% | 0.752 | 0.679 | 0.775 |
| Eye disorders | 61% | 0.745 | 0.671 | 0.824 |
| Musculoskeletal and connective tissue disorders | 70% | 0.744 | 0.643 | 0.872 |
| Renal and urinary disorders | 64% | 0.743 | 0.665 | 0.830 |
| Respiratory, thoracic and mediastinal disorders | 74% | 0.737 | 0.647 | 0.886 |
| Vascular disorders | 78% | 0.728 | 0.623 | 0.900 |
| Metabolism and nutrition disorders | 70% | 0.714 | 0.607 | 0.851 |
| Ear and labyrinth disorders | 46% | 0.709 | 0.627 | 0.712 |
| Injury, poisoning and procedural complications | 66% | 0.708 | 0.617 | 0.821 |
| Infections and infestations | 70% | 0.707 | 0.644 | 0.856 |
| Surgical and medical procedures | 15% | 0.704 | 0.588 | 0.358 |
| Social circumstances | 18% | 0.701 | 0.633 | 0.396 |
| Skin and subcutaneous tissue disorders | 92% | 0.699 | 0.545 | 0.962 |
| General disorders and administration site conditions | 91% | 0.688 | 0.553 | 0.956 |
| Immune system disorders | 72% | 0.682 | 0.630 | 0.853 |
| Pregnancy, puerperium and perinatal conditions | 9% | 0.674 | 0.599 | 0.259 |
| Congenital, familial and genetic disorders | 18% | 0.673 | 0.606 | 0.354 |
| Product issues | 2% | 0.547 | 0.592 | 0.032 |
