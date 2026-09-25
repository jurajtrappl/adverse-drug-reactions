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
| **structure rf + pharma rf** | 3 | structure + pharma | Average of a structure forest and a pharmacology forest | 0.736 ± 0.000 | +0.074 [+0.070, +0.076] | +0.175 |
| atc+ind+rf | 3 | pharma | ATC + indications, random forest | 0.714 ± 0.002 | +0.051 [+0.047, +0.054] | +0.160 |
| ind+rf | 3 | pharma | SIDER indications (what the drug treats), random forest | 0.688 ± 0.001 | +0.025 [+0.022, +0.027] | +0.142 |
| morgan+desc+atc+ind+rf | 3 | structure + pharma | Structure + ATC + indications in one random forest | 0.677 ± 0.004 | +0.014 [+0.013, +0.015] | +0.127 |
| morgan+desc+rf-multi | 1 | structure | Morgan + descriptors, one forest for all 27 ADRs (no class weights) | 0.665 ± 0.001 | +0.002 [+0.000, +0.003] | +0.114 |
| morgan+desc+rf | 1 | structure | Morgan + descriptors, random forest per ADR | 0.663 ± 0.002 | — | +0.111 |
| count-stacked rf | 3 | structure | Morgan + descriptors + predicted ADR count, random forest | 0.662 ± 0.003 | -0.001 [-0.003, +0.002] | +0.112 |
| mlp + rf ensemble | 2 | structure | Average of the multi-task MLP and the random forest | 0.660 ± 0.003 | -0.002 [-0.005, +0.002] | +0.112 |
| morgan+rf | 1 | structure | Morgan fingerprint, random forest per ADR | 0.660 ± 0.003 | -0.003 [-0.005, -0.001] | +0.111 |
| desc+rf | 1 | structure | RDKit descriptors, random forest per ADR | 0.658 ± 0.002 | -0.005 [-0.006, -0.003] | +0.108 |
| morgan+desc+lgbm | 1 | structure | Morgan + descriptors, LightGBM per ADR | 0.654 ± 0.002 | -0.008 [-0.013, -0.005] | +0.105 |
| mol2vec+rf | 2 | structure | Pretrained Mol2vec embedding, random forest per ADR | 0.644 ± 0.003 | -0.019 [-0.020, -0.018] | +0.098 |
| morgan+desc+mlp | 2 | structure | Morgan + descriptors, multi-task MLP (27 outputs) | 0.635 ± 0.002 | -0.027 [-0.029, -0.024] | +0.095 |
| chemprop+desc | 2 | structure | Chemprop D-MPNN + RDKit descriptors, multi-task | 0.633 ± 0.011 | -0.030 [-0.041, -0.024] | +0.085 |
| count-only | 3 | structure | Predicted ADR count only (from structure), logistic regression per ADR | 0.630 ± 0.003 | -0.033 [-0.037, -0.028] | +0.082 |
| mol2vec+mlp | 2 | structure | Pretrained Mol2vec embedding, multi-task MLP | 0.627 ± 0.003 | -0.035 [-0.038, -0.031] | +0.080 |
| mol2vec+logreg | 2 | structure | Pretrained Mol2vec embedding, logistic regression | 0.626 ± 0.001 | -0.037 [-0.039, -0.033] | +0.080 |
| chemprop | 2 | structure | Chemprop D-MPNN graph network, multi-task | 0.619 ± 0.009 | -0.044 [-0.051, -0.038] | +0.073 |
| knn5 tanimoto | 1 | structure | Copy labels of the 5 most similar training drugs (no training) | 0.613 ± 0.002 | -0.050 [-0.052, -0.048] | +0.068 |
| atc+rf | 3 | pharma | WHO ATC class (level 1+2, 45% of drugs matched), random forest | 0.611 ± 0.001 | -0.051 [-0.055, -0.049] | +0.079 |
| morgan+logreg | 1 | structure | Morgan fingerprint, logistic regression | 0.608 ± 0.002 | -0.055 [-0.055, -0.054] | +0.073 |
| size+logreg | 1 | structure | Heavy-atom count + SMILES length, logistic regression | 0.548 ± 0.003 | -0.114 [-0.119, -0.111] | +0.018 |
| prior | 1 | — | Predict each ADR's training frequency (coin flip by construction) | 0.500 ± 0.000 | -0.163 [-0.165, -0.161] | +0.000 |

## Scaffold vs random split (Phase 1)

The random split lets near-identical drugs sit on both sides, so it flatters every model.

| Model | ROC-AUC scaffold | ROC-AUC random |
| --- | --- | --- |
| prior | 0.500 | 0.500 |
| size+logreg | 0.548 | 0.550 |
| knn5 tanimoto | 0.613 | 0.637 |
| morgan+logreg | 0.608 | 0.632 |
| morgan+rf | 0.660 | 0.684 |
| desc+rf | 0.658 | 0.679 |
| morgan+desc+rf | 0.663 | 0.686 |
| morgan+desc+rf-multi | 0.665 | 0.687 |
| morgan+desc+lgbm | 0.654 | 0.674 |

## Per ADR, scaffold split

| ADR | Positive drugs | ROC-AUC `structure rf + pharma rf` | ROC-AUC `morgan+desc+rf` | PR-AUC `structure rf + pharma rf` |
| --- | --- | --- | --- | --- |
| Gastrointestinal disorders | 91% | 0.858 | 0.770 | 0.981 |
| Blood and lymphatic system disorders | 62% | 0.796 | 0.724 | 0.872 |
| Nervous system disorders | 91% | 0.789 | 0.766 | 0.974 |
| Hepatobiliary disorders | 52% | 0.787 | 0.718 | 0.789 |
| Neoplasms benign, malignant and unspecified (incl cysts and polyps) | 26% | 0.781 | 0.706 | 0.624 |
| Cardiac disorders | 69% | 0.773 | 0.687 | 0.889 |
| Reproductive system and breast disorders | 51% | 0.773 | 0.706 | 0.791 |
| Skin and subcutaneous tissue disorders | 92% | 0.765 | 0.660 | 0.971 |
| Psychiatric disorders | 71% | 0.764 | 0.677 | 0.893 |
| Investigations | 81% | 0.758 | 0.676 | 0.924 |
| Musculoskeletal and connective tissue disorders | 70% | 0.755 | 0.665 | 0.880 |
| Renal and urinary disorders | 64% | 0.750 | 0.670 | 0.833 |
| Endocrine disorders | 23% | 0.750 | 0.695 | 0.541 |
| Eye disorders | 61% | 0.746 | 0.671 | 0.827 |
| Vascular disorders | 78% | 0.738 | 0.644 | 0.905 |
| Respiratory, thoracic and mediastinal disorders | 74% | 0.735 | 0.646 | 0.886 |
| Ear and labyrinth disorders | 46% | 0.725 | 0.646 | 0.728 |
| Metabolism and nutrition disorders | 70% | 0.725 | 0.625 | 0.858 |
| Infections and infestations | 70% | 0.713 | 0.659 | 0.860 |
| Surgical and medical procedures | 15% | 0.712 | 0.606 | 0.365 |
| Injury, poisoning and procedural complications | 66% | 0.705 | 0.617 | 0.822 |
| Social circumstances | 18% | 0.703 | 0.632 | 0.399 |
| General disorders and administration site conditions | 91% | 0.698 | 0.597 | 0.956 |
| Immune system disorders | 72% | 0.694 | 0.645 | 0.858 |
| Pregnancy, puerperium and perinatal conditions | 9% | 0.683 | 0.580 | 0.246 |
| Congenital, familial and genetic disorders | 18% | 0.672 | 0.610 | 0.357 |
| Product issues | 2% | 0.529 | 0.593 | 0.027 |
