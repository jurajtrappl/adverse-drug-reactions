# Phase 1 results

Mean over 27 ADRs of the per-ADR ROC-AUC, averaged over 5 folds and 3 seeds.
PR-AUC lift = PR-AUC minus the ADR's prevalence (0 = no better than guessing).
Scaffold split = no Murcko scaffold is shared between train and test (the honest
'new drug' setting). Regenerate with `python scripts/run_phase1.py && python scripts/make_report.py`.

| Model | What it is | ROC-AUC scaffold | ROC-AUC random | PR-AUC lift scaffold |
| --- | --- | --- | --- | --- |
| prior | Predict each ADR's training frequency (coin flip by construction) | 0.500 ± 0.000 | 0.500 | +0.000 |
| size+logreg | Heavy-atom count + SMILES length, logistic regression | 0.532 ± 0.003 | 0.550 | +0.019 |
| knn5 tanimoto | Copy labels of the 5 most similar training drugs (no training) | 0.604 ± 0.002 | 0.637 | +0.059 |
| morgan+logreg | Morgan fingerprint, logistic regression | 0.590 ± 0.006 | 0.632 | +0.066 |
| morgan+rf | Morgan fingerprint, random forest per ADR | 0.645 ± 0.004 | 0.684 | +0.106 |
| desc+rf | RDKit descriptors, random forest per ADR | 0.642 ± 0.006 | 0.679 | +0.103 |
| **morgan+desc+rf** | Morgan + descriptors, random forest per ADR | 0.648 ± 0.005 | 0.685 | +0.106 |
| morgan+desc+rf-multi | Morgan + descriptors, one forest for all 27 ADRs | 0.634 ± 0.005 | 0.665 | +0.090 |
| morgan+desc+lgbm | Morgan + descriptors, LightGBM per ADR | 0.636 ± 0.008 | 0.674 | +0.099 |

## Per ADR: `morgan+desc+rf`, scaffold split

| ADR | Positive drugs | ROC-AUC | PR-AUC |
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
