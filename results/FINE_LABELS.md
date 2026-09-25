# Phase 4: specific side effects

1422 drugs, 579 MedDRA preferred terms listed for at least 50 drugs (prevalence 3.5%-84%). Scaffold split, 5 folds, 3 seeds; ROC-AUC per term and fold, averaged. Indication features that share a MedDRA term with any label are removed (see `pharma_without_label_terms`).

## Overall

| Model | Mean ROC-AUC | PR-AUC lift |
| --- | --- | --- |
| prior | 0.500 ± 0.000 | +0.000 |
| documentation flags + logreg | 0.622 ± 0.002 | +0.085 |
| knn5 tanimoto | 0.613 ± 0.002 | +0.075 |
| structure rf | 0.665 ± 0.004 | +0.141 |
| pharma rf | 0.689 ± 0.002 | +0.179 |
| structure rf + pharma rf | 0.726 ± 0.003 | +0.213 |

## By how common the side effect is

| Share of drugs listing the term | Terms | structure rf | pharma rf | structure rf + pharma rf |
| --- | --- | --- | --- | --- |
| <10% | 325 | 0.669 | 0.693 | 0.730 |
| 10-25% | 163 | 0.665 | 0.691 | 0.727 |
| 25-50% | 73 | 0.652 | 0.671 | 0.709 |
| >50% | 18 | 0.642 | 0.676 | 0.708 |

## Is it still one factor?

PC1 of the 579-term matrix explains 21% of the variance (27 organ classes: 33%) and correlates with the number of listed terms at |r| = 0.99. The 'long package insert' factor is still there.

## Where structure is strongest

Terms with the highest structure-only ROC-AUC. Most look like class effects of drug families a fingerprint recognises (antipsychotics, anticholinergics, antibiotics, steroid hormones).

| Term | Drugs listing it | Structure | Pharmacology | Structure - pharma |
| --- | --- | --- | --- | --- |
| Pseudomembranous colitis | 5% | 0.907 | 0.878 | +0.030 |
| Neuroleptic malignant syndrome | 4% | 0.879 | 0.918 | -0.039 |
| Accommodation disorder | 5% | 0.864 | 0.799 | +0.065 |
| Nephropathy toxic | 5% | 0.855 | 0.775 | +0.081 |
| Drug withdrawal syndrome | 4% | 0.853 | 0.889 | -0.036 |
| Akathisia | 5% | 0.853 | 0.857 | -0.004 |
| Menstruation irregular | 6% | 0.843 | 0.746 | +0.096 |
| Ileus paralytic | 4% | 0.837 | 0.667 | +0.171 |
| Extrapyramidal disorder | 7% | 0.830 | 0.868 | -0.037 |
| Breast enlargement | 6% | 0.824 | 0.789 | +0.035 |
| Opportunistic infection | 4% | 0.814 | 0.836 | -0.022 |
| Mydriasis | 8% | 0.804 | 0.766 | +0.039 |
| Muscle rigidity | 4% | 0.804 | 0.872 | -0.068 |
| Tubulointerstitial nephritis | 8% | 0.803 | 0.734 | +0.069 |
| Hirsutism | 6% | 0.801 | 0.748 | +0.053 |

Structure beats pharmacology on 163 of 579 terms. The 10 largest structure advantages:

| Term | Drugs listing it | Structure | Pharmacology |
| --- | --- | --- | --- |
| Ileus paralytic | 4% | 0.837 | 0.667 |
| Urine abnormality | 5% | 0.635 | 0.501 |
| Urine analysis abnormal | 5% | 0.641 | 0.508 |
| Induration | 4% | 0.681 | 0.566 |
| Feeling hot | 8% | 0.647 | 0.533 |
| Flank pain | 4% | 0.640 | 0.527 |
| Prothrombin level increased | 4% | 0.737 | 0.625 |
| Respiratory arrest | 4% | 0.728 | 0.616 |
| Thrombocytosis | 6% | 0.781 | 0.669 |
| Erythema nodosum | 4% | 0.761 | 0.654 |

## Where pharmacology dominates

| Term | Drugs listing it | Structure | Pharmacology |
| --- | --- | --- | --- |
| Joint swelling | 5% | 0.513 | 0.682 |
| Sudden death | 4% | 0.638 | 0.802 |
| Conjunctival hyperaemia | 4% | 0.594 | 0.739 |
| Hypotonia | 5% | 0.688 | 0.825 |
| Hypothermia | 5% | 0.620 | 0.756 |
| Body temperature decreased | 5% | 0.620 | 0.756 |
| Stupor | 7% | 0.634 | 0.764 |
| Muscle relaxant therapy | 4% | 0.703 | 0.832 |
| Foetor hepaticus | 5% | 0.510 | 0.636 |
| Febrile neutropenia | 6% | 0.768 | 0.894 |
