# Phase 4: specific side effects

1422 drugs, 579 MedDRA preferred terms listed for at least 50 drugs (prevalence 3.5%-84%). Scaffold split, 5 folds, 3 seeds; ROC-AUC per term and fold, averaged. Indication features that share a MedDRA term with any label are removed (see `pharma_without_label_terms`).

## Overall

| Model | Mean ROC-AUC | PR-AUC lift |
| --- | --- | --- |
| prior | 0.500 ± 0.000 | -0.000 |
| documentation flags + logreg | 0.625 ± 0.001 | +0.089 |
| knn5 tanimoto | 0.617 ± 0.002 | +0.080 |
| structure rf | 0.673 ± 0.004 | +0.146 |
| pharma rf | 0.693 ± 0.001 | +0.185 |
| structure rf + pharma rf | 0.733 ± 0.001 | +0.221 |

## By how common the side effect is

| Share of drugs listing the term | Terms | structure rf | pharma rf | structure rf + pharma rf |
| --- | --- | --- | --- | --- |
| <10% | 325 | 0.678 | 0.695 | 0.737 |
| 10-25% | 163 | 0.672 | 0.696 | 0.733 |
| 25-50% | 73 | 0.658 | 0.679 | 0.715 |
| >50% | 18 | 0.659 | 0.685 | 0.720 |

## Is it still one factor?

PC1 of the 579-term matrix explains 21% of the variance (27 organ classes: 33%) and correlates with the number of listed terms at |r| = 0.99. The 'long package insert' factor is still there.

## Where structure is strongest

Terms with the highest structure-only ROC-AUC. Most look like class effects of drug families a fingerprint recognises (antipsychotics, anticholinergics, antibiotics, steroid hormones).

| Term | Drugs listing it | Structure | Pharmacology | Structure - pharma |
| --- | --- | --- | --- | --- |
| Pseudomembranous colitis | 5% | 0.923 | 0.878 | +0.044 |
| Neuroleptic malignant syndrome | 4% | 0.891 | 0.912 | -0.021 |
| Drug withdrawal syndrome | 4% | 0.875 | 0.889 | -0.014 |
| Accommodation disorder | 5% | 0.868 | 0.808 | +0.060 |
| Akathisia | 5% | 0.861 | 0.845 | +0.017 |
| Ileus paralytic | 4% | 0.860 | 0.763 | +0.097 |
| Muscle rigidity | 4% | 0.847 | 0.875 | -0.027 |
| Extrapyramidal disorder | 7% | 0.845 | 0.861 | -0.015 |
| Mydriasis | 8% | 0.844 | 0.780 | +0.064 |
| Nephropathy toxic | 5% | 0.843 | 0.769 | +0.074 |
| Galactorrhoea | 6% | 0.837 | 0.810 | +0.027 |
| Skin striae | 4% | 0.835 | 0.809 | +0.027 |
| Menstruation irregular | 6% | 0.833 | 0.747 | +0.086 |
| Breast enlargement | 6% | 0.826 | 0.780 | +0.045 |
| Hirsutism | 6% | 0.825 | 0.739 | +0.086 |

Structure beats pharmacology on 168 of 579 terms. The 10 largest structure advantages:

| Term | Drugs listing it | Structure | Pharmacology |
| --- | --- | --- | --- |
| Urine abnormality | 5% | 0.663 | 0.481 |
| Urine analysis abnormal | 5% | 0.651 | 0.504 |
| Haematocrit decreased | 4% | 0.625 | 0.496 |
| Prothrombin level increased | 4% | 0.748 | 0.620 |
| Thrombocytosis | 6% | 0.793 | 0.667 |
| Transient ischaemic attack | 5% | 0.769 | 0.646 |
| Torsade de pointes | 6% | 0.805 | 0.697 |
| Respiratory arrest | 4% | 0.737 | 0.631 |
| Flank pain | 4% | 0.599 | 0.500 |
| Ileus paralytic | 4% | 0.860 | 0.763 |

## Where pharmacology dominates

| Term | Drugs listing it | Structure | Pharmacology |
| --- | --- | --- | --- |
| Joint swelling | 5% | 0.540 | 0.694 |
| Febrile neutropenia | 6% | 0.762 | 0.906 |
| Sudden death | 4% | 0.677 | 0.820 |
| Hypothyroidism | 6% | 0.614 | 0.755 |
| Muscle relaxant therapy | 4% | 0.706 | 0.839 |
| Hyperthyroidism | 4% | 0.599 | 0.730 |
| Sexual dysfunction | 5% | 0.674 | 0.805 |
| Conjunctival hyperaemia | 4% | 0.570 | 0.700 |
| Lethargy | 16% | 0.620 | 0.747 |
| Hyporeflexia | 4% | 0.638 | 0.763 |
