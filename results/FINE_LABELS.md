# Phase 4: specific side effects

1422 drugs, 579 MedDRA preferred terms listed for at least 50 drugs (prevalence 3.5%-84%). Scaffold split, 5 folds, 3 seeds; ROC-AUC per term and fold, averaged.

## Overall

| Model | Mean ROC-AUC | PR-AUC lift |
| --- | --- | --- |
| prior | 0.500 ± 0.000 | +0.000 |
| documentation flags + logreg | 0.622 ± 0.002 | +0.085 |
| knn5 tanimoto | 0.613 ± 0.002 | +0.075 |
| structure rf | 0.665 ± 0.004 | +0.141 |
| pharma rf | 0.709 ± 0.003 | +0.197 |
| structure rf + pharma rf | 0.735 ± 0.003 | +0.223 |

## By how common the side effect is

| Share of drugs listing the term | Terms | structure rf | pharma rf | structure rf + pharma rf |
| --- | --- | --- | --- | --- |
| <10% | 325 | 0.669 | 0.712 | 0.739 |
| 10-25% | 163 | 0.665 | 0.708 | 0.734 |
| 25-50% | 73 | 0.652 | 0.696 | 0.721 |
| >50% | 18 | 0.642 | 0.704 | 0.722 |

## Is it still one factor?

PC1 of the 579-term matrix explains 21% of the variance (27 organ classes: 33%) and correlates with the number of listed terms at |r| = 0.99. The 'long package insert' factor is still there.

## Where structure is strongest

Terms with the highest structure-only ROC-AUC. Most look like class effects of drug families a fingerprint recognises (antipsychotics, anticholinergics, antibiotics, steroid hormones).

| Term | Drugs listing it | Structure | Pharmacology | Structure - pharma |
| --- | --- | --- | --- | --- |
| Pseudomembranous colitis | 5% | 0.907 | 0.920 | -0.013 |
| Neuroleptic malignant syndrome | 4% | 0.879 | 0.942 | -0.063 |
| Accommodation disorder | 5% | 0.864 | 0.800 | +0.065 |
| Nephropathy toxic | 5% | 0.855 | 0.821 | +0.035 |
| Drug withdrawal syndrome | 4% | 0.853 | 0.908 | -0.056 |
| Akathisia | 5% | 0.853 | 0.877 | -0.024 |
| Menstruation irregular | 6% | 0.843 | 0.769 | +0.074 |
| Ileus paralytic | 4% | 0.837 | 0.717 | +0.120 |
| Extrapyramidal disorder | 7% | 0.830 | 0.876 | -0.046 |
| Breast enlargement | 6% | 0.824 | 0.778 | +0.046 |
| Opportunistic infection | 4% | 0.814 | 0.854 | -0.040 |
| Mydriasis | 8% | 0.804 | 0.811 | -0.006 |
| Muscle rigidity | 4% | 0.804 | 0.875 | -0.071 |
| Tubulointerstitial nephritis | 8% | 0.803 | 0.775 | +0.028 |
| Hirsutism | 6% | 0.801 | 0.752 | +0.049 |

Structure beats pharmacology on 97 of 579 terms. The 10 largest structure advantages:

| Term | Drugs listing it | Structure | Pharmacology |
| --- | --- | --- | --- |
| Urine abnormality | 5% | 0.635 | 0.496 |
| Urine analysis abnormal | 5% | 0.641 | 0.517 |
| Ileus paralytic | 4% | 0.837 | 0.717 |
| Nephrotic syndrome | 4% | 0.759 | 0.660 |
| White blood cell count decreased | 5% | 0.722 | 0.637 |
| Induration | 4% | 0.681 | 0.597 |
| Erythema nodosum | 4% | 0.761 | 0.677 |
| Jaundice cholestatic | 9% | 0.761 | 0.681 |
| Acidosis | 5% | 0.677 | 0.598 |
| Feeling hot | 8% | 0.647 | 0.568 |

## Where pharmacology dominates

| Term | Drugs listing it | Structure | Pharmacology |
| --- | --- | --- | --- |
| Conjunctival hyperaemia | 4% | 0.594 | 0.791 |
| Joint swelling | 5% | 0.513 | 0.696 |
| Hypothermia | 5% | 0.620 | 0.795 |
| Body temperature decreased | 5% | 0.620 | 0.795 |
| Stupor | 7% | 0.634 | 0.807 |
| Encephalopathy | 6% | 0.571 | 0.721 |
| Sexual dysfunction | 5% | 0.694 | 0.843 |
| Skin irritation | 4% | 0.562 | 0.709 |
| Leukoderma | 4% | 0.680 | 0.824 |
| Glossitis | 10% | 0.592 | 0.736 |
