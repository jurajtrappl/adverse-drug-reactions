# Phase 3 analysis: understanding the problem

1427 drugs, 27 ADR categories. Pharmacology coverage: 1422 drugs matched to a PubChem/STITCH id, 997 have SIDER indications (262 terms used), 643 matched to a WHO ATC code by name.

## 1. One factor dominates the labels

A drug lists 16 of the 27 ADR categories on median (quartiles 11-20, range 1-26).

| Principal component | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| Share of label variance | 33% | 6% | 5% | 4% | 4% |

PC1 correlates with the drug's ADR count at |r| = 0.995. Knowing only how many *other* categories a drug lists predicts each category at mean ROC-AUC **0.837**, far above any structure model (~0.65). The count behaves like a 'how long is the package insert' factor.

## 2. ADR categories that move together

Average-linkage clustering on 1 - correlation, cut into 6 groups. Mean within-group correlation in brackets.

| Group | Categories (share of drugs listing it) | Within r |
| --- | --- | --- |
| 1 | Skin and subcutaneous tissue disorders (92%), Immune system disorders (72%) | 0.27 |
| 2 | Nervous system disorders (91%), Gastrointestinal disorders (91%), General disorders and administration site conditions (91%), Investigations (81%), Vascular disorders (78%), Respiratory, thoracic and mediastinal disorders (74%), Psychiatric disorders (71%), Infections and infestations (70%), Musculoskeletal and connective tissue disorders (70%), Metabolism and nutrition disorders (70%), Cardiac disorders (69%), Injury, poisoning and procedural complications (66%), Renal and urinary disorders (64%), Blood and lymphatic system disorders (62%), Eye disorders (61%), Hepatobiliary disorders (52%), Reproductive system and breast disorders (51%), Ear and labyrinth disorders (46%) | 0.35 |
| 3 | Neoplasms benign, malignant and unspecified (incl cysts and polyps) (26%), Endocrine disorders (23%), Congenital, familial and genetic disorders (18%) | 0.28 |
| 4 | Social circumstances (18%), Surgical and medical procedures (15%) | 0.32 |
| 5 | Pregnancy, puerperium and perinatal conditions (9%) | — |
| 6 | Product issues (2%) | — |

## 3. Predicting the ADR count

Random-forest regression of the number of listed ADR categories (0-27), scaffold split, 3 seeds.

| Features | Spearman ρ | R² |
| --- | --- | --- |
| morgan+desc | 0.340 | 0.127 |
| atc+ind | 0.493 | 0.257 |
| morgan+desc+atc+ind | 0.413 | 0.178 |

## 4. Near-identical molecules, different labels

101 pairs of *different* molecules have Morgan Tanimoto ≥ 0.8. Their ADR label sets overlap by Jaccard 0.69 on average (median 0.75); 20% of pairs share less than half their labels. Pairs with the least overlap:

| Drug A | ADRs | Drug B | ADRs | Tanimoto | Label Jaccard |
| --- | --- | --- | --- | --- | --- |
| 5,8,11,14,17-Eicosapentaenoic acid | 2 | Epanova | 7 | 0.83 | 0.00 |
| (no name) | 14 | (no name) | 1 | 0.80 | 0.07 |
| CID 6328526 | 2 | Edetic Acid | 12 | 0.93 | 0.08 |
| 4,6-Diamino-2-{[3-o-(2,6-diamino-2,6-did | 9 | Paromomycin I | 1 | 0.98 | 0.11 |
| 5-[2-[1-(5-cyclopropyl-5-hydroxypent-3-e | 6 | trans-Doxercalciferol | 16 | 0.80 | 0.29 |
| Beconase AQ | 12 | Betamethasone dipropionate | 9 | 0.80 | 0.31 |
| [(17R)-11-hydroxy-17-(2-hydroxyacetyl)-1 | 6 | [11-hydroxy-17-(2-hydroxyacetyl)-10,13-d | 6 | 0.92 | 0.33 |
| 33-(4-Amino-3,5-dihydroxy-6-methyloxan-2 | 21 | 33-(4-Amino-3,5-dihydroxy-6-methyloxan-2 | 7 | 0.87 | 0.33 |
| Prednefrin | 7 | Pred Forte | 20 | 1.00 | 0.35 |
| Pentetic acid | 7 | Edetic Acid | 12 | 0.93 | 0.36 |
| Teduglutide | 15 | Glucagon | 7 | 0.84 | 0.38 |
| Tetraethylenepentamine | 12 | Triethylenetetramine | 6 | 1.00 | 0.38 |

Names are PubChem titles. The mismatches are mostly different *products* of the same chemistry: a nasal spray vs a skin cream (Beconase AQ vs betamethasone dipropionate), a combination eye drop vs the plain steroid (Prednefrin vs Pred Forte), a supplement vs a prescription drug (eicosapentaenoic acid vs Epanova). Route, dose and label length differ, and the structure can't see any of that.

## 5. Documentation depth is a feature

Three numbers with no chemistry and no drug-specific meaning (does the drug have parsed indications, was it matched to an ATC code, how many indications) give mean ROC-AUC **0.646** (logistic regression, scaffold split, 3 seeds), as good as the best structure model. Part of what the pharmacology features add is this 'well-documented drug' signal, which is the same factor as the ADR count in section 1.

## 6. Learning curves

Random forest, scaffold split (seed 0), trained on a random share of each training fold (100% ≈ 1142 drugs).

| Training drugs | 228 | 456 | 685 | 913 | 1142 |
| --- | --- | --- | --- | --- | --- |
| morgan+desc | 0.591 | 0.620 | 0.624 | 0.635 | 0.643 |
| atc+ind | 0.666 | 0.683 | 0.696 | 0.702 | 0.708 |

Both curves still rise, slowly: structure gains +0.023 from 456 to 1142 drugs. More drugs would help a little; nothing suggests a data size at which structure alone would catch up with pharmacology.

