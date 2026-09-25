"""Phase 4: specific side effects instead of 27 organ-class categories.

SIDER 4.1 (`meddra_all_se.tsv.gz`, dhimmel/SIDER4 mirror, CC BY-NC-SA 4.0) lists every side
effect found on each drug's package insert as MedDRA preferred terms (PT), e.g. "Nausea",
"Hypoglycaemia", "Electrocardiogram QT prolonged". The DeepChem SIDER file we used so far
rolls these ~4,250 terms up into 27 system organ classes; here we keep the terms.

Rows are the SIDER drugs that map to a STITCH id (1,422 of 1,427); labels are the terms
listed for at least MIN_DRUGS of them.
"""
import numpy as np
import pandas as pd

from adr import pharma

SE_URL = "https://raw.githubusercontent.com/dhimmel/SIDER4/master/download/meddra_all_se.tsv.gz"
MIN_DRUGS = 50  # ~3.5% of drugs: enough positives in every scaffold test fold


def load(df: pd.DataFrame, min_drugs: int = MIN_DRUGS):
    """Returns (keep, Y, terms): row mask over df, binary matrix for kept rows, term names."""
    se = pd.read_csv(pharma._download(SE_URL, "sider_meddra_all_se.tsv.gz"), sep="\t", header=None,
                     names=["cid", "cid_stereo", "umls", "type", "umls_meddra", "term"])
    se = se[se.type == "PT"]
    cids = pharma.drug_table(df)["Flat_CID"]
    keep = cids.notna().to_numpy()
    se = se[se.cid.isin(set(cids[keep]))]
    counts = se.groupby("term")["cid"].nunique()
    terms = sorted(counts[counts >= min_drugs].index)
    col = {t: j for j, t in enumerate(terms)}
    by_cid = se[se.term.isin(col)].groupby("cid")["term"].apply(set)
    kept_cids = cids[keep].to_numpy()
    Y = np.zeros((keep.sum(), len(terms)), dtype=np.int8)
    for i, cid in enumerate(kept_cids):
        for t in by_cid.get(cid, ()):
            Y[i, col[t]] = 1
    return keep, Y, terms


def restrict_folds(folds, keep):
    """Map folds over all df rows to folds over the kept rows only."""
    new_index = np.cumsum(keep) - 1
    return [new_index[f[keep[f]]] for f in folds]
