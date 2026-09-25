"""Phase 3: pharmacology features that structure alone cannot see.

Two sources, both downloaded on first use into .cache/ (not committed, see licences below):

* Indications - what each drug is prescribed for, from SIDER 4.1
  (`meddra_all_indications.tsv.gz`, mirrored in github.com/dhimmel/SIDER4; CC BY-NC-SA 4.0).
  Keyed by STITCH flat compound id, which `data/pubchem_fetch.csv` already carries for our drugs.
  Only rows detected as `NLP_indication` are used: `text_mention` rows can be side-effect
  mentions and would leak labels.
* ATC classes - WHO Anatomical Therapeutic Chemical codes (scrape of the WHO ATC index in
  github.com/fabkury/atcd), joined by drug name. Name matching finds only part of the drugs;
  unmatched drugs get all-zero ATC features plus a `has_atc` = 0 flag.

Caveat: both are known only for *marketed* drugs, so these features explain ADRs rather than
predict them for a brand-new molecule. Indications also come from the same package inserts
as the ADR labels.
"""
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
PUBCHEM = ROOT / "data" / "pubchem_fetch.csv"
INDICATIONS_URL = ("https://raw.githubusercontent.com/dhimmel/SIDER4/master/download/"
                   "meddra_all_indications.tsv.gz")
ATC_URL = "https://raw.githubusercontent.com/fabkury/atcd/master/WHO%20ATC-DDD%202026-04-25.csv"

MIN_DRUGS_PER_INDICATION = 5  # rarer indication terms are dropped (too few to learn from)

_SALT_WORDS = (r"\b(hydrochloride|dihydrochloride|hydrobromide|sodium|disodium|potassium|calcium|"
               r"magnesium|mesylate|mesilate|besylate|besilate|maleate|fumarate|tartrate|bitartrate|"
               r"citrate|succinate|sulfate|sulphate|phosphate|acetate|nitrate|bromide|chloride|"
               r"iodide|lactate|gluconate|hyclate|monohydrate|dihydrate|trihydrate|hemihydrate|"
               r"anhydrous|hcl|free base|base)\b")


def _download(url: str, name: str) -> Path:
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def _inchikey(smiles):
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return Chem.MolToInchiKey(m) if m is not None else None


def drug_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per SIDER row: STITCH flat id and PubChem title (NaN where unmatched).

    Joined on the InChIKey of the *raw* SIDER SMILES, which is how DeepChem built sider.csv.
    """
    pc = pd.read_csv(PUBCHEM)
    pc = pc.assign(ik=pc["InChIKey"]).drop_duplicates("ik")[["ik", "Flat_CID", "Title"]]
    out = pd.DataFrame({"ik": df["smiles_raw"].map(_inchikey)})
    return out.merge(pc, on="ik", how="left").drop(columns="ik").set_index(df.index)


def _norm_name(title):
    if not isinstance(title, str):
        return None
    t = re.sub(r"\(.*?\)", "", title.lower())
    t = re.sub(_SALT_WORDS, "", t)
    return re.sub(r"\s+", " ", t).strip(" ,-") or None


def indications(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Binary drug x indication-term matrix (MedDRA preferred terms) + a has_indication flag."""
    raw = pd.read_csv(_download(INDICATIONS_URL, "sider_meddra_all_indications.tsv.gz"),
                      sep="\t", header=None,
                      names=["cid", "umls", "method", "name", "type", "umls_meddra", "term"])
    raw = raw[(raw.method == "NLP_indication") & (raw.type == "PT")]
    cids = drug_table(df)["Flat_CID"]
    raw = raw[raw.cid.isin(set(cids.dropna()))]
    counts = raw.groupby("term")["cid"].nunique()
    terms = sorted(counts[counts >= MIN_DRUGS_PER_INDICATION].index)
    col = {t: j for j, t in enumerate(terms)}
    by_cid = raw[raw.term.isin(col)].groupby("cid")["term"].apply(set)
    X = np.zeros((len(df), len(terms) + 1), dtype=np.float32)
    for i, cid in enumerate(cids):
        for t in by_cid.get(cid, ()):
            X[i, col[t]] = 1
    X[:, -1] = X[:, :-1].any(axis=1)
    return X, [f"ind:{t}" for t in terms] + ["has_indication"]


def atc(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """One-hot ATC level 1 (14 anatomical groups) and level 2 (therapeutic subgroups) + has_atc."""
    who = pd.read_csv(_download(ATC_URL, "who_atc_ddd_2026-04-25.csv"))
    lvl5 = who[who.atc_code.str.len() == 7]
    by_name = lvl5.groupby(lvl5.atc_name.str.lower().str.strip())["atc_code"].apply(set)
    names = drug_table(df)["Title"].map(_norm_name)
    codes = names.map(lambda n: by_name.get(n, set()) if n else set())
    l1 = sorted(who.atc_code[who.atc_code.str.len() == 1])
    l2 = sorted(who.atc_code[who.atc_code.str.len() == 3])
    col = {c: j for j, c in enumerate(l1 + l2)}
    X = np.zeros((len(df), len(col) + 1), dtype=np.float32)
    for i, cs in enumerate(codes):
        for c in cs:
            X[i, col[c[0]]] = 1
            X[i, col[c[:3]]] = 1
        X[i, -1] = bool(cs)
    return X, [f"atc:{c}" for c in l1 + l2] + ["has_atc"]


def coverage(df: pd.DataFrame) -> dict:
    t = drug_table(df)
    ind, _ = indications(df)
    a, _ = atc(df)
    return {"drugs": len(df), "matched_cid": int(t.Flat_CID.notna().sum()),
            "with_indications": int(ind[:, -1].sum()), "with_atc": int(a[:, -1].sum()),
            "indication_terms": ind.shape[1] - 1}
