"""Load SIDER, strip counter-ions, attach Murcko scaffolds and duplicate groups."""
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "sider.csv"
CLEAN_PATH = ROOT / "data" / "sider_clean.csv"

_uncharger = rdMolStandardize.Uncharger()

# Fragments this big are treated as active ingredients, smaller ones as counter-ions/solvent.
# Keeps combination drugs (e.g. etonogestrel + ethinylestradiol) intact while dropping
# Na+, Cl-, water, acetate, mesylate, citrate (13), meglumine (14) ...
# Known casualty: small co-drugs such as levodopa (14 heavy atoms) in carbidopa/levodopa.
MIN_ACTIVE_HEAVY_ATOMS = 15


def clean_smiles(smiles: str) -> str:
    """Strip counter-ions and neutralise charges; keep stereo. Returns canonical SMILES.

    Rule: always keep the largest fragment, plus any other fragment with
    >= MIN_ACTIVE_HEAVY_ATOMS heavy atoms; identical fragments (2:1 salts) are merged.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles}")
    frags = sorted(Chem.GetMolFrags(mol, asMols=True), key=lambda m: -m.GetNumHeavyAtoms())
    keep = [frags[0]] + [f for f in frags[1:] if f.GetNumHeavyAtoms() >= MIN_ACTIVE_HEAVY_ATOMS]
    parts = sorted({Chem.MolToSmiles(_uncharger.uncharge(f)) for f in keep})
    return Chem.MolToSmiles(Chem.MolFromSmiles(".".join(parts)))


def scaffold_of(smiles: str) -> str:
    """Bemis-Murcko scaffold; '' for acyclic molecules."""
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)


def build_clean(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    labels = list(raw.columns[1:])
    df = pd.DataFrame({"smiles_raw": raw["smiles"]})
    df["smiles"] = df["smiles_raw"].map(clean_smiles)
    df["scaffold"] = df["smiles"].map(scaffold_of)
    # Same parent molecule sold as different salts -> one group, never split apart.
    df["group"] = df.groupby("smiles", sort=False).ngroup()
    return pd.concat([df, raw[labels]], axis=1)


def load(path: Path = CLEAN_PATH) -> tuple[pd.DataFrame, list[str]]:
    """Return the cleaned frame and the 27 label column names."""
    if not path.exists():
        build_clean().to_csv(path, index=False)
    df = pd.read_csv(path, keep_default_na=False)  # '' scaffold must stay ''
    labels = [c for c in df.columns if c not in ("smiles_raw", "smiles", "scaffold", "group")]
    return df, labels
