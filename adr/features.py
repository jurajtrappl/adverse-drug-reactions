"""Molecule featurizers. Each is stateless: one row per molecule, no fitting."""
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _mols(smiles):
    return [Chem.MolFromSmiles(s) for s in smiles]


def morgan_bitvects(smiles):
    """ECFP4-like bit vectors (RDKit objects), used for Tanimoto similarity."""
    return [_morgan.GetFingerprint(m) for m in _mols(smiles)]


def morgan(smiles) -> np.ndarray:
    """2048-bit Morgan fingerprint, radius 2: bit = 'this substructure is present'."""
    out = np.zeros((len(smiles), 2048), dtype=np.uint8)
    for i, fp in enumerate(morgan_bitvects(smiles)):
        DataStructs.ConvertToNumpyArray(fp, out[i])
    return out


def descriptors(smiles) -> np.ndarray:
    """~210 RDKit 2D descriptors (logP, TPSA, MW, ring counts, ...)."""
    rows = [[fn(m) for _, fn in Descriptors.descList] for m in _mols(smiles)]
    x = np.asarray(rows, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -1e6, 1e6).astype(np.float32)


def size(smiles) -> np.ndarray:
    """Two numbers: heavy-atom count and SMILES length. The 'dumb size' baseline."""
    return np.array([[m.GetNumHeavyAtoms(), len(s)] for m, s in zip(_mols(smiles), smiles)],
                    dtype=np.float32)


FEATURIZERS = {
    "size": size,
    "morgan": morgan,
    "desc": descriptors,
    "morgan+desc": lambda s: np.hstack([morgan(s).astype(np.float32), descriptors(s)]),
}
