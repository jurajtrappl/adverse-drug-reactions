"""Molecule featurizers. Each is stateless: one row per molecule, no fitting."""
import urllib.request
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
MOL2VEC_URL = ("https://raw.githubusercontent.com/samoturk/mol2vec/master/"
               "examples/models/model_300dim.pkl")
MOL2VEC_PATH = ROOT / ".cache" / "mol2vec_model_300dim.pkl"

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


def mol2vec(smiles) -> np.ndarray:
    """Pretrained Mol2vec (Jaeger et al. 2018): word2vec trained on ~20M molecules, where each
    atom's Morgan environment (radius 0 and 1) is a 'word' and a molecule is a 'sentence'.
    The molecule vector is the sum of its word vectors (300 dims); unseen words map to 'UNK'.
    This is what the 2023 autoencoder was aiming for, pretrained properly.
    """
    from gensim.models import word2vec

    if not MOL2VEC_PATH.exists():
        MOL2VEC_PATH.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(MOL2VEC_URL, MOL2VEC_PATH)
    wv = word2vec.Word2Vec.load(str(MOL2VEC_PATH)).wv
    unk = wv["UNK"]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=1)
    out = np.zeros((len(smiles), wv.vector_size), dtype=np.float32)
    for i, m in enumerate(_mols(smiles)):
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        gen.GetSparseCountFingerprint(m, additionalOutput=ao)
        words = [str(ident) for ident, envs in ao.GetBitInfoMap().items() for _ in envs]
        out[i] = np.sum([wv[w] if w in wv.key_to_index else unk for w in words], axis=0)
    return out


def mol2vec_hit_rate(smiles) -> float:
    """Share of a molecule set's Morgan words that exist in the Mol2vec vocabulary."""
    from gensim.models import word2vec

    wv = word2vec.Word2Vec.load(str(MOL2VEC_PATH)).wv
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=1)
    hits = total = 0
    for m in _mols(smiles):
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        gen.GetSparseCountFingerprint(m, additionalOutput=ao)
        for ident, envs in ao.GetBitInfoMap().items():
            total += len(envs)
            hits += len(envs) * (str(ident) in wv.key_to_index)
    return hits / total


FEATURIZERS = {
    "size": size,
    "morgan": morgan,
    "desc": descriptors,
    "morgan+desc": lambda s: np.hstack([morgan(s).astype(np.float32), descriptors(s)]),
    "mol2vec": mol2vec,
}
