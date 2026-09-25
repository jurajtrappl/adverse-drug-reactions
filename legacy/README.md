# Legacy: the 2023 autoencoder pipeline

Kept for reference. This is the original NAIL107 term project code (Krumm & Trappl, 2023):
a BiLSTM autoencoder trained on ~440k PubChem SMILES, whose encoder output was used as a
"word embedding" for SIDER classifiers.

Don't use its reported numbers. The audit in 2026 found that:

- minority-class rows were duplicated before the train/test split, so ~39% of test rows
  were copies of training rows (the 98-100% accuracies are memorisation);
- the SIDER SMILES were tokenized with a tokenizer re-fitted on SIDER, so token IDs didn't
  match the ones the encoder was trained on;
- the autoencoder regressed token IDs with MSE, so the embedding mostly encodes molecule
  size (it scores like a two-number size baseline, ROC-AUC ~0.53 on a scaffold split).

Contents:

| Path | What it is |
| --- | --- |
| `sider.ipynb` | Main notebook (paths adjusted to run from this folder) |
| `we_network.py` | Autoencoder training script (does not run as committed: callback typo, shape mismatch) |
| `models/` | SMILES-PE vocabulary and three autoencoder checkpoints |
| `data/smiles_embedding.csv` | ~444k PubChem SMILES used to train the autoencoder |
| `heatmap_tanimoto.png` | Pairwise Tanimoto similarity of SIDER drugs (generating code was not committed) |

The maintained pipeline lives in `adr/` and `scripts/` at the repository root.
