"""Phase 2 neural models: a multi-task MLP and a Chemprop D-MPNN.

Both follow the same interface as adr.models: fit(X, Y) with Y of shape (n, 27),
predict_proba(X) -> (n, 27). Both learn all 27 ADRs at once through shared layers,
so the "how many ADRs does this drug have" factor is learned once, not 27 times.
Early stopping uses a random 10% of the *training* fold; the test fold is never touched.
"""
import copy

import lightning.pytorch as pl
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn


def _inner_split(n: int, seed: int, frac: float = 0.1):
    idx = np.random.RandomState(seed).permutation(n)
    n_val = max(1, int(round(frac * n)))
    return idx[n_val:], idx[:n_val]


def _pos_weight(Y: np.ndarray) -> np.ndarray:
    """neg/pos per ADR: up-weights rare positives, down-weights 90%-positive ADRs."""
    pos = Y.sum(0)
    return np.clip((len(Y) - pos) / np.maximum(pos, 1), 0.05, 50.0)


def mean_auc(Y, P) -> float:
    """Mean per-ADR ROC-AUC (ADRs with one class in Y are skipped)."""
    aucs = [roc_auc_score(Y[:, j], P[:, j]) for j in range(Y.shape[1]) if Y[:, j].min() != Y[:, j].max()]
    return float(np.mean(aucs)) if aucs else float("nan")


class MultiTaskMLP:
    """Shared hidden layers -> 27 sigmoid outputs, class-weighted BCE.

    Early stopping on mean per-ADR validation ROC-AUC (patience 15 epochs).
    """

    def __init__(self, hidden=(512, 256), dropout=0.3, lr=1e-3, weight_decay=1e-5,
                 batch_size=64, max_epochs=200, patience=15, seed=0, n_jobs=-1):
        self.hidden, self.dropout, self.lr, self.wd = hidden, dropout, lr, weight_decay
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.seed = seed
        if n_jobs and n_jobs > 0:
            torch.set_num_threads(n_jobs)

    def _net(self, d_in, d_out):
        layers, d = [], d_in
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.dropout)]
            d = h
        layers.append(nn.Linear(d, d_out))
        return nn.Sequential(*layers)

    def _scale(self, X):
        return np.clip((X - self.mu_) / self.sd_, -6, 6).astype(np.float32)

    def fit(self, X, Y):
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float64)
        tr, va = _inner_split(len(X), self.seed)
        self.mu_ = X[tr].mean(0)
        self.sd_ = X[tr].std(0) + 1e-6
        Xt, Yt = torch.from_numpy(self._scale(X[tr])), torch.from_numpy(Y[tr].astype(np.float32))
        Xv = torch.from_numpy(self._scale(X[va]))
        self.net_ = self._net(X.shape[1], Y.shape[1])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(_pos_weight(Y[tr]), dtype=torch.float32))
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.wd)
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.net_.train()
            for b in torch.randperm(len(Xt), generator=gen).split(self.batch_size):
                opt.zero_grad()
                loss_fn(self.net_(Xt[b]), Yt[b]).backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                v = -mean_auc(Y[va], torch.sigmoid(self.net_(Xv)).numpy())
            if np.isnan(v):  # no ADR has both classes in the validation split
                continue
            if v < best - 1e-4:
                best, best_state, bad = v, copy.deepcopy(self.net_.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:  # None only if validation AUC was never defined
            self.net_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.from_numpy(self._scale(np.asarray(X, dtype=np.float64))))
        return torch.sigmoid(logits).numpy()


class _BestByMeanAUC(pl.Callback):
    """Keep the weights of the epoch with the best mean per-ADR validation ROC-AUC.

    Validation *loss* is a poor stopping signal here: it bottoms out after ~5 epochs
    (the model grows over-confident) while the ranking quality keeps improving.
    """

    def __init__(self, val_loader, Y_val):
        self.val_loader, self.Y_val = val_loader, Y_val
        self.score, self.state, self.epoch = -np.inf, None, -1

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        with torch.no_grad():
            P = torch.cat([pl_module(b.bmg, b.V_d, b.X_d) for b in self.val_loader])
        score = mean_auc(self.Y_val, P.reshape(len(self.Y_val), -1).numpy())
        if not np.isnan(score) and score > self.score:
            self.score, self.epoch = score, trainer.current_epoch
            self.state = copy.deepcopy(pl_module.state_dict())


class Chemprop:
    """Chemprop v2 D-MPNN: learns features from the molecular graph (atoms = nodes, bonds = edges).

    X is an array of SMILES strings, or rows of [smiles, descriptor vector] to feed extra
    RDKit descriptors into the output network. Defaults follow Chemprop's CLI (hidden 300,
    depth 3, mean aggregation, batch norm, 50 epochs); keeps the epoch with the best mean
    per-ADR validation ROC-AUC, as Chemprop v1 did for classification.
    """

    def __init__(self, max_epochs=50, batch_size=50, seed=0, n_jobs=-1):
        self.max_epochs, self.batch_size, self.seed = max_epochs, batch_size, seed
        if n_jobs and n_jobs > 0:
            torch.set_num_threads(n_jobs)

    @staticmethod
    def _unpack(X):
        """X is either SMILES (1-D) or an object array [smiles, descriptor vector] per row."""
        X = np.asarray(X, dtype=object)
        if X.ndim == 1:
            return X, None
        return X[:, 0], np.stack(X[:, 1]).astype(np.float64)

    def _dataset(self, X, Y=None):
        from chemprop import data, featurizers
        smiles, D = self._unpack(X)
        if D is not None:
            D = np.clip((D - self.mu_) / self.sd_, -6, 6).astype(np.float32)
        dps = []
        for i, s in enumerate(smiles):
            kw = {"x_d": D[i]} if D is not None else {}
            y = Y[i].astype(float) if Y is not None else None
            dps.append(data.MoleculeDatapoint.from_smi(s, y, **kw))
        return data.MoleculeDataset(dps, featurizers.SimpleMoleculeMolGraphFeaturizer())

    def fit(self, X, Y):
        from chemprop import data, models
        from chemprop import nn as cnn

        pl.seed_everything(self.seed, workers=True, verbose=False)
        tr, va = _inner_split(len(X), self.seed)
        _, D = self._unpack(X)
        n_desc = 0
        if D is not None:  # extra molecule-level descriptors, standardised on the train part
            self.mu_, self.sd_, n_desc = D[tr].mean(0), D[tr].std(0) + 1e-6, D.shape[1]
        train_loader = data.build_dataloader(self._dataset(X[tr], Y[tr]), batch_size=self.batch_size,
                                             num_workers=0, seed=self.seed)
        val_loader = data.build_dataloader(self._dataset(X[va], Y[va]), batch_size=self.batch_size,
                                           num_workers=0, shuffle=False)
        mp = cnn.BondMessagePassing()
        ffn = cnn.BinaryClassificationFFN(n_tasks=Y.shape[1], input_dim=mp.output_dim + n_desc)
        self.model_ = models.MPNN(mp, cnn.MeanAggregation(), ffn, batch_norm=True)
        best = _BestByMeanAUC(val_loader, Y[va])
        trainer = pl.Trainer(max_epochs=self.max_epochs, accelerator="cpu", devices=1, logger=False,
                             enable_checkpointing=False, enable_progress_bar=False,
                             enable_model_summary=False, callbacks=[best])
        trainer.fit(self.model_, train_loader, val_loader)
        if best.state is not None:
            self.model_.load_state_dict(best.state)
        self.best_epoch_, self.best_val_auc_ = best.epoch, best.score
        self._trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False,
                                   enable_progress_bar=False, enable_model_summary=False)
        return self

    def predict_proba(self, X):
        from chemprop import data
        loader = data.build_dataloader(self._dataset(X), batch_size=self.batch_size,
                                       num_workers=0, shuffle=False)
        preds = self._trainer.predict(self.model_, loader)
        return torch.cat(preds).numpy().reshape(len(np.asarray(X, dtype=object)), -1)
