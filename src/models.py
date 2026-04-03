"""
Model definitions: ML baselines (RF, SVM, PLS-DA) and DL models (1D-CNN, 1D-ResNet).
All models expose sklearn-compatible fit(X, y) / predict(X) interface.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupShuffleSplit

from src.config import (SEED, DL_LR, DL_EPOCHS, DL_PATIENCE, DL_BATCH, DL_WEIGHT_DECAY,
                        DL_VAL_FRAC, TASKS, KAN_GRID_SIZE, KAN_SPLINE_ORDER,
                        AUG_ENABLED, AUG_N)


# ====================== Traditional ML ======================

def get_rf():
    return RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        random_state=SEED, n_jobs=-1,
    )


def get_svm():
    return SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED)


class PLSDA:
    """PLS-DA with sklearn-like interface."""
    def __init__(self, n_components=10):
        self.pls = PLSRegression(n_components=n_components)
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        Y = pd.get_dummies(y).values.astype(float)
        self.pls.fit(X, Y)
        return self

    def predict(self, X):
        P = self.pls.predict(X)
        return self.classes_[np.argmax(P, axis=1)]


def get_plsda():
    return PLSDA(n_components=10)


# ====================== 1D-CNN (PyTorch) ======================

class _CNN1DNet(nn.Module):
    """4-block Conv1D + BN + ReLU + MaxPool → GAP → Dropout → Dense"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


class _ResBlock1D(nn.Module):
    """Residual block for 1D convolutions."""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class _ResNet1DNet(nn.Module):
    """1D-ResNet: stem → 3 stages (with stride-2 downsampling) → GAP → Dense"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.stage1 = nn.Sequential(
            _ResBlock1D(64), _ResBlock1D(64),
        )
        self.down1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, stride=2),
            nn.BatchNorm1d(128),
        )
        self.stage2 = nn.Sequential(
            _ResBlock1D(128), _ResBlock1D(128),
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, stride=2),
            nn.BatchNorm1d(256),
        )
        self.stage3 = nn.Sequential(
            _ResBlock1D(256), _ResBlock1D(256),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


class _DLWrapper:
    """Sklearn-compatible wrapper for PyTorch classification models."""
    def __init__(self, net_cls, lr=DL_LR, epochs=DL_EPOCHS, patience=DL_PATIENCE,
                 batch_size=DL_BATCH, seed=SEED):
        self.net_cls = net_cls
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_map = None
        self.inv_map = None

    def fit(self, X, y, groups=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = True

        self.label_map = {lab: i for i, lab in enumerate(sorted(set(y)))}
        self.inv_map = {i: lab for lab, i in self.label_map.items()}
        num_classes = len(self.label_map)
        yi = np.array([self.label_map[v] for v in y])

        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        yt = torch.tensor(yi, dtype=torch.long)

        # Internal early-stopping split (group-aware when groups provided)
        n = len(Xt)
        if groups is not None:
            gss = GroupShuffleSplit(n_splits=1, test_size=DL_VAL_FRAC, random_state=self.seed)
            t_idx, v_idx = next(gss.split(X, yi, groups))
        else:
            idx = np.random.permutation(n)
            n_val = max(int(n * DL_VAL_FRAC), 1)
            t_idx, v_idx = idx[n_val:], idx[:n_val]

        # Data augmentation (training set only)
        if AUG_ENABLED:
            from src.dataset import augment_spectra
            X_tr_np, y_tr_np = X[t_idx], yi[t_idx]
            X_tr_np, y_tr_np = augment_spectra(X_tr_np, y_tr_np, n_aug=AUG_N)
            Xt_train = torch.tensor(X_tr_np, dtype=torch.float32).unsqueeze(1)
            yt_train = torch.tensor(y_tr_np, dtype=torch.long)
        else:
            Xt_train, yt_train = Xt[t_idx], yt[t_idx]

        dl_t = DataLoader(TensorDataset(Xt_train, yt_train),
                          batch_size=self.batch_size, shuffle=True)
        dl_v = DataLoader(TensorDataset(Xt[v_idx], yt[v_idx]),
                          batch_size=256, shuffle=False)

        self.model = self.net_cls(X.shape[1], num_classes).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=DL_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        crit = nn.CrossEntropyLoss()

        best_loss = float('inf')
        wait = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in dl_t:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = crit(self.model(xb), yb)
                loss.backward()
                opt.step()

            self.model.eval()
            vloss = 0
            vn = 0
            with torch.no_grad():
                for xb, yb in dl_v:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    vloss += crit(self.model(xb), yb).item() * len(yb)
                    vn += len(yb)
            vloss /= max(vn, 1)
            scheduler.step(vloss)

            if vloss < best_loss:
                best_loss = vloss
                wait = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            preds = self.model(Xt).argmax(1).cpu().numpy()
        return np.array([self.inv_map[p] for p in preds])


def get_cnn1d():
    return _DLWrapper(_CNN1DNet)


def get_resnet1d():
    return _DLWrapper(_ResNet1DNet)


# ====================== KAN Layer ======================

class _KANLinear(nn.Module):
    """Kolmogorov-Arnold Network linear layer with B-spline basis functions.

    Instead of fixed activation + learned weight (MLP), KAN learns the
    activation function on each edge using B-spline parameterisation.
    output_j = sum_i [ phi_{ij}(x_i) ] where phi is a B-spline.
    A residual SiLU base path is added for stable training.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = 2.0 / grid_size
        grid = torch.linspace(-1 - h * spline_order, 1 + h * spline_order,
                              grid_size + 2 * spline_order + 1)
        self.register_buffer('grid', grid)

        n_bases = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, n_bases) * 0.1
        )
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(
                -1.0 / in_features ** 0.5, 1.0 / in_features ** 0.5
            )
        )

    def forward(self, x):
        bases = self._compute_bases(x)
        spline_out = torch.einsum('bin,oin->bo', bases, self.spline_weight)
        base_out = nn.functional.silu(x) @ self.base_weight.t()
        return spline_out + base_out

    def _compute_bases(self, x):
        """Cox-de Boor recursive B-spline basis computation."""
        x = torch.tanh(x).unsqueeze(-1)          # (batch, in, 1)
        grid = self.grid                           # (G,)
        # Order-0 indicator bases
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)  # (batch, in, G-1)
        # Recursive higher-order
        for k in range(1, self.spline_order + 1):
            n = bases.shape[-1]
            t_left   = grid[:n - 1]
            t_left_k = grid[k:k + n - 1]
            left  = (x - t_left) / (t_left_k - t_left + 1e-8)
            t_right_k1 = grid[k + 1:k + 1 + n - 1]
            t_right    = grid[1:1 + n - 1]
            right = (t_right_k1 - x) / (t_right_k1 - t_right + 1e-8)
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases  # (batch, in, grid_size + spline_order)


# ====================== Multi-Task Networks ======================

class _MultiTaskCNN1DNet(nn.Module):
    """Shared CNN backbone + task-specific Linear heads."""
    def __init__(self, input_dim, task_num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.heads = nn.ModuleDict({
            tid: nn.Linear(256, nc) for tid, nc in task_num_classes
        })

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.gap(feat).squeeze(-1)
        feat = self.dropout(feat)
        return {tid: head(feat) for tid, head in self.heads.items()}


class _MultiTaskResNet1DNet(nn.Module):
    """Shared ResNet backbone + task-specific Linear heads."""
    def __init__(self, input_dim, task_num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.stage1 = nn.Sequential(_ResBlock1D(64), _ResBlock1D(64))
        self.down1 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, stride=2), nn.BatchNorm1d(128))
        self.stage2 = nn.Sequential(_ResBlock1D(128), _ResBlock1D(128))
        self.down2 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, stride=2), nn.BatchNorm1d(256))
        self.stage3 = nn.Sequential(_ResBlock1D(256), _ResBlock1D(256))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.heads = nn.ModuleDict({
            tid: nn.Linear(256, nc) for tid, nc in task_num_classes
        })

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        feat = self.gap(x).squeeze(-1)
        feat = self.dropout(feat)
        return {tid: head(feat) for tid, head in self.heads.items()}


class _MultiTaskKANCNN1DNet(nn.Module):
    """Shared CNN backbone + task-specific KAN heads (innovation point).

    Replaces the standard Linear classifier head with a KAN layer that uses
    learnable B-spline activation functions, enabling the model to capture
    non-linear task-specific mappings from shared representations.
    """
    def __init__(self, input_dim, task_num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.heads = nn.ModuleDict({
            tid: _KANLinear(256, nc, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
            for tid, nc in task_num_classes
        })

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.gap(feat).squeeze(-1)
        feat = self.dropout(feat)
        return {tid: head(feat) for tid, head in self.heads.items()}


# ====================== Multi-Task Wrapper ======================

class _MultiTaskDLWrapper:
    """Sklearn-compatible wrapper for Multi-Task PyTorch models.

    Unlike _DLWrapper (one task per fit), this trains all 7 tasks simultaneously
    by optimizing the mean of per-task cross-entropy losses.  One fit = one
    forward/backward pass covering all tasks, so 5-fold CV needs only 5 fits
    instead of 7×5 = 35.
    """
    is_multitask = True

    def __init__(self, net_cls, tasks, lr=DL_LR, epochs=DL_EPOCHS,
                 patience=DL_PATIENCE, batch_size=DL_BATCH, seed=SEED):
        self.net_cls = net_cls
        self.tasks = tasks
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_maps = {}
        self.inv_maps = {}

    def fit(self, X, y_dict, groups=None):
        """
        X: (n_samples, n_features) numpy array.
        y_dict: {task_id: y_array} with original labels for every task.
        groups: optional group labels for group-aware internal val split.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = True

        # Encode labels per task
        task_num_classes = []
        encoded_y = {}
        for t in self.tasks:
            tid = t['id']
            labels = sorted(set(y_dict[tid]))
            self.label_maps[tid] = {lab: i for i, lab in enumerate(labels)}
            self.inv_maps[tid] = {i: lab for lab, i in self.label_maps[tid].items()}
            task_num_classes.append((tid, len(labels)))
            encoded_y[tid] = torch.tensor(
                [self.label_maps[tid][v] for v in y_dict[tid]], dtype=torch.long
            )

        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

        # Group-aware internal val split
        n = len(Xt)
        if groups is not None:
            gss = GroupShuffleSplit(n_splits=1, test_size=DL_VAL_FRAC,
                                   random_state=self.seed)
            t_idx, v_idx = next(gss.split(
                X, encoded_y[self.tasks[0]['id']].numpy(), groups
            ))
        else:
            idx = np.random.permutation(n)
            n_val = max(int(n * DL_VAL_FRAC), 1)
            t_idx, v_idx = idx[n_val:], idx[:n_val]

        # DataLoader: X + one label tensor per task
        # Data augmentation (training set only)
        if AUG_ENABLED:
            from src.dataset import augment_spectra_mt
            X_tr_np = X[t_idx]
            y_dict_tr = {t['id']: encoded_y[t['id']][t_idx].numpy() for t in self.tasks}
            X_tr_np, y_dict_tr = augment_spectra_mt(X_tr_np, y_dict_tr, n_aug=AUG_N)
            Xt_train = torch.tensor(X_tr_np, dtype=torch.float32).unsqueeze(1)
            train_tensors = [Xt_train] + [
                torch.tensor(y_dict_tr[t['id']], dtype=torch.long) for t in self.tasks
            ]
        else:
            train_tensors = [Xt[t_idx]] + [encoded_y[t['id']][t_idx] for t in self.tasks]
        val_tensors   = [Xt[v_idx]] + [encoded_y[t['id']][v_idx] for t in self.tasks]
        dl_t = DataLoader(TensorDataset(*train_tensors),
                          batch_size=self.batch_size, shuffle=True)
        dl_v = DataLoader(TensorDataset(*val_tensors),
                          batch_size=256, shuffle=False)

        self.model = self.net_cls(X.shape[1], task_num_classes).to(self.device)
        n_tasks = len(self.tasks)

        # Uncertainty weighting (Kendall et al. CVPR 2018):
        # learnable log-variance per task — harder tasks get lower precision ⇒ higher weight
        log_vars = nn.Parameter(torch.zeros(n_tasks, device=self.device))

        opt = torch.optim.Adam(
            list(self.model.parameters()) + [log_vars],
            lr=self.lr, weight_decay=DL_WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        crit = nn.CrossEntropyLoss()

        best_loss = float('inf')
        wait = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            for batch in dl_t:
                xb = batch[0].to(self.device)
                opt.zero_grad()
                logits = self.model(xb)
                # Uncertainty-weighted multi-task loss
                loss = sum(
                    torch.exp(-log_vars[i]) * crit(logits[self.tasks[i]['id']],
                                                    batch[i + 1].to(self.device))
                    + 0.5 * log_vars[i]
                    for i in range(n_tasks)
                )
                loss.backward()
                opt.step()

            self.model.eval()
            vloss = 0.0
            vn = 0
            with torch.no_grad():
                for batch in dl_v:
                    xb = batch[0].to(self.device)
                    logits = self.model(xb)
                    bloss = sum(
                        torch.exp(-log_vars[i]) * crit(logits[self.tasks[i]['id']],
                                                        batch[i + 1].to(self.device))
                        + 0.5 * log_vars[i]
                        for i in range(n_tasks)
                    )
                    vloss += bloss.item() * len(xb)
                    vn += len(xb)
            vloss /= max(vn, 1)
            scheduler.step(vloss)

            if vloss < best_loss:
                best_loss = vloss
                wait = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        """Returns {task_id: pred_array} with original labels."""
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        preds = {}
        with torch.no_grad():
            logits = self.model(Xt)
            for tid, lg in logits.items():
                idx = lg.argmax(1).cpu().numpy()
                preds[tid] = np.array([self.inv_maps[tid][i] for i in idx])
        return preds


def get_mt_cnn():
    return _MultiTaskDLWrapper(_MultiTaskCNN1DNet, TASKS)


def get_mt_resnet():
    return _MultiTaskDLWrapper(_MultiTaskResNet1DNet, TASKS)


def get_mt_kan_cnn():
    return _MultiTaskDLWrapper(_MultiTaskKANCNN1DNet, TASKS)


# ====================== Model Registry ======================

MODEL_REGISTRY = {
    'RF': get_rf,
    'SVM': get_svm,
    'PLS-DA': get_plsda,
    '1D-CNN': get_cnn1d,
    '1D-ResNet': get_resnet1d,
    'MT-CNN': get_mt_cnn,
    'MT-ResNet': get_mt_resnet,
    'MT-KAN-CNN': get_mt_kan_cnn,
}
