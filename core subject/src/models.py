"""
Model definitions for the SERS project.
Round 4 adds focal loss, inverse-frequency class weighting, and 3-class MT models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC

from src.config import (
    AUG_ENABLED,
    AUG_N,
    DL_BATCH,
    DL_EPOCHS,
    DL_LR,
    DL_PATIENCE,
    DL_SCHEDULER,
    DL_VAL_FRAC,
    DL_WEIGHT_DECAY,
    FOCAL_LOSS_GAMMA,
    KAN_GRID_SIZE,
    KAN_SPLINE_ORDER,
    LABEL_SMOOTHING,
    MIXUP_ALPHA,
    MIXUP_ENABLED,
    SEED,
    TASKS,
    TASKS_MT_FULL,
    USE_CLASS_WEIGHT,
)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        target_matrix = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1.0)

        if self.label_smoothing > 0:
            smooth = self.label_smoothing / num_classes
            target_matrix = target_matrix * (1.0 - self.label_smoothing) + smooth

        probs = torch.exp(log_probs)
        focal_weight = (1.0 - probs) ** self.gamma
        loss = -target_matrix * focal_weight * log_probs

        if self.weight is not None:
            loss = loss * self.weight.to(logits.device).unsqueeze(0)

        return loss.sum(dim=-1).mean()


def _compute_class_weights(y_encoded):
    classes, counts = np.unique(y_encoded, return_counts=True)
    weights = np.ones(int(classes.max()) + 1, dtype=np.float32)
    weights[classes] = len(y_encoded) / (len(classes) * counts)
    weights = weights / weights.max()
    return torch.tensor(weights, dtype=torch.float32)


def _make_loss(y_encoded, num_classes):
    weight = None
    if USE_CLASS_WEIGHT:
        weight = _compute_class_weights(y_encoded)
        if len(weight) < num_classes:
            padded = torch.ones(num_classes, dtype=torch.float32)
            padded[: len(weight)] = weight
            weight = padded
    return FocalLoss(
        gamma=FOCAL_LOSS_GAMMA,
        weight=weight,
        label_smoothing=LABEL_SMOOTHING,
    )


def get_rf():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1,
    )


def get_svm():
    return SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        class_weight='balanced',
        random_state=SEED,
    )


class PLSDA:
    def __init__(self, n_components=10):
        self.pls = PLSRegression(n_components=n_components)
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        Y = pd.get_dummies(y).values.astype(float)
        self.pls.fit(X, Y)
        return self

    def predict(self, X):
        probs = self.pls.predict(X)
        return self.classes_[np.argmax(probs, axis=1)]


def get_plsda():
    return PLSDA(n_components=10)


class _MLFeatureWrapper:
    """Wrap an sklearn classifier to use hand-crafted features (78-dim)."""
    is_feature_model = True

    def __init__(self, base_cls_fn):
        self.base_cls_fn = base_cls_fn
        self._clf = None
        self._wn = None

    def _extract(self, X):
        from src.feature_engineering import extract_all_features
        features, _ = extract_all_features(X, self._wn)
        return features

    def fit(self, X, y, groups=None, wn=None):
        if wn is not None:
            self._wn = wn
        feats = self._extract(X)
        self._clf = self.base_cls_fn()
        self._clf.fit(feats, y)
        return self

    def predict(self, X):
        feats = self._extract(X)
        return self._clf.predict(feats)


def get_rf_feat():
    return _MLFeatureWrapper(get_rf)


def get_svm_feat():
    return _MLFeatureWrapper(get_svm)


def get_plsda_feat():
    return _MLFeatureWrapper(get_plsda)


class _SE1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = x.mean(dim=-1)
        weights = self.fc(weights).unsqueeze(-1)
        return x * weights


class _CNN1DNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), _SE1D(16), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), _SE1D(32), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), _SE1D(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), _SE1D(64),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


class _ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.se = _SE1D(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.se(self.block(x)) + x)


class _ResNet1DNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.stage1 = nn.Sequential(_ResBlock1D(16), _ResBlock1D(16))
        self.down1 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=1, stride=2),
            nn.BatchNorm1d(32),
        )
        self.stage2 = nn.Sequential(_ResBlock1D(32), _ResBlock1D(32))
        self.down2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1, stride=2),
            nn.BatchNorm1d(64),
        )
        self.stage3 = nn.Sequential(_ResBlock1D(64), _ResBlock1D(64))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
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

        self.label_map = {label: idx for idx, label in enumerate(sorted(set(y)))}
        self.inv_map = {idx: label for label, idx in self.label_map.items()}
        yi = np.array([self.label_map[value] for value in y])
        num_classes = len(self.label_map)

        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        yt = torch.tensor(yi, dtype=torch.long)

        if groups is not None:
            gss = GroupShuffleSplit(n_splits=1, test_size=DL_VAL_FRAC, random_state=self.seed)
            t_idx, v_idx = next(gss.split(X, yi, groups))
        else:
            idx = np.random.permutation(len(Xt))
            n_val = max(int(len(Xt) * DL_VAL_FRAC), 1)
            t_idx, v_idx = idx[n_val:], idx[:n_val]

        if AUG_ENABLED:
            from src.dataset import augment_spectra, spectral_mixup

            X_tr_np, y_tr_np = X[t_idx], yi[t_idx]
            X_tr_np, y_tr_np = augment_spectra(X_tr_np, y_tr_np, n_aug=AUG_N)
            if MIXUP_ENABLED:
                X_mix, y_mix = spectral_mixup(
                    X_tr_np,
                    y_tr_np,
                    n_mix=len(X[t_idx]),
                    alpha=MIXUP_ALPHA,
                )
                X_tr_np = np.concatenate([X_tr_np, X_mix], axis=0)
                y_tr_np = np.concatenate([y_tr_np, y_mix], axis=0)
            Xt_train = torch.tensor(X_tr_np, dtype=torch.float32).unsqueeze(1)
            yt_train = torch.tensor(y_tr_np, dtype=torch.long)
        else:
            Xt_train, yt_train = Xt[t_idx], yt[t_idx]

        train_loader = DataLoader(
            TensorDataset(Xt_train, yt_train),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(Xt[v_idx], yt[v_idx]),
            batch_size=256,
            shuffle=False,
        )

        self.model = self.net_cls(X.shape[1], num_classes).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=DL_WEIGHT_DECAY,
        )
        if DL_SCHEDULER == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs,
                eta_min=1e-6,
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=10,
                factor=0.5,
            )

        train_loss = _make_loss(yt_train.cpu().numpy(), num_classes)
        val_loss = nn.CrossEntropyLoss()
        best_loss = float('inf')
        best_state = None
        wait = 0

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = train_loss(self.model(xb), yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            epoch_val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    epoch_val_loss += val_loss(self.model(xb), yb).item() * len(yb)
                    val_count += len(yb)

            epoch_val_loss /= max(val_count, 1)
            if DL_SCHEDULER == 'cosine':
                scheduler.step()
            else:
                scheduler.step(epoch_val_loss)

            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                wait = 0
                best_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            preds = self.model(Xt).argmax(1).cpu().numpy()
        return np.array([self.inv_map[pred] for pred in preds])


def get_cnn1d():
    return _DLWrapper(_CNN1DNet)


def get_resnet1d():
    return _DLWrapper(_ResNet1DNet)


class _DLFeatureWrapper:
    """Run any DL model on handcrafted features instead of raw spectra."""
    is_feature_model = True

    def __init__(self, net_cls, **kwargs):
        self._inner = _DLWrapper(net_cls, **kwargs)
        self._wn = None

    def _extract(self, X):
        from src.feature_engineering import extract_all_features
        features, _ = extract_all_features(X, self._wn)
        return features

    def fit(self, X, y, groups=None, wn=None):
        if wn is not None:
            self._wn = wn
        feats = self._extract(X)
        return self._inner.fit(feats, y, groups=groups)

    def predict(self, X):
        feats = self._extract(X)
        return self._inner.predict(feats)


def get_cnn1d_feat():
    return _DLFeatureWrapper(_CNN1DNet)


def get_resnet1d_feat():
    return _DLFeatureWrapper(_ResNet1DNet)


class _KANCNN1DNet(nn.Module):
    """CNN backbone + KAN head (replaces FC classifier with KAN layers)."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), _SE1D(16), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), _SE1D(32), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), _SE1D(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), _SE1D(64),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.kan_head = nn.Sequential(
            _KANLinear(64, 32, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER),
            nn.Dropout(0.2),
            _KANLinear(32, num_classes, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.kan_head(x)


def get_kan_cnn():
    return _DLWrapper(_KANCNN1DNet)


def get_kan_cnn_feat():
    return _DLFeatureWrapper(_KANCNN1DNet)


class _KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        step = 2.0 / grid_size
        grid = torch.linspace(
            -1 - step * spline_order,
            1 + step * spline_order,
            grid_size + 2 * spline_order + 1,
        )
        self.register_buffer('grid', grid)

        n_bases = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, n_bases) * 0.1
        )
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(
                -1.0 / in_features ** 0.5,
                1.0 / in_features ** 0.5,
            )
        )

    def forward(self, x):
        bases = self._compute_bases(x)
        spline_out = torch.einsum('bin,oin->bo', bases, self.spline_weight)
        base_out = F.silu(x) @ self.base_weight.t()
        return spline_out + base_out

    def _compute_bases(self, x):
        x = torch.tanh(x).unsqueeze(-1)
        bases = ((x >= self.grid[:-1]) & (x < self.grid[1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            n = bases.shape[-1]
            t_left = self.grid[:n - 1]
            t_left_k = self.grid[k:k + n - 1]
            left = (x - t_left) / (t_left_k - t_left + 1e-8)
            t_right_k1 = self.grid[k + 1:k + 1 + n - 1]
            t_right = self.grid[1:1 + n - 1]
            right = (t_right_k1 - x) / (t_right_k1 - t_right + 1e-8)
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases


class _MultiTaskCNN1DNet(nn.Module):
    def __init__(self, input_dim, task_num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), _SE1D(16), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), _SE1D(32), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), _SE1D(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), _SE1D(64),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.heads = nn.ModuleDict({
            task_id: nn.Linear(64, n_classes) for task_id, n_classes in task_num_classes
        })

    def forward(self, x):
        features = self.backbone(x)
        features = self.gap(features).squeeze(-1)
        features = self.dropout(features)
        return {task_id: head(features) for task_id, head in self.heads.items()}


class _MultiTaskResNet1DNet(nn.Module):
    def __init__(self, input_dim, task_num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.stage1 = nn.Sequential(_ResBlock1D(16), _ResBlock1D(16))
        self.down1 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=1, stride=2), nn.BatchNorm1d(32))
        self.stage2 = nn.Sequential(_ResBlock1D(32), _ResBlock1D(32))
        self.down2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=1, stride=2), nn.BatchNorm1d(64))
        self.stage3 = nn.Sequential(_ResBlock1D(64), _ResBlock1D(64))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.heads = nn.ModuleDict({
            task_id: nn.Linear(64, n_classes) for task_id, n_classes in task_num_classes
        })

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        features = self.gap(x).squeeze(-1)
        features = self.dropout(features)
        return {task_id: head(features) for task_id, head in self.heads.items()}


class _MultiTaskKANCNN1DNet(nn.Module):
    def __init__(self, input_dim, task_num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), _SE1D(16), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), _SE1D(32), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), _SE1D(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), _SE1D(64),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.heads = nn.ModuleDict({
            task_id: _KANLinear(64, n_classes, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
            for task_id, n_classes in task_num_classes
        })

    def forward(self, x):
        features = self.backbone(x)
        features = self.gap(features).squeeze(-1)
        features = self.dropout(features)
        return {task_id: head(features) for task_id, head in self.heads.items()}


class _MultiTaskDLWrapper:
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
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = True

        task_num_classes = []
        encoded_y = {}
        for task in self.tasks:
            task_id = task['id']
            labels = sorted(set(y_dict[task_id]))
            self.label_maps[task_id] = {label: idx for idx, label in enumerate(labels)}
            self.inv_maps[task_id] = {idx: label for label, idx in self.label_maps[task_id].items()}
            task_num_classes.append((task_id, len(labels)))
            encoded_y[task_id] = torch.tensor(
                [self.label_maps[task_id][value] for value in y_dict[task_id]],
                dtype=torch.long,
            )

        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        if groups is not None:
            gss = GroupShuffleSplit(n_splits=1, test_size=DL_VAL_FRAC, random_state=self.seed)
            t_idx, v_idx = next(gss.split(X, encoded_y[self.tasks[0]['id']].numpy(), groups))
        else:
            idx = np.random.permutation(len(Xt))
            n_val = max(int(len(Xt) * DL_VAL_FRAC), 1)
            t_idx, v_idx = idx[n_val:], idx[:n_val]

        if AUG_ENABLED:
            from src.dataset import augment_spectra_mt, spectral_mixup_mt

            X_tr_np = X[t_idx]
            y_dict_tr = {task['id']: encoded_y[task['id']][t_idx].numpy() for task in self.tasks}
            X_tr_np, y_dict_tr = augment_spectra_mt(X_tr_np, y_dict_tr, n_aug=AUG_N)
            if MIXUP_ENABLED:
                X_mix, y_mix = spectral_mixup_mt(
                    X_tr_np,
                    y_dict_tr,
                    n_mix=len(X[t_idx]),
                    alpha=MIXUP_ALPHA,
                )
                X_tr_np = np.concatenate([X_tr_np, X_mix], axis=0)
                y_dict_tr = {
                    task_id: np.concatenate([y_dict_tr[task_id], y_mix[task_id]])
                    for task_id in y_dict_tr
                }
            Xt_train = torch.tensor(X_tr_np, dtype=torch.float32).unsqueeze(1)
            train_tensors = [Xt_train] + [
                torch.tensor(y_dict_tr[task['id']], dtype=torch.long) for task in self.tasks
            ]
        else:
            train_tensors = [Xt[t_idx]] + [encoded_y[task['id']][t_idx] for task in self.tasks]

        val_tensors = [Xt[v_idx]] + [encoded_y[task['id']][v_idx] for task in self.tasks]
        train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(*val_tensors), batch_size=256, shuffle=False)

        self.model = self.net_cls(X.shape[1], task_num_classes).to(self.device)
        n_tasks = len(self.tasks)
        log_vars = nn.Parameter(torch.zeros(n_tasks, device=self.device))
        task_losses = [
            _make_loss(train_tensors[i + 1].cpu().numpy(), len(self.label_maps[self.tasks[i]['id']]))
            for i in range(n_tasks)
        ]
        val_loss = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + [log_vars],
            lr=self.lr,
            weight_decay=DL_WEIGHT_DECAY,
        )
        if DL_SCHEDULER == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs,
                eta_min=1e-6,
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=10,
                factor=0.5,
            )

        best_loss = float('inf')
        best_state = None
        wait = 0

        for _ in range(self.epochs):
            self.model.train()
            for batch in train_loader:
                xb = batch[0].to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = sum(
                    torch.exp(-log_vars[i]) * task_losses[i](logits[self.tasks[i]['id']], batch[i + 1].to(self.device))
                    + 0.5 * log_vars[i]
                    for i in range(n_tasks)
                )
                loss.backward()
                optimizer.step()

            self.model.eval()
            epoch_val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    xb = batch[0].to(self.device)
                    logits = self.model(xb)
                    batch_loss = sum(
                        val_loss(logits[self.tasks[i]['id']], batch[i + 1].to(self.device))
                        for i in range(n_tasks)
                    )
                    epoch_val_loss += batch_loss.item() * len(xb)
                    val_count += len(xb)

            epoch_val_loss /= max(val_count, 1)
            if DL_SCHEDULER == 'cosine':
                scheduler.step()
            else:
                scheduler.step(epoch_val_loss)

            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                wait = 0
                best_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        preds = {}
        with torch.no_grad():
            logits = self.model(Xt)
            for task_id, task_logits in logits.items():
                idx = task_logits.argmax(1).cpu().numpy()
                preds[task_id] = np.array([self.inv_maps[task_id][value] for value in idx])
        return preds


def get_mt_cnn():
    return _MultiTaskDLWrapper(_MultiTaskCNN1DNet, TASKS)


def get_mt_resnet():
    return _MultiTaskDLWrapper(_MultiTaskResNet1DNet, TASKS)


def get_mt_kan_cnn():
    return _MultiTaskDLWrapper(_MultiTaskKANCNN1DNet, TASKS)


def get_mt_cnn_3c():
    return _MultiTaskDLWrapper(_MultiTaskCNN1DNet, TASKS_MT_FULL)


def get_mt_resnet_3c():
    return _MultiTaskDLWrapper(_MultiTaskResNet1DNet, TASKS_MT_FULL)


def get_mt_kan_cnn_3c():
    return _MultiTaskDLWrapper(_MultiTaskKANCNN1DNet, TASKS_MT_FULL)


class _SpectrumKANNet(nn.Module):
    """Pure KAN on full spectrum (no CNN backbone)."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.kan1 = _KANLinear(input_dim, 128, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.kan2 = _KANLinear(128, 64, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.kan3 = _KANLinear(64, 32, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.kan4 = _KANLinear(32, num_classes, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = self.bn(x)
        x = self.kan1(x)
        x = self.dropout1(x)
        x = self.kan2(x)
        x = self.dropout2(x)
        x = self.kan3(x)
        x = self.dropout3(x)
        return self.kan4(x)


def get_spectrum_kan():
    return _DLWrapper(_SpectrumKANNet)


class _FeatureKANNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.kan1 = _KANLinear(input_dim, 64, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.kan2 = _KANLinear(64, 32, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.kan3 = _KANLinear(32, num_classes, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.bn(x)
        x = self.kan1(x)
        x = self.dropout1(x)
        x = self.kan2(x)
        x = self.dropout2(x)
        x = self.kan3(x)
        return x


class _FeatureKANWrapper:
    is_feature_model = True

    def __init__(self, lr=DL_LR, epochs=DL_EPOCHS, patience=DL_PATIENCE,
                 batch_size=DL_BATCH, seed=SEED):
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_map = None
        self.inv_map = None
        self._wn = None

    def _extract(self, X):
        from src.feature_engineering import extract_all_features

        features, _ = extract_all_features(X, self._wn)
        return features

    def fit(self, X, y, groups=None, wn=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if wn is not None:
            self._wn = wn
        feats = self._extract(X)

        self.label_map = {label: idx for idx, label in enumerate(sorted(set(y)))}
        self.inv_map = {idx: label for label, idx in self.label_map.items()}
        yi = np.array([self.label_map[value] for value in y])
        num_classes = len(self.label_map)

        if groups is not None:
            gss = GroupShuffleSplit(n_splits=1, test_size=DL_VAL_FRAC, random_state=self.seed)
            t_idx, v_idx = next(gss.split(feats, yi, groups))
        else:
            idx = np.random.permutation(len(feats))
            n_val = max(int(len(feats) * DL_VAL_FRAC), 1)
            t_idx, v_idx = idx[n_val:], idx[:n_val]

        if AUG_ENABLED:
            rng = np.random.RandomState(self.seed)
            F_tr, y_tr = feats[t_idx], yi[t_idx]
            aug_feats = [F_tr]
            aug_labels = [y_tr]
            for _ in range(AUG_N):
                noisy = F_tr.copy()
                noisy += rng.normal(0, 0.02, noisy.shape) * np.std(noisy, axis=0, keepdims=True)
                noisy *= rng.uniform(0.95, 1.05, (len(noisy), 1))
                aug_feats.append(noisy)
                aug_labels.append(y_tr.copy())
            F_tr = np.concatenate(aug_feats, axis=0)
            y_tr = np.concatenate(aug_labels, axis=0)
        else:
            F_tr, y_tr = feats[t_idx], yi[t_idx]

        train_loader = DataLoader(
            TensorDataset(torch.tensor(F_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(feats[v_idx], dtype=torch.float32), torch.tensor(yi[v_idx], dtype=torch.long)),
            batch_size=256,
            shuffle=False,
        )

        self.model = _FeatureKANNet(feats.shape[1], num_classes).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=DL_WEIGHT_DECAY)
        if DL_SCHEDULER == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        train_loss = _make_loss(y_tr, num_classes)
        val_loss = nn.CrossEntropyLoss()
        best_loss = float('inf')
        best_state = None
        wait = 0

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = train_loss(self.model(xb), yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            epoch_val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    epoch_val_loss += val_loss(self.model(xb), yb).item() * len(yb)
                    val_count += len(yb)

            epoch_val_loss /= max(val_count, 1)
            if DL_SCHEDULER == 'cosine':
                scheduler.step()
            else:
                scheduler.step(epoch_val_loss)

            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                wait = 0
                best_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        feats = self._extract(X)
        Xt = torch.tensor(feats, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(Xt).argmax(1).cpu().numpy()
        return np.array([self.inv_map[pred] for pred in preds])


def get_feature_kan():
    return _FeatureKANWrapper()


class _MTFeatureKANNet(nn.Module):
    def __init__(self, input_dim, task_num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.shared1 = _KANLinear(input_dim, 64, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.shared2 = _KANLinear(64, 48, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.heads = nn.ModuleDict({
            task_id: _KANLinear(48, n_classes, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
            for task_id, n_classes in task_num_classes
        })

    def forward(self, x):
        x = self.bn(x)
        x = self.shared1(x)
        x = self.dropout1(x)
        x = self.shared2(x)
        x = self.dropout2(x)
        return {task_id: head(x) for task_id, head in self.heads.items()}


class _MTFeatureKANWrapper:
    is_multitask = True
    is_feature_model = True

    def __init__(self, tasks, lr=DL_LR, epochs=DL_EPOCHS,
                 patience=DL_PATIENCE, batch_size=DL_BATCH, seed=SEED):
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
        self._wn = None

    def _extract(self, X):
        from src.feature_engineering import extract_all_features

        features, _ = extract_all_features(X, self._wn)
        return features

    def fit(self, X, y_dict, groups=None, wn=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if wn is not None:
            self._wn = wn
        feats = self._extract(X)

        task_num_classes = []
        encoded_y = {}
        for task in self.tasks:
            task_id = task['id']
            labels = sorted(set(y_dict[task_id]))
            self.label_maps[task_id] = {label: idx for idx, label in enumerate(labels)}
            self.inv_maps[task_id] = {idx: label for label, idx in self.label_maps[task_id].items()}
            task_num_classes.append((task_id, len(labels)))
            encoded_y[task_id] = torch.tensor(
                [self.label_maps[task_id][value] for value in y_dict[task_id]],
                dtype=torch.long,
            )

        if groups is not None:
            gss = GroupShuffleSplit(n_splits=1, test_size=DL_VAL_FRAC, random_state=self.seed)
            t_idx, v_idx = next(gss.split(feats, encoded_y[self.tasks[0]['id']].numpy(), groups))
        else:
            idx = np.random.permutation(len(feats))
            n_val = max(int(len(feats) * DL_VAL_FRAC), 1)
            t_idx, v_idx = idx[n_val:], idx[:n_val]

        if AUG_ENABLED:
            rng = np.random.RandomState(self.seed)
            F_tr = feats[t_idx]
            y_dict_tr = {task['id']: encoded_y[task['id']][t_idx].numpy() for task in self.tasks}
            aug_feats = [F_tr]
            aug_labels = {task['id']: [y_dict_tr[task['id']]] for task in self.tasks}
            for _ in range(AUG_N):
                noisy = F_tr + rng.normal(0, 0.02, F_tr.shape) * np.std(F_tr, axis=0, keepdims=True)
                noisy *= rng.uniform(0.95, 1.05, (len(noisy), 1))
                aug_feats.append(noisy)
                for task in self.tasks:
                    aug_labels[task['id']].append(y_dict_tr[task['id']].copy())
            F_tr = np.concatenate(aug_feats, axis=0)
            y_dict_tr = {task_id: np.concatenate(parts) for task_id, parts in aug_labels.items()}
        else:
            F_tr = feats[t_idx]
            y_dict_tr = {task['id']: encoded_y[task['id']][t_idx].numpy() for task in self.tasks}

        train_tensors = [torch.tensor(F_tr, dtype=torch.float32)] + [
            torch.tensor(y_dict_tr[task['id']], dtype=torch.long) for task in self.tasks
        ]
        val_tensors = [torch.tensor(feats[v_idx], dtype=torch.float32)] + [
            encoded_y[task['id']][v_idx] for task in self.tasks
        ]
        train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(*val_tensors), batch_size=256, shuffle=False)

        self.model = _MTFeatureKANNet(feats.shape[1], task_num_classes).to(self.device)
        n_tasks = len(self.tasks)
        log_vars = nn.Parameter(torch.zeros(n_tasks, device=self.device))
        task_losses = [
            _make_loss(train_tensors[i + 1].cpu().numpy(), len(self.label_maps[self.tasks[i]['id']]))
            for i in range(n_tasks)
        ]
        val_loss = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + [log_vars],
            lr=self.lr,
            weight_decay=DL_WEIGHT_DECAY,
        )
        if DL_SCHEDULER == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        best_loss = float('inf')
        best_state = None
        wait = 0

        for _ in range(self.epochs):
            self.model.train()
            for batch in train_loader:
                xb = batch[0].to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = sum(
                    torch.exp(-log_vars[i]) * task_losses[i](logits[self.tasks[i]['id']], batch[i + 1].to(self.device))
                    + 0.5 * log_vars[i]
                    for i in range(n_tasks)
                )
                loss.backward()
                optimizer.step()

            self.model.eval()
            epoch_val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    xb = batch[0].to(self.device)
                    logits = self.model(xb)
                    batch_loss = sum(
                        val_loss(logits[self.tasks[i]['id']], batch[i + 1].to(self.device))
                        for i in range(n_tasks)
                    )
                    epoch_val_loss += batch_loss.item() * len(xb)
                    val_count += len(xb)

            epoch_val_loss /= max(val_count, 1)
            if DL_SCHEDULER == 'cosine':
                scheduler.step()
            else:
                scheduler.step(epoch_val_loss)

            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                wait = 0
                best_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        feats = self._extract(X)
        Xt = torch.tensor(feats, dtype=torch.float32).to(self.device)
        preds = {}
        with torch.no_grad():
            logits = self.model(Xt)
            for task_id, task_logits in logits.items():
                idx = task_logits.argmax(1).cpu().numpy()
                preds[task_id] = np.array([self.inv_maps[task_id][value] for value in idx])
        return preds


def get_mt_feature_kan():
    return _MTFeatureKANWrapper(TASKS)


def get_mt_feature_kan_3c():
    return _MTFeatureKANWrapper(TASKS_MT_FULL)


class _EnsembleWrapper:
    """Soft-voting ensemble: RF(features) + Feature-KAN(features)."""
    is_feature_model = True

    def __init__(self, seed=SEED):
        self.seed = seed
        self._rf = None
        self._kan = None
        self._wn = None

    def _extract(self, X):
        from src.feature_engineering import extract_all_features
        features, _ = extract_all_features(X, self._wn)
        return features

    def fit(self, X, y, groups=None, wn=None):
        if wn is not None:
            self._wn = wn
        self._rf = _MLFeatureWrapper(get_rf)
        self._rf.fit(X, y, groups=groups, wn=wn)
        self._kan = _FeatureKANWrapper()
        self._kan.fit(X, y, groups=groups, wn=wn)
        return self

    def predict(self, X):
        feats = self._extract(X)
        # RF: use predict_proba for soft voting
        rf_proba = self._rf._clf.predict_proba(feats)
        # KAN: get logits → softmax
        import torch
        self._kan.model.eval()
        Xt = torch.tensor(feats, dtype=torch.float32).to(self._kan.device)
        with torch.no_grad():
            kan_logits = self._kan.model(Xt)
            kan_proba = torch.softmax(kan_logits, dim=-1).cpu().numpy()
        # Align class dimensions (RF may have fewer classes)
        n_classes = max(rf_proba.shape[1], kan_proba.shape[1])
        if rf_proba.shape[1] < n_classes:
            pad = np.zeros((rf_proba.shape[0], n_classes - rf_proba.shape[1]))
            rf_proba = np.hstack([rf_proba, pad])
        if kan_proba.shape[1] < n_classes:
            pad = np.zeros((kan_proba.shape[0], n_classes - kan_proba.shape[1]))
            kan_proba = np.hstack([kan_proba, pad])
        # Average probabilities
        avg_proba = 0.5 * rf_proba + 0.5 * kan_proba
        pred_idx = np.argmax(avg_proba, axis=1)
        # Map back to original labels
        inv_map = self._kan.inv_map
        return np.array([inv_map[idx] for idx in pred_idx])


def get_ensemble():
    return _EnsembleWrapper()


MODEL_REGISTRY = {
    # ML full-spectrum
    'RF': get_rf,
    'SVM': get_svm,
    'PLS-DA': get_plsda,
    # ML features
    'RF-feat': get_rf_feat,
    'SVM-feat': get_svm_feat,
    'PLS-DA-feat': get_plsda_feat,
    # DL full-spectrum
    '1D-CNN': get_cnn1d,
    '1D-ResNet': get_resnet1d,
    'Spectrum-KAN': get_spectrum_kan,
    'KAN-CNN': get_kan_cnn,
    # DL features
    '1D-CNN-feat': get_cnn1d_feat,
    '1D-ResNet-feat': get_resnet1d_feat,
    'Feature-KAN': get_feature_kan,
    'KAN-CNN-feat': get_kan_cnn_feat,
    # Ensemble
    'Ensemble': get_ensemble,
    # Multi-task (kept for optional use)
    'MT-CNN': get_mt_cnn,
    'MT-ResNet': get_mt_resnet,
    'MT-KAN-CNN': get_mt_kan_cnn,
    'MT-Feature-KAN': get_mt_feature_kan,
    'MT-CNN-3c': get_mt_cnn_3c,
    'MT-ResNet-3c': get_mt_resnet_3c,
    'MT-KAN-CNN-3c': get_mt_kan_cnn_3c,
    'MT-Feature-KAN-3c': get_mt_feature_kan_3c,
}