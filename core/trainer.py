#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import gc
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.registry import FULL_ALGO_MAP
from core.profiler import EdgeProfiler

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    _HAS_XGBOOST = False


# ============================================================
# 4-class target
# ============================================================
TARGET_CLASSES = ["eating", "ruminating", "standing", "walking"]
LABEL2ID = {"eating": 0, "ruminating": 1, "standing": 2, "walking": 3}
ID2LABEL = ["eating", "ruminating", "standing", "walking"]
FIXED_LABELS = np.arange(4, dtype=int)

# ============================================================
# Time columns (ONLY for sorting, NOT features)
# ============================================================
TIME_COL_CANDIDATES = [
    "TimeStamp_UNIX", "TimeStamp_JST",
    "timestamp", "dateTime",
    "datetime", "DateTime", "Timestamp", "TIME", "Time",
    "date", "Date", "time",
]

# ============================================================
# Group columns candidates (cow/animal identity)
# ============================================================
GROUP_COL_CANDIDATES = [
    "cow_id", "cowid", "CowID", "CowId", "cow",
    "animal_id", "animalid", "AnimalID", "AnimalId",
    "tag_id", "tagid", "TagID", "TagId",
    "device_id", "DeviceID", "device",
    "id", "ID"
]

# ============================================================
# Dataset-specific maps (optional)
# ============================================================
LABEL_MAPS = {
    "japan_cow": {
        "GRZ": "eating",
        "FES": "eating",
        "SLT": "eating",
        "DRN": "eating",
        "LCK": "eating",
        "MOV": "walking",
        "RES": "standing",
        "REL": "standing",
        "RUS": "ruminating",
        "RUL": "ruminating",
    },
    "actbecalf": {
        "standing": "standing",
        "lying": "standing",
        "lying-down": "standing",
        "rising": "standing",

        "walking": "walking",
        "backward": "walking",
        "sniff_walking": "walking",
        "running": "walking",

        "eating_forage": "eating",
        "eating_concentrates": "eating",
        "eating_bedding": "eating",
        "eating": "eating",

        "rumination": "ruminating",
        "rumination_lying": "ruminating",
        "ruminating": "ruminating",
    },
    "nose_ring": {
        # 0=Feeding, 1=Rumination, 2=Standing, 3=Lying, 4=Walking
        0: "eating",
        1: "ruminating",
        2: "standing",
        3: "standing",
        4: "walking",
    },
}

GENERIC_SYNONYMS = {
    # eating
    "grazing": "eating", "graze": "eating", "feed": "eating", "feeding": "eating",
    "eating": "eating", "chewing": "eating", "forage": "eating",
    "concentrate": "eating", "concentrates": "eating", "bedding": "eating",

    # walking
    "walking": "walking", "walk": "walking", "moving": "walking", "move": "walking",
    "locomotion": "walking", "run": "walking", "running": "walking",
    "backward": "walking", "sniff_walking": "walking",

    # standing (merge lying/resting into standing)
    "standing": "standing", "stand": "standing", "resting": "standing", "rest": "standing",
    "lying": "standing", "lie": "standing", "lying-down": "standing",
    "laying": "standing", "lay": "standing", "rising": "standing",
    "rise": "standing", "idle": "standing",

    # ruminating
    "ruminating": "ruminating", "ruminate": "ruminating", "rumination": "ruminating",
    "cud": "ruminating", "chew_cud": "ruminating",
}


def _normalize_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _safe_int(x):
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def _detect_label_col(df: pd.DataFrame) -> str:
    for c in ["Target", "target", "Label", "label", "Activity", "activity",
              "behavior", "behaviour", "class", "Class"]:
        if c in df.columns:
            return c
    return df.columns[-1]


def _auto_detect_dataset_key(label_series: pd.Series) -> str | None:
    s = label_series.dropna()
    if s.empty:
        return None

    sample_vals = s.sample(min(len(s), 2000), random_state=42).tolist()

    # numeric small codes => likely nose_ring
    numeric_vals = []
    numeric_ok = True
    for v in sample_vals:
        iv = _safe_int(v)
        if iv is None:
            numeric_ok = False
            break
        numeric_vals.append(iv)
    if numeric_ok and numeric_vals:
        mn, mx = min(numeric_vals), max(numeric_vals)
        if mn >= 0 and mx <= 10:
            return "nose_ring"

    # japan cow codes
    up = set(_normalize_str(v).upper() for v in sample_vals)
    if any(k in up for k in ["GRZ", "FES", "MOV", "RES", "REL", "RUS", "RUL"]):
        return "japan_cow"

    # actbecalf-like labels
    lo = set(_normalize_str(v).lower() for v in sample_vals)
    if any(k in lo for k in ["eating_forage", "eating_concentrates", "rumination_lying",
                             "sniff_walking", "lying-down"]):
        return "actbecalf"

    return None


def _map_to_4class(df: pd.DataFrame, label_col: str, dataset_key: str | None = None):
    raw = df[label_col]

    if dataset_key is None:
        dataset_key = _auto_detect_dataset_key(raw)

    mapping = LABEL_MAPS.get(dataset_key, None)

    mapped = []
    if dataset_key == "nose_ring" and mapping is not None:
        for v in raw.tolist():
            iv = _safe_int(v)
            mapped.append(mapping.get(iv, None))
    elif dataset_key == "japan_cow" and mapping is not None:
        for v in raw.tolist():
            k = _normalize_str(v).upper()
            mapped.append(mapping.get(k, None))
    elif dataset_key == "actbecalf" and mapping is not None:
        for v in raw.tolist():
            k = _normalize_str(v).lower()
            mapped.append(mapping.get(k, None))
    else:
        for v in raw.tolist():
            s = _normalize_str(v).lower()
            if s in TARGET_CLASSES:
                mapped.append(s)
                continue
            if s in GENERIC_SYNONYMS:
                mapped.append(GENERIC_SYNONYMS[s])
                continue

            # contains-based heuristics
            if ("rumin" in s) or ("cud" in s):
                mapped.append("ruminating")
            elif ("graz" in s) or ("feed" in s) or ("eat" in s) or ("forage" in s):
                mapped.append("eating")
            elif ("walk" in s) or ("mov" in s) or ("run" in s) or ("backward" in s):
                mapped.append("walking")
            elif ("stand" in s) or ("rest" in s) or ("lying" in s) or ("lay" in s) or ("idle" in s) or ("rise" in s):
                mapped.append("standing")
            else:
                mapped.append(None)

    mapped = pd.Series(mapped, index=df.index, dtype="object")
    keep = mapped.isin(TARGET_CLASSES)

    df2 = df.loc[keep].copy()
    y_std = mapped.loc[keep].astype(str)

    meta = {
        "dataset_key": dataset_key or "unknown",
        "kept_rows": int(keep.sum()),
        "dropped_rows": int((~keep).sum()),
        "kept_ratio": float(int(keep.sum()) / max(1, len(df))),
        "target_classes": ",".join(TARGET_CLASSES),
    }
    return df2, y_std, meta


def _encode_fixed_4(y_std: pd.Series) -> np.ndarray:
    """FIXED encoding to 0..3 so macro-F1 is always computed on 4 classes."""
    y = y_std.map(LABEL2ID)
    if y.isna().any():
        bad = y_std[y.isna()].unique()[:10]
        raise ValueError(f"Found unmapped labels after 4-class mapping: {bad}")
    return y.astype(int).to_numpy()


def _can_stratify(y: np.ndarray, test_size: float) -> bool:
    """stratify can crash if some classes too rare."""
    y = np.asarray(y)
    n = len(y)
    if n < 20:
        return False
    vals, cnts = np.unique(y, return_counts=True)
    if len(vals) < 2:
        return False
    if cnts.min() < 2:
        return False
    test_n = int(np.ceil(test_size * n))
    if test_n < len(vals):
        return False
    return True


class StudentTrainer:
    """
    Supported split_method:
      - random: random split (window-wise)
      - time  : chronological split (after sorting)
      - group : GroupShuffleSplit (group-wise, no group leakage)
      - loco  : Leave-One-Group-Out (train excludes test group entirely)
      - full_loco: run LOCO for every group + also export one *_final model trained on ALL data
      - all   : no split; train on ALL data and export *_final model
    """

    def __init__(self):
        self.models_dir = "deployment_builds"
        os.makedirs(self.models_dir, exist_ok=True)
        self.profiler = EdgeProfiler()

    # -----------------------------
    # Basic helpers
    # -----------------------------
    @staticmethod
    def _pick_label_col(df: pd.DataFrame) -> str:
        return _detect_label_col(df)

    @staticmethod
    def _coerce_numeric(df_feat: pd.DataFrame) -> np.ndarray:
        df_num = df_feat.apply(pd.to_numeric, errors="coerce")
        med = df_num.median(numeric_only=True)
        df_num = df_num.fillna(med).fillna(0.0)
        return df_num.to_numpy(dtype=np.float32)

    # -----------------------------
    # Auto-detect group_col
    # -----------------------------
    @staticmethod
    def _auto_pick_group_col(df_m: pd.DataFrame) -> str | None:
        """Try to find a reasonable animal/cow identity column."""
        n = len(df_m)
        if n < 10:
            return None

        for c in GROUP_COL_CANDIDATES:
            if c not in df_m.columns:
                continue
            s = df_m[c]
            nunq = s.nunique(dropna=True)

            if nunq < 2:
                continue

            ratio = float(nunq) / float(max(1, n))
            if ratio > 0.90:
                continue

            return c
        return None

    # -----------------------------
    # Sequence building
    # -----------------------------
    @staticmethod
    def _build_sequences(X: np.ndarray, y: np.ndarray, window: int, groups: np.ndarray | None = None):
        if window <= 1:
            return X, y, groups

        T = len(X)
        if T < window:
            raise ValueError(f"Not enough rows for windowing: T={T}, window={window}")

        Xs, ys = [], []
        gs = [] if groups is not None else None

        for i in range(T - window + 1):
            Xs.append(X[i:i + window])
            ys.append(y[i + window - 1])
            if groups is not None:
                gs.append(groups[i + window - 1])

        Xs = np.asarray(Xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.int64)
        if groups is not None:
            gs = np.asarray(gs, dtype=object)
        return Xs, ys, gs

    # -----------------------------
    # Sorting helpers (only for correct windowing)
    # -----------------------------
    @staticmethod
    def _extract_time_series(df_m: pd.DataFrame, dataset_key: str | None):
        # Prefer numeric unix when available
        if "TimeStamp_UNIX" in df_m.columns:
            return pd.to_numeric(df_m["TimeStamp_UNIX"], errors="coerce")

        # Otherwise try known datetime strings
        if dataset_key == "nose_ring" and "timestamp" in df_m.columns:
            return pd.to_datetime(df_m["timestamp"], errors="coerce")
        if dataset_key == "actbecalf" and "dateTime" in df_m.columns:
            return pd.to_datetime(df_m["dateTime"], errors="coerce")
        if "TimeStamp_JST" in df_m.columns:
            # usually HH:MM:SS.xxx; parsing still works but may drop date context
            return pd.to_datetime(df_m["TimeStamp_JST"], errors="coerce")

        found = None
        for c in TIME_COL_CANDIDATES:
            if c in df_m.columns:
                found = c
                break
        if found is None:
            return None

        if found == "TimeStamp_UNIX":
            return pd.to_numeric(df_m[found], errors="coerce")
        return pd.to_datetime(df_m[found], errors="coerce")

    @staticmethod
    def _sort_by_datetime(df_m: pd.DataFrame, y_std: pd.Series, groups: np.ndarray | None, dataset_key: str | None):
        tser = StudentTrainer._extract_time_series(df_m, dataset_key)
        if tser is None or tser.notna().sum() == 0:
            return df_m, y_std, groups

        tmp = df_m.copy()
        tmp["__time__"] = tser
        if groups is not None:
            tmp["__group__"] = pd.Series(groups, index=df_m.index, dtype="object")
            tmp = tmp.sort_values(by=["__group__", "__time__"], kind="mergesort")
        else:
            tmp = tmp.sort_values(by=["__time__"], kind="mergesort")

        order_idx = tmp.index
        df_sorted = df_m.loc[order_idx].copy()
        y_sorted = y_std.loc[order_idx].copy()

        if groups is not None:
            pos = df_m.index.get_indexer(order_idx)
            groups_sorted = groups[pos]
        else:
            groups_sorted = None

        return df_sorted, y_sorted, groups_sorted

    # -----------------------------
    # window -> tabular stats for classic ML
    # -----------------------------
    @staticmethod
    def _window_to_tabular_features(X: np.ndarray, window: int) -> np.ndarray:
        """Turn a sequence into tabular features for classic ML WITHOUT huge intermediate tensors.

        Output shape: (T-window+1, 6*F) where we concatenate:
        mean, std (ddof=0), min, max, range, energy(sum of squares)
        """
        if window <= 1:
            return X.astype(np.float32, copy=False)

        Xf = np.asarray(X, dtype=np.float32)
        T, F = Xf.shape
        if T < window:
            raise ValueError(f"Not enough rows for windowing: T={T}, window={window}")

        # ---- mean/std/energy via cumulative sums ----
        c1 = np.cumsum(Xf, axis=0, dtype=np.float64)
        c2 = np.cumsum(Xf * Xf, axis=0, dtype=np.float64)

        pad0 = np.zeros((1, F), dtype=np.float64)
        sum_w  = c1[window - 1:] - np.concatenate([pad0, c1[:-window]], axis=0)
        sum2_w = c2[window - 1:] - np.concatenate([pad0, c2[:-window]], axis=0)

        mean = (sum_w / float(window)).astype(np.float32)
        ms2  = (sum2_w / float(window)).astype(np.float32)
        var  = np.maximum(ms2 - (mean * mean), 0.0)
        std  = np.sqrt(var).astype(np.float32)

        eng = sum2_w.astype(np.float32)

        # ---- min/max via rolling ----
        dfX = pd.DataFrame(Xf)
        mn = dfX.rolling(window=window, min_periods=window).min().to_numpy(dtype=np.float32)[window - 1:]
        mx = dfX.rolling(window=window, min_periods=window).max().to_numpy(dtype=np.float32)[window - 1:]
        rng = (mx - mn).astype(np.float32)

        Xw = np.concatenate([mean, std, mn, mx, rng, eng], axis=1).astype(np.float32, copy=False)
        return Xw

    # -----------------------------
    # Splitting
    # -----------------------------
    def _split_data(self, X, y, split_method="random", groups=None, test_group=None, test_size=0.2):
        n = len(X)
        if n < 10:
            raise ValueError("Dataset too small for splitting.")

        if split_method == "all":
            return X, X, y, y

        if split_method == "time":
            cut = int((1 - test_size) * n)
            return X[:cut], X[cut:], y[:cut], y[cut:]

        if split_method == "group":
            if groups is None:
                raise ValueError("group split requires group_col to exist in the dataset.")
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            tr_idx, te_idx = next(gss.split(X, y, groups=groups))
            return X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

        if split_method == "loco":
            if groups is None:
                raise ValueError("LOCO split requires group_col to exist in the dataset.")
            if test_group is None:
                raise ValueError("LOCO split requires a selected test_group.")
            test_mask = (groups == test_group)
            if int(test_mask.sum()) == 0:
                raise ValueError(f"test_group={test_group} was not found in the dataset groups.")
            X_train, y_train = X[~test_mask], y[~test_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            return X_train, X_test, y_train, y_test

        strat = y if _can_stratify(y, test_size) else None
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=strat,
        )

    # ============================================================
    # MAIN TRAIN
    # ============================================================
    def train_model(
        self,
        algo_name: str,
        data_path: str,
        device_name: str,
        hp_epochs: int = 5,
        hp_batch: int = 32,
        hp_est: int = 50,
        window: int = 1,
        classic_window: int | None = None,
        split_method: str = "random",
        group_col: str | None = None,
        test_group=None,
        dataset_key: str | None = None,
    ):
        df = pd.read_csv(data_path)
        label_col = self._pick_label_col(df)

        # if user explicitly passed group_col, capture groups before filtering
        groups = None
        if group_col and group_col in df.columns:
            groups = df[group_col].astype(str).to_numpy(dtype=object)

        if test_group is not None:
            test_group = str(test_group)

        # 4-class mapping (keeps only valid rows)
        df_m, y_std, meta = _map_to_4class(df, label_col=label_col, dataset_key=dataset_key)

        # align passed-in groups to df_m after filtering
        if groups is not None:
            groups = groups[df_m.index.to_numpy()]

        # auto-detect group_col if user didn't pass it
        if groups is None and (group_col is None or group_col == ""):
            auto_gc = self._auto_pick_group_col(df_m)
            if auto_gc is not None:
                group_col = auto_gc
                groups = df_m[group_col].astype(str).to_numpy(dtype=object)
                gser = pd.Series(groups)
                print(f"[AUTO] detected group_col='{group_col}' nunique={gser.nunique()} sample={gser.unique()[:5]}")

        # sort BEFORE windowing/splitting (group-wise when groups exist)
        df_m, y_std, groups = self._sort_by_datetime(df_m, y_std, groups, meta.get("dataset_key"))

        # drop label, group, time, and obvious leakage columns from features
        drop_cols = {label_col}
        if group_col and group_col in df_m.columns:
            drop_cols.add(group_col)

        # time columns are ONLY for sorting, never features
        for c in TIME_COL_CANDIDATES:
            if c in df_m.columns:
                drop_cols.add(c)

        # leakage columns (duplicate label encodings / fine-grained labels)
        LEAK_COLS = {
            # common numeric/duplicate label encodings
            "behavior", "Behavior",
            "behaviour", "Behaviour",   # <-- IMPORTANT for AcTBeCalf (e.g., eating_concentrates)
            "activity_code", "ActivityCode",
            "label_id", "LabelID",
            "class_id", "ClassID",

            # other common label-like columns that may leak the answer
            "activity", "Activity",
            "target", "Target",
            "label", "Label",  # note: you already drop label_col, this is extra safety if duplicates exist
        }

        for c in LEAK_COLS:
            if c in df_m.columns:
                drop_cols.add(c)

        feat_cols = [c for c in df_m.columns if c not in drop_cols]
        if len(feat_cols) == 0:
            raise ValueError("No feature columns found. Check label/group/time/leak column names.")

        X = self._coerce_numeric(df_m[feat_cols])

        # FIXED 4-class encoding (0..3)
        y = _encode_fixed_4(y_std)
        n_classes = 4

        if len(np.unique(y)) < 2:
            raise ValueError(f"Not enough classes after 4-class mapping. present={np.unique(y).tolist()}")

        logic = FULL_ALGO_MAP.get(algo_name)
        if logic is None:
            raise ValueError(
                f"Unknown algo_name='{algo_name}'. Not found in FULL_ALGO_MAP. "
                "Fix core.registry mapping or the Streamlit algo list to match."
            )

        # window routing: sequence window for CNN/LSTM, classic window for ML
        seq_window = int(window)
        cls_window = int(classic_window) if classic_window is not None else int(window)
        if cls_window < 1:
            cls_window = 1
        if seq_window < 1:
            seq_window = 1

        print(f"[RUN] algo={algo_name} logic={logic} seq_window={seq_window} cls_window={cls_window} split={split_method} test_group={test_group}")
        print(f"[DATA] rows={len(df_m)} kept_ratio={meta.get('kept_ratio'):.3f} dataset_key={meta.get('dataset_key')}")

        # ============================================================
        # WINDOWING
        # ============================================================
        # Sequence models windowing: (N, T, F)
        if logic in ["rnn", "cnn"] and seq_window > 1:
            if groups is not None:
                X_all, y_all, g_all = [], [], []
                for gid in np.unique(groups):
                    mask = (groups == gid)
                    Xg, yg = X[mask], y[mask]
                    if len(Xg) < seq_window:
                        continue
                    Xs, ys, gs = self._build_sequences(
                        Xg, yg, seq_window, groups=np.full(len(yg), gid, dtype=object)
                    )
                    if len(Xs) > 0:
                        X_all.append(Xs)
                        y_all.append(ys)
                        g_all.append(gs)
                if not X_all:
                    raise ValueError("No sequences created. Sequence window may be too large for each group.")
                X = np.concatenate(X_all, axis=0)
                y = np.concatenate(y_all, axis=0)
                groups = np.concatenate(g_all, axis=0)
            else:
                X, y, _ = self._build_sequences(X, y, seq_window, groups=None)

        # Classic ML windowing -> tabular stats
        if logic not in ["rnn", "cnn"] and cls_window > 1:
            if groups is not None:
                X_all, y_all, g_all = [], [], []
                for gid in np.unique(groups):
                    mask = (groups == gid)
                    Xg, yg = X[mask], y[mask]
                    # IMPORTANT: classic ML uses cls_window, not seq_window
                    if len(Xg) < cls_window:
                        continue
                    Xw = self._window_to_tabular_features(Xg, cls_window)
                    yw = yg[cls_window - 1:]
                    X_all.append(Xw)
                    y_all.append(yw)
                    g_all.append(np.full(len(yw), gid, dtype=object))
                if not X_all:
                    raise ValueError("No windowed samples created for classic ML. Classic window may be too large per group.")
                X = np.concatenate(X_all, axis=0)
                y = np.concatenate(y_all, axis=0)
                groups = np.concatenate(g_all, axis=0)
            else:
                X = self._window_to_tabular_features(X, cls_window)
                y = y[cls_window - 1:]

        # ============================================================
        # Inner runner: train + export + metrics
        # ============================================================
        def _run_once(split_m: str, test_g: str | None, split_label: str, file_suffix: str):
            # Split ("all" => train on all, test on same)
            X_train, X_test, y_train, y_test = self._split_data(
                X, y, split_method=split_m, groups=groups, test_group=test_g, test_size=0.2
            )

            model = None
            model_size = 0
            start = time.time()

            # -------------------------
            # Deep Learning (TF)
            # -------------------------
            if logic in ["dnn", "cnn", "rnn"]:
                if logic == "dnn":
                    dim = X_train.shape[1]
                    Xtr, Xte = X_train, X_test
                    model = tf.keras.Sequential([
                        tf.keras.layers.Dense(64, activation="relu", input_shape=(dim,)),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(32, activation="relu"),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(n_classes, activation="softmax"),
                    ])
                else:
                    if X_train.ndim != 3:
                        raise ValueError("Sequence models require (N, T, F). Increase window > 1.")
                    T = X_train.shape[1]
                    F = X_train.shape[2]
                    Xtr, Xte = X_train, X_test

                    norm = tf.keras.layers.Normalization(axis=-1)
                    # adapt can be memory heavy; batching helps
                    try:
                        norm.adapt(Xtr, batch_size=max(128, min(2048, Xtr.shape[0])))
                    except Exception:
                        norm.adapt(Xtr)

                    if logic == "cnn":
                        model = tf.keras.Sequential([
                            tf.keras.layers.Input(shape=(T, F)),
                            norm,
                            tf.keras.layers.Conv1D(32, 3, activation="relu"),
                            tf.keras.layers.MaxPooling1D(2),
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(64, activation="relu"),
                            tf.keras.layers.Dropout(0.2),
                            tf.keras.layers.Dense(n_classes, activation="softmax"),
                        ])
                    else:
                        model = tf.keras.Sequential([
                            tf.keras.layers.Input(shape=(T, F)),
                            norm,
                            tf.keras.layers.LSTM(64),
                            tf.keras.layers.Dropout(0.2),
                            tf.keras.layers.Dense(n_classes, activation="softmax"),
                        ])

                # class_weight (robust to missing classes)
                class_weight = None
                try:
                    present = np.unique(y_train)
                    weights = compute_class_weight(class_weight="balanced", classes=present, y=y_train)
                    class_weight = {int(c): float(w) for c, w in zip(present, weights)}
                except Exception:
                    class_weight = None

                model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                model.fit(Xtr, y_train, epochs=hp_epochs, batch_size=hp_batch, verbose=0, class_weight=class_weight)

                prob = model.predict(Xte, verbose=0)
                preds = np.argmax(prob, axis=1).astype(int)

                out = os.path.abspath(os.path.join(self.models_dir, f"{algo_name}_{file_suffix}.tflite"))

                try:
                    converter = tf.lite.TFLiteConverter.from_keras_model(model)
                    converter.target_spec.supported_ops = [
                        tf.lite.OpsSet.TFLITE_BUILTINS,
                        tf.lite.OpsSet.SELECT_TF_OPS
                    ]
                    converter._experimental_lower_tensor_list_ops = False
                    tflite_model = converter.convert()
                    with open(out, "wb") as f:
                        f.write(tflite_model)
                    model_size = os.path.getsize(out)
                    print(f"[TFLite Saved] {out} ({model_size / 1024:.2f} KB)")
                except Exception as e:
                    print(f"[TFLite Export Failed] algo={algo_name} reason={e}")
                    model_size = 0

                approx_flops = float(getattr(model, "count_params", lambda: 10000)()) * 2.0
                artifact_path = out

                # important for full_loco RAM stability
                tf.keras.backend.clear_session()
                gc.collect()

            # -------------------------
            # Classic ML + XGBoost
            # -------------------------
            else:
                sw_train = None
                try:
                    present = np.unique(y_train)
                    w = compute_class_weight(class_weight="balanced", classes=present, y=y_train)
                    w_map = {int(c): float(v) for c, v in zip(present, w)}
                    sw_train = np.asarray([w_map[int(yy)] for yy in y_train], dtype=np.float32)
                except Exception:
                    sw_train = None

                if algo_name == "SVM":
                    model = make_pipeline(
                        StandardScaler(with_mean=True, with_std=True),
                        LinearSVC(dual="auto", max_iter=10000)
                    )
                elif algo_name == "KNN":
                    model = KNeighborsClassifier(n_neighbors=5)
                elif algo_name == "DecisionTree":
                    model = DecisionTreeClassifier(random_state=42)
                elif algo_name == "NaiveBayes":
                    model = GaussianNB()
                elif algo_name in ["XGBoost", "XGBoost-Tiny"]:
                    if not _HAS_XGBOOST:
                        raise RuntimeError("xgboost is not installed. Run: pip install xgboost")

                    if algo_name == "XGBoost":
                        n_estimators, max_depth, lr, subs, cols = 300, 6, 0.05, 0.9, 0.9
                    else:
                        n_estimators, max_depth, lr, subs, cols = 60, 3, 0.10, 0.8, 0.8

                    model = XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=lr,
                        subsample=subs,
                        colsample_bytree=cols,
                        reg_lambda=1.0,
                        tree_method="hist",
                        eval_metric="mlogloss",
                        n_jobs=-1,
                        random_state=42,
                        verbosity=0,
                        objective="multi:softprob",
                        num_class=n_classes,
                    )
                elif algo_name == "RandomForest":
                    model = RandomForestClassifier(n_estimators=hp_est, random_state=42)
                else:
                    raise ValueError(
                        f"Classic algo_name='{algo_name}' is not implemented in trainer.py "
                        f"(logic={logic}). Add it explicitly or fix the Streamlit algo name."
                    )

                # Fit with sample weights where supported
                if algo_name == "SVM" and sw_train is not None:
                    model.fit(X_train, y_train, linearsvc__sample_weight=sw_train)
                elif algo_name in ["XGBoost", "XGBoost-Tiny"] and sw_train is not None:
                    model.fit(X_train, y_train, sample_weight=sw_train)
                elif sw_train is not None and algo_name not in ["KNN"]:
                    try:
                        model.fit(X_train, y_train, sample_weight=sw_train)
                    except TypeError:
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)

                preds = model.predict(X_test)

                out = os.path.abspath(os.path.join(self.models_dir, f"{algo_name}_{file_suffix}.joblib"))
                try:
                    joblib.dump(model, out)
                    model_size = os.path.getsize(out)
                except Exception:
                    model_size = 0

                approx_flops = float(X_train.shape[1] * 2000)
                artifact_path = out

            end = time.time()

            # ============================================================
            # Metrics (STRICT 4-class)
            # ============================================================
            per_f1 = f1_score(y_test, preds, average=None, labels=FIXED_LABELS, zero_division=0)
            per_class_f1 = {
                "F1 (eating)": float(per_f1[0]),
                "F1 (ruminating)": float(per_f1[1]),
                "F1 (standing)": float(per_f1[2]),
                "F1 (walking)": float(per_f1[3]),
            }

            acc = float(accuracy_score(y_test, preds))
            macro_p = float(precision_score(y_test, preds, average="macro", labels=FIXED_LABELS, zero_division=0))
            macro_r = float(recall_score(y_test, preds, average="macro", labels=FIXED_LABELS, zero_division=0))
            macro_f = float(f1_score(y_test, preds, average="macro", labels=FIXED_LABELS, zero_division=0))
            w_f = float(f1_score(y_test, preds, average="weighted", labels=FIXED_LABELS, zero_division=0))

            hw = self.profiler.profile(float(approx_flops), device_name)

            return {
                "Algorithm": algo_name,
                "Split": split_label,
                "Test Group": (str(test_g) if split_label == "loco" and test_g is not None else ""),

                "Accuracy": acc,
                "Precision (Macro)": macro_p,
                "Recall (Macro)": macro_r,
                "F1 (Macro)": macro_f,
                "F1 (Weighted)": w_f,
                "Training Time (ms)": float((end - start) * 1000),

                **hw,
                **per_class_f1,
            }

        # ============================================================
        # full_loco: run LOCO for every group + export one *_final model
        # ============================================================
        if split_method == "full_loco":
            if groups is None:
                raise ValueError("full_loco requires a detected/provided group_col in the dataset.")

            uniq = sorted(pd.Series(groups).dropna().astype(str).unique().tolist())
            results = []
            for gid in uniq:
                print(f"[FULL_LOCO] test_group={gid}")
                res = _run_once(split_m="loco", test_g=str(gid), split_label="loco", file_suffix=f"loco_{gid}")
                results.append(res)

            # Final deployment model trained on ALL data
            print("[FULL_LOCO] training final deployment model on ALL data")
            res_final = _run_once(split_m="all", test_g=None, split_label="final", file_suffix="final")
            results.append(res_final)
            return results

        # ============================================================
        # all: no split; directly train final deployment model on ALL data
        # ============================================================
        if split_method == "all":
            return _run_once(split_m="all", test_g=None, split_label="all", file_suffix="final")

        # ============================================================
        # normal single run (random/time/group/loco)
        # ============================================================
        if split_method == "loco" and test_group is None:
            raise ValueError("split_method='loco' requires test_group")

        file_suffix = split_method if split_method != "loco" else f"loco_{test_group}"
        return _run_once(split_m=split_method, test_g=test_group, split_label=split_method, file_suffix=file_suffix)

