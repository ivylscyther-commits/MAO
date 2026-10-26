"""Data processing, model training, retrieval, and trial database for the MAO app.

The default model uses 11 controllable process settings. A separate current-aware
model uses those 11 settings plus initial/final current for post-trial analysis.
"""
from __future__ import annotations

import json
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

CORE_FEATURES = [
    "KOH (g/l)",
    "Na₃PO₄ (g/l)",
    "Na₂SiO₃ (g/l)",
    "Voltage (V) +ve",
    "Voltage (V) -ve",
    "Frequency (Hz)",
    "Time (min)",
    "polarity(unipolar-1/bipolar-2)",
    "duty cycle +ve (%)",
    "duty cycle -ve (%)",
    "basic duty cycle (%)",
]
CURRENT_FEATURES = ["initial current", "final current"]
ALL_FEATURES = CORE_FEATURES + CURRENT_FEATURES
THICKNESS_TARGETS = ["Thickness Min (μm)", "Thickness Max (μm)"]
ROUGHNESS_TARGETS = ["Ra Value Min (μm)", "Ra Value Max (μm)"]
ALL_TARGETS = THICKNESS_TARGETS + ROUGHNESS_TARGETS
NUMERIC_COLUMNS = ALL_FEATURES + ALL_TARGETS
REMARK_COLUMN = "Remark"
MODE_LABELS = {"process": "Before-trial estimate (process settings)", "current": "After-trial analysis (process settings + measured currents)"}
FAMILIES = {"Thickness": THICKNESS_TARGETS, "Ra": ROUGHNESS_TARGETS}


def normalize_dataset(frame: pd.DataFrame, source: str = "Historical") -> pd.DataFrame:
    """Return a predictable schema without mutating the user's source workbook."""
    df = frame.copy()
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if REMARK_COLUMN not in df.columns:
        df[REMARK_COLUMN] = ""
    df[REMARK_COLUMN] = df[REMARK_COLUMN].fillna("").astype(str)
    if "_source_label" not in df.columns:
        df["_source_label"] = [f"{source} #{i + 1}" for i in range(len(df))]
    else:
        df["_source_label"] = df["_source_label"].fillna(source).astype(str)
    return df.reset_index(drop=True)


def read_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place 'MAO DATASET.xlsx' beside app.py or upload it in the app."
        )
    return normalize_dataset(pd.read_excel(path), "Historical")


def ensure_database(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS trials ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "created_at TEXT NOT NULL, payload_json TEXT NOT NULL, remark TEXT NOT NULL DEFAULT ''"
            ")"
        )
        con.commit()


def list_trials(path: str | Path) -> list[dict[str, Any]]:
    ensure_database(path)
    with sqlite3.connect(path) as con:
        rows = con.execute(
            "SELECT id, created_at, payload_json, remark FROM trials ORDER BY id DESC"
        ).fetchall()
    out = []
    for trial_id, created_at, payload_json, remark in rows:
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            payload = {}
        out.append({"id": trial_id, "created_at": created_at, "payload": payload, "remark": remark})
    return out


def save_trial(path: str | Path, payload: dict[str, Any], remark: str = "") -> int:
    ensure_database(path)
    safe_payload = {}
    for key in NUMERIC_COLUMNS:
        val = payload.get(key)
        safe_payload[key] = None if val is None or (isinstance(val, float) and not np.isfinite(val)) else float(val)
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with sqlite3.connect(path) as con:
        cursor = con.execute(
            "INSERT INTO trials (created_at, payload_json, remark) VALUES (?, ?, ?)",
            (timestamp, json.dumps(safe_payload, ensure_ascii=False), str(remark or "")),
        )
        con.commit()
        return int(cursor.lastrowid)


def update_trial(path: str | Path, trial_id: int, payload_updates: dict[str, Any], remark: str | None = None) -> bool:
    ensure_database(path)
    with sqlite3.connect(path) as con:
        row = con.execute("SELECT payload_json, remark FROM trials WHERE id = ?", (int(trial_id),)).fetchone()
        if not row:
            return False
        payload = json.loads(row[0])
        for key, value in payload_updates.items():
            if key not in NUMERIC_COLUMNS:
                continue
            payload[key] = None if value is None else float(value)
        new_remark = row[1] if remark is None else str(remark)
        con.execute(
            "UPDATE trials SET payload_json = ?, remark = ? WHERE id = ?",
            (json.dumps(payload, ensure_ascii=False), new_remark, int(trial_id)),
        )
        con.commit()
    return True


def trials_to_frame(path: str | Path) -> pd.DataFrame:
    records = []
    for item in list_trials(path):
        row = dict(item["payload"])
        row[REMARK_COLUMN] = item["remark"]
        row["_source_label"] = f"Database trial #{item['id']}"
        row["_trial_id"] = item["id"]
        row["_created_at"] = item["created_at"]
        records.append(row)
    if not records:
        return normalize_dataset(pd.DataFrame(columns=NUMERIC_COLUMNS + [REMARK_COLUMN]), "Database")
    return normalize_dataset(pd.DataFrame(records), "Database")


def combine_training_data(base_df: pd.DataFrame, trial_df: pd.DataFrame | None = None) -> pd.DataFrame:
    base = normalize_dataset(base_df, "Historical")
    if trial_df is None or trial_df.empty:
        return base
    extra = normalize_dataset(trial_df, "Database")
    return pd.concat([base, extra], ignore_index=True, sort=False)


def _group_ids(X: pd.DataFrame) -> np.ndarray:
    # Repeated identical settings stay in the same CV fold to reduce duplicate leakage.
    rounded = X.copy()
    for col in rounded.columns:
        rounded[col] = rounded[col].map(
            lambda v: "<MISSING>" if pd.isna(v) else str(round(float(v), 6))
        )
    return rounded.astype(str).agg("|".join, axis=1).to_numpy()


def _make_splits(X: pd.DataFrame):
    if len(X) < 3:
        raise ValueError("At least three usable records are required for model validation.")
    groups = _group_ids(X)
    group_count = len(np.unique(groups))
    if group_count >= 3:
        splitter = GroupKFold(n_splits=min(5, group_count))
        return list(splitter.split(X, groups=groups))
    splitter = KFold(n_splits=min(3, len(X)), shuffle=True, random_state=42)
    return list(splitter.split(X))


def _candidate_estimators() -> dict[str, Pipeline]:
    return {
        "Extra Trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", ExtraTreesRegressor(
                n_estimators=160, min_samples_leaf=2, max_features=1.0,
                random_state=42, n_jobs=-1
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", RandomForestRegressor(
                n_estimators=160, min_samples_leaf=2, max_features=1.0,
                random_state=42, n_jobs=-1
            )),
        ]),
        "SVR": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", MultiOutputRegressor(SVR(C=10.0, epsilon=0.1, kernel="rbf"))),
        ]),
    }


def _pair_data(df: pd.DataFrame, feature_cols: list[str], pair_cols: list[str], mode: str):
    data = df.copy()
    data[pair_cols[0]] = pd.to_numeric(data[pair_cols[0]], errors="coerce")
    data[pair_cols[1]] = pd.to_numeric(data[pair_cols[1]], errors="coerce")
    valid = data[pair_cols].notna().all(axis=1)
    valid &= data[pair_cols[0]] >= 0
    valid &= data[pair_cols[1]] >= data[pair_cols[0]]
    if mode == "current":
        valid &= data[CURRENT_FEATURES].notna().all(axis=1)
    data = data.loc[valid].copy()
    X = data[feature_cols].astype(float)
    low = data[pair_cols[0]].to_numpy(dtype=float)
    high = data[pair_cols[1]].to_numpy(dtype=float)
    y = np.column_stack([(low + high) / 2.0, (high - low) / 2.0])
    return data, X, y, low, high


def _bounds_from_predictions(mid_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    midpoint = mid_delta[:, 0]
    delta = np.maximum(mid_delta[:, 1], 0.0)
    low = np.maximum(midpoint - delta, 0.0)
    high = np.maximum(midpoint + delta, low)
    return low, high


class ModelSuite:
    """Trained model collection, cross-validation evidence, and retrieval index."""

    def __init__(self, df: pd.DataFrame):
        self.data = normalize_dataset(df)
        self.models: dict[str, dict[str, Pipeline]] = {"process": {}, "current": {}}
        self.model_names: dict[str, dict[str, str]] = {"process": {}, "current": {}}
        self.metrics_rows: list[dict[str, Any]] = []
        self.validation_frames: list[pd.DataFrame] = []
        self.uncertainty: dict[str, dict[str, dict[str, float]]] = {"process": {}, "current": {}}
        self.train_counts: dict[str, dict[str, int]] = {"process": {}, "current": {}}
        self.bounds: dict[str, dict[str, tuple[float, float]]] = {"process": {}, "current": {}}
        self.neighbor_preprocessors: dict[str, Pipeline] = {}
        self.neighbor_models: dict[str, NearestNeighbors] = {}
        self.neighbor_frames: dict[str, pd.DataFrame] = {}
        self.novelty_thresholds: dict[str, float] = {}
        self.data_fingerprint = self._fingerprint()
        self._fit_all()
        self._build_retrieval_indexes()

    def _fingerprint(self) -> str:
        cols = [c for c in NUMERIC_COLUMNS if c in self.data.columns]
        payload = pd.util.hash_pandas_object(self.data[cols], index=False).values.tobytes()
        return hashlib.sha256(payload).hexdigest()

    def _fit_all(self) -> None:
        for mode, features in (("process", CORE_FEATURES), ("current", ALL_FEATURES)):
            for family, target_cols in FAMILIES.items():
                data, X, y, actual_low, actual_high = _pair_data(self.data, features, target_cols, mode)
                if len(data) < 20:
                    raise ValueError(
                        f"Not enough usable rows to train {family.lower()} model in {MODE_LABELS[mode]}."
                    )
                splits = _make_splits(X)
                family_results = []
                best = None
                best_score = float("inf")
                best_oof = None
                for name, template in _candidate_estimators().items():
                    oof = np.full((len(X), 2), np.nan, dtype=float)
                    for train_idx, test_idx in splits:
                        estimator = clone(template)
                        estimator.fit(X.iloc[train_idx], y[train_idx])
                        oof[test_idx] = estimator.predict(X.iloc[test_idx])
                    pred_low, pred_high = _bounds_from_predictions(oof)
                    mae_low = float(mean_absolute_error(actual_low, pred_low))
                    mae_high = float(mean_absolute_error(actual_high, pred_high))
                    score = (mae_low + mae_high) / 2.0
                    r2_low = float(r2_score(actual_low, pred_low)) if len(actual_low) > 1 else float("nan")
                    r2_high = float(r2_score(actual_high, pred_high)) if len(actual_high) > 1 else float("nan")
                    family_results.append({
                        "mode": mode, "Mode": MODE_LABELS[mode], "family": family,
                        "Model": name, "MAE Low": mae_low, "MAE High": mae_high,
                        "Mean Endpoint MAE": score, "R² Low": r2_low, "R² High": r2_high,
                        "Validation rows": len(data), "selected": False,
                    })
                    if score < best_score:
                        best_score = score
                        best = (name, template)
                        best_oof = oof.copy()
                assert best is not None and best_oof is not None
                for row in family_results:
                    row["selected"] = row["Model"] == best[0]
                    self.metrics_rows.append(row)
                final_model = clone(best[1])
                final_model.fit(X, y)
                self.models[mode][family] = final_model
                self.model_names[mode][family] = best[0]
                self.train_counts[mode][family] = len(data)
                pred_low, pred_high = _bounds_from_predictions(best_oof)
                endpoint_frames = []
                endpoint_map = [
                    (target_cols[0], actual_low, pred_low),
                    (target_cols[1], actual_high, pred_high),
                ]
                family_uncertainty: dict[str, dict[str, float]] = {}
                for target, actual, predicted in endpoint_map:
                    errors = np.abs(actual - predicted)
                    q90 = float(np.quantile(errors, 0.90))
                    mae = float(mean_absolute_error(actual, predicted))
                    r2 = float(r2_score(actual, predicted))
                    family_uncertainty[target] = {"MAE": mae, "Q90 absolute error": q90, "R²": r2}
                    endpoint_frames.append(pd.DataFrame({
                        "mode": mode, "Mode": MODE_LABELS[mode], "family": family,
                        "outcome": target, "Actual": actual, "Predicted": predicted,
                        "Absolute error": errors, "Model": best[0],
                    }))
                self.uncertainty[mode][family] = family_uncertainty
                self.validation_frames.extend(endpoint_frames)

            support_data = self.data
            if mode == "current":
                support_data = support_data.loc[support_data[CURRENT_FEATURES].notna().all(axis=1)]
            for feat in features:
                values = pd.to_numeric(support_data[feat], errors="coerce").dropna()
                if not values.empty:
                    self.bounds[mode][feat] = (float(values.min()), float(values.max()))

        self.metrics = pd.DataFrame(self.metrics_rows)
        self.validation_predictions = (
            pd.concat(self.validation_frames, ignore_index=True) if self.validation_frames
            else pd.DataFrame(columns=["mode", "Mode", "family", "outcome", "Actual", "Predicted", "Absolute error", "Model"])
        )

    def _build_retrieval_indexes(self) -> None:
        for mode, features in (("process", CORE_FEATURES), ("current", ALL_FEATURES)):
            eligible = self.data.dropna(subset=features).copy()
            if mode == "current":
                eligible = eligible.loc[eligible[CURRENT_FEATURES].notna().all(axis=1)].copy()
            if len(eligible) < 5:
                continue
            X = eligible[features].astype(float)
            pre = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ])
            transformed = pre.fit_transform(X)
            nn = NearestNeighbors(metric="euclidean", algorithm="auto")
            nn.fit(transformed)
            self.neighbor_preprocessors[mode] = pre
            self.neighbor_models[mode] = nn
            self.neighbor_frames[mode] = eligible.reset_index(drop=True)
            # Leave-one-out nearest distance is the second neighbour because first is itself.
            kn = min(2, len(transformed))
            distances, _ = nn.kneighbors(transformed, n_neighbors=kn)
            nearest_other = distances[:, 1] if kn > 1 else distances[:, 0]
            self.novelty_thresholds[mode] = float(np.quantile(nearest_other, 0.95))

    def predict(self, values: dict[str, Any], mode: str = "process") -> dict[str, float]:
        features = CORE_FEATURES if mode == "process" else ALL_FEATURES
        row = {name: values.get(name, np.nan) for name in features}
        X = pd.DataFrame([row], columns=features).astype(float)
        result: dict[str, float] = {}
        for family, targets in FAMILIES.items():
            mid_delta = np.asarray(self.models[mode][family].predict(X), dtype=float).reshape(1, 2)
            low, high = _bounds_from_predictions(mid_delta)
            result[targets[0]] = float(low[0])
            result[targets[1]] = float(high[0])
        return result

    def range_warnings(self, values: dict[str, Any], mode: str = "process") -> list[str]:
        features = CORE_FEATURES if mode == "process" else ALL_FEATURES
        warnings = []
        for feat in features:
            if feat not in self.bounds[mode] or values.get(feat) is None:
                continue
            try:
                value = float(values[feat])
            except (TypeError, ValueError):
                continue
            lo, hi = self.bounds[mode][feat]
            if value < lo or value > hi:
                warnings.append(f"{feat} = {value:g} is outside the observed training range [{lo:g}, {hi:g}].")
        distance = self.nearest_distance(values, mode)
        threshold = self.novelty_thresholds.get(mode)
        if distance is not None and threshold is not None and distance > threshold:
            warnings.append(
                f"This combination is unusual: normalized nearest-trial distance {distance:.2f} exceeds the historical 95th-percentile reference {threshold:.2f}."
            )
        return warnings

    def nearest_distance(self, values: dict[str, Any], mode: str = "process") -> float | None:
        if mode not in self.neighbor_preprocessors or mode not in self.neighbor_models:
            return None
        features = CORE_FEATURES if mode == "process" else ALL_FEATURES
        X = pd.DataFrame([{f: values.get(f, np.nan) for f in features}], columns=features).astype(float)
        transformed = self.neighbor_preprocessors[mode].transform(X)
        distance, _ = self.neighbor_models[mode].kneighbors(transformed, n_neighbors=1)
        return float(distance[0, 0])

    def similar_trials(self, values: dict[str, Any], mode: str = "process", n: int = 5) -> pd.DataFrame:
        if mode not in self.neighbor_preprocessors or mode not in self.neighbor_models:
            return pd.DataFrame()
        features = CORE_FEATURES if mode == "process" else ALL_FEATURES
        X = pd.DataFrame([{f: values.get(f, np.nan) for f in features}], columns=features).astype(float)
        transformed = self.neighbor_preprocessors[mode].transform(X)
        k = min(max(1, int(n)), len(self.neighbor_frames[mode]))
        distances, indices = self.neighbor_models[mode].kneighbors(transformed, n_neighbors=k)
        result = self.neighbor_frames[mode].iloc[indices[0]].copy()
        result.insert(0, "Similarity distance", distances[0])
        display_cols = ["_source_label", "Similarity distance"] + features + ALL_TARGETS + [REMARK_COLUMN]
        return result[[c for c in display_cols if c in result.columns]].reset_index(drop=True)

    def outlier_rows(self, contamination: float = 0.05) -> pd.DataFrame:
        cols = CORE_FEATURES + ALL_TARGETS
        source = self.data.copy()
        subset = source.loc[source[ALL_TARGETS].notna().sum(axis=1) >= 2].copy()
        if len(subset) < 20:
            return pd.DataFrame()
        X = subset[cols].replace([np.inf, -np.inf], np.nan)
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", IsolationForest(contamination=contamination, random_state=42, n_estimators=200)),
        ])
        labels = pipeline.fit_predict(X)
        subset["Outlier flag"] = np.where(labels == -1, "Review", "Typical")
        subset["Outlier score"] = pipeline.named_steps["model"].decision_function(
            pipeline.named_steps["imputer"].transform(X)
        )
        return subset.loc[subset["Outlier flag"] == "Review"].sort_values("Outlier score")

    def target_search(
        self,
        target_t_min: float,
        target_t_max: float,
        target_ra_min: float,
        target_ra_max: float,
        n_candidates: int = 600,
        seed: int = 73,
    ) -> pd.DataFrame:
        """Rank observed configurations and conservative one/two-factor perturbations.

        This is candidate screening, not an optimizer or a guarantee of a feasible
        or physically optimal MAO recipe.
        """
        rng = np.random.default_rng(seed)
        candidates_data = self.data.dropna(subset=CORE_FEATURES).copy()
        if candidates_data.empty:
            return pd.DataFrame()
        candidates_data = candidates_data.drop_duplicates(subset=CORE_FEATURES)
        source_rows = candidates_data[CORE_FEATURES].astype(float).to_dict("records")
        pool = list(source_rows)
        mutable = [f for f in CORE_FEATURES if f != "polarity(unipolar-1/bipolar-2)"]
        values_by_polarity: dict[float, dict[str, list[float]]] = {}
        for p in (1.0, 2.0):
            sub = candidates_data.loc[candidates_data["polarity(unipolar-1/bipolar-2)"] == p]
            values_by_polarity[p] = {}
            for feat in CORE_FEATURES:
                vals = pd.to_numeric(sub[feat], errors="coerce").dropna().unique().tolist()
                values_by_polarity[p][feat] = [float(v) for v in vals]
        mutations = max(0, int(n_candidates) - len(pool))
        attempts = 0
        while len(pool) < n_candidates and attempts < n_candidates * 5:
            attempts += 1
            base = dict(source_rows[int(rng.integers(0, len(source_rows)))])
            polarity = float(base.get("polarity(unipolar-1/bipolar-2)", 2.0))
            if polarity not in (1.0, 2.0):
                continue
            n_changes = int(rng.integers(1, 3))
            chosen = rng.choice(mutable, size=min(n_changes, len(mutable)), replace=False)
            for feat in chosen:
                observed = values_by_polarity.get(polarity, {}).get(feat, [])
                if observed:
                    base[feat] = float(rng.choice(observed))
            if polarity == 1.0:
                base["Voltage (V) -ve"] = 0.0
                base["duty cycle -ve (%)"] = 0.0
            pool.append(base)
        candidate_df = pd.DataFrame(pool).drop_duplicates(subset=CORE_FEATURES).reset_index(drop=True)
        prediction_rows = []
        for start in range(0, len(candidate_df), 256):
            batch = candidate_df.iloc[start:start + 256]
            for values in batch.to_dict("records"):
                pred = self.predict(values, "process")
                prediction_rows.append({**{f: values.get(f) for f in CORE_FEATURES}, **pred})
        ranked = pd.DataFrame(prediction_rows)
        scales = {
            THICKNESS_TARGETS[0]: max(abs(target_t_max - target_t_min), 5.0),
            THICKNESS_TARGETS[1]: max(abs(target_t_max - target_t_min), 5.0),
            ROUGHNESS_TARGETS[0]: max(abs(target_ra_max - target_ra_min), 1.0),
            ROUGHNESS_TARGETS[1]: max(abs(target_ra_max - target_ra_min), 1.0),
        }
        targets = {
            THICKNESS_TARGETS[0]: target_t_min,
            THICKNESS_TARGETS[1]: target_t_max,
            ROUGHNESS_TARGETS[0]: target_ra_min,
            ROUGHNESS_TARGETS[1]: target_ra_max,
        }
        penalties = [abs(ranked[col] - targets[col]) / scales[col] for col in targets]
        ranked["Target match score (lower is closer)"] = np.mean(np.column_stack(penalties), axis=1)
        ranked["Nearest-trial distance"] = [self.nearest_distance(r.to_dict(), "process") for _, r in ranked.iterrows()]
        ranked["Support warning"] = ranked["Nearest-trial distance"].map(
            lambda d: "Check support" if d is not None and d > self.novelty_thresholds.get("process", float("inf")) else "Within reference"
        )
        return ranked.sort_values("Target match score (lower is closer)").head(20).reset_index(drop=True)

    def save_artifacts(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data_fingerprint": self.data_fingerprint,
            "features": {"process": CORE_FEATURES, "current": ALL_FEATURES},
            "models": self.models,
            "model_names": self.model_names,
            "metrics": self.metrics,
            "uncertainty": self.uncertainty,
            "train_counts": self.train_counts,
        }
        joblib.dump(bundle, path)
