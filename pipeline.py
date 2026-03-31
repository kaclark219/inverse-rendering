import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSING_DIR = PROJECT_ROOT / "processing"

DATA_MASTER_CSV = DATA_DIR / "data_master.csv"
INVERSE_METADATA_CSV = DATA_DIR / "inverse_rendering_dataset" / "metadata.csv"
MASTER_WITH_PATHS_CSV = PROCESSING_DIR / "master_with_paths.csv"


def load_model_prefer_finetuned(base_name: str):
    finetuned = MODELS_DIR / f"{base_name}_finetuned.keras"
    base = MODELS_DIR / f"{base_name}.keras"

    if finetuned.exists():
        return keras.models.load_model(finetuned), finetuned.name
    if base.exists():
        return keras.models.load_model(base), base.name
    return None, None


def load_and_preprocess_image(path: str, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize(size, Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


def normalize_rows(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, eps, None)


def resolve_existing_path(row: pd.Series) -> str | None:
    candidates = []
    dataset_source = str(row.get("dataset_source", "")).strip()
    image_relpath = str(row.get("image_relpath", "")).strip()

    for raw in (row.get("resolved_image_relpath"), row.get("image_relpath")):
        if pd.isna(raw):
            continue
        raw = str(raw).strip()
        if not raw or raw.lower() == "nan":
            continue
        candidates.extend([
            raw,
            str(PROJECT_ROOT / raw),
            str(DATA_DIR / raw),
        ])

    if dataset_source == "render-lighting":
        candidates.append(str(DATA_DIR / "render-lighting" / image_relpath))
    elif dataset_source == "inverse_rendering_dataset":
        candidates.append(str(DATA_DIR / "inverse_rendering_dataset" / image_relpath))
        candidates.append(str(DATA_DIR / "inverse_rendering_dataset" / "images" / image_relpath))
    elif dataset_source == "spotlight-sphere-data":
        candidates.append(str(DATA_DIR / "spotlight-sphere-data" / image_relpath))
    elif dataset_source in {"two_object", "two_object_training"}:
        candidates.append(str(DATA_DIR / image_relpath))
        candidates.append(str(DATA_DIR / "two_object" / image_relpath))

    seen = set()
    for p in candidates:
        norm = os.path.normpath(p)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.exists(norm):
            return norm
    return None


def ensure_cam_space_single(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = [
        "light0_pos_cam_x", "light0_pos_cam_y", "light0_pos_cam_z",
        "light0_dir_cam_x", "light0_dir_cam_y", "light0_dir_cam_z",
    ]
    if set(required).issubset(out.columns) and out[required].notna().all().all():
        return out

    cam_pos = out[["cam_pos_x", "cam_pos_y", "cam_pos_z"]].to_numpy(dtype=np.float32)
    cam_right = normalize_rows(out[["cam_right_x", "cam_right_y", "cam_right_z"]].to_numpy(dtype=np.float32))
    cam_up = normalize_rows(out[["cam_up_x", "cam_up_y", "cam_up_z"]].to_numpy(dtype=np.float32))
    cam_forward = normalize_rows(out[["cam_forward_x", "cam_forward_y", "cam_forward_z"]].to_numpy(dtype=np.float32))
    cam_back = -cam_forward

    light_pos = out[["light0_pos_x", "light0_pos_y", "light0_pos_z"]].to_numpy(dtype=np.float32)
    light_dir = normalize_rows(out[["light0_dir_x", "light0_dir_y", "light0_dir_z"]].to_numpy(dtype=np.float32))
    rel = light_pos - cam_pos

    out["light0_pos_cam_x"] = np.einsum("ij,ij->i", rel, cam_right)
    out["light0_pos_cam_y"] = np.einsum("ij,ij->i", rel, cam_up)
    out["light0_pos_cam_z"] = np.einsum("ij,ij->i", rel, cam_back)

    out["light0_dir_cam_x"] = np.einsum("ij,ij->i", light_dir, cam_right)
    out["light0_dir_cam_y"] = np.einsum("ij,ij->i", light_dir, cam_up)
    out["light0_dir_cam_z"] = np.einsum("ij,ij->i", light_dir, cam_back)

    return out


def ensure_cam_space_tri(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    cam_pos = out[["cam_pos_x", "cam_pos_y", "cam_pos_z"]].to_numpy(dtype=np.float32)
    cam_right = normalize_rows(out[["cam_right_x", "cam_right_y", "cam_right_z"]].to_numpy(dtype=np.float32))
    cam_up = normalize_rows(out[["cam_up_x", "cam_up_y", "cam_up_z"]].to_numpy(dtype=np.float32))
    cam_forward = normalize_rows(out[["cam_forward_x", "cam_forward_y", "cam_forward_z"]].to_numpy(dtype=np.float32))
    cam_back = -cam_forward

    for i in range(3):
        pos_cols = [f"light{i}_pos_x", f"light{i}_pos_y", f"light{i}_pos_z"]
        dir_cols = [f"light{i}_dir_x", f"light{i}_dir_y", f"light{i}_dir_z"]

        if not set(pos_cols + dir_cols).issubset(out.columns):
            continue

        light_pos = out[pos_cols].to_numpy(dtype=np.float32)
        light_dir = normalize_rows(out[dir_cols].to_numpy(dtype=np.float32))
        rel = light_pos - cam_pos

        out[f"light{i}_pos_cam_x"] = np.einsum("ij,ij->i", rel, cam_right)
        out[f"light{i}_pos_cam_y"] = np.einsum("ij,ij->i", rel, cam_up)
        out[f"light{i}_pos_cam_z"] = np.einsum("ij,ij->i", rel, cam_back)

        out[f"light{i}_dir_cam_x"] = np.einsum("ij,ij->i", light_dir, cam_right)
        out[f"light{i}_dir_cam_y"] = np.einsum("ij,ij->i", light_dir, cam_up)
        out[f"light{i}_dir_cam_z"] = np.einsum("ij,ij->i", light_dir, cam_back)

    return out


def align_feature_width(df: pd.DataFrame, expected_dim: int) -> tuple[pd.DataFrame, list[str]]:
    current_dim = df.shape[1]
    if current_dim == expected_dim:
        return df, df.columns.tolist()

    if current_dim > expected_dim:
        trimmed = df.iloc[:, :expected_dim].copy()
        return trimmed, trimmed.columns.tolist()

    padded = df.copy()
    for i in range(expected_dim - current_dim):
        padded[f"__pad_{i}"] = 0.0
    return padded, padded.columns.tolist()


def build_angular_context(model):
    df = pd.read_csv(DATA_MASTER_CSV, low_memory=False)
    required_cols = [
        "num_active_lights",
        "light0_type",
        "cam_pos_x", "cam_pos_y", "cam_pos_z",
        "cam_right_x", "cam_right_y", "cam_right_z",
        "cam_up_x", "cam_up_y", "cam_up_z",
        "cam_forward_x", "cam_forward_y", "cam_forward_z",
        "light0_energy",
        "light0_color_r", "light0_color_g", "light0_color_b",
        "light0_spot_cone_deg",
        "light0_pos_x", "light0_pos_y", "light0_pos_z",
        "light0_dir_x", "light0_dir_y", "light0_dir_z",
    ]

    df["image_path"] = df.apply(resolve_existing_path, axis=1)
    df = df[df["image_path"].notna()].copy()
    df = df[df["num_active_lights"].fillna(0).astype(int) == 1].copy()
    df = df[df["light0_type"].astype(str).str.upper() == "SPOT"].copy()
    for col in required_cols:
        df = df[df[col].notna()]

    df = df.reset_index(drop=True)
    df = ensure_cam_space_single(df)
    df["light_type_label"] = df["light0_type"].astype(str).str.upper()

    target_cols = [
        "light0_pos_cam_x", "light0_pos_cam_y", "light0_pos_cam_z",
        "light0_dir_cam_x", "light0_dir_cam_y", "light0_dir_cam_z",
    ]

    exclude_cols = set(target_cols + [
        "image_relpath", "image_path", "camera_png",
        "light0_pos_x", "light0_pos_y", "light0_pos_z",
        "light0_dir_x", "light0_dir_y", "light0_dir_z",
    ])

    preprocess_path = MODELS_DIR / "angular_predictor_preprocessing.json"
    expected_tab_dim = model.input_shape[1][1]
    tab_mean = None
    tab_std = None
    feature_cols = None
    cat_cols = ["light_type_label"]
    target_means = None
    target_stds = None

    if preprocess_path.exists():
        with open(preprocess_path, "r", encoding="utf-8") as f:
            preprocess = json.load(f)
            saved_feature_cols = preprocess.get("feature_names", [])
            saved_feature_means = preprocess.get("feature_means", [])
            saved_feature_stds = preprocess.get("feature_stds", [])
            if len(saved_feature_cols) == expected_tab_dim:
                feature_cols = list(saved_feature_cols)
                tab_mean = np.array(saved_feature_means, dtype=np.float32).reshape(1, -1)
                tab_std = np.array(saved_feature_stds, dtype=np.float32).reshape(1, -1)
                tab_std[tab_std < 1e-8] = 1.0
            target_means = np.array(preprocess.get("target_means", []), dtype=np.float32)
            target_stds = np.array(preprocess.get("target_stds", []), dtype=np.float32)

    if feature_cols is None:
        base_feature_cols = [
            "num_active_lights",
            "cam_pos_x", "cam_pos_y", "cam_pos_z",
            "light0_energy",
            "light0_color_r", "light0_color_g", "light0_color_b",
            "light0_spot_cone_deg",
        ]
        cat_cols = ["light_type_label"]

        known_df = df[base_feature_cols + cat_cols].copy()
        known_df = pd.get_dummies(known_df, columns=cat_cols, drop_first=False)
        known_df, feature_cols = align_feature_width(known_df, expected_tab_dim)

        x_tab_raw = known_df.to_numpy(dtype=np.float32)
        idx = np.arange(len(df))
        idx_train, _ = train_test_split(idx, test_size=0.2, random_state=42)
        idx_train, _ = train_test_split(idx_train, test_size=0.2, random_state=42)

        tab_mean = x_tab_raw[idx_train].mean(axis=0, keepdims=True)
        tab_std = x_tab_raw[idx_train].std(axis=0, keepdims=True)
        tab_std[tab_std < 1e-8] = 1.0

    return {
        "df": df,
        "feature_cols": feature_cols,
        "exclude_cols": exclude_cols,
        "cat_cols": cat_cols,
        "tab_mean": tab_mean,
        "tab_std": tab_std,
        "target_cols": target_cols,
        "target_means": target_means,
        "target_stds": target_stds,
    }


def build_tri_context(model):
    df = pd.read_csv(MASTER_WITH_PATHS_CSV, low_memory=False)
    df = df[
        df["light_folder"].astype(str).isin({"Tri Lighting"})
        & (df["num_active_lights"].astype(int) == 3)
        & (df["material_folder"].astype(str) == "PlasticGlossy")
        & (df["batch_folder"].astype(str) == "Batch 1 - Cycles AGX")
    ].copy()

    cols = [
        "image_relpath", "shape_name", "material_folder", "light_folder", "batch_folder", "frame", "config_id", "camera_png", "camera_name",
        "cam_pos_x", "cam_pos_y", "cam_pos_z", "cam_forward_x", "cam_forward_y", "cam_forward_z",
        "cam_up_x", "cam_up_y", "cam_up_z", "cam_right_x", "cam_right_y", "cam_right_z", "focal_length_mm",
    ]
    for i in range(3):
        cols += [
            f"light{i}_energy", f"light{i}_color_r", f"light{i}_color_g", f"light{i}_color_b",
            f"light{i}_pos_x", f"light{i}_pos_y", f"light{i}_pos_z",
            f"light{i}_dir_x", f"light{i}_dir_y", f"light{i}_dir_z",
        ]

    image_df = df[[c for c in cols if c in df.columns]].copy()
    image_df = ensure_cam_space_tri(image_df)
    image_df["image_path"] = image_df["image_relpath"].astype(str).map(lambda p: str(DATA_DIR / "render-lighting" / p))
    image_df = image_df[image_df["image_path"].map(os.path.exists)].reset_index(drop=True)

    target_cols = []
    for i in range(3):
        target_cols.extend([
            f"light{i}_pos_cam_x", f"light{i}_pos_cam_y", f"light{i}_pos_cam_z",
            f"light{i}_dir_cam_x", f"light{i}_dir_cam_y", f"light{i}_dir_cam_z",
        ])

    exclude_cols = set(target_cols + ["image_relpath", "image_path", "camera_png"])
    for i in range(3):
        exclude_cols.update([
            f"light{i}_pos_x", f"light{i}_pos_y", f"light{i}_pos_z",
            f"light{i}_dir_x", f"light{i}_dir_y", f"light{i}_dir_z",
        ])

    known_df = image_df[[c for c in image_df.columns if c not in exclude_cols]].copy()
    cat_cols = [c for c in ["shape_name", "material_folder", "light_folder", "batch_folder", "camera_name"] if c in known_df.columns]
    known_df = pd.get_dummies(known_df, columns=cat_cols, drop_first=False)

    x_tab_raw = known_df.to_numpy(dtype=np.float32)
    idx = np.arange(len(image_df))
    idx_train, _ = train_test_split(idx, test_size=0.2, random_state=42)
    idx_train, _ = train_test_split(idx_train, test_size=0.2, random_state=42)

    tab_mean = x_tab_raw[idx_train].mean(axis=0, keepdims=True)
    tab_std = x_tab_raw[idx_train].std(axis=0, keepdims=True)
    tab_std[tab_std < 1e-8] = 1.0

    expected_tab_dim = model.input_shape[1][1]
    if x_tab_raw.shape[1] != expected_tab_dim:
        raise ValueError(f"Tri-angular tab dim mismatch: {x_tab_raw.shape[1]} vs {expected_tab_dim}")

    return {
        "df": image_df,
        "feature_cols": known_df.columns.tolist(),
        "exclude_cols": exclude_cols,
        "cat_cols": cat_cols,
        "tab_mean": tab_mean,
        "tab_std": tab_std,
        "target_cols": target_cols,
    }


def build_color_context(color_model_path: str):
    if "finetuned" in color_model_path:
        df = pd.read_csv(INVERSE_METADATA_CSV, low_memory=False)
        df = df[(df["num_active_lights"].astype(int) == 1) & (df["light0_type"].astype(str) == "SPOT")].copy()
        df["image_path"] = df["image_relpath"].astype(str).map(lambda p: str(DATA_DIR / "inverse_rendering_dataset" / p))
        df = df[df["image_path"].map(os.path.exists)].reset_index(drop=True)

        y_color_raw = df[["light0_color_r", "light0_color_g", "light0_color_b"]].to_numpy(dtype=np.float32)
        y_energy_raw = np.log(df["light0_energy"].clip(lower=1e-6).to_numpy(dtype=np.float32)).reshape(-1, 1)

        color_cols = ["light0_color_r", "light0_color_g", "light0_color_b"]
        energy_col = "light0_energy"
    else:
        legacy_csv = PROCESSING_DIR / "color_power_labels.csv"
        df = pd.read_csv(legacy_csv, low_memory=False)
        df["image_path"] = df["image_relpath"].astype(str).map(lambda p: str(DATA_DIR / "spotlight-sphere-data" / p))
        df = df[df["image_path"].map(os.path.exists)].reset_index(drop=True)

        y_color_raw = df[["spot_color_r", "spot_color_g", "spot_color_b"]].to_numpy(dtype=np.float32)
        y_energy_raw = np.log(df["spot_energy"].clip(lower=1e-6).to_numpy(dtype=np.float32)).reshape(-1, 1)

        color_cols = ["spot_color_r", "spot_color_g", "spot_color_b"]
        energy_col = "spot_energy"

    idx = np.arange(len(df))
    idx_train, _ = train_test_split(idx, test_size=0.2, random_state=42)
    idx_train, _ = train_test_split(idx_train, test_size=0.2, random_state=42)

    color_mean = y_color_raw[idx_train].mean(axis=0, keepdims=True)
    color_std = y_color_raw[idx_train].std(axis=0, keepdims=True)
    color_std[color_std < 1e-8] = 1.0

    energy_mean = y_energy_raw[idx_train].mean(axis=0, keepdims=True)
    energy_std = y_energy_raw[idx_train].std(axis=0, keepdims=True)
    energy_std[energy_std < 1e-8] = 1.0

    return {
        "df": df,
        "color_cols": color_cols,
        "energy_col": energy_col,
        "color_mean": color_mean,
        "color_std": color_std,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
    }


def build_spot_context():
    df = pd.read_csv(INVERSE_METADATA_CSV, low_memory=False)
    df = df[
        (df["num_active_lights"].astype(int) == 1)
        & (df["light0_type"].astype(str) == "SPOT")
        & (df["batch_folder"].astype(str).str.contains("spot_size", case=False, na=False))
    ].copy()
    df["image_path"] = df["image_relpath"].astype(str).map(lambda p: str(DATA_DIR / "inverse_rendering_dataset" / p))
    df = df[df["image_path"].map(os.path.exists)].reset_index(drop=True)

    y = df["light0_spot_cone_deg"].to_numpy(dtype=np.float32).reshape(-1, 1)

    idx = np.arange(len(df))
    idx_train, _ = train_test_split(idx, test_size=0.2, random_state=42)
    idx_train, _ = train_test_split(idx_train, test_size=0.2, random_state=42)

    y_mean = y[idx_train].mean(axis=0, keepdims=True)
    y_std = y[idx_train].std(axis=0, keepdims=True)
    y_std[y_std < 1e-8] = 1.0

    return {"df": df, "y_mean": y_mean, "y_std": y_std}


def build_prediction_csv_row(results: dict) -> dict:
    predictions = results.get("predictions", {})
    skipped = results.get("skipped", {})

    row = {
        "image_path": results.get("image_path", ""),
        "pred_light_count": None,
        "pred_light_count_confidence": None,
        "pred_light_type": None,
        "pred_light_type_confidence": None,
        "pred_color_r": None,
        "pred_color_g": None,
        "pred_color_b": None,
        "pred_energy": None,
        "pred_spot_cone_deg": None,
        "pred_light0_position_cam": None,
        "pred_light0_direction_cam": None,
        "pred_tri_lights": None,
        "skipped_color_power_predictor": skipped.get("color_power_predictor", ""),
        "skipped_spot_size_predictor": skipped.get("spot_size_predictor", ""),
        "skipped_angular_predictor": skipped.get("angular_predictor", ""),
        "skipped_tri_angular_predictor": skipped.get("tri_angular_predictor", ""),
    }

    count = predictions.get("light_count_detector")
    if count:
        row["pred_light_count"] = count.get("light_count")
        row["pred_light_count_confidence"] = count.get("confidence")

    light_type = predictions.get("light_type_classifier")
    if light_type:
        row["pred_light_type"] = light_type.get("light_type")
        row["pred_light_type_confidence"] = light_type.get("confidence")

    color_power = predictions.get("color_power_predictor")
    if color_power:
        rgb = color_power.get("predicted_color_rgb", [])
        if len(rgb) == 3:
            row["pred_color_r"] = rgb[0]
            row["pred_color_g"] = rgb[1]
            row["pred_color_b"] = rgb[2]
        row["pred_energy"] = color_power.get("predicted_energy")

    spot_size = predictions.get("spot_size_predictor")
    if spot_size:
        row["pred_spot_cone_deg"] = spot_size.get("predicted_spot_cone_deg")

    angular = predictions.get("angular_predictor")
    if angular:
        row["pred_light0_position_cam"] = json.dumps(angular.get("predicted_position_cam", []))
        row["pred_light0_direction_cam"] = json.dumps(angular.get("predicted_direction_cam", []))

    tri = predictions.get("tri_angular_predictor")
    if tri:
        row["pred_tri_lights"] = json.dumps(tri.get("lights", []))

    return row


def load_pipeline_models() -> dict:
    models = {}
    for name in [
        "light_count_detector",
        "light_type_classifier",
        "angular_predictor",
        "color_power_predictor",
        "tri_angular_predictor",
        "spot_size_predictor",
    ]:
        model, source = load_model_prefer_finetuned(name)
        if model is not None:
            models[name] = {"model": model, "source": source}
    return models


def load_light_type_mapping() -> dict[int, str]:
    mapping_path = MODELS_DIR / "light_type_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw["idx_to_light_type"].items()}


def predict_light_count(model, image_path: Path) -> dict:
    img_h, img_w = model.input_shape[1:3]
    img = load_and_preprocess_image(str(image_path), (img_w, img_h))
    pred = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    return {
        "light_count": int(np.argmax(pred) + 1),
        "confidence": float(np.max(pred)),
        "probabilities": pred.tolist(),
    }


def predict_light_type(model, image_path: Path, mapping: dict[int, str]) -> dict:
    img_h, img_w = model.input_shape[1:3]
    img = load_and_preprocess_image(str(image_path), (img_w, img_h))
    pred = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    pred_idx = int(np.argmax(pred))
    return {
        "light_type": mapping.get(pred_idx, f"IDX_{pred_idx}"),
        "confidence": float(np.max(pred)),
        "probabilities": {mapping.get(i, f"IDX_{i}"): float(v) for i, v in enumerate(pred)},
    }


def find_matching_row(df: pd.DataFrame, image_path: Path) -> pd.DataFrame:
    resolved = image_path.resolve()
    df = df.copy()

    if "image_path" in df.columns:
        df["__resolved_image_path"] = df["image_path"].astype(str).map(lambda p: str(Path(p).resolve()))
        exact = df[df["__resolved_image_path"] == str(resolved)]
        if not exact.empty:
            return exact.iloc[[0]].copy()

    basename = image_path.name
    if "image_path" in df.columns:
        base_match = df[df["image_path"].astype(str).map(lambda p: Path(p).name) == basename]
        if not base_match.empty:
            return base_match.iloc[[0]].copy()

    if "image_relpath" in df.columns:
        rel_match = df[df["image_relpath"].astype(str).map(lambda p: Path(p).name) == basename]
        if not rel_match.empty:
            return rel_match.iloc[[0]].copy()

    return df.iloc[0:0].copy()


def build_inference_row(
    ctx: dict,
    image_path: Path,
    overrides: dict | None = None,
    source_row: pd.Series | dict | None = None,
) -> pd.DataFrame:
    df = ctx["df"]
    if source_row is not None:
        one = pd.DataFrame([dict(source_row)]).copy()
    else:
        matched = find_matching_row(df, image_path)
        if not matched.empty:
            one = matched.iloc[[0]].copy()
        else:
            known_cols = [c for c in df.columns if c not in ctx["exclude_cols"]]
            row = {}

            for col in known_cols:
                series = df[col].dropna()
                if series.empty:
                    row[col] = 0.0
                    continue

                if pd.api.types.is_numeric_dtype(df[col]):
                    row[col] = float(pd.to_numeric(series, errors="coerce").dropna().mean())
                else:
                    mode = series.mode(dropna=True)
                    row[col] = str(mode.iloc[0] if not mode.empty else series.iloc[0])

            one = pd.DataFrame([row])

    one = one.copy()
    one["image_path"] = str(image_path.resolve())
    one["image_relpath"] = image_path.name

    if "resolved_image_relpath" in one.columns:
        one["resolved_image_relpath"] = str(image_path.resolve())

    if "light_type_label" in ctx.get("cat_cols", []) and "light_type_label" not in one.columns:
        raw_type = one.iloc[0].get("light0_type")
        if raw_type is not None and not pd.isna(raw_type):
            one["light_type_label"] = str(raw_type).upper()

    if "light0_type" in one.columns:
        one["light0_type"] = one["light0_type"].astype(str).str.upper()

    if "light_type_label" in one.columns:
        one["light_type_label"] = one["light_type_label"].astype(str).str.upper()

    if set(ctx.get("target_cols", [])).issubset(one.columns):
        one = ensure_cam_space_single(one)

    if overrides:
        for key, value in overrides.items():
            one[key] = value

    if "light_type_label" in ctx.get("cat_cols", []):
        raw_type = None
        if overrides:
            raw_type = overrides.get("light0_type")
        if raw_type is None:
            raw_type = one.iloc[0].get("light0_type")
        if raw_type is not None:
            one["light_type_label"] = str(raw_type).upper()

    return one


def build_single_light_angular_overrides(results: dict) -> dict:
    overrides = {
        "num_active_lights": 1,
        "light0_type": "SPOT",
    }

    color_power = results["predictions"].get("color_power_predictor")
    if color_power:
        color = color_power["predicted_color_rgb"]
        overrides.update(
            {
                "light0_energy": float(color_power["predicted_energy"]),
                "light0_color_r": float(color[0]),
                "light0_color_g": float(color[1]),
                "light0_color_b": float(color[2]),
            }
        )

    spot_size = results["predictions"].get("spot_size_predictor")
    if spot_size:
        overrides["light0_spot_cone_deg"] = float(spot_size["predicted_spot_cone_deg"])

    return overrides


def run_angular_predictor(
    model,
    ctx: dict,
    image_path: Path,
    overrides: dict | None = None,
    source_row: pd.Series | dict | None = None,
) -> dict:
    one = build_inference_row(ctx, image_path, overrides, source_row=source_row)

    img_h, img_w = model.input_shape[0][1:3]
    img = load_and_preprocess_image(str(image_path), (img_w, img_h))

    known = one[[c for c in one.columns if c not in ctx["exclude_cols"]]].copy()
    known = pd.get_dummies(known, columns=[c for c in ctx["cat_cols"] if c in known.columns], drop_first=False)
    known = known.reindex(columns=ctx["feature_cols"], fill_value=0.0)

    x_tab = known.to_numpy(dtype=np.float32)
    x_tab = (x_tab - ctx["tab_mean"]) / ctx["tab_std"]

    pred = model.predict([np.expand_dims(img, 0), x_tab], verbose=0)[0]
    
    # Denormalize predictions if target normalization parameters are available
    if ctx.get("target_means") is not None and ctx.get("target_stds") is not None:
        pred = pred * ctx["target_stds"] + ctx["target_means"]
    
    return {
        "predicted_position_cam": pred[:3].tolist(),
        "predicted_direction_cam": pred[3:6].tolist(),
        "tabular_inputs": overrides or {},
    }


def run_tri_angular_predictor(model, ctx: dict, image_path: Path) -> dict:
    one = find_matching_row(ctx["df"], image_path)
    if one.empty:
        return {"skipped": "No matching metadata row found for tri_angular_predictor."}

    img_h, img_w = model.input_shape[0][1:3]
    img = load_and_preprocess_image(str(image_path), (img_w, img_h))

    known = one[[c for c in one.columns if c not in ctx["exclude_cols"]]].copy()
    known = pd.get_dummies(known, columns=[c for c in ctx["cat_cols"] if c in known.columns], drop_first=False)
    known = known.reindex(columns=ctx["feature_cols"], fill_value=0.0)

    x_tab = known.to_numpy(dtype=np.float32)
    x_tab = (x_tab - ctx["tab_mean"]) / ctx["tab_std"]

    pred = model.predict([np.expand_dims(img, 0), x_tab], verbose=0)[0]

    lights = []
    for i in range(3):
        start = i * 6
        lights.append(
            {
                "light_index": i,
                "predicted_position_cam": pred[start:start + 3].tolist(),
                "predicted_direction_cam": pred[start + 3:start + 6].tolist(),
            }
        )

    return {"lights": lights}


def run_color_power_predictor(model, model_source: str, ctx: dict, image_path: Path) -> dict:
    img_h, img_w = model.input_shape[1:3]
    img = load_and_preprocess_image(str(image_path), (img_w, img_h))

    pred = model.predict(np.expand_dims(img, 0), verbose=0)
    if not isinstance(pred, list) or len(pred) != 2:
        raise ValueError("Color/power model output format unexpected.")

    pred_color_z, pred_energy_z = pred
    pred_color = np.clip(pred_color_z * ctx["color_std"] + ctx["color_mean"], 0.0, 1.0)[0]
    pred_energy = float(np.exp((pred_energy_z * ctx["energy_std"] + ctx["energy_mean"])[0, 0]))

    return {
        "model_source": model_source,
        "predicted_color_rgb": pred_color.tolist(),
        "predicted_energy": pred_energy,
    }


def run_spot_size_predictor(model, ctx: dict, image_path: Path) -> dict:
    img_h, img_w = model.input_shape[1:3]
    img = load_and_preprocess_image(str(image_path), (img_w, img_h))
    pred_z = model.predict(np.expand_dims(img, 0), verbose=0)
    pred_deg = float((pred_z * ctx["y_std"] + ctx["y_mean"])[0, 0])
    return {"predicted_spot_cone_deg": pred_deg}


def initialize_pipeline_runtime() -> dict:
    models = load_pipeline_models()
    if "light_count_detector" not in models or "light_type_classifier" not in models:
        raise ValueError("Both light_count_detector and light_type_classifier are required.")

    runtime = {
        "models": models,
        "type_mapping": load_light_type_mapping(),
        "contexts": {},
        "context_errors": {},
    }

    if "color_power_predictor" in models:
        try:
            color_source = models["color_power_predictor"]["source"]
            runtime["contexts"]["color_power_predictor"] = build_color_context(color_source)
        except Exception as e:
            runtime["context_errors"]["color_power_predictor"] = str(e)

    if "spot_size_predictor" in models:
        try:
            runtime["contexts"]["spot_size_predictor"] = build_spot_context()
        except Exception as e:
            runtime["context_errors"]["spot_size_predictor"] = str(e)

    if "angular_predictor" in models:
        try:
            runtime["contexts"]["angular_predictor"] = build_angular_context(models["angular_predictor"]["model"])
        except Exception as e:
            runtime["context_errors"]["angular_predictor"] = str(e)

    if "tri_angular_predictor" in models:
        try:
            runtime["contexts"]["tri_angular_predictor"] = build_tri_context(models["tri_angular_predictor"]["model"])
        except Exception as e:
            runtime["context_errors"]["tri_angular_predictor"] = str(e)

    return runtime


def route_predictors(image_path: Path, runtime: dict, metadata_row: pd.Series | dict | None = None) -> dict:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    models = runtime["models"]
    type_mapping = runtime["type_mapping"]
    contexts = runtime["contexts"]
    context_errors = runtime["context_errors"]

    results = {
        "image_path": str(image_path.resolve()),
        "predictions": {},
        "skipped": {},
    }

    count_result = predict_light_count(models["light_count_detector"]["model"], image_path)
    type_result = predict_light_type(models["light_type_classifier"]["model"], image_path, type_mapping)

    # Force DOUBLE_SPOT_SPOT into the single-spot path for all downstream routing.
    if type_result.get("light_type") == "DOUBLE_SPOT_SPOT":
        type_result = {**type_result, "light_type": "SPOT LIGHT"}
        count_result = {**count_result, "light_count": 1}

    predicted_count = count_result["light_count"]
    predicted_type = type_result["light_type"]

    results["predictions"]["light_count_detector"] = count_result
    results["predictions"]["light_type_classifier"] = type_result

    if predicted_type == "SPOT LIGHT":
        if "color_power_predictor" in models and "color_power_predictor" in contexts:
            color_source = models["color_power_predictor"]["source"]
            color_ctx = contexts["color_power_predictor"]
            results["predictions"]["color_power_predictor"] = run_color_power_predictor(
                models["color_power_predictor"]["model"],
                color_source,
                color_ctx,
                image_path,
            )
        elif "color_power_predictor" in context_errors:
            results["skipped"]["color_power_predictor"] = f"Context/model setup failed: {context_errors['color_power_predictor']}"
        else:
            results["skipped"]["color_power_predictor"] = "Model not found."
    else:
        results["skipped"]["color_power_predictor"] = f"Predicted type {predicted_type} is not routed to color_power_predictor."

    if predicted_type == "SPOT LIGHT":
        if "spot_size_predictor" in models and "spot_size_predictor" in contexts:
            spot_ctx = contexts["spot_size_predictor"]
            results["predictions"]["spot_size_predictor"] = run_spot_size_predictor(
                models["spot_size_predictor"]["model"],
                spot_ctx,
                image_path,
            )
        elif "spot_size_predictor" in context_errors:
            results["skipped"]["spot_size_predictor"] = f"Context/model setup failed: {context_errors['spot_size_predictor']}"
        else:
            results["skipped"]["spot_size_predictor"] = "Model not found."
    else:
        results["skipped"]["spot_size_predictor"] = f"Predicted type {predicted_type} is not routed to spot_size_predictor."

    if predicted_type == "SPOT LIGHT" and predicted_count == 1:
        if "angular_predictor" in models and "angular_predictor" in contexts:
            try:
                ang_ctx = contexts["angular_predictor"]
                angular_result = run_angular_predictor(
                    models["angular_predictor"]["model"],
                    ang_ctx,
                    image_path,
                    build_single_light_angular_overrides(results),
                    source_row=metadata_row,
                )
                target = "predictions" if "skipped" not in angular_result else "skipped"
                results[target]["angular_predictor"] = angular_result if target == "predictions" else angular_result["skipped"]
            except Exception as e:
                results["skipped"]["angular_predictor"] = f"Error while preparing/running angular_predictor: {e}"
        elif "angular_predictor" in context_errors:
            results["skipped"]["angular_predictor"] = f"Context/model setup failed: {context_errors['angular_predictor']}"
        else:
            results["skipped"]["angular_predictor"] = "Model not found."
    else:
        results["skipped"]["angular_predictor"] = (
            f"Predicted count/type ({predicted_count}, {predicted_type}) is not routed to angular_predictor."
        )

    if predicted_type == "TRI LIGHTING" and predicted_count == 3:
        if "tri_angular_predictor" in models and "tri_angular_predictor" in contexts:
            try:
                tri_ctx = contexts["tri_angular_predictor"]
                tri_result = run_tri_angular_predictor(models["tri_angular_predictor"]["model"], tri_ctx, image_path)
                target = "predictions" if "skipped" not in tri_result else "skipped"
                results[target]["tri_angular_predictor"] = tri_result if target == "predictions" else tri_result["skipped"]
            except Exception as e:
                results["skipped"]["tri_angular_predictor"] = f"Error while preparing/running tri_angular_predictor: {e}"
        elif "tri_angular_predictor" in context_errors:
            results["skipped"]["tri_angular_predictor"] = f"Context/model setup failed: {context_errors['tri_angular_predictor']}"
        else:
            results["skipped"]["tri_angular_predictor"] = "Model not found."
    else:
        results["skipped"]["tri_angular_predictor"] = (
            f"Predicted count/type ({predicted_count}, {predicted_type}) is not routed to tri_angular_predictor."
        )

    return results


def print_summary(results: dict) -> None:
    predictions = results["predictions"]
    skipped = results["skipped"]

    print(f"Image: {results['image_path']}")
    print()

    count = predictions["light_count_detector"]
    print("light_count_detector")
    print(f"  Predicted lights: {count['light_count']}")
    print(f"  Confidence: {count['confidence'] * 100:.2f}%")
    print()

    light_type = predictions["light_type_classifier"]
    print("light_type_classifier")
    print(f"  Predicted type: {light_type['light_type']}")
    print(f"  Confidence: {light_type['confidence'] * 100:.2f}%")
    print()

    for name in [
        "color_power_predictor",
        "spot_size_predictor",
        "angular_predictor",
        "tri_angular_predictor",
    ]:
        if name in predictions:
            print(name)
            print(json.dumps(predictions[name], indent=2))
            print()
        elif name in skipped:
            print(name)
            print(f"  Skipped: {skipped[name]}")
            print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the inverse-rendering pipeline on one image or the packaged test set."
    )
    parser.add_argument("image", type=Path, nargs="?", help="Path to a single input image")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result payload as JSON instead of a text summary.",
    )
    parser.add_argument(
        "--test-metadata",
        type=Path,
        default=PROJECT_ROOT / "test_data" / "test_metadata.csv",
        help="Metadata CSV used for batch test runs.",
    )
    parser.add_argument(
        "--batch-out",
        type=Path,
        default=PROJECT_ROOT / "test_predictions.csv",
        help="CSV path for batch test predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = initialize_pipeline_runtime()

    if args.image is not None:
        results = route_predictors(args.image, runtime)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print_summary(results)
        return

    metadata_csv = args.test_metadata
    test_df = pd.read_csv(metadata_csv, low_memory=False)
    rows = []

    for _, meta_row in test_df.iterrows():
        img_path_str = resolve_existing_path(meta_row)
        if not img_path_str:
            fallback = PROJECT_ROOT / "test_data" / str(meta_row.get("image_relpath", ""))
            if fallback.exists():
                img_path_str = str(fallback)

        if not img_path_str:
            rows.append(
                {
                    "image_path": str(meta_row.get("image_relpath", "")),
                    "skipped_color_power_predictor": "",
                    "skipped_spot_size_predictor": "",
                    "skipped_angular_predictor": "",
                    "skipped_tri_angular_predictor": "",
                    "error": f"Image not found for metadata row: {meta_row.get('image_relpath', '')}",
                }
            )
            continue

        try:
            results = route_predictors(Path(img_path_str), runtime, metadata_row=meta_row)
            pred_row = build_prediction_csv_row(results)
        except Exception as e:
            pred_row = {
                "image_path": str(img_path_str),
                "pred_light_count": None,
                "pred_light_count_confidence": None,
                "pred_light_type": None,
                "pred_light_type_confidence": None,
                "pred_color_r": None,
                "pred_color_g": None,
                "pred_color_b": None,
                "pred_energy": None,
                "pred_spot_cone_deg": None,
                "pred_light0_position_cam": None,
                "pred_light0_direction_cam": None,
                "pred_tri_lights": None,
                "skipped_color_power_predictor": "",
                "skipped_spot_size_predictor": "",
                "skipped_angular_predictor": "",
                "skipped_tri_angular_predictor": "",
                "error": str(e),
            }
        pred_row["actual_image_relpath"] = str(meta_row.get("image_relpath", ""))
        pred_row["actual_dataset_source"] = str(meta_row.get("dataset_source", ""))
        rows.append(pred_row)

    out_csv = args.batch_out
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved predictions CSV: {out_csv}")


if __name__ == "__main__":
    main()

# run python pipeline.py path/to/image.png
