import json
import math
import os
import secrets
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras

np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSING_DIR = PROJECT_ROOT / "processing"

DATA_MASTER_CSV = DATA_DIR / "data_master.csv"
INVERSE_METADATA_CSV = DATA_DIR / "inverse_rendering_dataset" / "metadata.csv"
MASTER_WITH_PATHS_CSV = PROCESSING_DIR / "master_with_paths.csv"

OUT_IMAGE = PROJECT_ROOT / "testing" / "test_result.png"


def load_model_prefer_finetuned(base_name: str):
    finetuned = MODELS_DIR / f"{base_name}_finetuned.keras"
    base = MODELS_DIR / f"{base_name}.keras"

    if finetuned.exists():
        return keras.models.load_model(finetuned), finetuned.name
    if base.exists():
        return keras.models.load_model(base), base.name
    return None, None


def load_required_models():
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
            print(f"Loaded {name} from {source}")
        else:
            print(f"Missing model: {name}")

    return models


def load_and_preprocess_image(path: str, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize(size, Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


def normalize_rows(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, eps, None)


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

    seen = set()
    for p in candidates:
        norm = os.path.normpath(p)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.exists(norm):
            return norm
    return None


def classify_light_type(row: pd.Series) -> str:
    light_folder = str(row.get("light_folder", "")).strip()
    num_lights = int(row.get("num_active_lights", 0) or 0)

    if num_lights == 2:
        l0 = str(row.get("light0_type", "")).upper()
        l1 = str(row.get("light1_type", "")).upper()
        if l0 and l1 and l0 != "NAN" and l1 != "NAN":
            return f"DOUBLE_{l0}_{l1}"
        return "DOUBLE_LIGHT"

    if light_folder.lower() == "single_light":
        return "SPOT LIGHT"

    if "HDRI" in light_folder.upper():
        return "HDRI"

    return light_folder.upper() if light_folder else "UNKNOWN"


def build_angular_context(model):
    inverse_df = pd.read_csv(INVERSE_METADATA_CSV, low_memory=False)
    inverse_df = inverse_df[(inverse_df["num_active_lights"].astype(int) == 1) & (inverse_df["light0_type"].astype(str) == "SPOT")].copy()
    inverse_df["image_path"] = inverse_df["image_relpath"].astype(str).map(lambda p: str(DATA_DIR / "inverse_rendering_dataset" / p))
    inverse_df = inverse_df[inverse_df["image_path"].map(os.path.exists)].reset_index(drop=True)
    inverse_df = ensure_cam_space_single(inverse_df)

    legacy_df = pd.read_csv(MASTER_WITH_PATHS_CSV, low_memory=False)
    legacy_df = ensure_cam_space_single(legacy_df)
    legacy_df = legacy_df[
        legacy_df["light_folder"].astype(str).isin({"Spot Light"})
        & (legacy_df["num_active_lights"].astype(int) == 1)
        & (legacy_df["material_folder"].astype(str) == "PlasticGlossy")
        & (legacy_df["batch_folder"].astype(str) == "Batch 1 - Cycles AGX")
    ].copy()

    legacy_cols = [
        "image_relpath", "shape_name", "material_folder", "light_folder", "batch_folder", "frame", "config_id",
        "camera_png", "camera_name", "cam_pos_x", "cam_pos_y", "cam_pos_z", "cam_forward_x", "cam_forward_y", "cam_forward_z",
        "cam_up_x", "cam_up_y", "cam_up_z", "cam_right_x", "cam_right_y", "cam_right_z", "focal_length_mm",
        "light0_energy", "light0_color_r", "light0_color_g", "light0_color_b", "light0_pos_x", "light0_pos_y", "light0_pos_z",
        "light0_dir_x", "light0_dir_y", "light0_dir_z", "light0_spot_cone_deg", "light0_spot_blend",
        "light0_pos_cam_x", "light0_pos_cam_y", "light0_pos_cam_z", "light0_dir_cam_x", "light0_dir_cam_y", "light0_dir_cam_z",
    ]

    legacy = legacy_df[[c for c in legacy_cols if c in legacy_df.columns]].copy()
    aligned = inverse_df[[c for c in legacy_cols if c in inverse_df.columns]].copy()

    target_cols = [
        "light0_pos_cam_x", "light0_pos_cam_y", "light0_pos_cam_z",
        "light0_dir_cam_x", "light0_dir_cam_y", "light0_dir_cam_z",
    ]

    exclude_cols = set(target_cols + [
        "image_relpath", "image_path", "camera_png",
        "light0_pos_x", "light0_pos_y", "light0_pos_z",
        "light0_dir_x", "light0_dir_y", "light0_dir_z",
    ])

    legacy_known = legacy[[c for c in legacy.columns if c not in exclude_cols]].copy()
    aligned_known = aligned[[c for c in aligned.columns if c not in exclude_cols]].copy()

    cat_cols = [c for c in ["shape_name", "material_folder", "light_folder", "batch_folder", "camera_name"] if c in legacy_known.columns]
    legacy_known = pd.get_dummies(legacy_known, columns=cat_cols, drop_first=False)
    feature_cols = legacy_known.columns.tolist()

    aligned_known = pd.get_dummies(aligned_known, columns=cat_cols, drop_first=False)
    aligned_known = aligned_known.reindex(columns=feature_cols, fill_value=0.0)

    x_tab_raw = aligned_known.to_numpy(dtype=np.float32)
    idx = np.arange(len(aligned))
    idx_train, _ = train_test_split(idx, test_size=0.2, random_state=42)
    idx_train, _ = train_test_split(idx_train, test_size=0.2, random_state=42)

    tab_mean = x_tab_raw[idx_train].mean(axis=0, keepdims=True)
    tab_std = x_tab_raw[idx_train].std(axis=0, keepdims=True)
    tab_std[tab_std < 1e-8] = 1.0

    expected_tab_dim = model.input_shape[1][1]
    if x_tab_raw.shape[1] != expected_tab_dim:
        raise ValueError(f"Angular tab dim mismatch: {x_tab_raw.shape[1]} vs {expected_tab_dim}")

    return {
        "df": inverse_df,
        "feature_cols": feature_cols,
        "exclude_cols": exclude_cols,
        "cat_cols": cat_cols,
        "tab_mean": tab_mean,
        "tab_std": tab_std,
        "target_cols": target_cols,
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


def choose_sample_rows():
    master = pd.read_csv(DATA_MASTER_CSV, low_memory=False)

    inv_single = master[(master["dataset_source"] == "inverse_rendering_dataset") & (master["num_active_lights"].astype(int) == 1)].copy()
    inv_single["resolved"] = inv_single.apply(resolve_existing_path, axis=1)
    inv_single = inv_single[inv_single["resolved"].notna()].reset_index(drop=True)

    inv_spot = inv_single[inv_single["batch_folder"].astype(str).str.contains("spot_size", case=False, na=False)].copy()

    tri = master[
        (master["dataset_source"] == "render-lighting")
        & (master["light_folder"].astype(str) == "Tri Lighting")
        & (master["num_active_lights"].astype(int) == 3)
    ].copy()
    tri["resolved"] = tri.apply(resolve_existing_path, axis=1)
    tri = tri[tri["resolved"].notna()].reset_index(drop=True)

    if inv_single.empty:
        raise ValueError("No inverse single-light test image found")
    if inv_spot.empty:
        raise ValueError("No inverse spot-size test image found")
    if tri.empty:
        raise ValueError("No tri-light test image found")

    return {
        "single": inv_single.iloc[secrets.randbelow(len(inv_single))].copy(),
        "spot": inv_spot.iloc[secrets.randbelow(len(inv_spot))].copy(),
        "tri": tri.iloc[secrets.randbelow(len(tri))].copy(),
    }


def test_light_count(model, row):
    img_size = model.input_shape[1:3]
    img = load_and_preprocess_image(row["resolved"], (img_size[1], img_size[0]))
    pred = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    pred_count = int(np.argmax(pred) + 1)
    true_count = int(row["num_active_lights"])

    return {
        "image_path": row["resolved"],
        "title": "light_count_detector",
        "summary": [
            f"Image: {Path(row['image_relpath']).name}",
            f"Actual lights: {true_count}",
            f"Predicted lights: {pred_count}",
            f"Confidence: {float(np.max(pred) * 100):.2f}%",
            f"Correct: {pred_count == true_count}",
        ],
    }


def test_light_type(model, row):
    mapping_path = MODELS_DIR / "light_type_mapping.json"
    if not mapping_path.exists():
        raise ValueError("light_type_mapping.json missing")

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    img_size = model.input_shape[1:3]
    img = load_and_preprocess_image(row["resolved"], (img_size[1], img_size[0]))
    pred = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    pred_idx = int(np.argmax(pred))
    pred_label = mapping["idx_to_light_type"].get(str(pred_idx), f"IDX_{pred_idx}")

    true_label = classify_light_type(row)
    return {
        "image_path": row["resolved"],
        "title": "light_type_classifier",
        "summary": [
            f"Image: {Path(row['image_relpath']).name}",
            f"Actual type: {true_label}",
            f"Predicted type: {pred_label}",
            f"Confidence: {float(np.max(pred) * 100):.2f}%",
            f"Correct: {pred_label == true_label}",
        ],
    }


def test_angular(model, ctx, row):
    relpath = row["image_relpath"]
    one = ctx["df"][ctx["df"]["image_relpath"].astype(str) == str(relpath)].copy()
    if one.empty:
        raise ValueError(f"Angular row not found for {relpath}")
    one = one.iloc[[0]].copy()

    img_h, img_w = model.input_shape[0][1:3]
    img = load_and_preprocess_image(one.iloc[0]["image_path"], (img_w, img_h))

    known = one[[c for c in one.columns if c not in ctx["exclude_cols"]]].copy()
    known = pd.get_dummies(known, columns=[c for c in ctx["cat_cols"] if c in known.columns], drop_first=False)
    known = known.reindex(columns=ctx["feature_cols"], fill_value=0.0)

    x_tab = known.to_numpy(dtype=np.float32)
    x_tab = (x_tab - ctx["tab_mean"]) / ctx["tab_std"]

    pred = model.predict([np.expand_dims(img, 0), x_tab], verbose=0)[0]
    true = one.iloc[0][ctx["target_cols"]].to_numpy(dtype=np.float32)

    pos_err = float(np.mean(np.abs(pred[:3] - true[:3])))
    pdir = pred[3:6] / max(np.linalg.norm(pred[3:6]), 1e-8)
    tdir = true[3:6] / max(np.linalg.norm(true[3:6]), 1e-8)
    cosang = float(np.clip(np.dot(pdir, tdir), -1.0, 1.0))
    ang_err = float(np.degrees(np.arccos(cosang)))

    return {
        "image_path": one.iloc[0]["image_path"],
        "title": "angular_predictor",
        "summary": [
            f"Image: {Path(relpath).name}",
            f"Actual pos: {np.round(true[:3], 3).tolist()}",
            f"Pred pos:   {np.round(pred[:3], 3).tolist()}",
            f"Pos MAE: {pos_err:.3f}",
            f"Dir angle err (deg): {ang_err:.3f}",
        ],
    }


def test_color_power(model, model_source, ctx, row):
    relpath = row["image_relpath"]
    one = ctx["df"][ctx["df"]["image_relpath"].astype(str) == str(relpath)].copy()
    if one.empty:
        raise ValueError(f"Color/power row not found for {relpath}")
    one = one.iloc[[0]].copy()

    img_h, img_w = model.input_shape[1:3]
    img = load_and_preprocess_image(one.iloc[0]["image_path"], (img_w, img_h))

    pred = model.predict(np.expand_dims(img, 0), verbose=0)
    if isinstance(pred, list) and len(pred) == 2:
        pred_color_z, pred_energy_z = pred[0], pred[1]
    else:
        raise ValueError("Color/power model output format unexpected")

    pred_color = np.clip(pred_color_z * ctx["color_std"] + ctx["color_mean"], 0.0, 1.0)[0]
    pred_energy = float(np.exp((pred_energy_z * ctx["energy_std"] + ctx["energy_mean"])[0, 0]))

    true_color = one.iloc[0][ctx["color_cols"]].to_numpy(dtype=np.float32)
    true_energy = float(one.iloc[0][ctx["energy_col"]])

    return {
        "image_path": one.iloc[0]["image_path"],
        "title": f"color_power_predictor ({model_source})",
        "summary": [
            f"Image: {Path(relpath).name}",
            f"Actual color: {np.round(true_color, 3).tolist()}",
            f"Pred color:   {np.round(pred_color, 3).tolist()}",
            f"Actual energy: {true_energy:.3f}",
            f"Pred energy:   {pred_energy:.3f}",
        ],
    }


def test_tri_angular(model, ctx, row):
    relpath = row["image_relpath"]
    one = ctx["df"][ctx["df"]["image_relpath"].astype(str) == str(relpath)].copy()
    if one.empty:
        raise ValueError(f"Tri row not found for {relpath}")
    one = one.iloc[[0]].copy()

    img_h, img_w = model.input_shape[0][1:3]
    img = load_and_preprocess_image(one.iloc[0]["image_path"], (img_w, img_h))

    known = one[[c for c in one.columns if c not in ctx["exclude_cols"]]].copy()
    known = pd.get_dummies(known, columns=[c for c in ctx["cat_cols"] if c in known.columns], drop_first=False)
    known = known.reindex(columns=ctx["feature_cols"], fill_value=0.0)

    x_tab = known.to_numpy(dtype=np.float32)
    x_tab = (x_tab - ctx["tab_mean"]) / ctx["tab_std"]

    pred = model.predict([np.expand_dims(img, 0), x_tab], verbose=0)[0]
    true = one.iloc[0][ctx["target_cols"]].to_numpy(dtype=np.float32)

    lines = [f"Image: {Path(relpath).name}"]
    for i, name in enumerate(["L0", "L1", "L2"]):
        s = i * 6
        e = s + 6
        p = pred[s:e]
        t = true[s:e]
        pos_err = float(np.mean(np.abs(p[:3] - t[:3])))
        pdir = p[3:6] / max(np.linalg.norm(p[3:6]), 1e-8)
        tdir = t[3:6] / max(np.linalg.norm(t[3:6]), 1e-8)
        ang_err = float(np.degrees(np.arccos(np.clip(np.dot(pdir, tdir), -1.0, 1.0))))
        lines.append(f"{name} pos MAE: {pos_err:.3f}, dir err: {ang_err:.2f}deg")

    return {
        "image_path": one.iloc[0]["image_path"],
        "title": "tri_angular_predictor",
        "summary": lines,
    }


def test_spot_size(model, ctx, row):
    relpath = row["image_relpath"]
    one = ctx["df"][ctx["df"]["image_relpath"].astype(str) == str(relpath)].copy()
    if one.empty:
        raise ValueError(f"Spot-size row not found for {relpath}")
    one = one.iloc[[0]].copy()

    img_h, img_w = model.input_shape[1:3]
    img = load_and_preprocess_image(one.iloc[0]["image_path"], (img_w, img_h))

    pred_z = model.predict(np.expand_dims(img, 0), verbose=0)
    pred_deg = float((pred_z * ctx["y_std"] + ctx["y_mean"])[0, 0])
    true_deg = float(one.iloc[0]["light0_spot_cone_deg"])

    return {
        "image_path": one.iloc[0]["image_path"],
        "title": "spot_size_predictor",
        "summary": [
            f"Image: {Path(relpath).name}",
            f"Actual spot cone (deg): {true_deg:.3f}",
            f"Pred spot cone (deg):   {pred_deg:.3f}",
            f"Abs error (deg):        {abs(pred_deg - true_deg):.3f}",
        ],
    }


def draw_report(results):
    n = len(results)
    if n == 0:
        raise ValueError("No test results to render")

    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, res in enumerate(results):
        ax = axes[i]
        img = load_and_preprocess_image(res["image_path"], (512, 512))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(res["title"], fontsize=12)
        text = "\n".join(res["summary"])
        ax.text(
            0.01,
            0.01,
            text,
            transform=ax.transAxes,
            fontsize=9,
            color="white",
            va="bottom",
            ha="left",
            bbox=dict(facecolor="black", alpha=0.65, boxstyle="round,pad=0.4"),
        )

    plt.tight_layout()
    OUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_IMAGE, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Testing available models with actual vs predicted outputs...")

    models = load_required_models()
    if not models:
        raise ValueError("No models found in models/")

    sample_rows = choose_sample_rows()

    results = []

    if "light_count_detector" in models:
        results.append(test_light_count(models["light_count_detector"]["model"], sample_rows["single"]))

    if "light_type_classifier" in models:
        results.append(test_light_type(models["light_type_classifier"]["model"], sample_rows["single"]))

    if "angular_predictor" in models:
        ang_ctx = build_angular_context(models["angular_predictor"]["model"])
        results.append(test_angular(models["angular_predictor"]["model"], ang_ctx, sample_rows["single"]))

    if "color_power_predictor" in models:
        color_source = models["color_power_predictor"]["source"]
        color_ctx = build_color_context(color_source)
        results.append(
            test_color_power(
                models["color_power_predictor"]["model"],
                color_source,
                color_ctx,
                sample_rows["single"],
            )
        )

    if "tri_angular_predictor" in models:
        tri_ctx = build_tri_context(models["tri_angular_predictor"]["model"])
        results.append(test_tri_angular(models["tri_angular_predictor"]["model"], tri_ctx, sample_rows["tri"]))

    if "spot_size_predictor" in models:
        spot_ctx = build_spot_context()
        results.append(test_spot_size(models["spot_size_predictor"]["model"], spot_ctx, sample_rows["spot"]))

    for res in results:
        print("\n" + "-" * 60)
        print(res["title"])
        for line in res["summary"]:
            print("  " + line)

    draw_report(results)
    print("\nSaved report image:", OUT_IMAGE)


if __name__ == "__main__":
    main()
