import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from testing.test_all_models import (
    MODELS_DIR,
    build_angular_context,
    build_color_context,
    build_spot_context,
    build_tri_context,
    load_and_preprocess_image,
    load_model_prefer_finetuned,
)

np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent


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


def build_inference_row(ctx: dict, image_path: Path, overrides: dict | None = None) -> pd.DataFrame:
    df = ctx["df"]
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

    row["image_path"] = str(image_path.resolve())
    row["image_relpath"] = image_path.name

    if overrides:
        row.update(overrides)

    return pd.DataFrame([row])


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


def run_angular_predictor(model, ctx: dict, image_path: Path, overrides: dict | None = None) -> dict:
    one = build_inference_row(ctx, image_path, overrides)

    img_h, img_w = model.input_shape[0][1:3]
    img = load_and_preprocess_image(str(image_path), (img_w, img_h))

    known = one[[c for c in one.columns if c not in ctx["exclude_cols"]]].copy()
    known = pd.get_dummies(known, columns=[c for c in ctx["cat_cols"] if c in known.columns], drop_first=False)
    known = known.reindex(columns=ctx["feature_cols"], fill_value=0.0)

    x_tab = known.to_numpy(dtype=np.float32)
    x_tab = (x_tab - ctx["tab_mean"]) / ctx["tab_std"]

    pred = model.predict([np.expand_dims(img, 0), x_tab], verbose=0)[0]
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


def route_predictors(image_path: Path) -> dict:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    models = load_pipeline_models()
    if "light_count_detector" not in models or "light_type_classifier" not in models:
        raise ValueError("Both light_count_detector and light_type_classifier are required.")

    type_mapping = load_light_type_mapping()
    results = {
        "image_path": str(image_path.resolve()),
        "predictions": {},
        "skipped": {},
    }

    count_result = predict_light_count(models["light_count_detector"]["model"], image_path)
    type_result = predict_light_type(models["light_type_classifier"]["model"], image_path, type_mapping)
    predicted_count = count_result["light_count"]
    predicted_type = type_result["light_type"]

    results["predictions"]["light_count_detector"] = count_result
    results["predictions"]["light_type_classifier"] = type_result

    if predicted_type == "SPOT LIGHT":
        if "color_power_predictor" in models:
            color_source = models["color_power_predictor"]["source"]
            color_ctx = build_color_context(color_source)
            results["predictions"]["color_power_predictor"] = run_color_power_predictor(
                models["color_power_predictor"]["model"],
                color_source,
                color_ctx,
                image_path,
            )
        else:
            results["skipped"]["color_power_predictor"] = "Model not found."
    else:
        results["skipped"]["color_power_predictor"] = f"Predicted type {predicted_type} is not routed to color_power_predictor."

    if predicted_type == "SPOT LIGHT":
        if "spot_size_predictor" in models:
            spot_ctx = build_spot_context()
            results["predictions"]["spot_size_predictor"] = run_spot_size_predictor(
                models["spot_size_predictor"]["model"],
                spot_ctx,
                image_path,
            )
        else:
            results["skipped"]["spot_size_predictor"] = "Model not found."
    else:
        results["skipped"]["spot_size_predictor"] = f"Predicted type {predicted_type} is not routed to spot_size_predictor."

    if predicted_type == "SPOT LIGHT" and predicted_count == 1:
        if "angular_predictor" in models:
            ang_ctx = build_angular_context(models["angular_predictor"]["model"])
            angular_result = run_angular_predictor(
                models["angular_predictor"]["model"],
                ang_ctx,
                image_path,
                build_single_light_angular_overrides(results),
            )
            target = "predictions" if "skipped" not in angular_result else "skipped"
            results[target]["angular_predictor"] = angular_result if target == "predictions" else angular_result["skipped"]
        else:
            results["skipped"]["angular_predictor"] = "Model not found."
    else:
        results["skipped"]["angular_predictor"] = (
            f"Predicted count/type ({predicted_count}, {predicted_type}) is not routed to angular_predictor."
        )

    if predicted_type == "TRI LIGHTING" and predicted_count == 3:
        if "tri_angular_predictor" in models:
            tri_ctx = build_tri_context(models["tri_angular_predictor"]["model"])
            tri_result = run_tri_angular_predictor(models["tri_angular_predictor"]["model"], tri_ctx, image_path)
            target = "predictions" if "skipped" not in tri_result else "skipped"
            results[target]["tri_angular_predictor"] = tri_result if target == "predictions" else tri_result["skipped"]
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
        description="Run the inverse-rendering pipeline on a single image."
    )
    parser.add_argument("image", type=Path, help="Path to the input image")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result payload as JSON instead of a text summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = route_predictors(args.image)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results)


if __name__ == "__main__":
    main()

# run python pipeline.py path/to/image.png
