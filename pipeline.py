import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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


def route_predictors(image_path: Path, runtime: dict) -> dict:
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
    runtime = initialize_pipeline_runtime()
    rows = []

    for img_path in PROJECT_ROOT.glob("test_data/images/*"):
        results = route_predictors(img_path, runtime)
        rows.append(build_prediction_csv_row(results))

    out_csv = PROJECT_ROOT / "test_predictions.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved predictions CSV: {out_csv}")


if __name__ == "__main__":
    main()

# run python pipeline.py path/to/image.png
