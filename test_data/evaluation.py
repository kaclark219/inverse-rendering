import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PRED_CSV = PROJECT_ROOT / "test_predictions.csv"
LEGACY_PRED_CSV = Path(__file__).resolve().parent / "test_predictions.csv"
DEFAULT_PRED_CSV = PIPELINE_PRED_CSV if PIPELINE_PRED_CSV.exists() else LEGACY_PRED_CSV
DEFAULT_ACTUAL_CSV = Path(__file__).resolve().parent / "test_metadata.csv"


def normalize_type(value: object) -> str:
	if pd.isna(value):
		return "UNKNOWN"
	raw = str(value).strip().upper()
	if raw in {"DOUBLE_SPOT_SPOT", "SPOT LIGHT", "SPOT"}:
		return "SPOT"
	if raw in {"TRI LIGHTING", "TRI"}:
		return "TRI"
	return raw


def image_key(series: pd.Series) -> pd.Series:
	return series.astype(str).map(lambda p: Path(p).name)


def safe_numeric(series: pd.Series) -> pd.Series:
	return pd.to_numeric(series, errors="coerce")


def parse_vector_cell(value: object, expected_len: int = 3) -> list[float] | None:
	if pd.isna(value):
		return None
	if isinstance(value, (list, tuple, np.ndarray)):
		raw = list(value)
	else:
		text = str(value).strip()
		if not text:
			return None
		try:
			raw = json.loads(text)
		except Exception:
			return None

	if not isinstance(raw, list) or len(raw) < expected_len:
		return None

	out: list[float] = []
	for i in range(expected_len):
		try:
			out.append(float(raw[i]))
		except Exception:
			return None
	return out


def vector_series_to_components(series: pd.Series, prefix: str) -> pd.DataFrame:
	rows = series.map(parse_vector_cell)
	return pd.DataFrame(
		{
			f"{prefix}_x": rows.map(lambda v: np.nan if v is None else v[0]),
			f"{prefix}_y": rows.map(lambda v: np.nan if v is None else v[1]),
			f"{prefix}_z": rows.map(lambda v: np.nan if v is None else v[2]),
		},
		index=series.index,
	)


def ensure_actual_cam_targets(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()

	for c in [
		"cam_pos_x", "cam_pos_y", "cam_pos_z",
		"cam_right_x", "cam_right_y", "cam_right_z",
		"cam_up_x", "cam_up_y", "cam_up_z",
		"cam_forward_x", "cam_forward_y", "cam_forward_z",
		"light0_pos_x", "light0_pos_y", "light0_pos_z",
		"light0_dir_x", "light0_dir_y", "light0_dir_z",
		"light0_dir_cam_x", "light0_dir_cam_y", "light0_dir_cam_z",
	]:
		if c in out.columns:
			out[c] = safe_numeric(out[c])

	pos_needed = {
		"cam_pos_x", "cam_pos_y", "cam_pos_z",
		"cam_right_x", "cam_right_y", "cam_right_z",
		"cam_up_x", "cam_up_y", "cam_up_z",
		"cam_forward_x", "cam_forward_y", "cam_forward_z",
		"light0_pos_x", "light0_pos_y", "light0_pos_z",
	}
	if pos_needed.issubset(out.columns):
		cam_pos = out[["cam_pos_x", "cam_pos_y", "cam_pos_z"]].to_numpy(dtype=float)
		# NORMALIZE camera axes to unit vectors (matching pipeline.py)
		cam_right = out[["cam_right_x", "cam_right_y", "cam_right_z"]].to_numpy(dtype=float)
		cam_right = cam_right / (np.linalg.norm(cam_right, axis=1, keepdims=True) + 1e-8)
		cam_up = out[["cam_up_x", "cam_up_y", "cam_up_z"]].to_numpy(dtype=float)
		cam_up = cam_up / (np.linalg.norm(cam_up, axis=1, keepdims=True) + 1e-8)
		cam_fwd = out[["cam_forward_x", "cam_forward_y", "cam_forward_z"]].to_numpy(dtype=float)
		cam_fwd = cam_fwd / (np.linalg.norm(cam_fwd, axis=1, keepdims=True) + 1e-8)
		cam_back = -cam_fwd
		light_pos = out[["light0_pos_x", "light0_pos_y", "light0_pos_z"]].to_numpy(dtype=float)
		rel = light_pos - cam_pos

		out["actual_light0_pos_cam_x"] = np.einsum("ij,ij->i", rel, cam_right)
		out["actual_light0_pos_cam_y"] = np.einsum("ij,ij->i", rel, cam_up)
		out["actual_light0_pos_cam_z"] = np.einsum("ij,ij->i", rel, cam_back)

	dir_cam_cols = {"light0_dir_cam_x", "light0_dir_cam_y", "light0_dir_cam_z"}
	if dir_cam_cols.issubset(out.columns):
		out["actual_light0_dir_cam_x"] = out["light0_dir_cam_x"]
		out["actual_light0_dir_cam_y"] = out["light0_dir_cam_y"]
		out["actual_light0_dir_cam_z"] = out["light0_dir_cam_z"]
	elif {
		"light0_dir_x", "light0_dir_y", "light0_dir_z",
		"cam_right_x", "cam_right_y", "cam_right_z",
		"cam_up_x", "cam_up_y", "cam_up_z",
		"cam_forward_x", "cam_forward_y", "cam_forward_z",
	}.issubset(out.columns):
		# NORMALIZE light direction to unit vector (matching pipeline.py)
		light_dir = out[["light0_dir_x", "light0_dir_y", "light0_dir_z"]].to_numpy(dtype=float)
		light_dir = light_dir / (np.linalg.norm(light_dir, axis=1, keepdims=True) + 1e-8)
		# NORMALIZE camera axes to unit vectors (matching pipeline.py)
		cam_right = out[["cam_right_x", "cam_right_y", "cam_right_z"]].to_numpy(dtype=float)
		cam_right = cam_right / (np.linalg.norm(cam_right, axis=1, keepdims=True) + 1e-8)
		cam_up = out[["cam_up_x", "cam_up_y", "cam_up_z"]].to_numpy(dtype=float)
		cam_up = cam_up / (np.linalg.norm(cam_up, axis=1, keepdims=True) + 1e-8)
		cam_back = -out[["cam_forward_x", "cam_forward_y", "cam_forward_z"]].to_numpy(dtype=float)
		cam_back = cam_back / (np.linalg.norm(cam_back, axis=1, keepdims=True) + 1e-8)

		out["actual_light0_dir_cam_x"] = np.einsum("ij,ij->i", light_dir, cam_right)
		out["actual_light0_dir_cam_y"] = np.einsum("ij,ij->i", light_dir, cam_up)
		out["actual_light0_dir_cam_z"] = np.einsum("ij,ij->i", light_dir, cam_back)

	return out


def direction_angle_metrics(df: pd.DataFrame) -> dict[str, float]:
	req = [
		"actual_light0_dir_cam_x", "actual_light0_dir_cam_y", "actual_light0_dir_cam_z",
		"pred_light0_dir_cam_x", "pred_light0_dir_cam_y", "pred_light0_dir_cam_z",
	]
	if any(c not in df.columns for c in req):
		return {"n": 0, "mae_deg": np.nan, "median_ae_deg": np.nan, "rmse_deg": np.nan}

	actual = df[["actual_light0_dir_cam_x", "actual_light0_dir_cam_y", "actual_light0_dir_cam_z"]].to_numpy(dtype=float)
	pred = df[["pred_light0_dir_cam_x", "pred_light0_dir_cam_y", "pred_light0_dir_cam_z"]].to_numpy(dtype=float)

	mask = np.isfinite(actual).all(axis=1) & np.isfinite(pred).all(axis=1)
	if mask.sum() == 0:
		return {"n": 0, "mae_deg": np.nan, "median_ae_deg": np.nan, "rmse_deg": np.nan}

	actual = actual[mask]
	pred = pred[mask]
	a_norm = np.linalg.norm(actual, axis=1)
	p_norm = np.linalg.norm(pred, axis=1)
	valid = (a_norm > 1e-8) & (p_norm > 1e-8)
	if valid.sum() == 0:
		return {"n": 0, "mae_deg": np.nan, "median_ae_deg": np.nan, "rmse_deg": np.nan}

	actual = actual[valid] / a_norm[valid, None]
	pred = pred[valid] / p_norm[valid, None]
	cosang = np.clip(np.sum(actual * pred, axis=1), -1.0, 1.0)
	angles = np.degrees(np.arccos(cosang))

	return {
		"n": int(len(angles)),
		"mae_deg": float(np.mean(np.abs(angles))),
		"median_ae_deg": float(np.median(np.abs(angles))),
		"rmse_deg": float(np.sqrt(np.mean(angles ** 2))),
	}


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
	mask = y_true.notna() & y_pred.notna()
	if mask.sum() == 0:
		return {
			"n": 0,
			"mae": np.nan,
			"median_ae": np.nan,
			"rmse": np.nan,
			"r2": np.nan,
			"mape_pct": np.nan,
			"smape_pct": np.nan,
			"nmae_pct_iqr": np.nan,
		}

	yt = y_true[mask].to_numpy(dtype=float)
	yp = y_pred[mask].to_numpy(dtype=float)
	err = yp - yt

	mae = float(np.mean(np.abs(err)))
	median_ae = float(np.median(np.abs(err)))
	rmse = float(np.sqrt(np.mean(err ** 2)))
	ss_res = float(np.sum(err ** 2))
	ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
	r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

	den = np.maximum(np.abs(yt), 1e-6)
	mape_pct = float(np.mean(np.abs(err) / den) * 100.0)
	smape_den = np.maximum((np.abs(yt) + np.abs(yp)) * 0.5, 1e-6)
	smape_pct = float(np.mean(np.abs(err) / smape_den) * 100.0)

	iqr = float(np.quantile(yt, 0.75) - np.quantile(yt, 0.25))
	nmae_pct_iqr = float((mae / iqr) * 100.0) if iqr > 1e-8 else np.nan

	return {
		"n": int(mask.sum()),
		"mae": mae,
		"median_ae": median_ae,
		"rmse": rmse,
		"r2": r2,
		"mape_pct": mape_pct,
		"smape_pct": smape_pct,
		"nmae_pct_iqr": nmae_pct_iqr,
	}


def classification_accuracy(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
	mask = y_true.notna() & y_pred.notna()
	if mask.sum() == 0:
		return {"n": 0, "accuracy": np.nan}
	acc = float((y_true[mask] == y_pred[mask]).mean())
	return {"n": int(mask.sum()), "accuracy": acc}


def bounded_score_from_error(error_value: float, scale: float) -> float:
	if pd.isna(error_value) or pd.isna(scale) or scale <= 0:
		return np.nan
	norm = float(error_value) / float(scale)
	return float(max(0.0, min(100.0, 100.0 * (1.0 - norm))))


def robust_scale_from_actual(y_true: pd.Series) -> float:
	v = safe_numeric(y_true).dropna()
	if v.empty:
		return np.nan
	q95 = float(v.quantile(0.95))
	q05 = float(v.quantile(0.05))
	scale = q95 - q05
	if scale <= 1e-8:
		scale = float(v.std())
	if scale <= 1e-8:
		scale = float(max(abs(v.mean()), 1.0))
	return scale


def light_count_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
	mask = y_true.notna() & y_pred.notna()
	if mask.sum() == 0:
		return {
			"n": 0,
			"exact_accuracy": np.nan,
			"within_one_accuracy": np.nan,
			"mae": np.nan,
			"rmse": np.nan,
		}

	yt = y_true[mask].to_numpy(dtype=float)
	yp = y_pred[mask].to_numpy(dtype=float)
	err = yp - yt

	return {
		"n": int(mask.sum()),
		"exact_accuracy": float(np.mean(np.round(yp) == np.round(yt))),
		"within_one_accuracy": float(np.mean(np.abs(np.round(yp) - np.round(yt)) <= 1)),
		"mae": float(np.mean(np.abs(err))),
		"rmse": float(np.sqrt(np.mean(err ** 2))),
	}


def plot_confusion_matrix(df: pd.DataFrame, out_path: Path) -> None:
	labels = sorted(set(df["actual_light_type_norm"].dropna()) | set(df["pred_light_type_norm"].dropna()))
	if not labels:
		return

	ctab = pd.crosstab(
		df["actual_light_type_norm"],
		df["pred_light_type_norm"],
		rownames=["Actual"],
		colnames=["Predicted"],
		dropna=False,
	)
	ctab = ctab.reindex(index=labels, columns=labels, fill_value=0)

	fig, ax = plt.subplots(figsize=(8, 6))
	im = ax.imshow(ctab.values, cmap="Blues")
	fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

	ax.set_xticks(np.arange(len(labels)))
	ax.set_yticks(np.arange(len(labels)))
	ax.set_xticklabels(labels, rotation=30, ha="right")
	ax.set_yticklabels(labels)
	ax.set_xlabel("Predicted")
	ax.set_ylabel("Actual")
	ax.set_title("Light Type Confusion Matrix")

	for i in range(len(labels)):
		for j in range(len(labels)):
			ax.text(j, i, str(int(ctab.values[i, j])), ha="center", va="center", color="black")

	fig.tight_layout()
	fig.savefig(out_path, dpi=180)
	plt.close(fig)


def plot_scatter_identity(
	df: pd.DataFrame,
	actual_col: str,
	pred_col: str,
	title: str,
	out_path: Path,
	log_scale: bool = False,
) -> None:
	x = safe_numeric(df[actual_col])
	y = safe_numeric(df[pred_col])
	mask = x.notna() & y.notna()
	if mask.sum() == 0:
		return

	xv = x[mask].to_numpy(dtype=float)
	yv = y[mask].to_numpy(dtype=float)

	fig, ax = plt.subplots(figsize=(7, 6))
	ax.scatter(xv, yv, alpha=0.6, s=22, edgecolors="none")

	lo = float(min(xv.min(), yv.min()))
	hi = float(max(xv.max(), yv.max()))
	ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="black", label="Ideal")

	if log_scale:
		ax.set_xscale("log")
		ax.set_yscale("log")

	ax.set_xlabel(f"Actual ({actual_col})")
	ax.set_ylabel(f"Predicted ({pred_col})")
	ax.set_title(title)
	ax.legend(loc="best")
	ax.grid(alpha=0.2)

	fig.tight_layout()
	fig.savefig(out_path, dpi=180)
	plt.close(fig)


def plot_color_error_hist(df: pd.DataFrame, out_path: Path) -> None:
	errs = []
	for ch in ["r", "g", "b"]:
		actual = safe_numeric(df[f"actual_color_{ch}"])
		pred = safe_numeric(df[f"pred_color_{ch}"])
		mask = actual.notna() & pred.notna()
		errs.extend(np.abs(pred[mask] - actual[mask]).tolist())

	if len(errs) == 0:
		return

	fig, ax = plt.subplots(figsize=(7, 5))
	ax.hist(errs, bins=30, color="#2b8cbe", edgecolor="white", alpha=0.9)
	ax.set_title("Absolute RGB Channel Error Distribution")
	ax.set_xlabel("|Pred - Actual|")
	ax.set_ylabel("Frequency")
	ax.grid(axis="y", alpha=0.2)
	fig.tight_layout()
	fig.savefig(out_path, dpi=180)
	plt.close(fig)


def plot_normalized_error_bars(metrics: dict, out_path: Path) -> None:
	items = [
		("Energy (MAPE%)", metrics["energy"]["mape_pct"]),
		("Spot (NMAE% IQR)", metrics["spot_cone_deg"]["nmae_pct_iqr"]),
		("Color R (MAE%)", metrics["color_r"]["mae"] * 100.0 if pd.notna(metrics["color_r"]["mae"]) else np.nan),
		("Color G (MAE%)", metrics["color_g"]["mae"] * 100.0 if pd.notna(metrics["color_g"]["mae"]) else np.nan),
		("Color B (MAE%)", metrics["color_b"]["mae"] * 100.0 if pd.notna(metrics["color_b"]["mae"]) else np.nan),
	]
	labels = [k for k, _ in items if pd.notna(_)]
	values = [v for _, v in items if pd.notna(v)]

	if not values:
		return

	fig, ax = plt.subplots(figsize=(8, 5))
	ax.bar(labels, values, color=["#3182bd", "#9ecae1", "#31a354", "#74c476", "#a1d99b"][: len(values)])
	ax.set_ylabel("Percent Error")
	ax.set_title("Normalized Error by Target")
	ax.grid(axis="y", alpha=0.2)
	for i, v in enumerate(values):
		ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
	ax.tick_params(axis="x", rotation=12)
	fig.tight_layout()
	fig.savefig(out_path, dpi=180)
	plt.close(fig)


def plot_model_scorecard(scorecard_df: pd.DataFrame, out_path: Path) -> None:
	if scorecard_df.empty:
		return

	df = scorecard_df.sort_values("score", ascending=True).copy()
	labels = df["model"].tolist()
	scores = df["score"].tolist()
	coverage = df["coverage_pct"].tolist()

	fig, ax = plt.subplots(figsize=(9, 5.5))
	colors = ["#2ca02c" if (not pd.isna(s) and s >= 80) else "#ff7f0e" if (not pd.isna(s) and s >= 60) else "#d62728" for s in scores]
	bars = ax.barh(labels, scores, color=colors, alpha=0.9)
	ax.set_xlim(0, 100)
	ax.set_xlabel("Model Score (0-100)")
	ax.set_title("Model Performance Scorecard")
	ax.grid(axis="x", alpha=0.2)

	for bar, score, cov in zip(bars, scores, coverage):
		if pd.isna(score):
			label = "N/A"
		else:
			label = f"{score:.1f}"
		if pd.notna(cov):
			label += f" | cov {cov:.1f}%"
		ax.text(min(99.0, bar.get_width() + 1.2), bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9)

	fig.tight_layout()
	fig.savefig(out_path, dpi=180)
	plt.close(fig)


def plot_overall_kpis(kpis: dict, out_path: Path) -> None:
	items = [
		("Overall Score", kpis.get("overall_score"), "pts"),
		("Type Accuracy", kpis.get("light_type_accuracy"), "%"),
		("Count Exact", kpis.get("light_count_exact_accuracy"), "%"),
		("Coverage", kpis.get("overall_coverage_pct"), "%"),
	]

	fig, ax = plt.subplots(figsize=(10, 2.8))
	ax.axis("off")

	xs = [0.12, 0.37, 0.62, 0.87]
	for x, (label, val, unit) in zip(xs, items):
		if pd.isna(val):
			text = "N/A"
		else:
			text = f"{val:.1f}{unit}"
		ax.text(x, 0.65, text, ha="center", va="center", fontsize=20, fontweight="bold")
		ax.text(x, 0.30, label, ha="center", va="center", fontsize=11)

	ax.set_title("Overall Evaluation KPIs", pad=8)
	fig.tight_layout()
	fig.savefig(out_path, dpi=180)
	plt.close(fig)


def build_eval_table(actual_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
	a = actual_df.copy()
	p = pred_df.copy()

	a["image_key"] = image_key(a["image_relpath"])
	p["image_key"] = image_key(p["image_path"])

	keep_pred_cols = [
		"image_key",
		"image_path",
		"pred_light_count",
		"pred_light_type",
		"pred_light_type_confidence",
		"pred_color_r",
		"pred_color_g",
		"pred_color_b",
		"pred_energy",
		"pred_spot_cone_deg",
		"pred_light0_position_cam",
		"pred_light0_direction_cam",
	]
	keep_actual_cols = [
		"image_key",
		"image_relpath",
		"num_active_lights",
		"light0_type",
		"light0_color_r",
		"light0_color_g",
		"light0_color_b",
		"light0_energy",
		"light0_spot_cone_deg",
		"cam_pos_x",
		"cam_pos_y",
		"cam_pos_z",
		"cam_right_x",
		"cam_right_y",
		"cam_right_z",
		"cam_up_x",
		"cam_up_y",
		"cam_up_z",
		"cam_forward_x",
		"cam_forward_y",
		"cam_forward_z",
		"light0_pos_x",
		"light0_pos_y",
		"light0_pos_z",
		"light0_dir_x",
		"light0_dir_y",
		"light0_dir_z",
		"light0_dir_cam_x",
		"light0_dir_cam_y",
		"light0_dir_cam_z",
	]

	merged = a[keep_actual_cols].merge(p[keep_pred_cols], on="image_key", how="left")

	merged = merged.rename(
		columns={
			"num_active_lights": "actual_light_count",
			"light0_type": "actual_light_type",
			"light0_color_r": "actual_color_r",
			"light0_color_g": "actual_color_g",
			"light0_color_b": "actual_color_b",
			"light0_energy": "actual_energy",
			"light0_spot_cone_deg": "actual_spot_cone_deg",
		}
	)

	merged["actual_light_type_norm"] = merged["actual_light_type"].map(normalize_type)
	merged["pred_light_type_norm"] = merged["pred_light_type"].map(normalize_type)

	if "pred_light0_position_cam" in merged.columns:
		pos_df = vector_series_to_components(merged["pred_light0_position_cam"], "pred_light0_pos_cam")
		merged = pd.concat([merged, pos_df], axis=1)
	if "pred_light0_direction_cam" in merged.columns:
		dir_df = vector_series_to_components(merged["pred_light0_direction_cam"], "pred_light0_dir_cam")
		merged = pd.concat([merged, dir_df], axis=1)

	merged = ensure_actual_cam_targets(merged)

	for col in [
		"actual_light_count",
		"pred_light_count",
		"actual_color_r",
		"actual_color_g",
		"actual_color_b",
		"pred_color_r",
		"pred_color_g",
		"pred_color_b",
		"actual_energy",
		"pred_energy",
		"actual_spot_cone_deg",
		"pred_spot_cone_deg",
		"actual_light0_pos_cam_x",
		"actual_light0_pos_cam_y",
		"actual_light0_pos_cam_z",
		"pred_light0_pos_cam_x",
		"pred_light0_pos_cam_y",
		"pred_light0_pos_cam_z",
		"actual_light0_dir_cam_x",
		"actual_light0_dir_cam_y",
		"actual_light0_dir_cam_z",
		"pred_light0_dir_cam_x",
		"pred_light0_dir_cam_y",
		"pred_light0_dir_cam_z",
	]:
		if col in merged.columns:
			merged[col] = safe_numeric(merged[col])

	merged["abs_err_energy"] = (merged["pred_energy"] - merged["actual_energy"]).abs()
	merged["abs_err_spot_cone_deg"] = (merged["pred_spot_cone_deg"] - merged["actual_spot_cone_deg"]).abs()
	merged["abs_err_color_r"] = (merged["pred_color_r"] - merged["actual_color_r"]).abs()
	merged["abs_err_color_g"] = (merged["pred_color_g"] - merged["actual_color_g"]).abs()
	merged["abs_err_color_b"] = (merged["pred_color_b"] - merged["actual_color_b"]).abs()
	merged["rgb_l2_error"] = np.sqrt(
		(merged["pred_color_r"] - merged["actual_color_r"]) ** 2
		+ (merged["pred_color_g"] - merged["actual_color_g"]) ** 2
		+ (merged["pred_color_b"] - merged["actual_color_b"]) ** 2
	)
	if all(c in merged.columns for c in ["pred_light0_pos_cam_x", "pred_light0_pos_cam_y", "pred_light0_pos_cam_z", "actual_light0_pos_cam_x", "actual_light0_pos_cam_y", "actual_light0_pos_cam_z"]):
		merged["light0_pos_cam_l2_error"] = np.sqrt(
			(merged["pred_light0_pos_cam_x"] - merged["actual_light0_pos_cam_x"]) ** 2
			+ (merged["pred_light0_pos_cam_y"] - merged["actual_light0_pos_cam_y"]) ** 2
			+ (merged["pred_light0_pos_cam_z"] - merged["actual_light0_pos_cam_z"]) ** 2
		)

	return merged


def run_evaluation(pred_csv: Path, actual_csv: Path, out_dir: Path) -> None:
	pred_df = pd.read_csv(pred_csv, low_memory=False)
	actual_df = pd.read_csv(actual_csv, low_memory=False)

	eval_df = build_eval_table(actual_df, pred_df)
	out_dir.mkdir(parents=True, exist_ok=True)

	metrics = {
		"light_type": classification_accuracy(eval_df["actual_light_type_norm"], eval_df["pred_light_type_norm"]),
		"light_count": light_count_metrics(eval_df["actual_light_count"], eval_df["pred_light_count"]),
		"energy": regression_metrics(eval_df["actual_energy"], eval_df["pred_energy"]),
		"spot_cone_deg": regression_metrics(eval_df["actual_spot_cone_deg"], eval_df["pred_spot_cone_deg"]),
		"color_r": regression_metrics(eval_df["actual_color_r"], eval_df["pred_color_r"]),
		"color_g": regression_metrics(eval_df["actual_color_g"], eval_df["pred_color_g"]),
		"color_b": regression_metrics(eval_df["actual_color_b"], eval_df["pred_color_b"]),
		"light0_pos_cam_x": regression_metrics(eval_df.get("actual_light0_pos_cam_x", pd.Series(dtype=float)), eval_df.get("pred_light0_pos_cam_x", pd.Series(dtype=float))),
		"light0_pos_cam_y": regression_metrics(eval_df.get("actual_light0_pos_cam_y", pd.Series(dtype=float)), eval_df.get("pred_light0_pos_cam_y", pd.Series(dtype=float))),
		"light0_pos_cam_z": regression_metrics(eval_df.get("actual_light0_pos_cam_z", pd.Series(dtype=float)), eval_df.get("pred_light0_pos_cam_z", pd.Series(dtype=float))),
		"light0_dir_angle_deg": direction_angle_metrics(eval_df),
	}
	metrics["rgb_l2_mean"] = float(eval_df["rgb_l2_error"].dropna().mean()) if eval_df["rgb_l2_error"].notna().any() else np.nan
	metrics["light0_pos_cam_l2_mean"] = float(eval_df["light0_pos_cam_l2_error"].dropna().mean()) if "light0_pos_cam_l2_error" in eval_df.columns and eval_df["light0_pos_cam_l2_error"].notna().any() else np.nan

	total_n = len(eval_df)
	coverage = {
		"light_type": float(100.0 * eval_df["pred_light_type_norm"].notna().mean()) if total_n > 0 else np.nan,
		"light_count": float(100.0 * eval_df["pred_light_count"].notna().mean()) if total_n > 0 else np.nan,
		"energy": float(100.0 * eval_df["pred_energy"].notna().mean()) if total_n > 0 else np.nan,
		"spot_cone_deg": float(100.0 * eval_df["pred_spot_cone_deg"].notna().mean()) if total_n > 0 else np.nan,
		"color": float(100.0 * eval_df[["pred_color_r", "pred_color_g", "pred_color_b"]].notna().all(axis=1).mean()) if total_n > 0 else np.nan,
		"light0_position_cam": float(100.0 * eval_df[["pred_light0_pos_cam_x", "pred_light0_pos_cam_y", "pred_light0_pos_cam_z"]].notna().all(axis=1).mean()) if total_n > 0 and all(c in eval_df.columns for c in ["pred_light0_pos_cam_x", "pred_light0_pos_cam_y", "pred_light0_pos_cam_z"]) else np.nan,
		"light0_direction_cam": float(100.0 * eval_df[["pred_light0_dir_cam_x", "pred_light0_dir_cam_y", "pred_light0_dir_cam_z"]].notna().all(axis=1).mean()) if total_n > 0 and all(c in eval_df.columns for c in ["pred_light0_dir_cam_x", "pred_light0_dir_cam_y", "pred_light0_dir_cam_z"]) else np.nan,
	}

	energy_score = bounded_score_from_error(metrics["energy"]["median_ae"], robust_scale_from_actual(eval_df["actual_energy"]))
	spot_score = bounded_score_from_error(metrics["spot_cone_deg"]["mae"], robust_scale_from_actual(eval_df["actual_spot_cone_deg"]))
	color_mae_mean = float(np.nanmean([metrics["color_r"]["mae"], metrics["color_g"]["mae"], metrics["color_b"]["mae"]]))
	color_median_mean = float(np.nanmean([metrics["color_r"]["median_ae"], metrics["color_g"]["median_ae"], metrics["color_b"]["median_ae"]]))
	color_score = bounded_score_from_error(color_mae_mean, 1.0)

	count_exact = metrics["light_count"]["exact_accuracy"]
	count_within_one = metrics["light_count"]["within_one_accuracy"]
	count_score = np.nan if pd.isna(count_exact) else float(100.0 * (0.7 * count_exact + 0.3 * (count_within_one if pd.notna(count_within_one) else count_exact)))

	type_score = np.nan if pd.isna(metrics["light_type"]["accuracy"]) else float(100.0 * metrics["light_type"]["accuracy"])

	# Position prediction score based on L2 error in camera space, scaled by the
	# spread of actual light positions so improvements are reflected across runs.
	pos_l2_mean = metrics["light0_pos_cam_l2_mean"]
	if all(c in eval_df.columns for c in ["actual_light0_pos_cam_x", "actual_light0_pos_cam_y", "actual_light0_pos_cam_z"]):
		actual_pos_l2 = np.sqrt(
			eval_df["actual_light0_pos_cam_x"] ** 2
			+ eval_df["actual_light0_pos_cam_y"] ** 2
			+ eval_df["actual_light0_pos_cam_z"] ** 2
		)
		pos_scale = robust_scale_from_actual(actual_pos_l2)
	else:
		pos_scale = np.nan
	if pd.isna(pos_scale) or pos_scale <= 1e-8:
		pos_scale = 15.0
	pos_score = bounded_score_from_error(pos_l2_mean, pos_scale)
	# Direction prediction score based on angular error against a fixed 90-degree baseline.
	dir_mae_deg = metrics["light0_dir_angle_deg"]["mae_deg"]
	dir_score = bounded_score_from_error(dir_mae_deg, 90.0)

	scorecard = [
		{"model": "light_type_classifier", "score": type_score, "coverage_pct": coverage["light_type"]},
		{"model": "light_count_detector", "score": count_score, "coverage_pct": coverage["light_count"]},
		{"model": "color_power_predictor", "score": color_score, "coverage_pct": coverage["color"]},
		{"model": "spot_size_predictor", "score": spot_score, "coverage_pct": coverage["spot_cone_deg"]},
		{"model": "energy_head", "score": energy_score, "coverage_pct": coverage["energy"]},
		{"model": "position_predictor", "score": pos_score, "coverage_pct": coverage["light0_position_cam"]},
		{"model": "direction_predictor", "score": dir_score, "coverage_pct": coverage["light0_direction_cam"]},
	]
	scorecard_df = pd.DataFrame(scorecard)

	weights = {
		"light_type_classifier": 0.20,
		"light_count_detector": 0.15,
		"color_power_predictor": 0.15,
		"spot_size_predictor": 0.10,
		"energy_head": 0.15,
		"position_predictor": 0.15,
		"direction_predictor": 0.10,
	}

	weighted = []
	for _, row in scorecard_df.iterrows():
		s = row["score"]
		w = weights[row["model"]]
		if pd.notna(s):
			weighted.append((float(s), w))

	if weighted:
		den = float(sum(w for _, w in weighted))
		overall_score = float(sum(s * w for s, w in weighted) / den)
	else:
		overall_score = np.nan

	overall_coverage_pct = float(np.nanmean(list(coverage.values()))) if coverage else np.nan

	kpis = {
		"overall_score": overall_score,
		"overall_coverage_pct": overall_coverage_pct,
		"light_type_accuracy": np.nan if pd.isna(metrics["light_type"]["accuracy"]) else float(100.0 * metrics["light_type"]["accuracy"]),
		"light_count_exact_accuracy": np.nan if pd.isna(count_exact) else float(100.0 * count_exact),
	}

	metrics["coverage_pct"] = coverage
	metrics["scorecard"] = scorecard
	metrics["kpis"] = kpis

	eval_csv = out_dir / "evaluation_table.csv"
	eval_df.to_csv(eval_csv, index=False)

	metrics_json = out_dir / "metrics_summary.json"
	with open(metrics_json, "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)

	scorecard_csv = out_dir / "model_scorecard.csv"
	scorecard_df.to_csv(scorecard_csv, index=False)

	plt.style.use("seaborn-v0_8-whitegrid")
	plot_confusion_matrix(eval_df, out_dir / "chart_confusion_matrix.png")
	plot_scatter_identity(
		eval_df,
		"actual_energy",
		"pred_energy",
		"Energy: Actual vs Predicted (Log Scale)",
		out_dir / "chart_energy_scatter.png",
		log_scale=True,
	)
	plot_scatter_identity(
		eval_df,
		"actual_spot_cone_deg",
		"pred_spot_cone_deg",
		"Spot Cone: Actual vs Predicted",
		out_dir / "chart_spot_cone_scatter.png",
	)
	plot_color_error_hist(eval_df, out_dir / "chart_rgb_error_hist.png")
	plot_normalized_error_bars(metrics, out_dir / "chart_normalized_error_by_target.png")
	plot_model_scorecard(scorecard_df, out_dir / "chart_model_scorecard.png")
	plot_overall_kpis(kpis, out_dir / "chart_overall_kpis.png")

	print(f"Saved evaluation table: {eval_csv}")
	print(f"Saved metrics summary: {metrics_json}")
	print(f"Saved model scorecard: {scorecard_csv}")
	print(f"Saved charts in: {out_dir}")
	print("\nKey metrics:")
	print(f"  Overall score (0-100): {kpis['overall_score']:.2f}")
	print(f"  Overall coverage (%): {kpis['overall_coverage_pct']:.2f}")
	print(f"  Light type accuracy (%): {kpis['light_type_accuracy']:.2f} (n={metrics['light_type']['n']})")
	print(f"  Light count exact acc (%): {kpis['light_count_exact_accuracy']:.2f}")
	print(f"  Light count within-1 acc (%): {metrics['light_count']['within_one_accuracy'] * 100.0:.2f}")
	print(f"  Light count MAE: {metrics['light_count']['mae']:.4f}")
	print(f"  Energy median AE: {metrics['energy']['median_ae']:.4f}")
	print(f"  Energy MAPE (%): {metrics['energy']['mape_pct']:.2f}")
	print(f"  Spot cone MAE: {metrics['spot_cone_deg']['mae']:.4f}")
	print(f"  Light0 pos cam MAE (x,y,z): {metrics['light0_pos_cam_x']['mae']:.4f}, {metrics['light0_pos_cam_y']['mae']:.4f}, {metrics['light0_pos_cam_z']['mae']:.4f}")
	print(f"  Light0 pos cam L2 mean: {metrics['light0_pos_cam_l2_mean']:.4f}")
	print(f"  Light0 dir angle MAE (deg): {metrics['light0_dir_angle_deg']['mae_deg']:.4f}")
	print(f"  Mean RGB median AE: {color_median_mean:.4f}")
	print(f"  Mean RGB MAE: {color_mae_mean:.4f}")
	print(f"  Mean RGB L2 error: {metrics['rgb_l2_mean']:.4f}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate predictions against actual metadata and produce charts.")
	parser.add_argument("--pred-csv", type=Path, default=DEFAULT_PRED_CSV, help="Prediction CSV path")
	parser.add_argument("--actual-csv", type=Path, default=DEFAULT_ACTUAL_CSV, help="Actual metadata CSV path")
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=PROJECT_ROOT / "test_data" / "evaluation_outputs",
		help="Output directory for metrics and charts",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	run_evaluation(args.pred_csv, args.actual_csv, args.out_dir)


if __name__ == "__main__":
	main()
