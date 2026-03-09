from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {
	".png",
	".jpg",
	".jpeg"
}

CATEGORY_FOLDERS = ("base", "color", "power", "spot_size")


def normalize_relpath(path_str: str) -> str:
	normalized = path_str.strip().replace("\\", "/")
	while normalized.startswith("./"):
		normalized = normalized[2:]
	return normalized


def load_metadata_paths(metadata_csv: Path) -> set[str]:
	paths: set[str] = set()
	with metadata_csv.open("r", encoding="utf-8-sig", newline="") as f:
		reader = csv.DictReader(f)
		if not reader.fieldnames:
			raise ValueError(f"No header found in CSV: {metadata_csv}")

		image_col = "image_relpath" if "image_relpath" in reader.fieldnames else reader.fieldnames[0]

		for row in reader:
			raw = row.get(image_col, "")
			if raw:
				paths.add(normalize_relpath(raw))
	return paths


def load_image_paths(images_dir: Path) -> set[str]:
	if not images_dir.exists():
		raise FileNotFoundError(f"Images directory not found: {images_dir}")

	dataset_root = images_dir.parent
	paths: set[str] = set()

	for file_path in images_dir.rglob("*"):
		if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
			rel = file_path.relative_to(dataset_root).as_posix()
			paths.add(normalize_relpath(rel))
	return paths


def count_by_category(paths: set[str]) -> dict[str, int]:
	counts = {category: 0 for category in CATEGORY_FOLDERS}
	counts["uncategorized"] = 0

	for rel in paths:
		parts = normalize_relpath(rel).split("/")
		matched = False
		for category in CATEGORY_FOLDERS:
			if category in parts:
				counts[category] += 1
				matched = True
				break

		if not matched:
			counts["uncategorized"] += 1

	return counts


def main() -> int:
	parser = argparse.ArgumentParser(description="Validate metadata.csv <-> images/ consistency")
	parser.add_argument(
		"--metadata",
		type=Path,
		default=Path(__file__).resolve().parent / "metadata.csv",
		help="Path to metadata.csv",
	)
	parser.add_argument(
		"--images-dir",
		type=Path,
		default=Path(__file__).resolve().parent / "images",
		help="Path to images directory",
	)
	args = parser.parse_args()

	metadata_csv = args.metadata.resolve()
	images_dir = args.images_dir.resolve()

	if not metadata_csv.exists():
		print(f"ERROR: Metadata CSV not found: {metadata_csv}", file=sys.stderr)
		return 2

	metadata_paths = load_metadata_paths(metadata_csv)
	image_paths = load_image_paths(images_dir)

	missing_on_disk = sorted(metadata_paths - image_paths)
	missing_in_metadata = sorted(image_paths - metadata_paths)
	metadata_counts = count_by_category(metadata_paths)
	image_counts = count_by_category(image_paths)

	print(f"Metadata entries: {len(metadata_paths)}")
	print(f"Image files:      {len(image_paths)}")
	print(f"Missing on disk (metadata -> image): {len(missing_on_disk)}")
	print(f"Missing in metadata (image -> metadata): {len(missing_in_metadata)}")

	print("\nCounts by folder category:")
	for category in CATEGORY_FOLDERS:
		print(
			f"{category:>9} | metadata={metadata_counts[category]:4d}"
			f" | on_disk={image_counts[category]:4d}"
		)
	if metadata_counts["uncategorized"] or image_counts["uncategorized"]:
		print(
			f"uncategorized | metadata={metadata_counts['uncategorized']:4d}"
			f" | on_disk={image_counts['uncategorized']:4d}"
		)

	if missing_on_disk:
		print("\nPaths present in metadata but missing on disk:")
		for p in missing_on_disk:
			print(p)

	if missing_in_metadata:
		print("\nImage files present on disk but missing from metadata:")
		for p in missing_in_metadata:
			print(p)

	if missing_on_disk or missing_in_metadata:
		return 1

	print("\nAll images and metadata entries match.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
