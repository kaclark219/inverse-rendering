# Research Log

Chronological lab notebook.

---

## Entry Template

### Date
YYYY-MM-DD

### Session Goal
(What did you intend to accomplish?)

### Changes Made
- Code/files changed:
- Data used/generated:
- Model/config used:

### Results
- Quantitative metrics:
- Qualitative observations:

### Interpretation
(What do these results likely mean?)

### Risks / Caveats
(Confounders, data leakage concerns, instability, etc.)

### Next Actions
- [ ]
- [ ]

---

## 2026-01-23

### Session Goal
Conduct literature review to ground inverse lighting project direction.

### Changes Made
- Code/files changed: None (reading + notes only).
- Data used/generated:
	- Research papers: Ramamoorthi & Hanrahan (2001), Yu, Yang & Xiao (2007), Sengupta et al. (2019), Zhang et al. (2021), Mengistu et al. (2025), Zaman et al. (2024), Aittala (2010).
- Model/config used: Conceptual review only.

### Results
- Quantitative metrics: ~6.5 hours literature review (Jan 23–24).
- Qualitative observations:
	- Lighting recoverability depends on frequency content (spherical harmonics framework).
	- Specular highlights preserve higher-frequency lighting information.
	- Modern neural inverse rendering separates geometry, reflectance, and lighting.
	- Lighting should be encoded separately from reflectance to avoid leakage.
	- Evaluation often includes relighting consistency and per-pixel error.
	- AR-style lighting models use low-dimensional directional + ambient light.

### Interpretation
Lighting representation should be low-dimensional, interpretable (direction, size, intensity, color, ambient), and recoverable via highlight/shadow cues. Synthetic training with real-scene validation is appropriate, with optimization-based inverse rendering as a possible supplement.

### Risks / Caveats
Inverse rendering is inherently under-constrained; learned priors may overfit to synthetic biases.

### Next Actions
- [x] Define lighting parameterization for project (direction, intensity, color, etc.).
- [x] Select subset of dataset for initial baseline.

---

## 2026-01-25

### Session Goal
Define data extraction workplan.

### Changes Made
- Code/files changed: Planning only.
- Data used/generated:
	- Selected shapes: cone, cube, cylinder, icosphere, sphere, torus.
	- Material: plastic glossy.
	- Light types: area, point, spotlight.
- Model/config used: Dataset design phase.

### Results
- Quantitative metrics: ~0.25 hrs summary reflection.
- Qualitative observations:
	- Need Blender script to extract light info, world light transform, and camera pose.
	- Dataset must support interpretable parameter extraction.

### Interpretation
Structured metadata extraction is required for supervised learning.

### Risks / Caveats
Incorrect coordinate frames (world vs camera) could silently corrupt labels.

### Next Actions
- [x] Implement Blender CSV export script.
- [x] Verify coordinate consistency.

---

## 2026-01-28

### Session Goal
Explore Blender file structure and begin data extraction.

### Changes Made
- Code/files changed: Initial Blender Python CSV export script.
- Data used/generated: 883 frames across 17 captures per material/environment.
- Model/config used: Raw Blender timeline data.

### Results
- Quantitative metrics: ~2 hrs.
- Qualitative observations:
	- File structured as keyframed timeline.
	- Materials controlled via switching nodes.
	- World nodes switch lighting modes.
	- Export appeared to repeat same lighting per frame.

### Interpretation
Lighting state may not have been properly toggled before export.

### Risks / Caveats
Silent label duplication could cause data leakage or degenerate training.

### Next Actions
- [x] Ensure lights explicitly toggled on/off before export.
- [x] Validate per-frame lighting changes.

---

## 2026-01-31

### Session Goal
Fix Blender data export formatting.

### Changes Made
- Code/files changed:
	- Rewrote Blender export script.
	- Organized output by file path naming.
- Data used/generated: Exported Cycles data.
- Model/config used: Plastic glossy material standardization.

### Results
- Quantitative metrics: ~3 hrs.
- Qualitative observations:
	- Ensured light activation per mode.
	- Organized space into area light, point light, spotlight, HDRI sunlight, HDRI overcast.
	- High contrast (“punchy”) render concerns emerged.
	- Several iterations were required to fix CSV duplication.

### Interpretation
Dataset export became structurally correct, but naming/path issues remained.

### Risks / Caveats
- High contrast lighting may bias learning.
- File naming inconsistencies may mislabel images.

### Next Actions
- [x] Fix file path naming system.
- [x] Prepare dataset for model ingestion.

---

## 2026-02-01

### Session Goal
Preprocess data and train baseline CNN.

### Changes Made
- Code/files changed:
	- Pandas preprocessing pipeline.
	- Baseline CNN implementation.
- Data used/generated: 7,956 images (224×224×3).
- Model/config used:
	- 10-dimensional lighting output.
	- ReLU activations.
	- Adam optimizer.
	- MSE + MAE metrics.
	- 44,529,994 trainable parameters.
	- 80/20 train/test split.
	- 20 epochs (~1 hour).

### Results
- Quantitative metrics: Successful training run.
- Qualitative observations: “Good” results, but possible data leakage across categories.

### Interpretation
Baseline model learned useful signal, but split strategy likely inflated performance.

### Risks / Caveats
Leakage between shape/material categories may inflate performance.

### Next Actions
- [x] Fix CSV file paths.
- [ ] Implement stricter split strategy.

---

## 2026-02-02

### Session Goal
Fix dataset labeling issues.

### Changes Made
- Code/files changed: Corrected master spreadsheet filepaths.
- Data used/generated: Revalidated dataset loading.
- Model/config used: Existing baseline pipeline.

### Results
- Quantitative metrics: ~1.5 hrs.
- Qualitative observations: Dataset now loads consistently.

### Interpretation
Dataset integrity improved.

### Risks / Caveats
Subtle leakage may still remain.

### Next Actions
- [x] Refactor model focus to reduce instability.

---

## 2026-02-03

### Session Goal
Refactor model to single spotlight.

### Changes Made
- Code/files changed:
	- Rewrote baseline model.
	- Separated loss terms per predicted component.
- Data used/generated: Spotlight-only subset.
- Model/config used: Separate loss scaling for position/direction/etc.

### Results
- Quantitative metrics: ~3 hrs.
- Qualitative observations:
	- More stable learning focus.
	- Prevented large-magnitude terms dominating loss.

### Interpretation
Single-light formulation better matched stage-light project focus.

### Risks / Caveats
May oversimplify future multi-light scenarios.

### Next Actions
- [x] Improve angular prediction stability.

---

## 2026-02-05

### Session Goal
Improve training stability.

### Changes Made
- Code/files changed: Warm-up + finetuning training schedule.
- Data used/generated: Same spotlight dataset.
- Model/config used: Modified training schedule.

### Results
- Quantitative metrics: ~2.25 hrs.
- Qualitative observations: Angle prediction remained unstable.

### Interpretation
Angle regression was still the primary difficulty.

### Risks / Caveats
Angle representation may need vector normalization or alternative encoding.

### Next Actions
- [x] Perform visual validation in Blender.

---

## 2026-02-08

### Session Goal
Visually validate predictions in Blender.

### Changes Made
- Code/files changed: Blender scripting to reconstruct scenes from predictions.
- Data used/generated: Rendered predicted scenes.
- Model/config used: Angular + position model.

### Results
- Quantitative metrics: ~2.5 hrs.
- Qualitative observations:
	- Axis convention issues.
	- Renderings were visually similar but numerically misaligned.

### Interpretation
Coordinate frame mismatch existed between training labels and Blender scene conventions.

### Risks / Caveats
Camera vs world space confusion could invalidate evaluation.

### Next Actions
- [x] Standardize camera-space representation.

---

## 2026-02-10

### Session Goal
Reset model scope.

### Changes Made
- Code/files changed:
	- Scrapped complex models.
	- Re-scoped to single spotlight (position + direction).
- Data used/generated: Camera-space standardized labels.
- Model/config used: Direction vector normalized.

### Results
- Quantitative metrics: ~3 hrs.
- Qualitative observations:
	- Simplified problem definition.
	- Improved conceptual clarity.

### Interpretation
Over-complex modeling had been harming progress.

### Risks / Caveats
Scope reduction delayed color/intensity modeling.

### Next Actions
- [x] Evaluate angular error in Blender.

---

## 2026-02-13

### Session Goal
Evaluate model via Blender simulation.

### Changes Made
- Code/files changed: Blender scene recreation script.
- Data used/generated: Simplified recreated dataset scenes.
- Model/config used: Angular + position spotlight model.

### Results
- Quantitative metrics: Direction angle error: 2.79°.
- Qualitative observations:
	- Major improvement.
	- Visual outputs were extremely similar.

### Interpretation
Camera-space normalization plus simplified scope significantly improved angular learning.

### Risks / Caveats
Evaluation remained limited to simplified synthetic scenes.

### Next Actions
- [x] Extend to color + intensity regression.

---

## 2026-02-15

### Session Goal
Create color/intensity dataset.

### Changes Made
- Code/files changed: Blender script generating randomized RGB + intensity.
- Data used/generated: 100 images + CSV file.
- Model/config used: Regression model for color + energy.

### Results
- Quantitative metrics: Poor regression performance.
- Qualitative observations:
	- Likely weak specular cues.
	- Dataset likely too small.

### Interpretation
Color/intensity recovery likely needs stronger lighting cues and larger datasets.

### Risks / Caveats
Diffuse surfaces may not encode enough signal.

### Next Actions
- [x] Expand dataset size.
- [ ] Increase specular visibility.

---

## 2026-02-16

### Session Goal
Scale dataset for color + energy training.

### Changes Made
- Code/files changed: Blender automation scripts (multiple iterations).
- Data used/generated:
	- 12 dataset attempts.
	- Final 1,000 image dataset.
- Model/config used: Multiple material/background/light variations.

### Results
- Quantitative metrics: ~5 hrs (Feb 16, 18, 21).
- Qualitative observations:
	- Extensive experimentation with shapes, material parameters, backgrounds, light ranges, and plug-ins.

### Interpretation
Dataset quality strongly affects regression viability.

### Risks / Caveats
Distribution shifts across attempts may complicate training.

### Next Actions
- [x] Train color/energy regression on 1,000-image dataset.

---

## 2026-02-22

### Session Goal
Train color/energy regression model.

### Changes Made
- Code/files changed: Color/energy training pipeline.
- Data used/generated: 1,000 image dataset + CSV.
- Model/config used: Regression model predicting RGB + energy.

### Results
- Quantitative metrics: Ongoing refinement (~2 hrs).
- Qualitative observations: Early-stage results under refinement.

### Interpretation
Dataset scale now supports meaningful training experiments.

### Risks / Caveats
Energy scaling and perceptual color differences may require log-space or normalized targets.

### Next Actions
- [ ] Evaluate performance in log and linear energy space.
- [x] Perform Blender-based visual validation.

---

## 2026-02-23

### Session Goal
Design and train a tri-light angular regression model predicting key, fill, and back light directions in camera space.

### Changes Made
- Code/files changed:
	- Implemented tri-light angular regression architecture.
	- Defined structured output for three light sources (key, fill, back).
	- Modified loss function to handle multiple light heads.
- Data used/generated:
	- Tri-light subset of synthetic dataset.
	- Camera-space direction vectors as regression targets.
- Model/config used:
	- Multi-output regression model.
	- Camera-space normalization.
	- Adjusted loss weighting per light.

### Results
- Quantitative metrics: Successful training run (no divergence).
- Qualitative observations:
	- Model converged stably after adjusting multi-light loss.
	- Sample predictions produced plausible direction vectors.
	- Visual comparison between predicted vs. ground truth on unseen sample showed reasonable alignment.

### Interpretation
Structured output per light source and camera-space prediction improved stability. Separating lights in the loss prevented one source from dominating gradients.

### Risks / Caveats
- Angular error not yet fully quantified across dataset.
- Camera-space dependency may limit generalization if metadata preprocessing differs.
- Inference requires tabular metadata alignment.

### Next Actions
- [x] Integrate tri-light model into unified testing framework.

---

## 2026-02-27

### Session Goal
Train light count detector and light type classifier for structured lighting prediction pipeline.

### Changes Made
- Code/files changed:
	- Implemented light count detector (binary: 1 vs. 3 lights in current dataset).
	- Implemented light type classifier (Area, Point, Spot, Tri).
	- Adjusted training parameters across multiple runs.
- Data used/generated:
	- ~20k labeled samples for light count training.
	- Categorized lighting dataset for type classification.
- Model/config used:
	- Classification architectures.
	- Softmax outputs.
	- Adjusted epochs, batch size, and learning rate.
	- Attempted 50-epoch training (crashed).

### Results
- Quantitative metrics:
	- Light count model trained successfully on ~20k samples.
	- Light type classifier achieved usable accuracy after parameter tuning.
- Qualitative observations:
	- Training light type classifier was time-intensive.
	- Increasing epochs to 50 caused instability/crash.
	- Hyperparameter adjustments improved confidence calibration.

### Interpretation
Structured classification layer successfully segments lighting scenario before regression. Light type prediction is a necessary gating step for routing to correct regression head.

### Risks / Caveats
- Class imbalance may bias count detection.
- Crashes at high epoch counts suggest memory or instability issues.
- Classification errors will cascade into regression routing.

### Next Actions
- [x] Save stable model checkpoints.
- [x] Prepare unified inference script.

---

## 2026-03-01

### Session Goal
Integrate saved models into full system testing pipeline and validate on random image samples.

### Changes Made
- Code/files changed:
	- Implemented unified testing script to load saved models.
	- Routed predictions based on count/type outputs.
	- Added error handling for incompatible model inputs.
- Data used/generated: Random image samples from render-lighting dataset.
- Model/config used: Pretrained count, type, and regression models.

### Results
- Quantitative metrics: Successful loading of saved models.
- Qualitative observations:
	- Numerous errors due to inconsistent input requirements:
	  - Image-only models vs. tabular + image models.
	  - Different preprocessing pipelines per lighting setup.
	- Required multiple workarounds to handle heterogeneous data structures.

### Interpretation
The pipeline architecture works conceptually but requires stricter standardization of preprocessing. Inconsistent metadata handling is now the primary bottleneck.

### Risks / Caveats
- Angular models cannot run without matching tabular preprocessing.
- Model interoperability depends on consistent camera-space conventions.
- High risk of silent failure if routing logic mismatches light type.

### Next Actions
- [x] Create documentation for all work completed thus far.
- [x] Find/create more data for a more diversified set & generalized results.

---

## 2026-03-03

### Session Goal
Initialize thesis-style documentation framework for reproducible progress tracking.

### Changes Made
- Created a dedicated `docs/` folder for preparation.
- Added overview, research-log, and struggles/decisions documents.

### Results
- Documentation scaffold is in place.
- Future sessions can now be logged in a consistent structure.

### Interpretation
A standard note format should reduce memory-based reporting errors and simplify writing documentation later.

### Risks / Caveats
Backfilling older experiments may be incomplete where metadata was not recorded.

---

## 2026-03-08

### Session Goal
Create new image dataset with controlled variations in color, power, and spot size for single and double spotlight configurations.

### Changes Made
- Code/files changed: 
	- Blender scripts to generate synthetic lighting dataset.
	- Automated rendering pipeline with randomized light parameters.
- Data used/generated:
	- Created inverse_rendering_dataset with 2,313 rendered images total.
	- Images organized by light count (single/double) and parameter type (base/color/power/spot_size).
	- Verified `inverse_rendering_dataset/metadata.csv` against `inverse_rendering_dataset/images/` using `check_images.py`.
- Model/config used: 
	- Icosphere with Shiny Metal PBR material.
	- CYCLES render engine, AgX view transform.
	- Fixed camera position with 50mm focal length.

### Results
- Quantitative metrics:
	- **Total images rendered**: 2,313 PNG files
	  - Single light: 840 images (base: 120, color: 273, power: 360, spot_size: 360)
	  - Double light: 840 images (base: 120, color: 360, power: 360, spot_size: 360)
	- **CSV entries**: 2,313 rows
	  - Single light: 1,113 entries
	  - Double light: 1,200 entries
- Qualitative observations:
	- Dataset integrity check passed for the current snapshot.

### Interpretation
The dataset is currently in a consistent, train-ready state for experiments that require strict one-to-one metadata/image mapping.

### Next Actions
- [x] Combine all data to one common location.

---

## 2026-03-10

### Session Goal
Merge three independent CSV datasets (render-lighting, spotlight-sphere-data, inverse_rendering_dataset) into a unified master for cross-dataset experiment capability.

### Changes Made
- Code/files changed:
	- Created merge script to align three datasets with different schemas.
	- Applied safe field mapping for spotlight-sphere-data → shared lighting schema.
- Data used/generated:
	- Source: `processing/master_with_paths.csv` (render-lighting, 31,824 rows)
	- Source: `processing/color_power_labels.csv` (spotlight-sphere-data, 1,000 rows)
	- Source: `data/inverse_rendering_dataset/metadata.csv` (inverse_rendering_dataset, 2,313 rows)
	- Output: `data/data_master.csv` (unified, 35,137 rows)
- Model/config used: None (data curation only).

### Results
- Quantitative metrics:
	- **Total rows merged**: 35,137 (31,824 + 1,000 + 2,313 = 35,137 ✓)
	- **Total columns in merged file**: 107 (union of all source columns + 3 audit columns)
	- **Data loss**: 0 non-empty cells lost across all sources
	- **Row count deltas**: 0 for each source
	- **Column presence**: 100% of source columns retained in merged file
- Field mapping (spotlight-sphere-data):
	- `spot_name → light0_name`: 1,000/1,000 perfect matches
	- `spot_energy → light0_energy`: 1,000/1,000 perfect matches
	- `spot_color_r/g/b → light0_color_r/g/b`: 1,000/1,000 perfect matches
	- `spot_loc_x/y/z → light0_pos_x/y/z`: 1,000/1,000 perfect matches
	- `light0_type = SPOT`: 1,000/1,000 rows
	- `num_active_lights = 1`: 1,000/1,000 rows
- Qualitative observations:
	- Merge used safe strategy: only populated empty target fields, never overwrote existing values.
	- Zero conflicts during mapping.
	- Source file paths normalized into `resolved_image_relpath` for consistent future lookup.
	- New `dataset_source` column enables per-source analysis and filtering downstream.

### Interpretation
All three datasets now coexist in a single queryable file without information loss. Cross-dataset experiments (e.g. comparing render-lighting and spotlight-sphere-data models) are now feasible. The audit confirms structural integrity and mapping completeness.

### Risks / Caveats
- Data from different sources has different image quality, resolution, and material/lighting assumptions; models trained on this combined file must account for source-specific biases.
- Spotlight-sphere-data was mapped from its native schema into shared lighting fields; users should verify mappings match their domain understanding.

### Next Actions
- [x] Train &/or finetune models using combined data.

---

## 2026-03-13

### Session Goal
Redo `light_classifiers` training using the full merged dataset and remove redundant filtering logic that could cause duplicate-check warnings.

### Changes Made
- Code/files changed:
	- Updated `models/light_classifiers.ipynb` data-filtering logic.
	- Removed redundant material gate for render-lighting rows when batch membership already implies PlasticGlossy.
	- Simplified training-row inclusion to avoid double-checking the same source criteria.
- Data used/generated:
	- Input source: `data/data_master.csv` (combined master dataset).
	- Included training rows from:
		- render-lighting entries whose `batch_folder` is in the PlasticGlossy batch list.
		- inverse_rendering_dataset entries via `dataset_source`.
- Model/config used:
	- Light-count classifier and light-type classifier pipeline in `light_classifiers.ipynb`.
	- Existing CNN architecture/training callbacks retained; focus was on data selection correctness.

### Results
- Quantitative metrics: Pending fresh retrain/evaluation after filter update.
- Qualitative observations:
	- Filtering logic is now single-pass and no longer checks overlapping conditions for the same render-lighting subset.
	- Expected behavior is cleaner dataset selection with no duplicate warning triggered by redundant criteria checks.

### Interpretation
The classifier pipeline now aligns better with the merged-dataset design: one canonical source (`data_master.csv`) with non-duplicative inclusion rules. This reduces label-selection ambiguity before retraining.

### Risks / Caveats
- Removing redundant checks assumes PlasticGlossy batches are authoritative and consistently named.
- If batch naming changes in future exports, filter coverage should be revalidated.

---

## 2026-03-15

### Session Goal
Finetune `angular_predictor` and `color_power_predictor` on `inverse_rendering_dataset`, diagnose high error, and align the pipeline to match original model training behavior.

### Changes Made
- Code/files changed:
	- Updated `models/finetune_angle+color.ipynb` end-to-end.
	- Added robust preprocessing for camera-space targets (`light0_pos_cam_*`, `light0_dir_cam_*`).
	- Fixed angular tabular feature mismatch by reconstructing the exact legacy 31-feature layout expected by `angular_predictor.keras`.
	- Added training-curve visualization and compact diagnostics cells (per-dimension MAE, baseline comparison, denormalized color/energy metrics).
	- Reworked finetuning hyperparameters to match original training recipes (optimizer/loss/epochs/callbacks and train-split normalization strategy).
- Data used/generated:
	- Input metadata: `data/inverse_rendering_dataset/metadata.csv`.
	- Legacy schema alignment references: `processing/master_with_paths.csv`, `processing/color_power_labels.csv`.
	- Filtered training subset: single-light SPOT rows from inverse dataset.
- Model/config used:
	- Start from pretrained checkpoints: `models/angular_predictor.keras`, `models/color_power_predictor.keras`.
	- Saved outputs: `models/angular_predictor_finetuned.keras`, `models/color_power_predictor_finetuned.keras`.

### Results
- Quantitative metrics:
	- Angular:
		- Test loss: `1.5164`
		- Test MAE: `0.8009`
		- Per-dim MAE: `[2.1643, 0.7168, 0.9405, 0.5439, 0.1798, 0.2603]`
		- Baseline per-dim MAE: `[2.2071, 1.5948, 2.0513, 0.5518, 0.3987, 0.5128]`
		- Model improved over baseline on all six outputs.
	- Color/Power:
		- Eval vector: `[0.6014, 0.3047, 0.3823, 0.3117, 0.4079]`
		- Color absolute MAE (0-1 scale): `[0.0690, 0.0684, 0.0803]`
		- Energy log-MAE: `0.1945`
		- Energy MAE (linear): `830.45`
		- Energy MAPE: `19.52%`
- Qualitative observations:
	- Initial high error was caused by schema/normalization mismatch and inconsistent preprocessing relative to original model training.
	- After restoring original-style preprocessing/training, both models converged stably and produced substantially better results.

### Interpretation
The major failure mode was pipeline mismatch rather than pure model incapacity. Matching the original feature layout, coordinate-space targets, and normalization/training recipes was critical for transfer learning on the new dataset.

### Risks / Caveats
- Domain shift between legacy data and inverse dataset remains (especially for angular position ranges), so absolute error may still be bounded by dataset differences.
- Energy errors should be interpreted with scale-aware metrics (log-MAE, MAPE), not linear MAE alone.

### Next Actions
- [x] Spotlight size predictor model using new data.
- [x] Create test files for all models.

---

## 2026-03-16

### Session Goal
Implement a dedicated spot-size predictor using `inverse_rendering_dataset` and upgrade the unified test harness to run all current models with visual actual-vs-predicted reporting.

### Changes Made
- Code/files changed:
	- Created `models/spot_size_predictor.ipynb` using the same template style as other predictor notebooks.
	- Added single-light SPOT + `batch_folder` spot-size filtering and trained regression target `light0_spot_cone_deg`.
	- Updated `testing/test_all_models.py` to load all model families and prefer finetuned checkpoints when available.
	- Added per-model inference paths for:
		- `light_count_detector`
		- `light_type_classifier`
		- `angular_predictor` (finetuned preferred)
		- `color_power_predictor` (finetuned preferred)
		- `tri_angular_predictor`
		- `spot_size_predictor`
	- Added report image generation (`testing/test_result.png`) with tested image panels and actual vs predicted values/metrics overlaid.
	- Fixed tri-angular test cardinality bug by forcing single-row metadata selection when duplicate `image_relpath` matches occur.
	- Switched test image selection to random sampling each run.
- Data used/generated:
	- `data/inverse_rendering_dataset/metadata.csv` for spot-size predictor training/eval.
	- `data/data_master.csv` + `processing/master_with_paths.csv` for cross-model test row selection and tabular feature reconstruction.
	- Output artifact: `testing/test_result.png`.
- Model/config used:
	- Finetuned preference logic:
		- `angular_predictor_finetuned.keras` over `angular_predictor.keras`
		- `color_power_predictor_finetuned.keras` over `color_power_predictor.keras`
	- Base checkpoints used where no finetuned variant exists.

### Results
- Quantitative metrics:
	- Spot-size training/eval pipeline now logs MAE/RMSE in degrees in notebook output.
	- Unified test script now executes inference across all available models in one run (after tri cardinality fix).
- Qualitative observations:
	- Multi-model test output is now consolidated into one visual report image.
	- Actual vs predicted values are visible directly on the figure for quick qualitative inspection.
	- Randomized image selection improves test coverage across runs.

### Interpretation
The project now has complete model-family smoke testing with interpretable visual outputs, and a dedicated spot-size regressor aligned with the newer inverse dataset.

### Risks / Caveats
- Different model families still rely on different preprocessing and dataset contexts; unified orchestration works, but the underlying pipelines remain heterogeneous.
- Duplicate metadata rows by `image_relpath` can still occur in merged sources; explicit single-row selection is required during inference.

### Next Actions
- [x] Fix angular model inputs.
- [x] Make testing actually work as a pipeline system.

---

## 2026-03-17

### Session Goal
Fix `angular_predictor` inference so the model receives the right tabular inputs and can be used reliably in the current workflow.

### Changes Made
- Code/files changed:
	- Reworked angular inference/preprocessing alignment around the expected feature layout.
	- Corrected the `angular_predictor` input handling so the tabular branch matches model expectations.
	- Cleaned up the single-light SPOT inference path to use the repaired angular setup consistently.
- Data used/generated:
	- Existing metadata context from `data/inverse_rendering_dataset/metadata.csv`.
	- Legacy alignment references from `processing/master_with_paths.csv`.
- Model/config used:
	- `models/angular_predictor_finetuned.keras` when available, otherwise `models/angular_predictor.keras`.

### Results
- Qualitative observations:
	- Angular inference no longer fails because of the earlier feature-layout mismatch.
	- The model can now be routed again as part of the broader inverse-rendering workflow.

### Interpretation
The main blocker for angular prediction was input-schema mismatch rather than the model weights themselves. Restoring the expected feature structure brought the predictor back into a usable state.

### Risks / Caveats
- The angular model still depends on tabular context, so pipeline integration must supply those values consistently at inference time.

### Next Actions
- [x] Repair angular predictor input path.

---

## 2026-03-18

### Session Goal
Update `pipeline.py` so the pipeline can run from a photo using outputs from the other models, and make the testing path behave like the intended end-to-end system.

### Changes Made
- Code/files changed:
	- Updated `pipeline.py` so single-light SPOT angular inference no longer requires a matching metadata row for the input image.
	- Added synthetic angular tabular-input construction using stable context defaults plus upstream predictions from:
		- `light_count_detector`
		- `light_type_classifier`
		- `color_power_predictor`
		- `spot_size_predictor`
	- Routed the single-photo pipeline through the repaired angular branch using the inferred light properties.
- Data used/generated:
	- Existing context statistics from the model-building helpers in `testing/test_all_models.py`.
	- No new dataset artifacts generated; this was an inference-pipeline integration change.
- Model/config used:
	- Pipeline model loading still prefers finetuned checkpoints where present.

### Results
- Qualitative observations:
	- The single-light SPOT pipeline can now run conceptually from just the photo plus the information predicted by the upstream models.
	- Angular inference is no longer coupled to image-path lookup in the metadata table for that branch.
	- `pipeline.py` passes syntax validation after the update.

### Interpretation
This closes the main gap between individual-model testing and actual pipeline-style inference for the single-light SPOT case. The system now behaves more like a real photo-to-parameters pipeline instead of a dataset-row replay.

### Risks / Caveats
- Tri-light angular inference still relies on metadata lookup and is not yet photo-only.
- Some angular tabular fields still use context defaults because they are not directly predicted from the image.

### Next Actions
- [ ] Do extensive testing for results.

---

## 2026-03-22

### Session Goal
Build a packaged test set with matching metadata and add a formal evaluation pass so pipeline behavior can be measured across models instead of only eyeballed.

### Changes Made
- Code/files changed:
	- Expanded `pipeline.py` into a batch-style testing workflow that writes a predictions CSV for the packaged test set.
	- Created `test_data/evaluation.py` to compare predictions against actual metadata and compute metrics across the available model outputs.
	- Added evaluation outputs under `test_data/evaluation_outputs/`, including charts, KPI summaries, and tabular scorecards.
	- Added the first packaged prediction/evaluation tables:
		- `test_data/test_predictions.csv`
		- `test_data/evaluation_outputs/evaluation_table.csv`
		- `test_data/evaluation_outputs/metrics_summary.json`
		- `test_data/evaluation_outputs/model_scorecard.csv`
	- Added a packaged benchmark image set in `test_data/images/`.
	- Added `test_data/test_metadata.csv` as the paired ground-truth metadata file for those images.
- Data used/generated:
	- Generated packaged benchmark images: `test_data/images/render_0000.png` through `render_0499.png`.
	- Generated evaluation metadata: `test_data/test_metadata.csv`.
	- Generated first-pass evaluation artifacts in `test_data/evaluation_outputs/`.
- Model/config used:
	- Batch testing used the current pipeline model/routing stack in `pipeline.py`.

### Results
- Quantitative metrics:
	- A 500-image packaged benchmark set was added, along with CSV-based prediction/evaluation outputs.
- Qualitative observations:
	- The project moved from ad hoc spot-checking to a reusable benchmark/evaluation workflow.
	- Pipeline behavior could now be tracked in one place across model families.

### Interpretation
This session established the testing/evaluation infrastructure that later made it possible to diagnose notebook-vs-pipeline mismatches more clearly.

### Risks / Caveats
- The packaged test workflow depends on `test_data/images` and `test_data/test_metadata.csv` staying synchronized.
- Metric quality is constrained by the routing logic and scoring rules used by the pipeline at the time.

### Next Actions
- [x] Add new datasets into the canonical metadata flow.
- [x] Revisit notebook-vs-pipeline alignment using the packaged benchmark.

---

## 2026-03-28

### Session Goal
Expand the canonical dataset to include the new `two_object` renders and retrain the angular model on the enlarged merged single-light SPOT pool.

### Changes Made
- Code/files changed:
	- Appended `data/two_object_metadata.csv` into `data/data_master.csv`, adding 1,350 new rows under `dataset_source = two_object`.
	- Updated `models/angular_predictor.ipynb` so image resolution includes `data/<image_relpath>` paths, allowing the notebook to pick up `two_object` images from the merged master table.
	- Retrained `models/angular_predictor.keras` and refreshed `models/angular_predictor_preprocessing.json` using the expanded single-light SPOT pool from `data/data_master.csv`.
- Data used/generated:
	- Merged master metadata: `data/data_master.csv`.
	- Added dataset directory: `data/two_object/`.
	- Refreshed outputs:
		- `models/angular_predictor.keras`
		- `models/angular_predictor_preprocessing.json`
- Model/config used:
	- Angular retraining now uses the merged single-light SPOT pool visible from `data/data_master.csv`.

### Results
- Quantitative metrics:
	- `angular_predictor.ipynb` now reports 8,127 usable single-light SPOT rows:
		- `render-lighting`: 5,304
		- `inverse_rendering_dataset`: 1,473
		- `two_object`: 1,350
- Qualitative observations:
	- `two_object` is now part of the canonical master metadata instead of living outside the main workflow.
	- The angular notebook can now train on the new dataset rather than silently skipping those images.

### Interpretation
This was the data-expansion stage. The key step was making `two_object` part of the same master table the angular notebook already understands, which made retraining straightforward.

### Risks / Caveats
- Notebook-level gains do not automatically imply pipeline-level gains if runtime conditioning still differs from notebook conditioning.

### Next Actions
- [x] Retrain color/power on the expanded merged data sources.
- [x] Align runtime angular inference more closely with the retrained notebook setup.

---

## 2026-03-29

### Session Goal
Extend color/power training to use the merged `two_object` labels and reduce the mismatch between notebook angular evaluation and runtime angular inference.

### Changes Made
- Code/files changed:
	- Updated `models/color_power_predictor.ipynb` to train from `data/data_master.csv` instead of `processing/color_power_labels.csv`, filtering to:
		- all `spotlight-sphere-data` rows with valid spot labels
		- all `two_object` rows with valid spot labels
	- Reworked `pipeline.py` angular context construction to use the merged `data/data_master.csv` single-light SPOT subset instead of the earlier split between `inverse_rendering_dataset/metadata.csv` and `processing/master_with_paths.csv`.
	- Added shared image-path resolution in `pipeline.py` so `two_object` rows can be resolved consistently in runtime code as well as notebooks.
	- Changed angular inference row building so the pipeline prefers a matched metadata row when one exists, and only falls back to a synthetic mean/mode row when necessary.
- Data used/generated:
	- `data/data_master.csv` as the shared source of truth for retraining and runtime context.
	- Refreshed output:
		- `models/color_power_predictor.keras`
- Model/config used:
	- Color/power retraining used the combined labeled set from `spotlight-sphere-data` + `two_object`.
	- Angular runtime context was rebuilt around the same merged single-light SPOT schema used by the notebook.

### Results
- Quantitative metrics:
	- `color_power_predictor.ipynb` now resolves a 2,350-row labeled training pool:
		- `spotlight-sphere-data`: 1,000
		- `two_object`: 1,350
- Qualitative observations:
	- The angular notebook and runtime pipeline now draw from much closer metadata distributions than before.
	- Color/power no longer ignores the new two-object data.

### Interpretation
This stage was mostly about alignment. Instead of retraining on one world and inferring in another, more of the workflow now points at the merged master metadata.

### Risks / Caveats
- Runtime inference can still diverge from notebook evaluation whenever the pipeline has to synthesize tabular context for unseen images.
- End-to-end gains still depend on upstream count/type/color/spot predictions being stable.

---

