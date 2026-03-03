# Research Log

Use this file as a chronological lab notebook.

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
Conduct literature review to ground inverse lighting thesis direction.

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
- [x] Define lighting parameterization for thesis (direction, intensity, color, etc.).
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
Single-light formulation better matched stage-light thesis focus.

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
- [ ] Build comprehensive evaluation metrics for tri-light angular predictions across dataset.
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
- [ ] Test classifier routing on diverse lighting scenarios (quality assurance).
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
- [ ] Find/create more data for a more diversified set & generalized results.

---

## 2026-03-03

### Session Goal
Initialize thesis-style documentation framework for reproducible progress tracking.

### Changes Made
- Created a dedicated `docs/` folder for thesis preparation.
- Added overview, research-log, and struggles/decisions documents.

### Results
- Documentation scaffold is in place.
- Future sessions can now be logged in a consistent structure.

### Interpretation
A standard note format should reduce memory-based reporting errors and simplify writing thesis chapters later.

### Risks / Caveats
Backfilling older experiments may be incomplete where metadata was not recorded.

### Next Actions
- [ ] 
