# Struggles and Decisions Log

Tracking blockers and key technical decisions to explain why the project evolved the way it did.

---

## Blocker Template

### Date
YYYY-MM-DD

### Blocker
(What went wrong?)

### Symptoms
- 

### Suspected Root Cause
- 

### Attempts Tried
1. 
2. 

### Outcome
(Resolved / Partially resolved / Unresolved)

---

## Decision Template

### Date
YYYY-MM-DD

### Decision
(What was chosen?)

### Context
(What constraints or evidence informed the decision?)

### Alternatives Considered
1. 
2. 

### Rationale
(Why this option?)

### Expected Tradeoffs
- Pros:
- Cons:

### Validation Plan
(How will you check if this was the right choice?)

---

## 2026-02-03 — Blocker

### Blocker
Dataset CSV paths and model input references did not consistently align with actual batch folder structure.

### Symptoms
- Commit message explicitly notes: "fixed csv file & model to use all the correct batches' filepaths".
- Large near-rewrite in path metadata (`processing/master_with_paths.csv`) suggested systemic path mismatch.
- Current CSV still includes many `image_exists=False` rows, indicating path validation remains fragile across environments.

### Suspected Root Cause
- Naming/layout drift between exported render folders and generated metadata paths.
- Multiple render variants (engine/view-transform combinations) increased chance of path schema inconsistency.

### Attempts Tried
1. Regenerated and corrected master path CSV.
2. Updated baseline modeling pipeline to consume corrected paths.

### Outcome
Resolved

---

## 2026-02-08 — Decision

### Decision
Adopt an iterative "train then immediately test with Blender-rendered examples" loop.

### Context
Commits added `processing/blender_script.py`, test notebooks, and held-out rendered examples for sanity checks.

### Alternatives Considered
1. Train on full dataset first, evaluate later in large batches.
2. Build a more formal benchmark pipeline before any manual inspection.

### Rationale
Fast qualitative feedback from controlled renders reduced blind training cycles and exposed failures quickly.

### Expected Tradeoffs
- Pros: Rapid debugging, visual intuition, faster iteration.
- Cons: Risk of overfitting decisions to a small hand-picked test subset.

### Validation Plan
Track whether model changes validated on ad-hoc tests also hold on broader test sets.

---

## 2026-02-15 — Decision

### Decision
Pivot from early baseline workflow to angle prediction in camera space as a focused intermediate target.

### Context
Commit history explicitly records a technique change: "starting by predicting only angle in camera space", with previous artifacts moved under `legacy/` and a new `models/angular_predictor.*` line established.

### Alternatives Considered
1. Continue with earlier broader baseline formulation.
2. Jump directly to full multi-attribute prediction without a focused angular phase.

### Rationale
Reducing task complexity created a cleaner supervision target and a clearer path to diagnose geometric/pose sensitivity.

### Expected Tradeoffs
- Pros: Easier loss interpretation, clearer failure modes.
- Cons: Requires later reintegration with other targets (count/type/color-power).

### Validation Plan
Compare angular-only model consistency across object categories before reintegrating multitask components.

---

## 2026-02-22 — Blocker

### Blocker
Color/power prediction appeared highly sensitive to render and scene characteristics, requiring multiple dataset attempts.

### Symptoms
- Dataset assets and filenames in `data/color-attempts/` indicate repeated edge-case generation (`blown-out`, `no-specular`, `odd-spot`, etc.).
- Separate `color_power_labels.csv` variants were introduced before refinement of `models/color_power_predictor.*`.

### Suspected Root Cause
- Target leakage/confounding from tone mapping and highlight behavior.
- Domain shift between seemingly similar renders due to subtle shading and world-light differences.

### Attempts Tried
1. Built multiple targeted color/power datasets and tested them separately.
2. Refined the color/power predictor after introducing a broader test set.

### Outcome
Resolved

---

## 2026-02-15 — Decision

### Decision
Expand color/power dataset from 100 to 1,000 images after initial regression showed poor performance on sparse training data.

### Context
First color/power prediction attempt on 100 synthetic images yielded poor regression results. Root analysis identified weak specular highlight cues and insufficient dataset diversity to capture color/energy variation.

### Alternatives Considered
1. Accept poor performance and focus only on geometric lighting (count, type, direction).
2. Switch to a different target formulation (e.g., HSL space, log intensity).
3. Expand dataset with more varied material and lighting parameters.

### Rationale
Lighting color and intensity are critical for full inverse rendering. Expanding dataset allowed systematic exploration of how material properties, light positions, and background affect RGB/energy recovery. Dataset diversity was the binding constraint, not the model.

### Expected Tradeoffs
- Pros: Comprehensive exploration of color/energy prediction feasibility; larger dataset enables better generalization.
- Cons: ~5 hours of dataset engineering; 12 failed attempts before acceptable diversity.

### Validation Plan
Compare performance (regression loss, visual similarity) across the 12 dataset variants; final 1,000-image set should show low variance in validation error.

---

## 2026-02-22 — Blocker

### Blocker
Energy target representation ambiguity: unclear whether lighting energy should be predicted in linear or log space.

### Symptoms
- Research note: "Energy scaling and perceptual color differences may require log-space or normalized targets."
- No clear precedent in dataset generation (renders exported in linear color space).
- Initial model training showed large variance in energy predictions without clear loss landscape.

### Suspected Root Cause
- Energy values span orders of magnitude (1.0 → 1000.0 in Blender units); linear regression struggles with high dynamic range.
- Perceptual color difference (Delta E) is nonlinear; MSE loss may not match human visual judgment.

### Attempts Tried
1. Trained initial models predicting energy in linear space.
2. Noted instability and high variance in predictions.

### Outcome
Resolved

---

## 2026-02-27 — Blocker

### Blocker
Classification model training became unstable at higher epoch counts, causing training crashes.

### Symptoms
- Successful training at default/low epoch counts (e.g., 20-30 epochs).
- Attempted 50-epoch training for light type classifier caused crash/out-of-memory.
- Class imbalance in lighting dataset (e.g., far more single-light than tri-light examples).

### Suspected Root Cause
- Long training schedules exhausted GPU memory, likely due to accumulated loss history or large batch accumulation.
- Class imbalance caused one class to dominate gradients, destabilizing later epochs.

### Attempts Tried
1. Trained at standard epochs (20-30); success.
2. Increased epochs to 50; crash.
3. Adjusted batch size and learning rate on multiple runs.

### Outcome
Resolved

---

## 2026-03-01 — Decision

### Decision
Move to a modular model family (count classifier, type classifier, angular predictors, color-power predictor) plus a unified test harness.

### Context
Commits introduced `light_count_detector`, `light_type_classifier`, `tri_angular_predictor`, mapping JSON, and `test_all_models.py` for consolidated loading/testing.

### Alternatives Considered
1. One monolithic multitask model for all outputs.
2. Continue isolated notebook-only experiments without a unified inference script.

### Rationale
Modular heads allow independent iteration cadence per task and simplify targeted retraining.

### Expected Tradeoffs
- Pros: Better debugging isolation, easier model replacement, clear component ownership.
- Cons: Integration complexity (different input sizes/metadata contracts across heads).

### Validation Plan
Maintain one canonical script to run all available models on the same image slice and archive outputs per release.

---

## 2026-03-01 — Blocker

### Blocker
Full end-to-end integration remains incomplete because not all predictors consume identical input modalities.

### Symptoms
- In `test_all_models.py`, angular and tri-angular predictors are currently skipped with note: requires tabular metadata matching training preprocessing.
- Combined "all models" testing exists, but only subset of heads can be evaluated directly from image-only input.

### Suspected Root Cause
- Heterogeneous training pipelines: some models are image-only, others depend on structured metadata features.

### Attempts Tried
1. Built unified model-loading and single-image test script.
2. Added explicit skip logic to prevent invalid inference calls.

### Outcome
Resolved

---

## 2026-03-16 — Blocker + Decision

### Blocker
Difficulty creating high-quality Blender materials that produced consistent, realistic render behavior for training data.

### Symptoms
- Repeated dissatisfaction with material quality/consistency during scene setup.
- Material iteration was slowing dataset and experiment progress.
- Asked peers for help with material setup, but did not get a volunteer to assist.

### Suspected Root Cause
- Personal experience with Blender is very limited.

### Attempts Tried
1. Manual material tweaking and repeated visual checks in Blender.
2. Reached out for help/collaboration on material setup.

### Outcome
Partially resolved

### Decision
Use BlenderKit materials as the primary source for scene materials going forward.

### Context
Progress on model/data work was being blocked by material quality issues and lack of immediate support for custom material development.

### Alternatives Considered
1. Continue building fully custom materials from scratch.
2. Pause data generation until a collaborator could help with material work.

### Rationale
BlenderKit provides production-ready materials quickly, reducing iteration time and unblocking dataset generation/training workflows.

### Expected Tradeoffs
- Pros: Faster setup, better baseline realism, less time spent on shader engineering.
- Cons: Less full control over material internals; potential licensing/style consistency considerations across assets.

### Validation Plan
Compare a small controlled set of renders before/after BlenderKit adoption for consistency, realism, and downstream model stability.
