# Project Overview

## Final Title
Inverse Rendering of Scene Lighting for Theatrical Lighting Reconstruction

## Project Summary
This project investigates whether interpretable theatrical lighting parameters can be reconstructed from images using a synthetic inverse-rendering pipeline. Rather than representing illumination as an environment map or latent feature space, the system models lighting as a small set of discrete light sources with explicit attributes such as light type, count, position, direction, color, intensity, and spotlight cone angle.

The full approach is modular. Instead of one multitask network, the pipeline separates scene-structure prediction from downstream geometric and photometric estimation. This keeps the system interpretable, easier to debug, and closer to how theatrical lighting is actually specified in production workflows.

## Problem Statement
Recover structured theatrical lighting parameters from rendered images in a form that can be reused for reconstruction, visualization, and analysis. The target is not just plausible relighting, but physically meaningful estimates of discrete light sources.

## Motivation
- Support documentation and reconstruction of theatrical lighting setups.
- Explore whether synthetic data can provide enough supervision for inverse lighting in a controlled setting.
- Preserve lighting outputs in a form that is directly usable in design and production workflows.
- Study which lighting properties are easy to recover from images and which remain ambiguous.

## Final Scope
- Domain focus: theatrical and stage-style lighting reconstruction.
- Core lighting representation: discrete lights with interpretable parameters.
- Primary scene assets:
  - Render-lighting dataset objects: cone, cube, cylinder, icosphere, sphere, torus.
  - Additional custom scenes: spotlight sphere, reflective icosphere, and two-object scenes.
- Primary materials used across experiments:
  - Plastic Glossy
  - Shiny Metal PBR
  - Glossy / metallic variants for targeted datasets
- Lighting configurations modeled:
  - Area light
  - Point light
  - Spot light
  - Double spot configurations
  - Tri-light theatrical setups

## Main Contributions
1. A theatrical-lighting-focused inverse-rendering formulation based on discrete, interpretable light sources.
2. A modular prediction pipeline that separates classification and regression into tractable sub-tasks.
3. Multiple Blender-generated synthetic datasets with explicit ground-truth lighting labels for training and evaluation.

## Pipeline Summary
1. Generate synthetic scenes in Blender with controlled lighting, materials, and camera views.
2. Export structured metadata including camera pose and per-light parameters.
3. Convert lighting labels into a stable representation, prioritizing camera-space geometry for prediction.
4. Train specialized models for scene structure, angular geometry, color/intensity, and spotlight size.
5. Route predictions through a staged inference pipeline and merge outputs into one structured lighting record.
6. Evaluate both numerically and by reconstructing predicted lighting back in Blender.

## Model Family
- `light_count_detector`
  - Predicts the number of active lights.
- `light_type_classifier`
  - Predicts the lighting configuration class.
- `angular_predictor`
  - Predicts single-spotlight position and direction in camera space.
- `tri_angular_predictor`
  - Predicts position and direction for three-light setups.
- `color_power_predictor`
  - Predicts RGB color and light energy.
- `spot_size_predictor`
  - Predicts spotlight cone angle.

## Data Summary
- Total images available in the codebase data section: 36,487
- Filtered dataset for `light_count_detector`: 23,529 images
- Filtered dataset for `light_type_classifier`: 34,137 images
- Training set for `angular_predictor`: 8,127 images
- Training set for `tri_angular_predictor`: 1,326 images
- Training set for `color_power_predictor`: 2,350 images
- Training set for `spot_size_predictor`: 480 images
- Final held-out benchmark: 500 Blender-generated spotlight images

## Final Evaluation Snapshot
On the 500-image held-out test set, the pipeline recovered coarse lighting structure very reliably while showing moderate error on continuous parameter estimation.

- Light type accuracy: 100%
- Light count accuracy: 99.6%
- Direction angular MAE: 21.81 degrees
- Position mean L2 error: 1.72
- RGB mean L2 error: 0.261
- Energy MAE: 2055.13
- Spot size MAE: 10.55 degrees
- Aggregate pipeline score: 86.35
- Prediction coverage: 99.89%

## Key Findings
- High-level scene structure is much easier to recover than exact geometric or photometric values.
- Camera-space geometry was more stable than world-space prediction.
- Modular task decomposition improved training stability and simplified debugging.
- Energy and precise angular recovery remain the hardest parts of the problem due to ambiguity in shading, highlights, and dynamic range.
- Predicted reconstructions are often visually plausible even when exact numeric estimates are imperfect.

## Limitations
- The entire system is trained and evaluated primarily on synthetic data.
- Real-world theatrical imagery has not yet been used for full validation.
- Staged routing introduces error propagation when upstream classifiers are wrong.
- Multi-light precision remains more difficult than coarse structural recovery.

## Future Work
- Expand dataset diversity for broader geometry, materials, and lighting conditions.
- Improve geometric supervision for direction and position prediction.
- Reduce photometric ambiguity in color and energy estimation.
- Evaluate the system on real stage imagery.
- Extend reconstruction quality for more complex multi-light scenes.