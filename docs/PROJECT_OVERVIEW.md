# Project Overview (Thesis Draft Base)

## Working Title
Inverse Rendering for Lighting Parameter Estimation from Synthetic Object Renders

## Problem Statement
Estimate scene lighting parameters (light type, count, direction/angular setup, and color power characteristics) from rendered images of simple geometric objects under controlled materials and lighting setups.

## Motivation
- Build a reproducible inverse-rendering pipeline grounded in synthetic data.
- Evaluate how geometry, material, and rendering engine/tone mapping affect lighting inference.
- Produce a foundation that can later transfer to more realistic or mixed-domain imagery.

## Current Scope
- Objects: Cone, Cube, Cylinder, Icosphere, Sphere, Torus
- Material focus (current): PlasticGlossy
- Lighting conditions include: Area, Point, Spot, Tri Lighting, HDRI variants
- Models currently present:
  - Light type classifier
  - Light count detector
  - Angular predictors (single/tri)
  - Color-power predictor

## Research Questions (Draft)
1. Which lighting properties are most reliably inferred across object categories?
2. How does renderer/view transform choice (Cycles vs Eevee, AGX/Filmic variants) impact model generalization?
3. What error patterns are systematic vs data-specific?
4. What is the minimum dataset diversity needed for robust prediction?

## Pipeline Summary
1. Data generation/render export
2. Label extraction and dataset assembly
3. Model training per sub-task
4. Cross-condition testing and error analysis
5. Iterative data/model refinement
