# Inverse Rendering Documentation Index

Last updated: 2026-04-22

## Purpose
This folder documents the development of the project, from early experimentation through the final paper:

**Inverse Rendering of Scene Lighting for Theatrical Lighting Reconstruction**

The documentation here complements the final paper by preserving project framing, research progress, technical decisions, blockers, and lessons learned that informed the final system.

## Documents

1. `PROJECT_OVERVIEW.md`
   - Final high-level summary of the project.
   - Describes the problem framing, modular pipeline, datasets, model family, final evaluation snapshot, limitations, and future work.

2. `RESEARCH_LOG.md`
   - Chronological project notebook.
   - Tracks literature review, dataset generation, model experiments, testing sessions, and interpretation of results over time.

3. `STRUGGLES_AND_DECISIONS.md`
   - Record of technical blockers and important design choices.
   - Explains why the project moved toward camera-space prediction, modular models, iterative Blender validation, and specialized datasets.

## How These Docs Relate To The Final Paper
- The paper provides the polished project narrative, methodology, results, and discussion.
- `PROJECT_OVERVIEW.md` gives a concise repository-facing summary of the final project.
- `RESEARCH_LOG.md` captures the week-to-week research process behind the finished system.
- `STRUGGLES_AND_DECISIONS.md` preserves rationale that is useful for future continuation or project handoff.

## Core Project Focus
- Reconstruct theatrical lighting from images using synthetic inverse rendering.
- Represent lighting as discrete, interpretable sources rather than latent illumination fields.
- Use a modular prediction pipeline for light count, light type, geometry, color/intensity, and spotlight size.
- Evaluate both quantitative accuracy and reconstruction plausibility.

## Written Material
- Final paper PDF:
  - `Inverse_Rendering_of_Scene_Lighting_for_Theatrical_Lighting_Reconstruction.pdf`

## References
1. C. Luthy, “Clifton Taylor shares importance of stage lighting design, how it’s changing,” `www.uncsa.edu`, Sep. 23, 2019. https://www.uncsa.edu/news/20190923-stage-lighting-design.aspx
2. N. Hunt, “The Virtual Opera House: hybrid realities in lighting design processes for large-scale opera,” *Theatre and Performance Design*, vol. 6, no. 4, pp. 341-355, Oct. 2020. DOI: https://doi.org/10.1080/23322551.2020.1856302
3. Ravi Ramamoorthi and P. Hanrahan, “A signal-processing framework for inverse rendering,” *International Conference on Computer Graphics and Interactive Techniques*, Aug. 2001. DOI: https://doi.org/10.1145/383259.383271
4. J. Yu, X. Yang, and S. Xiao, “Interactive Image Based Relighting with Physical Light Acquisition,” *Lecture Notes in Computer Science*, pp. 288-293, 2007. DOI: https://doi.org/10.1007/978-3-540-74873-1_35
5. S. Sengupta, J. Gu, K. Kim, G. Liu, D. W. Jacobs, and J. Kautz, “Neural Inverse Rendering of an Indoor Scene from a Single Image,” `arXiv.org`, 2019. https://arxiv.org/abs/1901.02453
6. X. Zhang, P. P. Srinivasan, B. Deng, P. Debevec, W. T. Freeman, and J. T. Barron, “NeRFactor,” *ACM Transactions on Graphics*, vol. 40, no. 6, pp. 1-18, Dec. 2021. DOI: https://doi.org/10.1145/3478513.3480496
7. S. F. Mengistu, F. Bergamasco, and M. Pistellato, “A Neural Reflectance Field Model for Accurate Relighting in RTI Applications,” *ACM Transactions on Graphics*, vol. 45, no. 1, pp. 1-19, Oct. 2025. DOI: https://doi.org/10.1145/3759452
8. M. Aittala, “Inverse lighting and photorealistic rendering for augmented reality,” *The Visual Computer*, vol. 26, no. 6-8, pp. 669-678, Apr. 2010. DOI: https://doi.org/10.1007/s00371-010-0501-7
9. K. A. U. Zaman, A. Islam, and M. A. Sayed, “Render lighting dataset: A collection of rendered images with varied lighting conditions using blender render engines,” *Data in Brief*, vol. 54, p. 110331, Jun. 2024. DOI: https://doi.org/10.1016/j.dib.2024.110331
