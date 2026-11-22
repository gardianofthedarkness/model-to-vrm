# model-to-vrm

Turn rough AI-generated 3D models of one character into a **fully parted, rigged, physics-driven figure** (VRM and
baked motion), with agents doing the judging that hand rules cannot.

## Current state: bones and muscles inside a generated figure

**Demo:** https://gardianofthedarkness.github.io/model-to-vrm/ (a recording and an interactive 3D viewer).

The part of the plan that works today fits an anatomy atlas inside one textured 3D model: a full skeleton, about 680
muscle and tendon pieces and the nose's cartilages, inside the figure's own surface.

    python model_to_vrm/pipeline.py <model.glb> <out dir> --anatomy

runs every step from the input model to `anatomy_dense/muscle_repair.npz` and `anatomy_dense/result.jpg`, skipping
steps whose outputs exist. It needs one-time atlas data that is not in this repository (the exported Z-Anatomy atlas,
its slice planes, the atlas head, CT tissue-depth statistics; see `docs/anatomy-sources.md`), a local GPU for SAM 3,
and keys for the outside services it calls (a vision LLM for labels, the Jev decision API for boxes, Hugging Face Jobs
for the slice fit). Keys are read from the system credential store, never from files.

What to know before trusting it:

- Only one figure has been run end to end.
- The pipeline still contains numbers set by hand. `docs/hardcoded-numbers.md` lists every one found in an audit, what
  it does and what should replace it.
- Known faults of the current result are listed on the demo page.
- `docs/interior-model-handout.md` is a proposal (not built) for a learned model that predicts the interior on 2D
  slices to guide the optimiser.

## Input

One character, given as one or more 3D models of **the same person in the same clothes**:

- models with a rough texture (for example Tripo / Hunyuan3D output: colours roughly right, blurry, edges smeared);
- models with parts (Tripo's segmented pieces, which often mix things: face skin and hair in one piece);
- a T-pose "twin" (clear structure, sometimes off in proportion) and a model in the original pose (loyal proportions);
- a **description per model**: what it is, and where exactly it is off ("texture smeared at the left cuff",
  "hair and face share one piece", "eyes blurry").
- optionally the original drawing (the truth for colours and parts).

See `examples/rikka/manifest.yaml` for the format.

## Output

- every part independent: skin, hair locks, each garment, buttons, stripes, eyes (eye white, iris, pupil, lashes),
  brows, mouth, each with its own flat colour and material;
- a parts tree (entity matching: all skin one entity, one garment one entity, details hung under what they sit on);
- a skeleton with body, hand, face and eye bones (VRM humanoid plus extra bones);
- physics: tetrahedral flesh, cloth, hair chains, baked offline;
- export to VRM / glTF, and baked motion from text (NVIDIA ARDY).

## Pipeline

| stage | what | how | code |
|---|---|---|---|
| 1 | read the models | parts, texture colours, normals onto one point set | `model_to_vrm/parts/glb_parts.py`, `part_colours.py`, `part_normals.py` |
| 2 | big parts | colour differencing on the zoomed-out model: Lab smoothing over the surface graph, GMM (count by BIC), connected pieces, Otsu size merge | `parts/carve_full.py`, `carve_show.py` |
| 3 | detail parts | SAM 3 text prompts on close-up renders (head, eyes, torso) from several directions; raw masks saved once; votes on the full-detail points; Otsu splits | `parts/views_crops.py`, `eye_crops.py`, `sam3_masks.py`, `masks_vote.py` |
| 4 | combined part list | colour regions for big parts, SAM 3 for every detail part | `parts/merge_full.py` |
| 5 | parts tree | an agent (Pi on DeepSeek, with memory) builds the tree with entity matching and checks it on renders | `agents/parts_tree/` |
| 6 | flat colours, materials | the agent gives each entity one flat colour from measured candidates, material and stiffness | `agents/flat_colours/`, `parts/flat_colours.py` |
| 7 | rig | shape-only fit of the T-pose twin to the posed model; weights carried; hands from range-of-motion data; face and iris bones | `rig/`, `face/` |
| 8 | style correction | non-rigid ICP of the twin onto the original, texture carried back | `style/` |
| 9 | motion | ARDY text-to-motion on the SOMA skeleton, retargeted | `motion/` |
| 10 | physics | fTetWild tets + Newton VBD flesh; cloth and hair chains (to do) | `flesh/` |
| 11 | export | VRM 1.0 / glTF (to do as one step) | |

The longer design notes are in `docs/bottom-up-plan.md`.

## Rules

- No hand-tuned numbers: every split comes from the data (Otsu, BIC, connected pieces) or from an agent's judgement on
  renders.
- Use existing tools first (SAM 3, fTetWild, Newton, ARDY, FLAME...), write our own only for the glue.
- Keys live in Windows Credential Manager (`model_to_vrm/common/vault.py`) and are only passed to child processes.

## Status

Built and tested on one character (Rikka). The scripts still carry that run's absolute paths (`E:\figures\...`); the
next step is a per-character config so every stage reads its paths from the manifest.

## Environments (this machine)

- `E:\splat\.venv` - torch (CUDA), transformers (SAM 3), numpy/scipy/opencv
- `E:\splat\.venv-sim` - NVIDIA Newton (flesh)
- `E:\splat\.venv-ardy` - ARDY
- Blender 5.3 (headless renders), Node + `@mariozechner/pi-coding-agent` (agents)
