---
license: cc-by-nc-4.0
library_name: pytorch
tags:
  - 3d
  - anatomy
  - game-assets
  - physics
  - flow-matching
  - generative
pipeline_tag: other
---

# MARROW-1.0-head

**MARROW-1.0-head generates what is inside a head from the head's outer surface alone.** Given a character's closed head shell, it draws 14 coarse **physics compartments**: skull, jaw, teeth, eyes, brain, chewing muscles, tongue, airway and so on. Each compartment is meant to be one material with one set of attachments in a game or simulation, for example a rigid skull, a jaw that hinges, soft fat that jiggles and eyes that stay put.

It is the coarse "game" stage of MARROW, a shell-to-interior model in the [model-to-vrm](https://github.com/gardianofthedarkness/model-to-vrm) project. The finer medical stage comes later.

![Input shell, generated compartments, and the shell half-transparent over them](example/elf_input_vs_output.png)

*Left: the only input, the closed head shell of our elf character, hair and pointed ears included. Middle: what MARROW-1.0-head generates inside it (the skull drawn at 35% so the brain shows). Right: the shell at 28% opacity over the output. The skull, jaw, teeth, eyes and chewing muscles fit inside her face, while the volume of her hair and ears stays empty.*

## Compartments

`outside, fat_layer, skull, mandible, teeth, eyes, brain, chew_left, chew_right, tongue, airway, spine, other_bone, other_soft`

- **chew_*:** masseter, temporalis and both pterygoids on each side.
- **airway:** sinuses, nasal cavity, pharynx, trachea, ear canals and other air.
- **other_bone:** hyoid and small pieces.
- **other_soft:** everything else that is soft tissue.

## How it works

1. **Shell tokens.** The head is cut into 16 mm cells. Each cell that holds body becomes a token. Its features are the signed distance to the skin (pooled 4×4×4) and the share of the cell that has data.
2. **Generator.** A flow-matching transformer (8 layers, width 384) puts the shell tokens and noise tokens in one sequence. It denoises the noise tokens into 8-number codes in 16 Euler steps.
3. **Decoder.** A patch-transformer VAE (16 mm patches, 8 latent channels) decodes the codes into per-voxel compartment probabilities at 1 mm.
4. **Meshes.** Optionally, one smooth surface per compartment is cut where that compartment's probability overtakes all the others. The surfaces are written as a coloured GLB.

**Hair and cloth.** During training the shell was randomly thickened with fake hair or cloth (up to 2.5 L) while the interior stayed the same. The model therefore learned to leave volume outside the skull empty instead of filling it with fat. This matters for game characters, which nearly always have hair:

| Model on the elf (ml) | Fat layer | Other soft tissue |
|---|---|---|
| No hair augmentation | 1976 | 2274 |
| **MARROW-1.0-head** | 394 | 1504 |

The generator is stochastic: different `--seed` values give different, equally plausible interiors.

## Quick start

```bash
pip install torch numpy scipy scikit-image safetensors trimesh fast-simplification
python marrow_head.py example/elf_shell.npz out --glb --max-faces 20000
```

The command writes three files:
- `out/compartments.npz`: `labels` (uint8 compartment ids), `probs` (float16, 14 × X × Y × Z) and `names`.
- `out/volumes.json`: volume of each compartment in ml.
- `out/compartments.glb`: one named, coloured mesh per compartment in metres, glTF axes (+Y up, face towards +Z). It also has a `skin_input` node, the input shell at 35% opacity, so input and output can be viewed together in any glTF viewer or engine.

The elf takes about 4 minutes on a T4 and about 6 minutes on a GTX 1650 (4 GB). It also runs on CPU with `--cpu`, more slowly.

## Input format

The input is an `.npz` with `labels`: a uint8 volume at **1 mm** in MARROW's **head frame**.
- **Frame:** the origin is the midpoint of the ears. +x points from the left ear to the right ear, +y towards the midpoint of the eyes, and +z up.
- **Volume:** the example is 228 × 324 × 324 voxels.
- **Values:** `0` = outside the body; `255` = unknown (for example below the neck, where training heads had no data); any other value = inside the body.

The mesh has no real-world scale, so the head must be sized to real millimetres. [`elf_shell.py`](https://github.com/gardianofthedarkness/model-to-vrm/blob/main/research/interior_model/elf_shell.py) builds this volume from a character mesh and body keypoints. It voxelises the mesh with a winding number, places the frame from the ears and eyes, and scales the head to real training heads.

## Example: the elf

`example/` holds our own generated character, an anime-style elf:
- `elf_shell.npz`: the input.
- `elf_out/`: the output, as `compartments.npz`, `volumes.json`, the full-resolution `compartments.glb` and the 20k-face-per-part `compartments_game.glb`.
- `elf_slices.png`: three slices through the eyes.
- `elf_input_vs_output.png`: the picture above, rendered from `compartments_game.glb` with [`render_demo.py`](render_demo.py) in Blender.
- `elf_compartments_3d.png`: a plain 3D preview of the compartments.

Drop `compartments_game.glb` into any glTF viewer (for example https://gltf-viewer.donmccurdy.com) to turn it around: the half-transparent `skin_input` node shows the shell over the compartments.

![Elf slices](example/elf_slices.png)

## Scores

Scores are on 15 held-out real CT heads. Overlap is the Dice score, where 1.0 is perfect. The volume ratio is predicted volume divided by true volume.

| Compartment | Overlap | Volume ratio |
|---|---|---|
| brain | 0.95 | 0.97 |
| eyes | 0.81 | 1.23 |
| chew_left / chew_right | 0.77 / 0.76 | 1.21 / 1.24 |
| skull | 0.76 | 1.02 |
| mandible | 0.76 | 1.11 |
| tongue | 0.74 | 0.95 |
| other_soft | 0.68 | — |
| spine | 0.62 | — |
| fat_layer | 0.59 | 1.06 |
| airway | 0.59 | — |
| teeth | 0.54 | 1.32 |
| other_bone | 0.13 | — |

The model trained without hair augmentation scores 0.01–0.04 higher on these clean (hairless) CT heads, but fills hair with fat on real characters.

At 8 mm tokens the model scored the same overall (tongue 0.69, teeth 0.50) while its autoencoder rebuilt heads better. The generator, not the token size, is the current limit.

## Limits

- **Coarse by design.** Expect about 1–3 mm median surface error, and more for thin parts. Teeth come out about 1.3× their true volume, and `other_bone` is unreliable.
- **Head and upper neck only.**
- **Trained on 61 + 42 adult CT heads,** mostly from clinical scanning. Unusual anatomy, children and extreme stylisation (very large eyes, tiny jaws) are outside the training range. On stylised characters the result is a plausible human interior fitted to the shell, not a medical prediction.
- **Not a medical device.** Do not use it for diagnosis or treatment.

## Training data and licence

The weights are released under **CC BY-NC 4.0, for non-commercial use only**, because of the terms of the training data:

- **[TotalSegmentator CT dataset v2.0.1](https://zenodo.org/records/10047292)** (CC BY 4.0): 61 heads. Their head labels (craniofacial structures, head muscles, head glands and cavities) were produced with TotalSegmentator's licensed head models, whose outputs are for non-commercial use.
- **[HaN-Seg](https://zenodo.org/records/7442914)** (CC BY-ND 4.0): 42 heads. Its licence forbids distributing adapted material, and whether trained weights count as adapted material is legally unsettled. If you need certainty, do not use these weights. A later release may drop HaN-Seg.

No scan, label volume or render of any real person is included in this repository. The only example is our own fictional character.

## Citation

```
@misc{marrow2026head,
  title  = {MARROW-1.0-head: shell-conditioned generation of head physics compartments},
  author = {gardianofthedarkness},
  year   = {2026},
  url    = {https://huggingface.co/gardianofthedarkness/MARROW-1.0-head}
}
```

Internal registry code: `MARROW-S1.0-t16h-k25` (VAE 12k steps, generator 25k steps).
