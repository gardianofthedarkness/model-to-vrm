# Interior model: design v8 (shell-conditioned flow matching)

**Status, 2026-11-24.** This amends the generator of v7 ([interior-model-plan-v7.md](interior-model-plan-v7.md), sections 5.3 and 13). Everything else in v7 still applies: the product, the tissue-label target, the label autoencoder, the data, the rig and the infrastructure rules. Nothing in v8 has been trained yet. Its one-shot baseline (section 3) is being run.

**What changed and why.** v7 planned a masked bidirectional transformer that predicts every interior token in one pass, with a generative head added later (milestone 20). Milestones 15 and 16 were run, and the one-shot transformer lost to copying the most similar training person (section 2). It also painted small structures at several times their size and averaged wherever the surface does not decide the interior. v8 makes the generator generative from the start: **the body shell is the reference, placed in the same token sequence as noise tokens that one transformer denoises by flow matching.** This is the conditioning used by current image-editing models (Stable Diffusion 3, FLUX Kontext, Qwen-Image-Edit; the user reports the same design in MiniMax H3), where the reference sits in the sequence and is attended to at every step, so the output follows it closely.

---

## 1. Summary of the design

| Piece | v7 | v8 |
|---|---|---|
| Label autoencoder | 3D convolutional, frozen; 8 numbers per token | unchanged |
| Tokens | part-relative grids: pelvis 12 mm, face 8 mm | unchanged |
| Generator | one pass: surface features in, token codes out | flow matching: noise tokens denoised over several steps |
| How the shell enters | added to each interior token | its own tokens in the same sequence (the reference) |
| Facts about the person | planned | a few tokens in the same sequence |
| Output | the average where unsure | one sharp, plausible interior per draw |
| Nearest training person | input in the v2 baseline | not used in the main design; tested as a second reference (F1b) |

## 2. What milestones 15 and 16 measured

**Milestone 15, Tokens (label autoencoder, done).** Median Dice over structures after compressing and rebuilding the labels (test people; face 15, pelvis 42):

| Part | Token size | Median Dice | Thin structures (0–3 mm) | Tokens per person |
|---|---|---|---|---|
| Face | 4 mm | 0.86 | 0.76 | about 66,000 |
| Face | 8 mm | 0.75 | 0.60 | about 8,300 |
| Face | 12 mm | 0.64 | 0.50 | about 2,500 |
| Pelvis | 12 mm | 0.97 | 0.69 | about 8,600 |

Decision: 8 mm tokens for the face now, 4 mm kept for a later detail pass (the eye lens survives only at 4 mm).

**Milestone 16, Transformer, version 1 (one-shot, done).** 14 M parameters, 20,000 steps on a T4, 417 / 28 / 55 people, about $2.40. Test, median Dice over structures:

| Part | Transformer | Copy of the nearest person | Compression ceiling |
|---|---|---|---|
| Pelvis | 0.38 | 0.56 | 0.97 |
| Face | 0.23 | 0.32 | 0.75 |

Findings:
- **Still learning.** Validation loss fell from 9.1 to 1.8 and its best value was the last step.
- **Over-painting.** Predicted structures were about 1.6 times their true volume; the right eye 5.5 times, the optic nerve 4.7 times. Inside the true eyeballs 92–95% of voxels were right, so the eyes were found and then spread. The likely cause is the loss weight that boosted rare classes.
- **Averaging.** Where the shell does not fix the interior (the bowel), the output is one smooth blob.
- **Label conventions differ between sources.** HaN-Seg never draws the brain, trachea or neck vertebrae; TotalSegmentator never draws 14 others (lips, lacrimal glands and so on). The model was charged for drawing a brain in a HaN-Seg head.
- **Left and right:** 0.34 against 0.29 over 15 paired structures, mostly the over-painted right eye.

## 3. The one-shot baseline (transformer version 2)

Kept as the baseline that v8 must beat, with the same data and inputs. `research/interior_model/transformer_v2.py`:
- predicts a correction on the nearest training person's tokens, so it starts at that baseline;
- cross-entropy without the rare-class boost; each voxel weighs 1 + exp(−distance to skin / token size) + exp(−distance to bone / token size);
- a Dice term over all classes with a one-voxel floor, which charges a class painted where it does not exist or at several times its size;
- classes a person's source never drew are not charged;
- training heads are also used mirrored, with left and right names swapped (face only; the pelvis and torso are not mirrored because organs are one-sided);
- cached preparation, validation 40 times per run, checkpoints uploaded at each validation.

Smoke run passed (8 people per split, 300 steps). The 20,000-step run on an A10G-small is queued behind the preparation job.

## 4. The generator

### 4.1 Sequence

One sequence per person, three kinds of token:

1. **Shell tokens (the reference).** One per token cell that the body surface passes through or lies next to. Content: the signed distance to the skin on that cell's voxels (pooled to 4×4×4), the share of the cell inside the scan, and the part it belongs to. Measured on prepared people: 22–29% of body tokens are cut by the surface and 41–59% lie within one token of it, so the shell block is about half the length of the interior block.
2. **Fact tokens.** Age, sex, and which classes this person's source draws, each with a known/unknown flag. Scanner facts in training only (v7 section 5.1).
3. **Noise tokens.** One per interior token position. They start as Gaussian noise and end as the autoencoder's 8-number code for that position.

Each kind has its own input projection, because the shell is geometry and the interior is anatomy code: they are different spaces and nothing can be copied across. In the first layers each kind may also keep its own weights (as Stable Diffusion 3 does for text and image); whether that is needed is test F2. **All tokens share one 3D position encoding** in the part frame: a shell token and an interior token at the same place carry the same position, and that shared coordinate is the link between the two spaces.

### 4.2 Transformer

Joint bidirectional attention over the whole sequence at every step. Same size as the baseline to begin with (8 layers, width 384, 14 M parameters), so the comparison measures the method, not the size. The time step enters as an embedding added to the noise tokens.

### 4.3 Training: flow matching

For a person with true codes $z_1$ (normalised, section 6), draw noise $z_0 \sim \mathcal N(0, I)$ and a time $t \in [0, 1]$, form $z_t = (1 - t)\, z_0 + t\, z_1$, and train the network to output the velocity:
$$ \mathcal L = \mathbb E \,\big\| v_\theta(z_t, t \mid \text{shell}, \text{facts}) - (z_1 - z_0) \big\|^2 . $$
The transformer never has to jump from geometry to anatomy in one pass. It only learns how to move a noisy interior a little closer to a real one, inside the anatomy space, while reading the shell.

Kept from v7 and the baseline:
- **Partial reveal.** For a random share of people some interior tokens are given clean, with a flag marking them as given. The velocity loss is computed on the other tokens only, and at sampling the given tokens are held fixed at every step. This trains the later uses of v7: extend a region, edit a region, condition on a revealed part. Whether held codes keep the decoded anatomy intact at region borders is checked after decoding.
- **Weighting.** The loss is averaged over the unknown tokens within each person first, then over people; how often each part is sampled is set explicitly.
- **Voxel-level terms.** The one-step estimate $\hat z_1 = z_t + (1 - t)\, v_\theta$ is decoded on a random crop and scored with the baseline's weighted cross-entropy and Dice. Its gradient reaches the velocity scaled by $(1 - t)$, so it vanishes at the very end of the path; it is therefore applied over a broad range of $t$, not only near 1. This keeps the surface-and-bone emphasis and the charge for inhuman sizes. Whether it helps over the plain velocity loss is test F3.
- **Fact dropout.** Facts are hidden at random so the model works without them and guidance strength can be chosen at sampling.
- **Mirrored heads** as in the baseline.

### 4.4 Sampling

Start from noise at every interior position and integrate the velocity field with a fixed-step solver. The number of steps is chosen on validation people by the scores of section 5, not by hand. Each draw is one interior; several draws give the spread.

### 4.5 Unlabelled classes

The target codes carry each source's drawing convention (a HaN-Seg head has "other lean tissue" where the brain is). The codes cannot be masked class by class. So the convention is a fact: the list of classes the person's source draws is given as fact tokens. At inference the full list is asked for. Test F4 checks that a HaN-Seg test head generated with the full list gets a brain where TotalSegmentator heads have one.

## 5. Scoring

A sharp structure in the wrong place scores as badly on Dice as a blur, so Dice alone can hide the gain. Each run reports:

1. **Dice per structure**, per draw and for the per-voxel majority over several draws, against the same baselines as before (nearest person, one-shot version 2, compression ceiling).
2. **Plausibility against real people:** per structure, the volume, the number of connected pieces and the centre position, compared with the training people's ranges. Reported as the share of structures inside the range. This is the measure for "clearly inhuman".
3. **Shell fidelity:** the generated body outline (all non-outside labels) against the input shell.
4. **Spread:** variation between draws per structure, against the variation between real people of similar shell.

## 6. Data preparation

Checked on 2026-11-24.

| Item | State | Work needed |
|---|---|---|
| Label volumes, splits, frames | done (v7) | none |
| Label autoencoders (face 8 mm, pelvis 12 mm) | done, frozen | none |
| Per-person tokens, codes, shell features, loss weights, mirrored heads | job running; stored once as `m16_full/prep_v2_TF8` | none after it lands; shared with the baseline |
| **Code normalisation** | missing | per part and per channel mean and spread over training people. Measured spread differs by part (face about 1.0, pelvis about 0.58; channels 0.28–1.28). Flow matching does not require unit variance; equal scales make the noise level mean the same thing in every channel and part. Minutes of CPU on the cache |
| **Shell token list** | missing | pick the cells at or next to the surface from the cached distance features; no new computation on the scans |
| **Facts: age and sex** | pelvis index has both; TotalSegmentator heads have them in the census table but not in the head index; HaN-Seg: not held | join the census table into the head index; mark HaN-Seg as unknown unless its release carries them (to check) |
| **Facts: drawn-class list** | in the head index (`known`) | pass through as fact tokens; the pelvis set has one convention |
| **Plausibility reference table** | missing | per structure volume, piece count and centre over training people; CPU on the label volumes |
| **Noise tolerance of the codes** | measured | see below; no work unless test F1 shows a problem |
| More heads | 103 labelled heads in total (61 TotalSegmentator, 42 HaN-Seg), all in use | a search for further public head CT with labels; not started |

**Noise tolerance of the codes (3 people, laptop).** Median Dice after adding Gaussian noise to the true codes, as a share of each channel's spread:

| Noise | 0 | 10% | 20% | 30% | 50% | 100% |
|---|---|---|---|---|---|---|
| Pelvis (2 people) | 0.85, 0.80 | 0.84, 0.79 | 0.81, 0.74 | 0.75, 0.67 | 0.70, 0.65 | 0.38, 0.28 |
| Face (1 person) | 0.79 | 0.79 | 0.78 | 0.76 | 0.74 | 0.63 |

The decoder tolerates independent errors of about a tenth of the spread with almost no loss. This measures the decoder's sensitivity only. It does not say a good draw must land within a tenth of the held-out person's code: a draw may be a different, equally plausible interior. If decoded draws look noisy, the fix is to fine-tune the decoder on slightly noisy codes (an hour on a small card), not to change the generator.

## 7. Tests, in order

Each step is a smoke run on a small card first, then a cost table before any full run (v7 section 7).

| # | Test | Passes if | Cost |
|---|---|---|---|
| F0 | One-shot baseline, version 2, 20,000 steps | reference numbers exist | about $2, A10G-small (queued) |
| F1 | Flow matching on the same tokens, shell and fact tokens in the sequence, plain velocity loss, same size and steps | beats F0 on plausibility and is no worse on Dice of the majority over draws | smoke under $1; full about $2–4 plus sampling |
| F1b | Flow matching with a retrieved training person as a second reference beside the shell (pelvis first). The retrieved person's codes enter as their own token block; the flow runs on the difference between the target's codes and the retrieved person's. The target and their mirror are excluded from retrieval in training. Together with a one-shot run without the nearest person (F0b), this completes the two-by-two: one-shot or flow, each with and without a reference person | shows whether a gain comes from flow or from the reference; kept as the generator if it beats F1 on Dice, shell-swap and plausibility | about the same as F1; F0b about $2 |
| F2 | Separate early weights per token kind against shared weights | kept only if it improves F1's scores | same as F1 |
| F3 | Voxel-level terms near the end of the path | kept only if eyes, bone edges and sizes improve | same as F1 |
| F4 | Unlabelled classes as facts | HaN-Seg test heads get a brain when the full class list is asked for | no extra training |
| F5 | Sampling steps and guidance strength | chosen on validation people by section 5 | sampling only |

After F1–F5 the v7 milestones continue with this generator: Context (17), Trunk (18), Seams (19), Rig (21).

## 8. Risks

- **Small data.** The image models this design comes from are pretrained on billions of images. This one trains from scratch on 76 heads (152 with mirrors) and 345 pelvises. It may largely reproduce training people. That is acceptable for plausibility and is measured by the spread score; it limits how far the model adapts to an unusual shell.
- **A weak condition.** A shell says less about the interior than a reference image says about an edit. The model can only be as right as the shell allows; the rest is a plausible draw. Dice will stay well below the compression ceiling for organs the shell does not constrain.
- **Sequence length.** Shell tokens add about half to the sequence, so attention costs about 2.2 times the one-shot model per step, and sampling runs the network once per step. At 8 mm face tokens this still fits the measured limits of v7 section 5.5.
- **Latent quality.** The autoencoder was trained for reconstruction only, with no smoothness constraint on its codes. Section 6's noise test suggests it is usable; F1 decides.

## 9. Changes to the milestone table (v7 section 13)

| # | Name | Change |
|---|---|---|
| 15 | Tokens | done (section 2) |
| 16 | Transformer | version 1 done and failed against the nearest-person baseline; version 2 is the one-shot baseline (F0) |
| 16b | Flow | new: shell-conditioned flow matching (F1–F5); replaces the one-shot model as the generator |
| 20 | Diffusion | merged into 16b |

## 10. External review, 2026-11-24

A reviewer read sections 1–9. Both papers it put forward exist and are the closest published work:
- **Surf2CT** (Yoon et al., 2025, arXiv 2505.22511): flow matching from a body-surface signed distance and demographics to torso CT, coarse at 8 mm then refined by patches; 3,198 training CTs. The nearest precedent for this generator, and for a coarse pass followed by local detail.
- **From Surface to Viscera** (Atici et al., MIDL 2026): one encoder reads the body surface, another a mean-shape point cloud of 20 organs, and the model predicts how that anatomy deforms. It starts from recognisable organs instead of noise.

**Correction of this document.** Section 2 first gave the pelvis as about 300 tokens per person. The right figure is about 8,600 (12 mm tokens on 3 mm labels); a pelvis and a face at 8 mm are about the same length. The reviewer's point that one face outweighs a pelvis 28 times followed from that error. The remedy it proposed is adopted anyway (section 4.3, weighting).

**Adopted** (already written into the sections above or the tests below):
- loss on unknown tokens only and held tokens at sampling for partial reveal; voxel-level terms over a broad range of $t$; per-person averaging; the narrower reading of the noise test; unit variance is a convenience, not a requirement;
- **shell-swap test (F6):** fixed random seed, shells exchanged between held-out people matched for part, coverage and facts. Bone centres, bone widths and skin-to-bone depths must follow the person whose shell was used. A memorised training body passes the plausibility score of section 5 and fails this;
- the single exported draw is scored on its own, beside the majority over draws;
- **hidden-structure test (replaces F4's pass condition):** take a structure that a source really draws, hide it under an artificial convention, keep its mask for scoring, then ask for the full class list. Recovery of the structure and damage to its neighbours are measured, which the original F4 could not do.

**Open, for decision:**
1. **Reference anatomy inside the flow model — decided 2026-11-24: run it.** F0 against F1 changes two things at once: one-shot against flow, and with against without the nearest person. The full two-by-two is run on the pelvis (tests F0, F0b, F1, F1b in section 7). The shell stays the primary reference of the design; F1b tests whether a retrieved person as a second reference adds to it.
2. **Face labels at the source.** A HaN-Seg head has "other lean tissue" where its brain is, and the frozen autoencoder has already turned that into the target code. Telling the generator the convention (section 4.5) does not teach it what a fully labelled head is. Two repairs, both before any face flow run: fill the missing classes in the data (the brain, trachea and neck vertebrae of the HaN-Seg CTs with the licensed TotalSegmentator tasks; the reverse direction needs a model trained on HaN-Seg's classes), and retrain the face autoencoder with a marginal loss so an ambiguous voxel accepts any class compatible with its source (Shi et al., marginal and exclusion loss for partially labelled data). The pelvis has one convention and is unaffected, which is one more reason to run the flow comparison on the pelvis first.

The reviewer's earlier warning on v7, that class-weighted cross-entropy moves boundaries and inflates rare classes, matches what version 1 then did (structures at about 1.6 times their volume).
