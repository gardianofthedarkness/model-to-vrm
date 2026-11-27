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

**Order (user decision, 2026-11-24).** F1b and any other use of a retrieved person are a last backup, not a next step. The current reading is that the limit is the amount of training data and that flow matching has not yet been judged at full training. So: judge F1 at full training on pelvis and face together (every run includes the face), then more data; retrieval only if those fall short.

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

## 11. Results and decisions, 2026-11-24 (evening)

### 11.1 A fault in test-time decoding, found and fixed

Tokens with no body in them were decoded as zeros. The frozen decoder reads neighbouring tokens, and zeros are not a code it knows, so it painted right parotid, right eye and brainstem over plain tissue next to the body's edge. On one test head the true codes decoded this way gave the right parotid at 10.5 times its volume. Those tokens now carry the autoencoder's codes of the shell-only volume (outside, unknown, one generic tissue inside), which is available at inference. Training was not affected. This was the cause of the apparent left/right gap and of the speckle.

Scores before the fix (sections 2 and 3 of this plan, and the first F0/F1 numbers) are superseded by:

| Part | Flow F1, one draw | Flow F1, majority of 3 | One-shot version 2 | Nearest person |
|---|---|---|---|---|
| Pelvis | 0.583 | 0.636 | 0.528 | 0.544 |
| Face | 0.339 | 0.393 | 0.365 | 0.310 |

Median Dice over classes, 55 test people, both models at 20,000 steps and 14.6 M parameters, pelvis and face trained together. For one flow draw on the pelvis, 99% of structures have a volume, 96% a piece count and 96% a centre inside the range of training people; on the face 80%, 77% and 81%. The majority over draws scores higher on Dice but breaks structures into pieces (67% and 53%), so the single draw is the output for rigging. A model six times larger (86.5 M) gained one Dice point in the smoke run.

### 11.2 First generated character (the elf)

- **Pelvis:** plausible from the full mesh (overlap 0.74 with a real person): hip bones, sacrum, femoral heads, lumbar spine, gluteal muscles, iliopsoas, fat layer, on both sides.
- **Head, first try: failed** (eyes 0.2 and 1.0 ml, jaw 18 ml). Two causes, both in how the input was prepared, not in the model:
  1. **Size.** The head was sized with her hair included, so her hair was sized like a skull and her face came out 19% too small.
  2. **Frame.** Ear and eye keypoints put the midline 9 mm off; her eye keypoints are themselves uneven.
- **Head, corrected** (size from bare skin only, midline from the mesh's symmetry plane): eyes 4–8 ml at ±35 mm, jaw 43–54 ml, teeth and an 635–867 ml brain, all at the places real heads have them. Still wrong: skull 1,025–1,118 ml and brain too small, because her hair counts as head; no eye lenses.
- **Hair:** on three real heads a 10 or 25 mm hair-like cover inflates the skull (about 800 to 1,200 ml) and leaves eyes and jaw intact. Hair is a skull problem, not a face problem.
- Head and pelvis asked for nearly the same size (1,791 and 1,844 mm per mesh unit), so one size for the whole figure is workable for this character.

### 11.3 Decisions (user)

- **Every training run includes the face.** Pelvis-only runs are plumbing checks.
- **A retrieved nearest person (F1b) is a last backup.** First: flow at full training, then more data.
- **No regenerated bald or re-dressed versions of a character** to get the surface under hair or cloth: they shift the face. Wanted: a learned guess of the body under hair and cloth from the covered shell. External tools read and parked as heavier than the need: VoxHammer on TRELLIS (region-preserving 3D editing, 40 GB card) and ETCH (body under clothing as a SMPL fit; weights non-commercial).
- **Production is end to end:** the full body shell goes in, the filled body comes out.

### 11.4 Added tests

| # | Test | Passes if | Cost |
|---|---|---|---|
| P1 | Sensitivity of the head model to size (±10, 20%), shift (5, 10 mm) and turn (5, 10°) on real test heads | gives the precision the input normalisation must reach and the jitter training needs | laptop, running |
| P2 | Input normalisation from the shell alone: upright, midline from the symmetry plane, one size from bare skin; no ear or eye keypoints | same frame on CT people as the anatomy-based one, within P1's tolerance | CPU |
| P3 | Train on shell-derived frames with size, shift and turn jitter | scores hold under the errors P1 measured | one run as F1 |
| P4 | Covers: real people with synthetic hair- and cloth-like covers on the shell, flagged per token; target under the cover is "outside" | covered real heads keep their true skull; the elf's skull falls to a real size | preparation on CPU plus one run as F1 |
| F7 | Continuous decoder: the autoencoder's decoder answers "which structure is at this point" for any point; meshes are extracted from the boundaries between regions (one answer per point, so no overlaps or gaps). The generator and tokens are unchanged | smoother surfaces and thin structures (eye lens, muscle sheets) survive; kept if the rig needs it | autoencoder retraining only |

Direct triangle generation (face-by-face mesh models) is not planned: it gives no guarantee that many nested structures are closed and do not intersect. A fixed template mesh bent to each person is kept in mind for bones only, where a consistent vertex layout helps the rig.

## 12. Fascia and tendons, 2026-11-25

User priority: fascia and tendons matter for the flesh physics; vessels and nerves do not. The model has no fascia or
tendon class. No open CT label set marks them, and on CT they look like muscle. The R3 teacher expects fat and fascia
from this model.

### 12.1 Mini test F8a: fascia as boundaries between predicted regions (local, CPU)

`research/interior_model/fascia_boundaries.py`. On the 42 test pelvises of the F1 run (3 mm voxels, predictions after
the decoding fix), three surfaces are read off the labels: fat | muscle (deep fascia), muscle | another muscle (septa),
muscle | bone (attachments). They are compared with the same surfaces in the true labels. The baseline is another test
person's true labels in the same frame, i.e. a population guess. Medians over people:

| Surface | Model: within 6 mm | Other person: within 6 mm | Model: median distance | Other person: median distance |
|---|---|---|---|---|
| fat \| muscle | 0.60 | 0.33 | 5.2 mm | 9.5 mm |
| muscle \| muscle | 0.53 | 0.51 | 6.0 mm | 6.0 mm |
| muscle \| bone | 0.72 | 0.56 | 4.2 mm | 5.2 mm |

Findings:
- The deep fascia surface carries real information about the person: about half the population guess's error and a
  90th percentile of 12 mm against 26 mm. The model's predicted area is close to the truth (2,546 cm² against 2,618 cm²).
- Septa are no better than the population guess, and the model draws only about two thirds of their true area
  (1,316 cm² against 1,951 cm²). In the truth they are also incomplete, because "muscle_other" lumps many muscles together.
  The shell says little about where one muscle ends and the next begins, so septa need a prior, not the boundary alone.
- Attachments to bone are better than the guess, but a 4 mm median is more than one voxel.
- Distances are in steps of 3 mm voxels. The continuous decoder (F7) is expected to tighten all three; measure it with the same script.

### 12.2 Mini test F8b: where the atlas puts its fascia (Z-Anatomy, CC BY-SA, local)

Distance from each atlas fascia sheet to the atlas's own muscles, skin and bone (median / 90th percentile):

| Sheet | to muscle | to skin | to bone |
|---|---|---|---|
| Fascia lata | 3.2 / 8.2 mm | 14 / 31 mm | 41 mm |
| Iliotibial tract | 4.1 / 11.4 mm | 17 / 31 mm | 21 mm |
| Thoracolumbar fascia, posterior layer | 1.5 / 2.5 mm | 22 / 32 mm | 15 mm |
| Temporal fascia, superficial layer | 0.9 / 1.7 mm | 4.5 / 8.0 mm | 4.6 mm |
| Masseteric fascia | 1.1 / 2.3 mm | 6.7 / 9.2 mm | 4.8 mm |
| Iliopsoas fascia | 9.0 / 28.5 mm | 76 / 91 mm | 17 mm |
| Epicranial aponeurosis | 20 / 55 mm | 5.2 / 8.3 mm | 4.2 mm |

Findings:
- Most sheets wrap muscle within a few millimetres, so their place is the fat | muscle surface the model already gives
  (12.1). The atlas is needed only to say which patch is which sheet (fascia lata, IT tract), and each sheet's thickness
  and stiffness should come from published measurements.
- Two do not follow the muscle: the epicranial aponeurosis (scalp) runs between skin and bone, and the iliopsoas fascia
  needs checking. They need their own placement.
- Tendons: the exported atlas has origin and insertion areas for the hip muscles (gluteus max/med/min, iliopsoas,
  piriformis, obturators, gemelli, quadratus femoris, tensor fasciae latae, sartorius, rectus femoris) and for the
  chewing and some expression muscles (masseter, temporalis, zygomaticus, orbicularis oris and oculi, frontalis). A
  tendon can be placed as the path from the predicted muscle's end to its attachment area on the predicted bone.

### 12.3 Proposed steps (not built)
- F8 fascia: deep fascia = the predicted fat | muscle surface (after F7). Atlas fascia sheets are fitted onto it only to
  name the patches. Thickness and stiffness per sheet come from the literature.
- F9 septa and tendons: from the atlas muscles fitted to the predicted bones and named muscles, the same fit that places
  the expression muscles. Score the septa with 12.1's script against the population guess.

## 13. The v3 face "failure" was a decoding bug, 2026-11-25

The full v3 run (40,000 steps, L40S) scored the face below nearest person: 0.216 against 0.269. Its sinus, teeth and oral
cavity were nearly absent. A chain of checks found the cause:
- The model was as bad on heads it trained on as on test heads. Its stored codes, decoded, reached 0.67. So it was not
  overfitting, and the targets were there.
- **Root cause: decode tile size.** The autoencoders use GroupNorm, which normalises over the whole tile. They were trained
  on 64-voxel crops, but sampling and scoring decoded in 24-token tiles (192 voxels at 8 mm tokens). The same codes for one
  test head give sinus 0.90 in 8-token tiles and 0.26–0.37 in 24-token tiles (`tile_check.py`). F1's autoencoder is less
  sensitive, so F1 looked fine. Its scores were still underestimates.
- Code noise is not the issue: all three autoencoders keep their Dice under added code noise (`code_fragility.py`).
- Fix: decode in tiles the size of the training crop, `transformer_smoke.dec_tile` (face 8 tokens, pelvis 16). Commit 6fa77cc.

A/B on identical v3 face data (face only, 10,000 steps, test heads, one draw / majority of 3; nearest person 0.269):

| Arm | Old decoding | Fixed decoding | Sinus | Oral cavity | Hair cleared |
|---|---|---|---|---|---|
| A: v3 face autoencoder | 0.177 / 0.154 | 0.265 / 0.266 | 0.61 | 0.31 | 49% |
| B: F1 face autoencoder | 0.237 / 0.264 | **0.280 / 0.323** | 0.65 | 0.62 | 96% |
| C: tissue types (new autoencoder) | 0.509 / 0.521 | 0.572 / 0.598 (nearest 0.597) | air 0.47 | — | 81% |

Decisions and notes:
- Keep F1's face autoencoder for named classes for now. It generates better, especially on covered heads.
- Tissue types (user direction, section 14 to come): level with nearest person on clean heads, ahead on covered heads
  (0.515 against 0.422). Eyes (0.63 against 0.73) and lean (0.58 against 0.65) lag. Fat under the skin got worse with the
  fixed tiles (0.47 to 0.36).
- Next autoencoder: make it independent of tile size (no GroupNorm, or a norm with fixed statistics) and encode in
  crop-sized tiles too. The data preparation still encodes in 96-voxel tiles.
- Lesson: an autoencoder's own reconstruction score must use the same encode and decode path as sampling.

Full v3 run (pelvis and face, 40,000 steps) re-scored with the fixed decoding, one draw / majority of 3, test people:

| Part | Old decoding | Fixed decoding | Nearest person |
|---|---|---|---|
| Face | 0.216 / 0.236 | **0.274 / 0.304** | 0.269 |
| Face, covered by hair | 0.224 / 0.248 | 0.285 / 0.323 | 0.111 |
| Pelvis | 0.502 / 0.545 | 0.512 / 0.552 | 0.454 |
| Pelvis, covered | 0.467 / 0.508 | 0.444 / 0.471 | 0.438 |

The face is now above nearest person, sinus 0.63 against 0.69. The covered pelvis got slightly worse with the 16-token
pelvis tiles. F1's scores (old decoding) are not comparable: different test people and frame, and they are also
underestimates.

## 14. Next phase: more heads, then two architectures side by side (user direction, 2026-11-25)

Targets are tissue types (fat, lean, bone, air, eye, organ), with names fitted afterwards from Z-Anatomy. Surfaces come
from the decoded probabilities (50% iso-surface), shown as an interactive 3D page of the elf for each model.

### 14.1 Data first: 400-500 heads with the face in view
Today: 103 heads (42 HaN-Seg, 61 TotalSegmentator). Only open sources, no emailed agreements. Candidates to count before
any download beyond a sample:

| Source | Size | Face in view? | Licence / access | Tissue labels |
|---|---|---|---|---|
| CQ500 (head CT, Qure.ai) | 491 scans | brain-protocol CT: orbits and nose likely, jaw often cut (to be measured) | CC BY-NC-SA 4.0; download page to be checked for a click-through | CT brightness + TotalSegmentator (brain, eyes) |
| TotalSegmentator CT v2.0.1, heads dropped by the strict completeness rule | to count | partial | CC BY 4.0, already held | same |
| IDC whole-body PET/CT with a whole skull | about 7% of scans (3 of 40 in the FDG sample) | yes | CC BY | same |
| IXI MRI | 581 | yes (not defaced; chin near the edge) | CC BY-SA 3.0, already held | fat and lean from MRI brightness; bone and air look alike on MRI, so a separate step |

Voxels outside a scan stay unknown, so a head cut below the nose still teaches what it shows.

### 14.2 Two directions at 30,000 steps each, run at the same time on the same heads
- **D1, current pipeline plus shape terms (plan item F3, revised).** The conv autoencoder (with a channel norm, so no
  tile dependence) and the flow transformer stay. F3's per-voxel terms are replaced by per-tissue shell terms on the
  decoded one-step estimate: position (soft centroid), size (soft volume), shape (surface distance via distance
  transforms) and topology (number of pieces and holes; e.g. clDice or Betti matching). Existing implementations first
  (MONAI losses, published topology losses).
- **D2, transformer VAE.** A VecSet-style shape VAE (3DShape2VecSet; the design behind Hunyuan3D-2's ShapeVAE and
  similar to TRELLIS, which is MIT): a fixed set of latent vectors with a KL term, decoded by cross-attention at any query
  point into tissue probabilities, so meshes come out directly. The flow transformer generates the latent set from the shell.
  Licences: Hunyuan3D-2 is under Tencent's community licence (not valid in the EU, UK and South Korea); 3DShape2VecSet's
  licence still to be checked; TRELLIS is MIT.
- **Comparison.** Test scores per tissue (Dice, surface distance), agreement between draws, the shell-swap test (F6),
  consistency across reruns, and the elf's 3D page. A cost table comes before either 30,000-step run.

### 14.3 Findings from tests 2 and 2.5 (2026-11-25), and the seam test

**Test 2.5, a head model continued on head and pelvis:** arm B's face model, plus 10,000 steps on both parts. Face kept:
0.285 / 0.327 (one draw / majority) against 0.280 / 0.323 before; covered heads 0.257 against 0.242. Pelvis learned from
zero to 0.402 (40,000-step v3: 0.512). No forgetting when the head stays in the training mix.

**Test 2, partial heads (CQ500-like: only the top 148 mm in view, the rest unknown).** Same 30 test heads, face priority
parts (eyes, nose, teeth, mouth, lips, tongue, jaw, masseters, sinus):

| Arm | Priority median | Jaw | Oral cavity | Lips |
|---|---|---|---|---|
| Half the people, full heads | 0.628 | 0.59 | 0.72 | 0.47 |
| Same plus the other half cut | 0.502 | 0.50 | 0.52 | 0.34 |
| All people full (arm B) | 0.587 | 0.59 | 0.62 | 0.45 |

Cut heads fed without telling the model lowered the lower face by 0.09-0.20 and did not help the upper head. Likely reason:
the model sees shells that end at the nose and learns lower faces as uncertain. Fixes to test: a coverage fact per head
(like age and sex), lower loss weight for partial heads. CQ500's upper heads join only after one of these removes the harm.
Note: "half, full" scored above "all, full", so single runs differ by about 0.04; a repeat of arm B measures the spread,
and the architecture comparison uses 2-3 seeds per arm.

**Seam test (added on the user's request).** Part windows end at fixed places in their frames, the same at training and
inference, so they do not cause the harm above. Seams have three other problems: less context near a window edge; two
separately generated parts that disagree (vertebrae, neck muscles, fat thickness); different frames. Test: on whole-body
PET/CT (IDC FDG and PSMA collections, skull base to thigh), a trunk window overlapping the pelvis window by about 5 cm,
generated (a) independently and (b) with the pelvis overlap revealed (partial reveal, section 4.3, resampled between
frames). Score: disagreement in the overlap (bone positions, fat thickness, muscle edges). Needs a trunk part first
(v7 milestones Context 17, Trunk 18, Seams 19). Runs after the current tests.

### 14.4 Head data census results (2026-11-25)

| Source (private copy) | Heads | Jaw whole | Face intact | MRI pair | Use |
|---|---|---|---|---|---|
| CQ500 (cq500-ct, 41 GB, CC BY-NC-SA) | 491 | 0 of 20 sampled; covers about the top 148 mm | yes | no | upper head only, and only after the coverage fact removes the harm of test 2 |
| SynthRAD2025 head-and-neck (synthrad2025, Zenodo record 14918214, centres A, B, C, E, CC BY-NC 4.0; centre D's restricted licence does not apply) | 156 | 20 of 20 sampled; upper teeth 18, skull whole 20; about 250 mm covered | yes (nose median 27 mm ahead, 19 of 20 above 10 mm) | 20 of 20 | full heads (CT) and MRI-CT pairs for test 1 |
| IXI (ixi-t1-t2) | 581 MRI | chin at the edge | yes | MRI only | full faces once MRI-to-tissue works (test 1) |

Test 1, first run (42 HaN-Seg pairs, 5 folds, face and neck region): jaw bone 0.81 (1.5 mm), lean 0.84, air 0.70, all bone
0.69, fat under the skin 0.65, thin fat 0, eyes 0. Rerun v2 with crops centred on eyes and thin fat and class weights.
Noise between identical 10,000-step runs: about 0.01-0.02 on the face medians (arm B repeated).

**Test 1 answer (MRI run v2: a share of crops centred on eyes and thin fat, class weights by rarity; 42 pairs, 5 folds):**
eyes 0.79 (against expert outlines 0.77), jaw bone 0.81, air 0.74, all bone 0.72, lean 0.83, fat under the skin 0.68,
thin fat 0.16; surfaces 1.5-2.1 mm. MRI aligns with CT for every priority part. The model trained on all 42 pairs is
saved as hanseg-ct-mr/mri_tissue_v2/segresnet_all.pt for IXI. IXI comes from other scanners: check a sample first.

**SynthRAD labelled (2026-11-26):** 128 of 156 heads in the head frame (tissue types; synthrad2025/face_v1_headframe,
index_synthrad.json; split 102 train, 13 val, 13 test). Centre A: 85 of 91, complete (18% of the window outside the scan).
Centre C: 43 of 65, top of the skull cut (53% outside the scan; face and jaw present). 28 failed, all for a missing eye or
ear canal at the top edge of the scan (22 from centre C); placing them by the shell fit instead could recover them.
Complete full-face heads: 42 HaN-Seg + 61 TotalSegmentator + 85 SynthRAD = 188; plus 43 top-cut.

**IXI local check (6 heads, MRI tissue model v2):** eyes found as two 6-10 ml pieces in 5 of 6; scalp, face fat, skull ring
and airway read cleanly. Inside the brain the model paints bone and fat patches (IXI's brain-research contrast differs from
the HaN-Seg radiotherapy MRIs). Fixes: brain mask from SynthStrip (FreeSurfer) set to the organ class; test and retrain
the MRI model on SynthRAD's MRI-CT pairs (other scanners).

### 14.5 Precision test: how much boundary accuracy the labelling loses (user idea, 2026-11-26)

`research/interior_model/precision_test.py`, HaN-Seg heads 01, 03, 05 (0.56-0.68 mm in-plane, 2 mm slices), locally.
A. Current CT tissue labels (median filter, hard per-voxel classes, 1 mm nearest) against the exact surface taken from the
raw CT at its own resolution: fat|lean (-30 HU) 0.99-1.10 mm median, 1.73-1.92 mm 90th percentile; bone (150 HU)
0.89-0.98 / 1.57-1.74 mm. Most of it comes from the 2 mm slices and the hard classes, little from the 1 mm resampling.
B. Named regions against the expert outlines: TotalSegmentator eyes 0.84-1.0 mm median (90th 1.0-1.5), mandible 0.63-0.84
(1.0-1.2): about the size of expert disagreement. Snapping to image edges (random walker in a band around the label) made
eyes worse (90th 2.7-3.9 mm, volume 11-13 ml against 8.5: leaks into the eyelids, which have no edge on CT) and shrank the
mandible (spongy bone lost).
Decision: identity from the labelling models; tissue boundaries from the raw CT as continuous fields (signed distance per
tissue at the CT's own resolution, carried into the head frame by interpolation, not by nearest labels). Target: tissue
surfaces under 0.5 mm. Edge snapping only where a real edge exists; not now.
Continuous-field test (`tissue_fields_test.py`, surfaces compared by dense even sampling, 3 heads): current labels
fat|lean 0.47-0.50 mm median, 1.44-2.32 mm 90th percentile; bone 0.41-0.43 / 0.82-0.89. Raw CT brightness interpolated
onto 1 mm, surface at the HU level: fat|lean 0.32-0.35 / 0.57-0.61; bone 0.29-0.31 / 0.50-0.53. At 0.5 mm only slightly
better (0.29-0.32 / 0.51-0.57). The worst 10% improves 2-4 times; 1 mm is enough. 3D jaw from the field is smooth (teeth,
nasal opening visible); from hard labels it is a staircase of 2 mm slices.

### 14.6 Head data correction and the field experiment (2026-11-26)

**SynthRAD is defaced.** Every SynthRAD2025 head-and-neck CT checked (6 of 6 mid-sagittal views, raw archive) has a wedge
over nose, lips, front of the eyes and chin filled with exactly -1024 HU, the scan minimum; its real air is noisy around
-1000. SynthRAD eye volumes were half of HaN-Seg's (median 8.8 against 18.6 ml). The census in 14.4 missed it. Decision:
exact-minimum voxels (opened by 3^3) are unknown, for SynthRAD only (HaN-Seg's scanner clamps real air to the minimum
too, so the rule would wrongly mark its air). SynthRAD stays as partial heads; it never scores the face. Corrected count
of complete faces: HaN-Seg 42 + TotalSegmentator 61 (raw not yet re-fetched), not 188.

**New open sources searched** (no emailed agreements). Kept: **PDDCA** (48 radiotherapy head-and-neck CTs, CC BY 3.0
mirror; all 48 show nose, lips and chin; head frame placed for 48 of 48, eye median 18.9 ml). Rejected: **VSDFullBody**
(Zenodo 8270365, post-mortem CT, copied privately as vsd-fullbody-ct for later trunk work): head series have a 212-272 mm
field of view that clips the face, post-mortem gas and fractures, whole-body series start at the shoulders. Not open:
TCIA head-and-neck collections (NIH controlled access since April 2025), NMDID, HECKTOR, SegRap, OASIS-3. Defaced: open
FDG PET/CT copy, SynthRAD2023, the 120-person head CT-MRI set on Zenodo. Not whole heads: MUG500+ (skull only),
ToothFairy (dental CBCT), HNTS-MRG. Full-face heads now: 90 labelled (HaN-Seg 42 + PDDCA 48).

**Field experiment (suite1, one L40S):** same full-face training heads (HaN-Seg + PDDCA train), same test heads (HaN-Seg
+ PDDCA test), 10,000 flow steps, conv autoencoder with channel norm (12,000 steps) and the flow transformer:
A0 old hard labels; N new continuous field, tissue + eye; NS N + shell loss (soft Dice, size, centroid, soft clDice on
the decoded one-step estimate, weight set each step so the shell part equals the flow loss: no set number); NSP NS + 85
SynthRAD heads with the masked face unknown. Seeds 0 and 1 for N, NS, NSP. NSP ran out of GPU memory beside the other
five (full-size runs peak at 6-9 GB, not the smoke test's 3-4) and runs as a follow-up reusing the same autoencoders
and preparation. Lesson: cap processes per card by measured peak memory (MAXPAR). Results go in 14.10.

### 14.7 What MAGI-1 teaches for D2 (SandAI, "MAGI-1: Autoregressive Video Generation at Scale", 2025; code Apache-2.0)

Read for ideas only (no weights used; weight licence not stated in the README). Their scale (614M VAE, 4.5B/24B
generator, petabytes of video) is far from ours (~15M generator, under 200 heads): we take designs, not sizes.

| MAGI-1 | Our use | Decision |
|---|---|---|
| **Transformer VAE**, not conv: a 3D conv patch embedding (8x8x4, stride = kernel), 24 transformer blocks with QK-norm, LayerNorm + linear to 16 latent channels (mean, log-variance: KL), symmetric decoder ending in pixel shuffle + one 3x3x3 conv. Faster than conv VAEs of similar size on modern GPUs; decode 12 ms per clip | D2 | **D2 becomes a patch transformer VAE first**: 8^3-voxel (8 mm) patches, so its tokens are exactly the flow transformer's tokens and the generator stays unchanged; QK-norm; KL latent; pixel-shuffle head for grids (training, scoring). The VecSet cross-attention point decoder (14.2) is kept as a second head for direct meshes, tested after the grid head works. A whole head is about 2-5k tokens: full attention, no tiling (tiling caused our decode bug) |
| VAE losses L1 + KL + LPIPS + GAN | D2 losses | soft cross-entropy + KL + the shell loss (our analogue of a perceptual loss). No GAN or LPIPS: they are image measures |
| **Chunk-wise generation**: each chunk has its own noise level, earlier chunks less noisy; full attention inside a chunk, causal across chunks; clean earlier chunks cached (KV cache), memory constant per chunk | seams between body parts (14.3) | **The seam test's arm (b) uses MAGI's scheme**: body regions as chunks (head, neck, trunk, pelvis, limbs), each region's tokens with their own timestep, block-causal attention across regions, the finished region cached. This replaces separate part models joined afterwards |
| Clean conditioning chunks get up to 5% noise, no loss on them | partial reveal (4.3) and masked heads | revealed tokens get a little noise during training (against over-trust of given tokens); loss only on tokens being generated. Masked partial heads (SynthRAD) fit the same frame: the measured part is "revealed", the masked face is generated |
| Image-to-video as continuation with only the first frame clean | partial scans | same idea for dental or cut scans: whatever is measured is given, the rest generated; one model for full and partial inputs |
| **Shortcut distillation**: one model trained so one large step equals two half steps; 8-64 steps at inference, guidance distilled in | live runtime (flesh at minimum compute) | after the architecture choice, distil the flow to 1-4 steps for games and VRChat (now 16 Euler steps) |
| Logit-normal timestep sampling (most compute at low noise); QK-norm; sandwich norm; softcapped adaLN scale | flow transformer (D1 and D2) | adopt QK-norm and logit-normal t in the next phase for both directions (cheap, stabilising). Not in suite1, which keeps conditions fixed |
| Guidance on the previous chunk (w_prev) and on text, strength by timestep | shell conditioning | test classifier-free guidance on the shell tokens (drop the shell in some training steps) after D1/D2 |
| MagiAttention for up to 4M tokens | none | not needed: our sequences are thousands of tokens |

### 14.8 MAGI-2-preview (SandAI, blog 2026-08-05; code and weights Apache-2.0; "intermediate research release")

What changed from MAGI-1, and what we take:

| MAGI-2-preview | Our use | Decision |
|---|---|---|
| **Dropped MAGI-1's chunk-wise autoregression**: one single-stream transformer generates the whole clip, audio and video together | seams (14.3, 14.7) | prefer **one stream over the whole body** while the tokens fit (a head is thousands of tokens); keep MAGI-1-style chunks only as the fallback when a whole body does not fit |
| Audio and video tokens in the **same stream** (not one conditioning the other) | shell and interior | test putting shell tokens and interior tokens in one stream (joint attention) instead of the interior cross-attending to the shell; later other channels (MRI brightness, joint positions) as more token types |
| **Two stages**: low-resolution generation, then a refiner to full resolution (100 + 5 steps) | fine boundaries | a **refiner stage**: the 8 mm generator places tissues, a small refiner sharpens boundaries at 1 mm (or 0.5 mm) toward the continuous-field precision (14.5). Fits the thin fat layers |
| Reused an existing VAE (Wan2.2) and **distilled a fast "turbo" decoder** | live runtime | keep the gated transformer VAE as the tokenizer; distil a small fast decoder for games and VRChat, beside step distillation of the generator |
| MoE backbone (114B, 6B active), Head Parallel, fused kernels | none | not at our scale |
| **Muon-type optimizer** (MagiMuon, orthogonalised per head/expert matrix; Adam for small parameters) | training speed | cheap test: Muon vs AdamW on the transformer VAE and the flow, same steps, gate and test scores |
| Deterministic-kernel switch (bit-exact runs) | seed tests | make runs reproducible where cheap (fixed seeds already; deterministic attention optional) |
| Multi-shot identity consistency, reported qualitatively only | none yet | no evidence to use |

### 14.9 What the newest 3D generators teach (survey 2026-11-26)

Sources: Hunyuan3D 2.0/2.1 (arXiv 2501.12202, 2506.15442), LATTICE / Hunyuan3D 3.0 (2512.03052), Hunyuan3D-Omni
(2509.21245), UltraShape 1.0 (2512.21185), TRELLIS.2 (2512.14692), TripoSG (2502.06608), Step1X-3D (2505.07747),
Direct3D-S2 (2505.17412), Seed3D 1.0/2.0 (2510.19944, 2605.13862), SAM 3D Objects (2511.16624), DeepMesh (2503.15265),
online RL for 3D flow models (2610.01233). Loss weights and schedules partly unstated in the sources.

**The paradigm:** a VAE turns shapes into tokens and a flow-matching (rectified-flow) transformer generates them; ours
has the same shape. In 2025 set latents (VecSet: tokens decoded by querying any point) and sparse-voxel latents
competed; by 2026 they merged into set tokens anchored to a coarse voxel grid (LATTICE, UltraShape, Seed3D 2.0), which
our 8 mm patch tokens already are. Nearly all generate coarse first, then refine near surfaces. Standard recipe: token
or resolution curriculum, condition dropout about 10% (guidance), logit-normal timesteps, a last fine-tune on the
cleanest data, guidance and step distillation, and lately reward or preference post-training.

**Decisions, in order of expected value:**
1. **Point-query decoder** for the transformer VAE: any (x, y, z) -> tissue probabilities by cross-attention to nearby
   tokens, trained on points sampled densely near tissue boundaries and jittered inside cells. Smooth surfaces at any
   resolution (the probability-surface view, 14.1), no 8 mm block edges. Replaces the pixel-shuffle head after a gate
   comparison.
2. **Signed distance per tissue** as a target beside the shares (TripoSG: SDF + normal + small eikonal + small KL),
   for sharp fascia and fat boundaries.
3. **Refiner** near boundaries (Seed3D 2.0, UltraShape; 14.8): trained on noised true coarse codes so it tolerates the
   generator's errors.
4. **Cheap changes:** logit-normal timesteps, condition dropout and guidance on the shell, shell as points with normals
   and a type embedding (Hunyuan3D-Omni), token curriculum.
5. **Later, reward post-training** for anatomical plausibility (punishing clearly inhuman predictions, a standing rule)
   once the base model is stable.
Not for us: billion-parameter models, mixture of experts, millions of assets, image conditioning, textures, mesh-token
generation, starting from their weights (large domain gap; Hunyuan's licence excludes the EU, UK and South Korea).

**Speed (user, 2026-11-26):** distillation stays in the plan. An interior is generated once per asset, so generator
speed is not critical; the real-time cost is the physics and motion engine that runs on the generated interior, which
must be fast. Speed work therefore targets the runtime (a compact interior representation, a fast distilled decoder,
the flesh simulation itself); step distillation of the generator stays as an option.

### 14.10 Results of the field experiment (2026-11-26)

**suite1 (conv autoencoders):** the soft autoencoder stalled (validation flat for 10,000 of 12,000 steps) and could not
rebuild eyes or air even from the true input, so N and NS scored 0 on both; A0's hard autoencoder trained. Lesson:
gate every autoencoder on per-tissue reconstruction before training a generator on it (`ae_check.py`).

**Autoencoder gate (4 validation heads, overlap / median surface mm):** the conv autoencoder stalls at random (3 of 5
runs, hard and soft alike). The **patch transformer VAE (D2, 14.7)** trained in 3 of 3 runs and beat the best conv one on
every tissue: eye 0.87-0.89, air 0.79-0.82, thin fat 0.39-0.41 (0.7 mm), fat under the skin 0.78-0.79 (worst 10% 7-9
mm against 14-17), lean 0.95, bone 0.94 (0.3 mm against 0.7). It is now the tokenizer (field_exp/ae_check2/tvae_s1).

**suite2 (transformer VAE; 14 full-face test heads; overlap, two seeds):**

| Tissue | A0 old input (conv) | **N new input** | NS + shell loss | NSP + masked SynthRAD faces |
|---|---|---|---|---|
| Eye | 0.82 | **0.82 / 0.83** | 0.12 / 0.08 | 0.08 / 0.07 |
| Air | 0.05 | **0.69 / 0.69** (0.5 mm) | 0.27 / 0.26 | 0.19 / 0.21 |
| Thin fat | 0.06 (3.4 mm) | **0.28 / 0.28** (1.4 mm) | 0.05 / 0.05 | 0.05 / 0.05 |
| Lean | 0.80 | **0.81 / 0.81** | 0.68 / 0.67 | 0.68 / 0.68 |
| Bone | 0.69 | **0.70 / 0.70** | 0.42 / 0.41 | 0.42 / 0.40 |
| Fat under the skin | **0.58** | 0.56 / 0.56 | 0.28 / 0.27 | 0.29 / 0.26 |

Seeds agree within about 0.01. **Decisions:** (1) the new input (continuous field, tissue + eye) with the transformer
VAE is the base: air and thin fat are many times better, pieces about 40% fewer, eyes equal. (2) The shell loss as
built hurts badly; suspected causes: applied at every noise level (the one-step estimate is blurry early on), and
possibly non-zero even for perfect codes (pulling codes off the decoder's space). Next: apply it only near the end of
the noise schedule and check its value on true codes, as a small test. (3) Masked partial faces changed little next to
NS (air a little worse, plausibly the masked nasal airway); a fair test adds them to N, without the shell loss.

### 14.11 Two goals, two branches (user, 2026-11-27)

**What the views showed.** On a real test head (HaN-Seg case_01, never seen in training) N's generated interior looks
right: skull, orbits and eyes, jaw and airway in place. Faults: the transformer VAE over-calls thin fat about 5.6x (741
against 132 ml; the generator inherits it; likely the rare-class weights in autoencoder training) and eyes and bone
slightly. On the elf the result looked noisy and filled: her shell counts her hair as head (9.0 L against 6.0-7.4 L for
real heads) and her proportions are stylised, a gap in shell types that more CT heads will not close. The earlier F1
elf (named structures) looked more coherent to the user.

**Data bug found and fixed:** a one-voxel band at each scan's field-of-view edge was old-label unknown (255) but not in
the unknown mask, so the continuous field read it as -1000 HU air: about 44 ml of fake air on case_01 (true air 48 ml).
The models learned to draw it, and **N's reported air overlap (0.69) is inflated by it**; air scores must be redone
after retraining with the fix (`field_exp.load_head`: old-label 255 is unknown).

**Decision: two branches.** Medicine needs the detailed interior of one real person; games need the right physics
compartments in plausible places and sizes. Same base (transformer VAE, flow transformer, gate, band fix, continuous
field for fat thickness):
- **Game branch (first).** Coarse only. Targets: about 10 physics compartments taken from the existing named labels
  (as F1 used), boundaries smoothed to about 4 mm: skull, jaw, teeth, eyes, brain, chewing muscles (masseter and
  temporalis, one piece per side), tongue, airway, the fat layer under the skin (as a thickness map), other soft tissue.
  Scored on what physics needs: compartment volumes, centres, one piece per muscle and side, fat-layer thickness, and
  the user's look at the elf; not voxel overlap. Feeds the flesh simulation (rigid bone, muscle volumes with
  attachments, fat thickness, cavities). Shell augmentation (smoothing, scale, proportions) and a hair-free shell for
  stylised characters belong here.
- **Medical branch (later).** The current fine tissue-type model plus the refiner stage (14.8, 14.9), after the decoder
  bias is fixed (volume check in the gate).
No training until the user says so.
