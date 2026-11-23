# Interior model: design v7 (tissue-label target)

**Status, 2026-11-23 (revision b: label sources, licences, limb data, parallel workstreams; see sections 3.1, 4.1, 12).** This is a full rewrite. It replaces the learned part of v6.x ([interior-model-plan-v6.md](interior-model-plan-v6.md)), which stays as the record of the Gaussian and intensity-code experiments. The mechanics, clothing and rig sections of v5.1 ([interior-model-handout.md](interior-model-handout.md)) still apply where cited. Nothing here has been trained beyond the tests in section 9.

**What changed and why.** The model's main target is no longer CT brightness. It is **which tissue occupies each point**: a label volume of bone, muscle, fat, organs and named structures. Brightness within one tissue turned out to be a small and largely unpredictable detail (section 9.4). Predicting it voxel by voxel made a deterministic model blur everything (section 9.3). A classifier that is unsure still puts each boundary at its most likely place, so the generated anatomy stays sharp, and the layers the rig needs (bone, muscle, fascia, fat, skin) are read straight off the labels.

---

## 1. The product

Input: one generated character, as its outside surface in a declared pose, optionally with a short description ("a lean woman in her thirties").

Output: the character's interior, consistent everywhere and saved as one object:
1. a **tissue-label volume** in the character's frame: what lies at every point;
2. a **density per tissue**, so CT-like slices can be drawn at any angle;
3. **rig meshes**: closed surfaces with fixed topology for each bone and main muscle, plus the layer surfaces (skin, fat, fascia), bound to the skeleton the existing pipeline fits.

Final goal: **one model that runs on a whole body**. The pelvis is the first proof only.

## 2. The object and its contract

For character $p$:
$$ \mathcal A_p = \big(\, \ell_p(x),\; \rho_p(k),\; T_p,\; \text{meshes}_p \,\big) $$
- $\ell_p(x) \in \{1..K\}$: the tissue label at point $x$, on a 1.5 mm grid in the model frame. In practice the model stores per-class probabilities and exports the most likely class.
- $\rho_p(k)$: one mean density (HU) per tissue class $k$, plus its uncertainty.
- $T_p$: model frame to mesh frame.
- meshes: fixed-topology templates fitted into $\ell_p$ (section 6).

A CT-like slice on any plane $\Pi$ is $I(x) = \rho_p(\ell_p(x))$, optionally plus a texture term (section 5.4). All slices come from one frozen object, so they agree by construction.

What the contract promises: one body, consistent from every angle; placement on the character through $T_p$ and the surface condition. What it does not promise: anatomical accuracy. That is scored on held-out real people only (section 8).

## 3. Tissue classes

Built per training scan from TotalSegmentator's labels (117 structures) plus brightness for the unlabelled soft tissue:

| Class group | Source | Notes |
|---|---|---|
| Air outside the body | body mask | |
| Subcutaneous fat | brightness −190 to −30 HU, between skin and the first muscle | Needed for the rig's fat layer. No public label set has it. |
| Other fat (visceral, intermuscular) | brightness in the fat range, deeper | |
| Named bones | TotalSegmentator: hip bones, sacrum, femurs, vertebrae C1–L5, ribs, humeri, scapulae, clavicles, skull, sternum | Kept separate: the rig needs each bone. |
| Named muscles | gluteus maximus, medius, minimus; iliopsoas; autochthon (back); others where labelled | |
| Other lean tissue | brightness −29 to 150 HU, not labelled | Unlabelled muscle, skin, connective tissue. |
| Organs and bowel | TotalSegmentator organ labels, grouped | Grouped to a few classes. Not needed for the rig, but they occupy space. |
| Gas in bowel | brightness below −500 inside the body | |

The **fascia** is then the boundary between the subcutaneous-fat class and the first lean class under it. It is read from the labels, not picked on each ray, which removes the 3–5 mm per-line noise of v6 (T0b).

**Existing solution.** NVIDIA's MAISI includes an anatomy-mask autoencoder and a mask-generation diffusion model over 132 classes: the body, all vertebrae and ribs, hip bones, sacrum, femurs, humeri, scapulae, and gluteal, iliopsoas and back muscles. It has no fat class and no forearm or lower-leg bones, and its weights are research-only. It is a candidate starting point for section 5.2, to be tested against our own small model.

### 3.1 Label sources, in versions

| Version | Source | Covers | Status |
|---|---|---|---|
| labels v1 | TotalSegmentator's 117 structure masks; brightness (3×3×3 median) for the rest: subcutaneous fat = fat connected to the outside through fat only; other fat; other lean; gas; dense unlabelled | all 1,226 adult CTs, on the crop and grid of the saved codes | running (4 CPU jobs, about $0.05) |
| tissue v1 | TotalSegmentator's licensed `tissue_4_types` model: subcutaneous fat, torso fat, skeletal muscle, intermuscular fat | the 409 whole-pelvis adults first | smoke running on an L4 card |
| head MRI | GRACE (11 head tissues from T1 MRI: skin, fat, muscle, spongy and cortical bone, air, eyes, brain, CSF, vessels) | IXI and HaN-Seg MRIs | H3b running: checked against the same person's CT |
| licensed extras | TotalSegmentator `appendicular_bones`, `thigh_shoulder_muscles`, head tasks | limbs and head, where scans reach | not started |

A2 runs on labels v1 and is then repeated on tissue v1, which also measures how much the label source matters.

**Licences of labels (decided 2026-11-23: non-commercial academic project).** TotalSegmentator's licensed tasks (academic key in the user's vault) and GRACE's weights (non-commercial research only; no derivative models for deployment) make every model trained on their labels a **research model**. A commercial product later needs its own labelling. TotalSegmentator's open structures and our brightness classes carry no such restriction.

## 4. Data

**TotalSegmentator CT v2.0.1:** 1,228 research CT scans (1,226 adults), CC BY 4.0, official split train 1,080 / validation 57 / test 89. Held in a private Hugging Face repo.

Body coverage in adults (structure whole in the scan):

| Region | Adults | Region | Adults |
|---|---|---|---|
| Chest | 575 | Shoulder | 173 |
| Abdomen | 557 | Skull | 110 |
| Pelvis | 409 | Upper arm | 87 |
| Neck | 215 | Thigh | 75 (29 long) |
| Forearm, hand, shin, foot | none | | |

No scan is a whole body. Head to mid-thigh is covered by overlapping regional scans. Arms, thighs and the skull are thin. Below the elbow and the knee there is nothing (section 10.1).

**Already prepared** (`codes_v1/` in the private repo, all 1,226 adults): per scan the body mask, the outside-surface distance on a 6 mm grid, body-frame numbers measured from the surface, bone landmarks (for scoring only, including femoral-head centres), age, sex, split, and MAISI brightness codes (kept for the optional texture term). **To prepare next:** the grouped label volumes of section 3 at 1.5 mm and 3 mm, with a CPU job (cents).

### 4.1 Limb and whole-body data (desk research, 2026-11-23; not downloaded)

| Dataset | Coverage | People | Labels | Licence and access | Notes |
|---|---|---|---|---|---|
| VSDFullBody (Zenodo) | whole-body post-mortem CT (arm coverage unverified) | about 109 entries (true count unverified) | none | CC BY-NC-SA 3.0 CH, open, 116 GB | **deceased** (forensic institutes) |
| VSDFullBodyBoneReconstruction | lower body incl. shin and foot bones | about 20 adults | bone masks and meshes | CC BY-NC-SA 4.0, open | deceased |
| BoneHub vsd-feet-seg | feet and ankles | 30 CTs | 56 bones | CC BY-NC-SA 4.0 | deceased |
| NMDID | head-to-toe CT, 1 mm | 15,249 | none | data use agreement **bans derivative works**; institutional application; over 50 TB | deceased, no consent; 1,171 under 18 |
| Visible Human male and female | CT, MRI and cryosection, head to toe | 2 | third-party bone masks | NLM terms | deceased donors; reference only |
| SAROS | 299 whole-body CTs (mostly cut at the upper legs) of 900 | 900 | every 5th slice: fat, muscle, bone; body parts | labels CC BY 4.0; source scans mixed | |
| CADS | head to knee | 22,022 | 167 structures | mixed licences; gated; 1.9 TB | |
| autoPET | mostly skull base to mid-thigh | 900 | lesions only | CC BY 4.0 | cancer patients |

**Reading.** No open whole-body MRI with fat and muscle labels was found. The only open sources for shins, feet, forearms and hands are post-mortem CT (VSD sets) and two Visible Human bodies; NMDID has the scale but bans derivative works, and its terms must be clarified in writing before any use. **Deceased subjects are a new consent category** for this project and need the user's decision (section 12, decision D2). Licensed `appendicular_bones` can label limb bones in any of these scans.

**Privacy and licences.** The CT data stay in private storage, and only aggregate numbers and the odd illustrative slice (CC BY, attributed) are shown. MAISI weights are research-only, so a product needs our own networks, trained on training people only.

## 5. The model

### 5.1 Inputs (conditions)
- **The outside surface** as a signed-distance volume (inside positive).
- **Position in the body:** each location's coordinates in a body frame measured from the surface (body centre, widths and depths per height, landmarks of the outline). Training scans and characters are aligned this way without resampling. At inference the frame comes from the character's fitted skeleton and outline.
- **Facts,** each with a known/unknown flag: age, sex, a broad build class. Scanner settings (tube voltage, contrast) are given in training only, so the model does not mistake a scanner difference for a person difference.

### 5.2 Label representation
Two options, compared in test A3 (section 8):
- **Direct:** class probabilities on a 3 mm grid, with an upsampling head to 1.5 mm.
- **Latent:** a label autoencoder compresses one-hot labels into a code grid (as MAISI's mask autoencoder does), and the generator works on codes. Smaller tokens make a whole body feasible.

### 5.3 Generator
- **Architecture:** a bidirectional transformer over tokens (3D patches of the label grid or label codes), conditioned on the surface, body-frame position and facts.
- **Training by masking, per person:** each example is one person's scan. A random share of interior tokens is hidden, from some to all, and predicted from the surface plus that person's unhidden tokens. With all hidden, the task is the product case (surface only). Checkpoints are chosen on that case.
- **Self-supervised pretraining** on all 1,080 training scans, with no extra labels needed beyond those built in section 4.
- **Output head:** first per-token class probabilities with cross-entropy, deterministic (stage A2). Then a generative head (diffusion or flow on label codes, each token with its own noise level), so different draws are different plausible bodies and each draw is sharp.
- **Whole body from regional scans:** the model is trained on whatever span each scan shows, with positions in the body frame. A character is generated trunk first, then extended region by region, each step conditioned on what is already generated. Training with noisy context (diffusion forcing) guards against drift. Seams are tested (T4).
- **Size:** start at 30–60 million parameters. Larger would overfit at this data size.

### 5.4 Density head
- **Per tissue class:** mean HU and its spread, predicted from facts, scanner and outline features. Section 9.4 gives what is learnable: bone about 12%, muscle about 33%, subcutaneous fat about 53% of the variation between people. The rest is reported as uncertainty, not guessed.
- **Optional texture:** for realistic-looking slices only, a decoder adds within-tissue texture (the saved MAISI codes can serve here). It is not part of the anatomy and never scored as anatomy.

## 6. From labels to the rig

1. **Layers:** skin = outer boundary of the body; subcutaneous fat; fascia = fat–lean boundary; lean compartment; bone surfaces. All are read from the label volume, so they are consistent with each other.
2. **Fixed-topology meshes:** for each bone and main muscle, the atlas's closed template mesh is deformed to fit its generated label region (the T0b registration machinery: similarity, then affine, then a smooth non-rigid step). Topology is guaranteed by the template, not invented by the network.
3. **Checks:** template folding and inversion (v6 saw 18–19% folds when offsetting along normals; fitting to labels should avoid that, and test T5 measures it), surface distance to the label boundary, and layer order.
4. **Rig:** joint frames, muscle attachments and skin binding as in v5.1 section 10. The anatomy is frozen; motion clips drive the bones.

## 7. Infrastructure rules

- **Every GPU job:** first a smoke run on a small card with a few people (it has caught 5 bugs so far). Then a cost and speed table over card types, read live from the Hub. The most efficient card is picked, which is not always the biggest: the encoding run was CPU-bound, so the 23-core RTX PRO 6000 beat the A100.
- **CPU-heavy preparation runs on CPU jobs** ($0.03 an hour); GPUs only train.
- **Containers report the host's cores and RAM;** jobs read their own cgroup limits.
- **At most one upload per job** (the Hub allows 256 commits an hour).
- **Split by person before anything is trained,** autoencoders included.

## 8. Tests, in order

| Test | Question | Data | Rough cost |
|---|---|---|---|
| **A2** Labels from the surface, pelvis, deterministic | Does a classifier from surface to labels beat the baselines on held-out people? | pelvis 351 / 16 / 42 (official split) | $2–4 |
| **A3** Direct or latent labels | Does a label autoencoder keep thin structures (cortex, fascia, small muscles)? MAISI's mask autoencoder against our own. | pelvis | $2–5 |
| **T1** Masked transformer, surface only, pelvis | The decisive test, with the full three-run benchmark below. | pelvis | about $10–20 |
| **T2** Context | Local surface patches against the whole surface. Revealing the chest's interior: how much does it help the pelvis? This answers whether regional scans are enough for a whole body. | chest-to-pelvis scans | small |
| **T3** Many regions in one model | Does training on chest, abdomen and pelvis together help the pelvis? | all trunk scans | GPU hours |
| **T4** Seams | Do neighbouring regions agree; do long bones stay continuous? | long scans | small |
| **T5** Rig export | Templates fitted into generated labels: distance, folds, layer order. | held-out people | CPU |
| **T6** Limbs and head | How far thin regions go; which data is needed. | this set, then 10.1 | small |
| **T7** Characters | Supine real CT against stylised standing characters: plausibility and repeatability on the swimsuit base. | our characters | small |
| **D** Generative head | Diffusion on labels: sharp, varied, calibrated draws. | trunk | GPU hours |

**Benchmark** (every test from A2 on), on the same held-out people:

| Run | Known at inference | Role |
|---|---|---|
| Surface-fitted atlas | outside only | the baseline to beat |
| Nearest training body (by outline) | outside only | the strong simple baseline |
| The model | outside only, facts hidden | the product |
| The model with facts | outside plus age and sex | does the description help |
| Bone-registered atlas | outside plus true bones | upper reference, not a competitor |

**Scores:**
- per-structure Dice and surface distance (mm);
- landmark distances: femoral-head centres, sacrum top, iliac crests, vertebral bodies;
- fat thickness and fascia depth along the skin (mm);
- per-tissue density error against the predictable floor (9.4).

## 9. What has been measured so far

### 9.1 Atlas placed into 409 pelvises (T0b)
- **Bones:** hips 4.7 mm, sacrum 2.8 mm, femurs 3.0 mm.
- **Skin layer:** fitted 0.1 mm; held-out 1.5 mm.
- **Fascia picked on each ray:** 3.2 mm fitted, 4.6–8.2 mm held out. Neighbouring rays already differ by 3.7 mm, which is why v7 reads the fascia from labels.
- **Folding:** 18–19% of triangles fold when the surface is offset along normals.

### 9.2 Gaussian representation (v6, dropped)
- **Holds a thigh:** 53k Gaussians reach 0.19 mm bone edge.
- **Fails between sparse slices:** worse than interpolation, because a container has no prior.
- **Saved file:** did not reload exactly.

### 9.3 Brightness code from the surface (stage A, dropped as the main target)
- **Autoencoder holds anatomy:** MAISI frozen, bone edge 0.24–0.39 mm, fat Dice 0.92 in the pelvis.
- **Deterministic surface-to-code model:** brightness better than the nearest body for 34 of 42 test people, and better than the average for all 42.
- **But bones blurred:** bone Dice 0.03, the averaging failure that motivates v7.
- **Cost:** 80 minutes on an A100, about $5.

### 9.4 Density within a tissue (all 1,226 adults, CPU, about $0.01)

| Tissue | Spread between people | Explained by age and sex | by outline | by all, with scanner |
|---|---|---|---|---|
| L1 spongy bone (154 HU) | ±68 | 11% | 0% | 12% |
| Muscle (38 HU) | ±19 | 14% | 15% | 33% |
| Subcutaneous fat (−100 HU) | ±12 | 0% | 42% | 53% |

Within-tissue differences (±8 to ±64 HU unexplained) are small next to the differences between tissues (fat −100, muscle +40, bone +150 to over +1000). So tissue type and place carry the anatomy; density is a per-tissue detail.

### 9.5 Infrastructure
- **Encoding:** all 1,226 scans in 4 parallel RTX PRO 6000 jobs, 7.6–7.9 minutes each, about $1.40.
- **Failures found along the way:** two out-of-memory crashes. All labels were held at once, and the containers report the host's 192 cores. One padding bug: the autoencoder needs sides that are multiples of 16.

## 9b. The head

The face is where characters are judged, and the rig needs the jaw, the eyes and the thin facial soft tissue. The head therefore gets its own data plan.

### 9b.1 Data held (all in private repositories)

| Set | People | Modality | Head coverage | Head labels | Licence |
|---|---|---|---|---|---|
| TotalSegmentator CT v2.0.1 | skull whole in 110 adults (90 / 7 / 13 by split), present in 472; brain whole in 156; C1–C2 in about 215 | CT | whole skull in 110; often with neck (75 reach C4) | skull (whole, incl. jaw), brain, C1, C2, carotids, thyroid, trachea, oesophagus; **no face soft-tissue, eye or muscle labels** | CC BY 4.0 |
| HaN-Seg | 42 adults (36–79) | **CT and T1 MRI of the same person** | face in view in all 42 (130 mm); top of the head never; facial soft tissue in front of bone: median 12 mm (10–17) | 30 expert structures: mandible, eyes (anterior and posterior), lips, oral cavity, buccal mucosa, parotid and submandibular glands, optic nerves, brainstem, larynx… | **CC BY-ND 4.0** (no derivatives) |
| IXI | 581 T1 heads, adults 20–86, with height and weight | MRI, about 1 mm | whole head; face clear of the edge in 489 | none | CC BY-SA 3.0 |
| TotalSegmentator MRI v3.0.0 | 1,296 scans, mostly body; many children; 276 without age | MRI, mixed sequences, often 3–6 mm slices | brain in some | 50 structures, mostly body | not yet verified |

### 9b.2 How the head enters the model
- **Same target, same model.** Head scans contribute tissue labels exactly like body scans. With labels as the target, MRI needs no intensity calibration: fat is "fat" in either modality. The density head learns from CT only.
- **CT gives bone,** which is nearly black on MRI and therefore indistinguishable from air there. **MRI gives soft tissue in bulk,** with 581 IXI heads against 110 CT skulls.
- **Labels for MRI heads** (no open model covers the face):
  - fat and lean from T1 brightness, after bias-field correction, split by a two-class mixture fitted per scan;
  - the brain from TotalSegmentator's open MRI model (Apache-2.0);
  - bone and air together as one "dark" class, with the skull's shape supplied by CT heads through the shared model;
  - eyes, jaw and the small facial structures from HaN-Seg's expert labels (42 people).
- **Validation:** on HaN-Seg's 42 people, MRI-derived labels are compared with CT of the same person after registration (test H3). This decides how far MRI labels can be trusted, per class.
- **Position in the body:** head tokens get body-frame positions like all others; the neck scans (C1–C4) link the head to the trunk.

### 9b.3 Exclusions, licences, privacy
- **Minors excluded.** People under 18, and TotalSegmentator MRI scans without a recorded age (276), are excluded. IXI and HaN-Seg are all adults.
- **HaN-Seg is no-derivatives.** Private training is the intended use; nothing derived from it (labels, registered volumes) is published.
- **IXI is share-alike.** Anything derived from it that we share carries the same licence.
- **Faces.** Head scans can reconstruct a real person's face. They stay in private storage, and no face rendering leaves a private job (consent rule of 2026-11-22).
- **Licensed segmenters.** TotalSegmentator's face, fat and limb MRI models require a licence (free for academic, non-commercial use). They are not used without a decision on that licence.

### 9b.4 First result: simple MRI labels do not match CT (H3 smoke, 4 people, 2026-11-23)

Each person's T1 MRI was rigidly registered to their CT (mutual information, two starting alignments, best kept) and compared outside the skull. Inside the skull the brain dominates MRI brightness, so the comparison was restricted to the face, scalp and neck. MRI classes came from bias correction plus a three-class split of brightness; CT classes from HU.

| Measure | Result (4 people) |
|---|---|
| Registration, body overlap | 0.92–0.98 in 3, failed in 1 (0.37) |
| Fat overlap, CT vs MRI | 0.06–0.26 |
| Lean overlap | 0.53–0.69 |
| Dark (bone + air) overlap | 0.40–0.47 |
| MRI "dark" share against CT bone + air share | 29–41% against 11–12% |
| Facial soft-tissue depth, CT / MRI | 13–19 mm / 5–10 mm |

**Conclusion:** brightness rules do not turn T1 MRI into tissue labels that agree with CT. Likely causes: fatty bone marrow and glands are bright on T1 but bone or gland on CT; signal falls off towards the coil's edge; partial volume in thicker MRI slices. The full 42-person run was not made: the method itself fails.

**Revised head plan:**
1. **CT first for the head:** 110 whole and 362 partial skulls (TotalSegmentator), plus HaN-Seg's 42 CTs, which carry the expert face labels (eyes, jaw, lips, oral cavity, glands) on the CT grid. CT fat and lean come from HU, which works.
2. **MRI labels need a learned mapping,** not rules. Options:
   - train a small MRI-to-label network on HaN-Seg's 42 registered pairs, with CT-derived labels as targets (few people; checked by leave-out);
   - TotalSegmentator's licensed MRI tissue model (free for academic, non-commercial use; a licence decision);
   - leave MRI out until the CT head model shows a need.
3. **Registration** needs a failure check that does not rely on a fixed number. Candidate: agreement of the two body outlines compared with the spread over all pairs. One of four failed here.

### 9b.5 Head tests

| Test | Question | Cost |
|---|---|---|
| **H1** CT head census | Done: the table above. | free |
| **H2** MRI head labels on IXI | Only after H3 shows a mapping that agrees with CT. | later |
| **H3** MRI against CT on HaN-Seg | Smoke on 4 people done (9b.4): brightness rules fail. Next version: a learned MRI-to-label mapping, scored the same way. | CPU or small GPU, cents to $1 |
| **H4** Head in the model | Head tokens from CT and MRI in the shared model; scored on held-out CT heads (skull, brain, facial fat depth) and HaN-Seg structures. | after T1 |

## 10. Open problems

1. **Limb and head data.** Nothing below the elbow or knee in the main set; few arms, thighs and skulls. Section 4.1 lists the candidates; all open limb sources are post-mortem CT (decision D2). Until then the atlas fills these parts.
2. **Licences.** MAISI weights are research-only. The product path needs our own label autoencoder and generator.
3. **Domain gap.** CT people lie supine and are real; characters stand and are stylised. Posture (v5.1 T7) and stylisation policy (v5.1 13.2) remain open.
4. **The muscle–fat split** is not visible from outside. The model can only learn it from the population and the description (9.4). It is reported with its uncertainty.
5. **Editing** (thicker thighs, bigger eyes, matching a portrait) comes after T1. With labels, an edit is a region and class change, which makes "more muscle or more fat?" an explicit choice.

## 11. Questions for review

0. (Revision b) Labels from licensed or research-only segmenters (TotalSegmentator tissue tasks, GRACE) as training targets: how should label noise from a model, rather than from experts, be handled in the loss and in scoring?

1. Direct 3 mm label prediction or a label autoencoder for a whole body: which keeps cortex and fascia at a token count a transformer can handle?
2. Masked-token generation with a per-token diffusion head over discrete labels: diffusion on label codes, or discrete masked generation (one class per voxel, iterative unmasking)?
3. How should the body-frame position be defined from the outline alone, so that it transfers from supine CT to standing characters?
4. Are 1,080 regional scans enough for a whole-trunk model, given T2's answer?
5. Template fitting into generated labels against generating shells directly: when would the second be worth building?


## 12. Workstreams: what blocks what, and what runs in parallel (2026-11-23)

| Workstream | Depends on | Status | Cost so far / next |
|---|---|---|---|
| W1 Labels v1, all CT (brightness + structures) | codes_v1 crops | running, 4 CPU jobs | about $0.05 |
| W2 Licensed tissue labels, 409 pelvises | licence key | smoke running (L4) | next: card choice from the smoke's seconds per scan |
| W3 **A2**: surface to labels, pelvis, deterministic | W1 | blocked on W1 only | about $2–4 (smoke, then the card picked by the cost table) |
| W4 A2 on tissue labels (label-source comparison) | W2, W3 | after both | small |
| W5 **A3**: label autoencoder capacity (MAISI mask autoencoder against our own) | W1 | can start as soon as W1 lands, parallel to W3 | small GPU |
| W6 Head: H3b GRACE against CT, 42 people | none | running (L4, about $0.70) | |
| W7 Head CT labels: HaN-Seg's 42 CTs (expert face labels) plus TotalSegmentator's 110 skulls, with licensed head tasks | licence key | not blocked | CPU plus small GPU |
| W8 Limb data | decision D2 | desk research done (4.1) | none until D2 |
| W9 Review of this plan and the training-target proposal | none | with the user | — |
| W10 Masked transformer T1 | W3 passing | after A2 | GPU hours |

**Not blocked and worth running in parallel now:** W5 (once W1 lands), W7, and the review W9. **Blocked:** W3 and W5 on W1 (about 25 minutes), W4 on W2, W8 on decision D2, W10 on A2.

**Decisions waiting for the user:**
- **D1 (taken):** non-commercial academic project; licensed TotalSegmentator tasks and GRACE allowed; models trained on them are research models.
- **D2 (open):** may post-mortem CT of deceased people (VSD sets, Visible Human; possibly NMDID after its terms are clarified) be used for limbs?
- **D3 (open):** MRI heads via GRACE labels, if H3b shows agreement with CT.
