# Interior model: plan version 6.4

> **Read section 12 first.** Version 6.4 (2026-11-22) changes what the model generates: a learned 3D code of the CT volume, not Gaussians, and sets the final goal as one model for the whole body. Sections 2, 3 and step 4a/4b below describe the Gaussian design and are kept as the record of what was tried.

**Status: the working plan from 2026-11-22.** It replaces the learned part of version 5.1 ([interior-model-handout.md](interior-model-handout.md)) and keeps that document's mechanics, clothing and rig sections, which are cited here by section number. Version 6 comes from the eighth review round, which proposed the persistent 3D field and the Gaussian head, and from the T0b trials run the same day. Version 6.1 adds the ninth round: bidirectional creation first, the optional description, coarse-to-fine generation, modality noise and editing (rows 8–12 of section 1). Version 6.2 folds in the tenth round, a review of 6.1 (rows 13–18 of section 1; all six accepted). Nothing in it has been trained.

## 0. The idea in one paragraph

World Labs' Atlas builds one persistent 3D asset (Gaussian splats) from posed views, refines that asset, and exports it to Blender. Every view of it agrees because every view is rendered from the same object. Version 6 does the same for the inside of a body. For each person or character the model produces **one 3D object**, a set of Gaussians in the character's own coordinates. Every CT or MRI slice, axial or oblique, is a **cross-section rendered from that object**. Training supervises only those rendered slices against real ones. Inference fits the object to the character's surface, refines it, freezes it and exports it. The export can be reopened and cut at a plane that nobody asked for during generation.

## 1. What changed from version 5.1

| # | Change | Why |
|---|---|---|
| 1 | The learned object is a persistent 3D Gaussian set per person, not a set of per-slice predictions or thickness fields alone. | Slices from one object agree by construction. Repainting slices one at a time would lose that. |
| 2 | The training target is the slice image: CT intensities (HU), MRI intensities, and TotalSegmentator label maps rendered as extra channels. | No ground-truth Gaussians and no hand-picked boundaries are needed. T0b showed that hand-picked fascia lines carry 3–5 mm of noise per line (section 9). |
| 3 | The Gaussians are **anchored to the layered atlas** (section 3). Free Gaussians are kept as the comparison. | Anchoring gives the same Gaussian *k* in every person, which a diffusion model and the rig both need. Our proposal, not the reviewer's: it is tested, not assumed. |
| 4 | Each MRI scan has its own intensity map (gain, offset, smooth bias field). | MRI intensities are not calibrated across scanners or across positions in one scan. CT HU are. |
| 5 | The experiments are now the seven steps of section 7, thigh first for the method and pelvis for the population. | The thigh is small and fast to test. Only 17 adults have a whole thigh, while 408 have a whole pelvis. |
| 6 | The v5.1 layered state, garment clearance and rig sections stay. They become the anchor frame (section 3), the surface term at inference (section 6) and the conversion to the animation asset (step 7). | Nothing in them is contradicted. They now serve the field instead of being the whole state. |
| 7 | T0b (oracle fit of the layered atlas) becomes the population-average baseline and the atlas-placement tool. | Its placement code is what step 7 and the anchors need. |
| 8 | **Version one is bidirectional and creates one fixed character** from one mesh and an optional description. No autoregressive order over body regions. | The user's input is a single static 3D model. A thigh should use the pelvis and the knee at once; there is no natural "next" region. Agreed with the reviewer and the user. |
| 9 | **The main training task is exterior only**: the scan's own skin surface and the person facts go in, interior slices are the targets. Context slices are an auxiliary task. | It matches what the user will actually supply. Exterior-only results are always reported apart from slice-assisted ones, so that good interpolation between real slices can't hide a weak character workflow. |
| 10 | **Optional description** enters as structured facts with unknown flags (section 5b), now, at no meaningful training cost. | The scans carry no free text, only a few facts. Compute isn't the limit; the facts are. |
| 11 | Two ideas from the papers the reviewer checked become later experiments, each changing one thing: coarse-to-fine Gaussian generation (from Latent Forcing) and independent noise per modality (from Modality Forcing). | They answer different questions from the Gaussian head and shouldn't be mixed into the baseline. |
| 12 | **Editing** (thicker thighs, bigger eyes, match a portrait) is a later step with its own training task, a causal order over *edits* and exact preservation outside the edited region. | Needed for the product, not for the first milestone. |
| 13 | **The exterior-only benchmark must not see the held-out person's bones.** T0b places the atlas with the CT bone labels, so a "population average" placed that way already knows where the femur is. Three runs on the same held-out people replace it (section 5). The population average is built from training people only. | A leak in 6.1. It would have made the baseline too strong and, through anchor initialisation, could have reached the network. |
| 14 | **Description facts measured from the scan are a separate, labelled run.** A user can state age, sex and a broad build. Exact fat or muscle ranks computed from the held-out CT reveal part of the answer. | The same leak, through the description. |
| 15 | **Version one trains on rendered HU alone.** Labels are used for scoring. Label supervision is a named ablation. With HU-only training, exported tissue names come from the anchors. | Keeps one supervised target type, as the user prefers, and shows what labels add. |
| 16 | **A test that bidirectional context helps**: local exterior patches against the whole exterior, on the same people (step 4c). The checkpoint is chosen on exterior-only validation. | A frozen field passes consistency checks whether or not the model used the rest of the body. |
| 17 | **Diffusion needs a defined set of clean states.** A fixed anchor gives Gaussian $k$ an index, not a unique fitted value: overlapping Gaussians can trade offset, width and intensity and render the same slices. Step 4b now repeats the same fit and compares the parameters. | Decides between diffusion on the Gaussian parameters and diffusion on a compact latent with a shared decoder. |
| 18 | Smaller corrections: $k = 1$ is a representation test, not a held-out test; the reload test also covers the native Gaussian file; the slice-profile width is an assumption recorded in the asset, not a header field; the MRI loss compares against the rendered MRI signal; step 5 is split into CT first, then MRI. | Errors and loose wording in 6.1. |

## 2. The asset (the output contract)

For person or character $p$, the asset is
$$ \mathcal G_p = \{(\mu_i, \Sigma_i, w_i, \ell_i)\}_{i=1}^{N}, \qquad T_p: \text{model frame} \to \text{mesh frame}, $$
with centre $\mu_i$, covariance $\Sigma_i = R_i \operatorname{diag}(s_i)^2 R_i^\top$, intensity coefficient $w_i$ (on the CT scale; MRI is mapped per scan, section 4) and label logits $\ell_i$ over tissue classes. The field is
$$ V(x) = b + \sum_i w_i \, \exp\!\big(-\tfrac12 (x-\mu_i)^\top \Sigma_i^{-1} (x-\mu_i)\big), \qquad L_c(x) = \operatorname{softmax}_c\Big(\sum_i \ell_{ic}\, \phi_i(x)\Big), $$
with $\phi_i$ the $i$-th Gaussian kernel.

**Slices are cross-sections, not projections.** A plane $\Pi$ is read by evaluating $V$ on its grid. A conventional splat renderer composites along camera rays and would give a projection. A thick acquired slice with a Gaussian slice profile of width $\sigma_s$ along the normal $n$ is exact in closed form: replace $\Sigma_i$ by $\Sigma_i + \sigma_s^2 n n^\top$ and scale $w_i$ by the matching factor $\sqrt{\det \Sigma_i / \det(\Sigma_i + \sigma_s^2 n n^\top)}$. The scan header gives the nominal slice thickness and spacing, not $\sigma_s$. The profile shape and the rule from thickness to $\sigma_s$ are an assumption, written into the renderer and the asset's metadata.

**Export.** Three files:
- the native Gaussian set together with the field definition above and $T_p$, so that a second program reproduces any slice exactly;
- a sampled volume (NIfTI with its voxel-to-world affine) for inspection;
- named meshes, which come from the labels at step 7.

A standard visual splat file is not enough on its own, because it leaves the intensity meaning unspecified.

**What the contract promises and what it does not.**

| Promise | Holds |
|---|---|
| All slices describe one body. | By construction, for a frozen $\mathcal G_p$. |
| The interior sits where the character is. | Through $T_p$ and the surface term at inference. Under cloth, only within the garment's bounds (v5.1 section 3.3). |
| The interior is anatomically right. | **Not by construction.** A frozen field can show a consistent femur in the wrong place. Accuracy is scored on held-out real scans only. |

## 3. Atlas-anchored Gaussians

Each Gaussian belongs to a tissue (bone, lean, subcutaneous, dermis) and has a fixed **anchor** in the layered atlas: a template skin point $X_i$ and a layer depth $\lambda_i \in [0,1]$ inside its tissue's column. With the v5.1 layered state $z = (z^B, t^L, t^F, f^A)$,
$$ \mu_i = \gamma_{\tau(i)}\big(X_i, \lambda_i; z\big) + \delta_i, $$
where $\gamma_\tau$ is the point at depth $\lambda_i$ between the bounding surfaces of tissue $\tau$ (v5.1 section 3.1), and $\delta_i$ is a learned residual offset. The anchor set is fixed once on the template. Its count per tissue follows step 4's capacity curve; we don't choose it.

What this buys:
1. **Correspondence.** Gaussian $k$ is the same material point in every person. A diffusion or flow model then works over a fixed-length list, $(\delta_i, \log s_i, R_i, w_i, \ell_i)_{i \le N}$, plus $z$.
2. **Named meshes for free.** The labels are the anchors' tissues, and step 7 doesn't need to fit the atlas into the volume again.
3. **Animation.** Each anchor moves with the layers, and the layers move with the bones (v5.1 section 10). Each Gaussian's covariance co-rotates: $\Sigma_i(t) = R(t)\Sigma_i R(t)^\top$.
4. **The prior stays inspectable.** The layered state is still there, so one can ask why a sleeve changed the femur.

What it may cost: fine structure that the anchor columns don't follow, such as the intermuscular septa, bowel, and vessels. The residual $\delta_i$ and an optional pool of free Gaussians cover these. Step 4 measures the cost directly by comparing anchored and free Gaussians at equal count.

## 4. Intensities and modalities

- **CT.** $V$ is in HU. One shared scale for everyone.
- **MRI.** For scan $s$ with sequence $m$:
$$ I_s(x) = a_s + c_s\, \beta_s(x)\, h_m\big(V(x), L(x)\big), $$
with $h_m$ a learned map from tissue to signal for sequence $m$, a per-scan gain $c_s$ and offset $a_s$, and a smooth bias field $\beta_s$ (low-order spline, fitted per scan).

Mixing CT and MRI in training is then consistent. The anatomy is shared, and every difference in intensity is absorbed by $h_m$ and the scan's own map. People in the MRI set are new subjects, not pairs. Where the same person has both CT and MRI, those pairs test the CT/MRI alignment.

## 5. The model

**Inputs**:
- the exterior: the mesh with material labels; on CT subjects, the skin surface taken from the scan;
- the description: person facts $g$ with known or unknown flags (section 5b);
- zero or more **context planes**, each with its pose (origin, orientation, pixel size, slice profile) and its image. With zero planes this is the main task (exterior only); with planes it is the auxiliary reconstruction task.

**Attention is bidirectional** over the whole character: every region sees every other. Attention decides which evidence a region may use; diffusion (step 5) decides how a plausible interior is sampled given that evidence. The two are independent choices.

**State**: one latent $Z$ per person, from a transformer over surface patches and context-plane patches with positional encoding in the shared frame. **Head**: $Z \mapsto (z, \{\delta_i, s_i, R_i, w_i, \ell_i\})$. Either a deterministic version (step 4) or a conditional diffusion or flow model over the same quantities (step 5).

**Training task (masked planes).** From one scan, draw context planes and withheld target planes. Mix the cases: dense neighbours, sparse or distant planes, and no interior planes at all. Some draws withhold a whole height band, so that copying from the nearest neighbour can't solve every task. The loss is computed only on rendered target planes:
$$ \mathcal L = \sum_{\Pi \in \text{targets}} \rho\big(\hat I|_\Pi - I|_\Pi\big) + \text{(prior terms on } z\text{)}, $$
with $\rho$ a robust loss and $\hat I$ the rendered signal for that plane's modality: $V$ for CT, the mapped signal of section 4 for MRI. **Version one is CT with this intensity loss alone.** TotalSegmentator labels are used only to score bone placement, fat overlap and boundary error; exported tissue names come from the anchors. Adding a label term $\kappa\,\mathrm{CE}(L|_\Pi, \text{labels}|_\Pi)$ is a named ablation, with $\kappa$ set by validation. Split by person before drawing any plane. Record the share of zero-context examples, and choose the checkpoint on exterior-only validation, so the easier slice-assisted examples don't pick the model.

**Baselines.** Where context planes exist: linear interpolation between them. For the exterior-only case, three runs on exactly the same held-out people:

| Run | Known at inference | What it measures |
|---|---|---|
| Surface-fitted population atlas | Exterior only | What a template recovers from body shape. |
| The model | The same exterior only, all optional facts hidden | What the learned model adds for the real workflow. |
| Bone-registered atlas (T0b) | Exterior plus the true CT bone labels | How much placement improves when the bones are given. An upper reference, not a competitor. |

The population average is built from training people only. Held-out bones enter the first two runs only when scoring. The anchors of a held-out person are initialised from the surface fit, never from T0b's bone registration.

## 5b. The optional description

**What the data holds.** The CT set records, per person: age, sex, a pathology class (none, tumour, vascular, trauma, inflammation, unclear) with its body region, and scanner settings. There are no radiology reports or other free text.

**Design.** The description is a short vector of facts, each with a known/unknown flag:
- recorded facts: age, sex;
- facts measured from the scan itself, expressed as a rank within the training people of the same sex: subcutaneous fat thickness (from T0a), labelled muscle volume against bone size, overall size.

During training each fact is hidden at random, so the model works with any subset, including none. At inference an LLM call turns the user's sentence ("a lean woman in her thirties") into those facts and leaves the rest unknown. Scanner settings go to the intensity side (section 4), not to the anatomy.

**Why not a text embedding.** A frozen text encoder would cost almost nothing to train with (one precomputed vector per person). But the only sentences we could write for training are templates of the same few facts, so the encoder could not teach the model any word the facts don't cover. Structured facts say the same thing, can be checked one at a time, and make "unknown" explicit. If a dataset with real reports arrives, a text encoder can be added beside the facts.

**What it can and cannot do.** It can shift the hidden split between muscle and fat, which the exterior doesn't show (v5.1 section 3.1). That is the one place a description adds information the mesh lacks. It cannot yet follow creative instructions such as "make the thighs more muscular"; those are edits (step 8).

**Test.** On held-out people, exterior only, three labelled runs: all facts hidden; the facts a user could state (age, sex, a broad build in a few classes); the exact ranks measured from that person's CT. The third reveals part of the quantity being predicted, so it is an upper reference and is never reported as the product's result. If the second shows no gain over the first, the description is dropped from version one.

## 6. Inference on a character

1. Fit $Z$ (through $z$ and the residuals) to the character's surface:
   - bare skin enters through the surface likelihood (v5.1 section 7.2);
   - covered skin enters through the computed garment clearance (v5.1 section 3.3);
   - the swimsuit base model stands in for the body (base characters are swimsuit generations; outfits are separate generations).
2. Refine against any observation that is available: an artist-approved slice or a supplied contour.
3. Freeze. Sample one $\mathcal G$, or several draws in step 5's model, each kept whole across all queries.
4. Export (section 2) with the original mesh transform preserved.

## 7. Steps

Each step has a pass test. The ordering follows the review: export first, then one person, then across people.

| Step | Build | Pass test | Status |
|---|---|---|---|
| **1. Asset and frame** | One thigh, one reference posture (supine as scanned). Record voxel-to-world and model-to-mesh transforms. Export a real CT crop as NIfTI with the skin mesh. | A separate process reopens both and cuts random oblique planes. The mesh section lies on the volume's skin edge to about a voxel. Two controls must fail: the mesh mirrored about the scanner midline, and the mesh in voxel units instead of millimetres. The native Gaussian file is reopened the same way and evaluated from its written definition. | Running with step 4a. |
| **2. Dataset by person** | Per person: volume, modality or sequence, spacing, field of view, exterior, age and sex where known, label maps, usable region. Split by person. | Counts known per region. Pelvis: 408 adults (T0a). Whole thigh: 17 adults, 29 femurs. | Pelvis and thigh counts done. MRI set fetched; children still to be excluded. |
| **3. Masked-plane loader** | Context and target planes with poses and slice profiles, in the mixes of section 5. | One training example shown end to end, with its planes drawn in 3D. | To do. |
| **4a. One-person Gaussian fit** (no network) | Fit $\mathcal G$ directly to one real thigh CT using only context axial slices (every $k$-th, $k = 1, 2, 4, 8$). Gaussian counts are set as voxels per Gaussian. | On withheld axial and oblique planes, inside the thigh: intensity error, bone-edge and skin-edge distance, fat Dice. Compared with interpolation from the same slices. $k = 1$ uses every acquired slice, so it is a representation and round-trip test (the capacity curve), not a held-out one; the comparison with interpolation uses the withheld native slices at $k = 2, 4, 8$. | Synthetic check passed; real job running. Subject: the only long femur whose 12 T0a planes all passed and that has no contrast (s0783 left). |
| **4b. Anchored vs free, and repeatability** | The same with atlas-anchored Gaussians (section 3), placed by T0b's registration (allowed here: one person, no held-out claim). Then the anchored fit repeated from several seeds on the same thigh. | Equal count, same scores; anchoring is kept if its loss in score is within the spread between people. Across seeds: rendered slices agree, and the fitted parameters are compared after removing equivalent covariance forms, with the groin anchors inspected. Stable parameters → per-person fits can train a diffusion model directly. Stable slices but unstable parameters → diffusion on a compact latent with a shared decoder. | After 4a. |
| **4c. Smallest network** | Encoder → $Z$ → deterministic head. First overfit one person, then train across people (pelvis, 408). | Exterior only, on held-out people: beats the surface-fitted atlas (three-run table, section 5). **Context ablation:** the same proximal-thigh target predicted from local exterior patches only against the whole pelvis and upper-thigh exterior, at matched capacity and budget; femur-centre displacement, skin-to-bone depth and fat thickness in millimetres, paired by person. Then one real slice revealed: gain plotted against distance from that slice. | After 3 and 4b. |
| **5. Generative head (CT), then MRI** | 5a: conditional diffusion or flow on CT, over the Gaussian parameters or a compact latent as 4b decides. 5b: MRI through section 4, after testing on the paired head-and-neck set that one tissue-to-signal map per sequence is enough (two places with the same $V$ and tissue must get the same signal before the scan's own correction). | For one draw, intersecting planes agree. Across draws, the spread in hidden anatomy is measured and calibrated (interval coverage on held-out people). Results reported separately for exterior-only, sparse and dense context. | Later. |
| **6. Fit, refine, export a character** | Section 6 on the swimsuit base, then on the clothed generation. | Reopen the export and cut an unseen plane. On a held-out person, overlay that cut on real CT and score the femur boundary, skin-to-bone depth and fascia. On a character, score surface agreement and consistency. Base against clothed is a repeatability test only (v5.1 T6). | Later. |
| **7. Animation asset** | Named meshes from the anchors' labels. Joint frames, attachments and skin binding as in v5.1 section 10. | Thigh, pelvis and lower leg through a slow hip and knee bend. Bone lengths fixed, attachments on their material points, visible skin continuous (v5.1 T8). | The T-rig audit found what is missing (v5.1 section 10.3). |

**Later experiments and editing** (same columns: build, pass test, status):

| Step | Build | Pass test | Status |
|---|---|---|---|
| **B. Coarse to fine** | Same Gaussian budget and compute as 4c or 5, but the coarse part (the layered state $z$ and the broad Gaussians) is settled before the fine residuals. | Better boundary placement on unseen planes than generating all scales together. Otherwise dropped. | After 5. Runs on CT alone. |
| **C. Noise per modality** | On same-person registered CT and MRI, vary the quality of each stream independently. | Beats the same model with plain modality dropout at making one real observation improve the other modality. | After 5. Needs paired data: the head-and-neck CT/MR set (42 people) is the only pairing we hold. |
| **8. Editing** | A second task on the same head: approved asset + instruction or portrait + editable region → a local update. Training pairs come from known 3D deformations of training volumes. Then short edit sequences that start from the model's own earlier output. | Requested change achieved; no change outside the permitted region after export and reload; earlier edits retained after later ones. | Later. |

**Editing notes, recorded now so the asset format allows for it.**
- A broad Gaussian centred inside the edited region still reaches outside it. Exact preservation needs the update defined on the field with a mask, $V_{k+1}(x) = V_k(x) + M_k(x)\,\Delta V_k(x)$, with the mask saved in the asset, or compact-support primitives.
- With anchored Gaussians, many edits are changes to the layered state in a region. "Thicker thighs" is more lean thickness $t^L$ or more subcutaneous thickness $t^F$ with the femur fixed, which also makes the user say which one they mean.
- "Bigger eyes" has to distinguish a wider eyelid opening from a larger eyeball. A portrait is one camera's evidence and leaves depth open, so the side profile is kept unless profile evidence is given.

**The first milestone needs the pelvis, not the thigh.** The reviewer's milestone is a held-out thigh from its exterior alone against a population average. Only 17 adults have a whole thigh, too few to train across people or to hold any out. The thigh therefore serves steps 1 and 4a (one person, no network). The exterior-only milestone is run on the pelvis (408 adults), with the upper thigh that those scans include.

**The three runs that matter first:**
1. the real-CT round trip (step 1);
2. the one-person reconstruction (step 4a);
3. a held-out person from the exterior alone, then with one real slice revealed, scored on a *different* withheld plane.

The third run is v5.1's T1b in the new form. It tests whether the model uses spatial context the way we want.

## 8. What carries over from version 5.1

| v5.1 section | Role in v6 |
|---|---|
| 3.1 Layered state $z$ | Anchor frame of the Gaussians; prior over bones and thicknesses. |
| 3.2 Tissue displacement $u$, 4 Mechanics | Unchanged. $u = 0$ until indentation (T2). |
| 3.3 Garment clearance, 5 Clothing, 7.2 Surface | The surface term at inference (section 6). |
| 8 Uncertainty | Applies to step 5's draws. The exact Gaussian benchmark stays the first check. |
| 10 Rig | Step 7. |
| 11 T-rig, T0a | Done; results in v5.1 section 11b. |
| 11 T0b, T0c | T0b becomes the registration tool and population-average baseline. T0c is folded into step 4b. |
| 11 T1, T1b | The third first-run of section 7, and step 6. |
| 11 T4, T5 | Step 5 (heads) and section 5 (the shared state with slice supervision is v5.1's paradigm B, now chosen). |

## 9. Results so far

**T0a** (measurement): see v5.1 section 11b. Pelvis soft layers repeat to 0.3–1 mm on 408 adults.

**T0b** (atlas placed into CT by its bones; six-person trials):
- **Atlas fix.** Half of the atlas skin triangles were wound inward. With every face oriented away from its nearest bone, usable skin lines rose from 37% to 73% of the skin vertices.
- **Bones.** The atlas is placed on the CT bone labels to 4.8 mm for the hip bones, 2.8 mm for the sacrum and 3.1 mm for the femurs.
- **Skin.** The layered skin offset holds the CT skin to 0.6 mm when fitted. At random held-out vertices it is off by 1.2–1.8 mm, the same as the spread between neighbouring lines.
- **Fascia** (taken as the first muscle-thick lean run, split by Otsu per person):
  - 4 mm fitted, 5–8 mm held out, against a layer 20–40 mm thick;
  - neighbouring lines already differ by 2.7–5.3 mm per line;
  - so the limit is the line-by-line pick or real detail finer than the atlas mesh, not the layered form;
  - sampling along the CT's own normal instead of the atlas normal did not help.
- **Folds.** Offsetting along normals folds about 20% of triangles at creases (groin, gluteal fold). This happens on the skin as much as on the fascia.

What this means for v6: supervising with images (section 5) instead of picked boundaries removes the fascia pick from the training target. The folding argues for the residual offsets $\delta_i$ in section 3.

**T0b on everyone** (409 people selected, none failed; medians across people, with the 10th to 90th percentile):

| Quantity | Result |
|---|---|
| Bones, CT surface to placed atlas | hip bones 4.7 mm, sacrum 2.8 mm, femurs 3.0 mm |
| Atlas scale against the person | 1.09 (1.03 to 1.16) |
| Skin vertices with a usable line | 74% (54 to 84%) |
| Carried atlas skin to CT skin, before the layer | 10.9 mm (−4.8 to 38.1) |
| Skin, fitted | 0.1 mm (0 to 0.9) |
| Skin, random held-out vertices | 1.5 mm (1.2 to 2.1) |
| Skin, held-out height band | 4.1 mm (3.1 to 5.9) |
| Subcutaneous thickness | 28.9 mm (17.6 to 50.8) |
| Fascia, fitted | 3.2 mm (1.2 to 6.5) |
| Fascia, random held-out vertices | 4.6 mm (3.4 to 7.2) |
| Fascia, held-out height band | 8.2 mm (5.9 to 11.6) |
| Fascia, spread between neighbouring lines | 3.7 mm (2.7 to 5.3) |
| Triangles folded, skin / fascia | 18% / 19% |

The six-person picture holds across the population. The count is 409 where T0a reported 408; the one-person difference in selection has not been traced.

**Steps 1 and 4a, one thigh** (s0783 left; crop 152 × 115 × 315 voxels at 1.5 mm, about 2.6 hours on one A10G):

*Round trip: passed.* The reopened volume and skin mesh agree on random oblique planes to 1.5 mm (one voxel; 95th percentile 2.1 mm). Both controls fail as they must: 18 mm with the mesh mirrored about the scanner midline, 54 mm in voxel units.

*Capacity ($k = 1$, all slices, scored on the same slices):*

| Gaussians | HU error | Bone edge | Skin edge | Fat Dice |
|---|---|---|---|---|
| 13,000 | 64 | 0.47 mm | 0.11 mm | 0.77 |
| 53,000 | 52 | 0.19 mm | 0.08 mm | 0.81 |
| 105,000 | 49 | 0.17 mm | 0.09 mm | 0.82 |

About 50,000 Gaussians hold this thigh's bone and skin edges to a fraction of a voxel. The remaining intensity error is mostly the fine texture of fat and CT noise.

*Sparse context, scored on withheld native axial slices (bone edge / fat Dice):*

| Context | Linear interpolation | Gaussians, 13k | 53k | 105k |
|---|---|---|---|---|
| every 2nd slice | 0.15 mm / 0.78 | 0.53 / 0.77 | 0.84 / 0.80 | 1.6 / 0.77 |
| every 4th | 0.20 / 0.76 | 2.5 / 0.72 | 3.4 / 0.62 | 9.7 / 0.39 |
| every 8th | 0.33 / 0.74 | 11.9 / 0.31 | 9.6 / 0.31 | 10.6 / 0.31 |

Free Gaussians fitted to one person's sparse slices are worse than interpolation everywhere, and worse with more Gaussians: they fit the given slices (training error falls) and leave the gaps between them unconstrained. Nothing in a lone fit says how anatomy continues between slices. That is the prior's job, the network's or the atlas anchors'. The k = 2, 4, 8 rows therefore test the prior, not the capacity. Step 4b reruns them with anchored Gaussians.

*Native asset reload: not yet exact.* The 105,000-Gaussian file, reopened in a separate process and evaluated from its written definition, differs from the in-process render by 14 HU, and its bone edge against CT is 4.7 mm. Training cut each Gaussian at its voxel box, while the file's definition sums every Gaussian to 5σ. Fix: one written truncation rule, used by training and by the reader alike, before step 4b.

## 10. Constraints

- No hand-set decision numbers in the pipeline. Experiment settings (Gaussian-count grid, context spacings $k$, iteration caps) are declared in each job's header and swept, never tuned to a result.
- CT and MRI of real people stay in private storage: no renders leave it and no per-person data is published.
- No model of a real person is used. Base characters are swimsuit generations.
- GPU work runs as Hugging Face Jobs. A pilot costs a few dollars.

## 11. Open questions for the next review round

1. **Capacity.** How many Gaussians a pelvis needs to hold cortical bone, the skin–air edge and the fat–muscle edge at a useful error. Whether mixed primitives (Gaussians plus a signed-distance surface per tissue) beat Gaussians alone at the same budget.
2. **Anchoring.** Whether the layered atlas is a good enough frame, where it fails (inner thigh, groin, bowel), and how many free Gaussians must be added.
3. **Generative model over a Gaussian list.** A diffusion or flow model over about $10^4$–$10^5$ anchored Gaussians: architecture, conditioning on context planes, sampling cost.
4. **MRI intensity model.** Whether $h_m$ plus a per-scan gain, offset and bias field is enough across the sequences we hold, and what to do with sequences that only one site uses.
5. **Thick slices and spacing.** Using the slice-profile closed form in training (section 2), and whether anisotropic CT spacing biases the fitted shapes.
6. **Scoring the exterior-only case.** The right baseline and interval measure when the exterior leaves the muscle–fat split unobserved (v5.1 section 3.1).
7. **Verification** of the cited works (Atlas documentation, Gaussian slice-to-volume work, Latent Forcing, Modality Forcing, Diffusion Forcing) from primary sources. The reviewer could not confirm that the two forcing methods are what keeps Atlas from drifting.
8. **Edit training pairs.** Which synthetic deformations of real volumes are anatomically believable enough to train an editor, and how the mask is stored compactly in the asset.


## 12. Version 6.4: generate a volume code; train for the whole body

### 12.1 What was decided, and on what evidence

| Evidence | Result |
|---|---|
| Thigh round trip (step 1) | Passed: 1.5 mm, one voxel; both planted errors caught. |
| Gaussian capacity (step 4a) | 53k Gaussians hold the thigh: bone edge 0.19 mm, skin edge 0.08 mm, fat Dice 0.81. |
| Gaussians fitted to sparse slices | Far worse than linear interpolation (bone edge 3.4 mm against 0.20 mm at every 4th slice). A container has no prior. |
| Saved Gaussian file | Did not read back exactly: 3.7 HU from cut-off tails, 10 HU from the test's own interpolation. |
| Reviewer (round 11) | Option A (volume code) for the first population run; a named-shell trial and a hybrid as side experiments. |
| **E1a, frozen MAISI autoencoder** (NV-Generate-CT `autoencoder_v1.pt`, no training, 16 times fewer numbers than voxels) | Thigh: bone edge 0.24 mm, skin edge 0.08 mm, outer / inner femur wall 0.18 / 0.30 mm, wall-thickness error 0.08 mm, fat Dice 0.67. Ten pelvises (median): bone edge 0.39 mm, skin edge 0.17 mm, fat Dice 0.92, wall-thickness error 0.05 mm. |

**Decision (user, 2026-11-22): option A.** The model generates the autoencoder's 3D code; no further Gaussian runs. The thigh's low fat Dice is mostly scanner noise crossing the fat threshold inside muscle, seen in the pictures; a fair fat measure (fat thickness or share per region) replaces voxel Dice. The inner-wall score is used on shafts only: it breaks in spongy bone such as the femoral head.

Known limits of MAISI's autoencoder: intensities above 1000 HU are clipped; decoded air near a crop's border is not exactly air; the weights are research-only (NVIDIA Open Model License), so a product needs our own autoencoder, trained on training people only.

### 12.2 The design now

1. **Object.** For a character: a code grid $c$ (4 channels, a quarter of the voxel grid per axis) in the character's frame, plus the frozen decoder $D$. $V = D(c)$ is the CT-like volume; slices at any angle are cuts of $V$.
2. **Condition.** The outside surface as a volume on the code grid (inside/outside distance), the position of the window in the body (section 12.3), and the optional facts of section 5b.
3. **Generator.** Deterministic network first, then conditional diffusion or flow on $c$. Bidirectional attention over the window, with neighbouring windows as context.
4. **Export.** Decode; write NIfTI with the mesh transform; fit the layered atlas into $V$ with the T0b machinery for named meshes and the rig.

Dropped: Gaussians as the generated object, the reload fixes, anchoring and repeat-fit tests (old steps 4a, 4b). Kept: exterior-only main task, the three-run benchmark without leaks, the description test, the context test, sections 4 (MRI), 5b, 6 and 10.

### 12.3 The whole body, not the pelvis only

**Final goal (user): one model trained for the whole body.** The pelvis is only the first proof.

**What the data covers.** 1,226 adults; no scan is a whole body. Counts of adults with the structure whole in the scan (present at all in brackets):

| Region | Whole | (Present) | Official split, train / val / test |
|---|---|---|---|
| Skull | 110 | (472) | 90 / 7 / 13 |
| Neck (C4) | 215 | (250) | 192 / 9 / 14 |
| Chest (T6) | 575 | (732) | 510 / 30 / 35 |
| Abdomen (L3) | 557 | (643) | 483 / 21 / 53 |
| Pelvis (both hip bones and sacrum) | 409 | (612) | 351 / 16 / 42 |
| Shoulder (scapula) | 173 | (809) | 150 / 8 / 15 |
| Upper arm (humerus) | 87 | (614) | 73 / 5 / 9 |
| Thigh (femur) | 75, of which 29 long | (564) | 62 / 5 / 8 |
| Forearm, hand, shin, foot | no labels; rarely in the scans | | |

So this dataset supports head to mid-thigh, with arms and thighs thin and nothing beyond elbow and knee. A whole-body model needs more data for the limbs (section 12.6) or falls back to the atlas there.

**How one model covers the whole body from partial scans.**
- **Train on windows.** Each training example is a fixed-size window of a scan (code grid plus its surface volume). All 1,226 adults contribute whatever they show; nothing requires a whole-body scan.
- **Tell the model where the window is.** Each window carries its position in a body frame. In training the frame comes from the scan's labelled bones; at inference it comes from the character's own skeleton, which the existing pipeline already fits. No hand-set numbers: the frame is landmarks.
- **Generate a body as overlapping windows** that see their neighbours' codes as context, coarse to fine: a low-resolution pass for the whole figure, then windows at full resolution. Seams are tested (12.4, test T4).
- **Split by person, once, for everything:** the dataset's own official split (train 1,080 / val 57 / test 89 adults). The autoencoder, if we train our own, sees training people only.

### 12.4 Tests left before the full training

| Test | Question | Data | Rough cost |
|---|---|---|---|
| **T1 (was E2)** Outside only, pelvis | Does the model beat an atlas fitted to the same surface, on held-out people? Femur position, skin-to-bone depth, fat thickness, paired by person. The decisive test. | 351 / 16 / 42 | about a GPU day |
| **T2 (was E3)** Context | Local surface patches against the whole window and its neighbours. | same | two more runs |
| **T3** One model, many regions | Does training on chest, abdomen and pelvis windows together help or hurt the pelvis result? | all trunk windows | about a GPU day |
| **T4** Seams | Do neighbouring windows agree where they overlap, and does the coarse pass keep long bones straight across windows? | held-out long scans | hours |
| **T5** Export | Atlas layers fitted into decoded volumes: within the T0b noise? Folding at the groin? | held-out pelvises | CPU |
| **T6** Limbs and head | How far do 87 upper arms, 29 long thighs and 110 skulls go; which extra data is needed. | this set, then 12.6 | small |
| **T7** Character surface | CT bodies are supine and real; characters are stylised and stand. Run on the swimsuit base and check plausibility and repeatability. | our characters | small |

Only after T1 passes is larger money spent. If T1 fails, the question becomes whether the exterior carries the information at all, before any change of architecture.

### 12.5 Work before the first training run (T1)

1. Fix the split (the official one). Done from the census.
2. Encode every adult scan once with the frozen encoder, whole scan, so that pelvis and whole-body training read the same codes. Store per scan: code grid, body mask, outside-surface distance volume, body-frame landmarks, facts.
3. A surface-only atlas baseline: place the atlas from the skin surface alone (no bones), population average from training people only.
4. The fair fat measure and the three-run scoring code.
5. The window loader and the deterministic network.

### 12.6 Open: data for limbs and head

To check, with licences, before T6: whole-body CT sets that include legs and arms (for example the autoPET FDG-PET/CT collection), lower-limb CT angiography sets, and the head sets already held (42-person head and neck CT/MR). None verified yet.
