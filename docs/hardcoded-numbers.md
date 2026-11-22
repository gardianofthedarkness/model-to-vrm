# Hand-set numbers still in the pipeline

The rule of this project is that no decision value is a number someone picked by hand: every value should come from the
model's own data, a learned or statistical model, an LLM or the Jev decision API, or an optimisation with data-derived
scales. **The pipeline does not meet that rule yet.** This file is the open list: an audit of 41 pipeline scripts
(2026-11-22) with every hand-set number found, what it does, and what should replace it. It is kept here so that nobody
has to take "no hard-coded numbers" on trust.

## Done since the audit

- `anatomy/muscle_inside.py`: the 2 mm skin thickness is replaced by the CT study's own thinnest measured tissue
  (`tissue_depth_ct/summary.json`, 5th percentile at the thinnest site), scaled to the figure.
- Sealing loops and penalty loops (`solids.py`, `body_assembly.py`, `part_bone_fit.py`, `skull_line_fit.py`,
  `muscle_carry.py`, `muscle_depth.py`, `muscle_inside.py`) now print a WARNING when they run out of tries instead of
  silently using the last value. The schedules themselves (H1, H7 below) are still hand-set.
- `anatomy/skull_line_fit.py`, `anatomy/skull_stage2.py`: which face lines are used, and which run over cartilage, now
  comes from the naming step's answer (`agents/landmark_lines.py`, read by `anatomy/face_lines.py`) instead of a list
  written in the code.
- Jev and SAM 3 model names are in `model_to_vrm/common/models.py` for the new steps (`parts/loose_names.py`,
  `face/philtrum_area.py`); older scripts still name their models inline.

## Order of work

1. `skull_stage2.py` weights and control points (H2, H3): the philtrum trouble came from here.
2. The "nearest 1%" rule (H5) and the penalty schedules (H1).
3. The muscle depth weight search (H6) and the sealing loops (H7).
4. Grid and sampling sizes, clustering ranges, word lists and name patterns.

Each change alters results and is checked with its own rerun before the next.

---

## The audit as written


Read-only audit of the 41 listed files (all read in full). Nothing was run or modified.
Line numbers are the file's own. "Matters" = how much the value can change the result.

Legend: V = VIOLATION (tuned, no data source), C = CITED, ? = unsure (explained).

---

## 1. HIGH importance violations

### H1. Penalty schedules "halve / double until inside" (4 files, same pattern)
| file:line | fragment | what it does |
|---|---|---|
| anatomy/part_bone_fit.py:138, 147 | `for outer in range(16)` ... `sig_in /= 2` | inside-penalty scale halved per round, at most 16 rounds (2^16) |
| anatomy/part_bone_fit.py:146 | `if worst < hstep and outer > 0` | forces at least two rounds |
| anatomy/skull_line_fit.py:108, 117 | `for outer in range(16)` ... `sig_in /= 2` | same |
| anatomy/body_assembly.py:350, 357 | `for outer in range(16)` ... `sig_in /= 2` | same (also 354 `outer > 0`) |
| anatomy/muscle_inside.py:45, 46, 51 | `w = 1.0` ... `range(40)` ... `w *= 2` | hold-weight starts at 1 (Laplacian units) and doubles, at most 40 rounds |

The factor 2 is a schedule, not a definition, and the cap (16 / 40) is reachable: when it is reached the last iterate is used silently. The stopping tolerance (one grid cell) is data-derived and fine.
Replace with: (a) a true hard constraint as skull_stage2 already does (cvxpy: `phi + grad . step >= 0`), which removes the penalty and its schedule; or (b) a 1-D root find on log(sig_in) for `worst(sig_in) = hstep` (scipy `brentq`), which has no step factor and ends by its own bracket test; raise if not reached rather than keeping the last round. For muscle_inside the starting weight should be the Laplacian's own scale (e.g. median diagonal of `LtL`) instead of 1.0.

### H2. skull_stage2.py - the convex problem's hand weights
| line | fragment | what it does | suggestion |
|---|---|---|---|
| 324 | `(cp.sum(sl) + cp.sum(s2)) / (hst / s0) * 1e3` | weight of the slack on the inside rule and the CT hard bounds | make both truly hard (no slack) with the step bound below handling infeasibility; or exact penalty: weight = largest dual value of the hard-constrained solve (Clarabel returns duals) - data-derived |
| 324 | `cp.abs(SHAPE @ zv) <= 3 * SIG_S * SCL[...]` | stretch cap "three spreads" (docstring 23-24) | the CT skull set that gave SIG_S also gives the observed extreme: use the largest proportion deviation actually seen among the 388 skulls (store it in skull_shape_spread.json) |
| 302 | `range(-1, 8)` | at most 8 loosening stages = loosening capped at 2^8 (docstring line 44 mentions 256) - this limits the result | loop already ends by `not lo.any() and not newly.any()` (341); make the cap a pure safety limit or drop it |
| 335, 342 | `SCL[newly] /= 2`, `SCL[lo] *= 2` | loosening factor per stage | `unmet()` already returns how far a node is from its standard in units of that standard: scale by `u` (the measured shortfall) instead of 2 |
| 322 | `BH / (SIG_L * 2)` | step damping at twice the line scale (docstring says "at the line scale") | trust radius = one grid cell `hst` (where the SDF linearisation holds), or SIG_L itself |
| 319 | `act = ph < 3 * SIG_L * s0` | which hull points get an inside constraint | all hull points, or those within the round's own step bound |
| 328, 304 | `if move < hst / 10`, `range(12)` | convergence "a tenth of a grid cell", at most 12 rounds (cap can bind) | `hst` (the grid's resolution, as every other file uses) or the detector noise `NOISE * s0`; cap as safety only |
| 314, 315 | `S_['tau'] - S_['sig']`, `-(S_['tau'] + S_['sig'])` | hard depth bounds at median -/+ ONE spread | CT quantiles per site (the summary already carries a p5: `thinnest_p5_mm`); store p5/p95 per site |
| 251 | `.get('thinnest_p5_mm', 0)` | 5th percentile as the minimum tissue under bare skin; silent 0 fallback | percentile choice lives in the CT stats file - mark it there as chosen, or use the CT minimum; fail if missing |

### H3. skull_stage2.py - deformation-graph construction
| line | fragment | what it does | suggestion |
|---|---|---|---|
| 114 | `k = min(4, len(nd))` | each bone point moves with its 4 nearest nodes | all nodes of the bone with harmonic weights on the bone mesh (muscle_carry.py already solves exactly this, "no falloff number") |
| 117 | `1.5 * dk[:, -1:]` | falloff radius when the bone has no (k+1)th node | disappears with harmonic weights |
| 139 | `np.argsort(d, 1)[:, 1:min(4, len(nd))]` | 3 nearest nodes form the stiff edges | Delaunay / mesh-geodesic neighbours of the nodes |
| 80 | `max(1, int(round((ext[b] / d0) ** 2)))` | node count per bone = (extent / median extent)^2 | ? rule is self-relative but the exponent and the "median bone = 1 node" are choices; alternative: node spacing = the landmark spacing `sp0_` (97) or the CT site spacing |
| 88 | `abs(gi[0]) < sp / 4` | node snapped to the midline within a quarter node spacing | snap when nearer the plane than its mirror partner would be to itself: `abs(x) < mesh edge` |
| 182 | `min(len(P), 40)` | 40 contact points sampled per seam - sets each seam's weight in the objective | use all contact points and divide the block by sqrt(n) so a seam's weight does not depend on a count |
| 188 | `np.sort(dd)[max(1, len(dd) // 100)]` | jaw hinge = nearest 1% of mandible points (comment says "tenth") | Otsu split of log distances, as line 176 does for seams |
| 229 | `k=min(4000, ...)`, `distance_upper_bound=wid` | socket bones = within ONE eye width, at most 4000 points | `query_ball_point(eA, wid)` (no count cap); the "one eye width" rule is a self-relative choice (?) |

### H4. anatomy/muscle_inside.py:29 - `SKIN_ATLAS_M = 0.002`
Skin thickness "about 2 mm (1 to 4 mm by place)" stated in a comment with no source. Sets the repair margin for every muscle and bone. Either cite a source next to the constant, or take it from data already in the pipeline: the CT summary (`thinnest_p5_mm`), or the atlas's own skin-to-outermost-muscle depth (rho in muscle_depth.py).

### H5. "nearest 1 %, at least 3" contact rule - `max(3, len(P) // 100)`
part_bone_fit.py:60, 61, 73; body_assembly.py:97, 221; toe_lines.py:54; agents/point_names.py:70, 74, 76 (and skull_stage2.py:188 above).
Defines every joint middle, every free bone end, the knee's two sides and every toe tip - i.e. the targets of the bone fits. Replace with the Otsu split of the log nearest-distances, which the pipeline already uses for the same purpose in skeleton_fit.py:74, segments.py:96 and skull_stage2.py:176.

### H6. anatomy/muscle_depth.py:79-88 - weight search
`w = 1.0`; `for _ in range(8)`; `w / 16`; `w *= 16`. Factor 16 and 8 steps bound the weight to 16^+-8 and give a 16x-coarse answer. Replace by `brentq` on log w for `r(w) = hstep` (monotone), started from the Laplacian's own scale.

### H7. Solid sealing loop (9 places)
`for seal in range(1, 17)` solids.py:20, 73; body_assembly.py:113; muscle_carry.py:100; muscle_depth.py:34.
`range(1, 33)` part_bone_fit.py:95; skull_line_fit.py:91. `range(1, 5)` solids.py:58; body_assembly.py:154.
Criterion `fill.sum() - thick.sum() > thick.sum()` ("inside larger than the shell", ratio 1) in all of them.
Three different caps (16 / 32 / 4) for one construction; when the loop runs out, the last `seal` is used silently except in the two `range(1,5)` places. The cap decides whether a limb piece "closes". Replace with: seal = the largest jump of filled volume against seal (the volume curve's own split), loop bound = half the grid's smallest side (a real safety limit), and raise when no inside appears.

---

## 2. MEDIUM importance violations

| file:line | fragment | what it does | suggestion |
|---|---|---|---|
| solids.py:18; body_assembly.py:108; muscle_carry.py:98 | `step = 2 * median edge` | whole-body grid at two edges (comment: one edge does not fit memory); parts use one edge | derive from memory: largest of one edge and (bbox volume / affordable cells)^(1/3); a memory size is an allowed parameter, the 2 is not |
| solids.py:62; body_assembly.py:158 | `L >= big / 2` | a limb piece's "real openings" = loops at least half the longest | Otsu split of the log loop lengths |
| body_assembly.py:230 | `K_['B'] - fwd_w * est` | kneecap front sits ONE surface edge under the skin bump | atlas's own skin-to-patella depth times s1 (W['F_skin'] is loaded) or a CT tissue depth |
| body_assembly.py:327-336 | `lam = 1e-3`, `range(100)`, `range(12)`, `lam *= 4`, `lam / 4`, `1e-9`, `gain < 1e-5` | hand-written Levenberg-Marquardt: start damping, factors, stop tolerance | scipy `least_squares` (as the other fits) or stop on "largest hull move < hstep" as skeleton_fit does; the 1e-5 relative gain decides where the fit stops |
| part_bone_fit.py:143 | `max(..., np.radians(1))` | floor of 1 degree on the turn scale | floor from data: detector spread / bone length (already computed as `noise` at 119) |
| part_bone_fit.py:126 | `sig_r = sig_r or 1.0` | first-round turn scale 1 rad when no stretch exists | the body skeleton fit's own turn spread, or the hand's measured value before the foot runs (order the steps) |
| part_bone_fit.py:139; skeleton_fit.py:139 | `max_nfev=200`, `max_nfev=60` | evaluation caps that can stop a solve early | keep as safety only but log/raise when `r.status == 0` (cap hit) |
| skeleton_fit.py:139 | `loss='soft_l1'` (default `f_scale=1.0`) | robust knee at 1 scaled unit | ? implicit: residuals are already in sigma units so the knee is "one sigma" - still a choice; same for `cp.huber(..., 1)` fit_slices.py:149, 151 |
| skeleton_fit.py:65 | `thr = median(nearest gaps)` | touching pairs = gap under the MEDIAN nearest-other-bone gap | ? data-derived but the 50 % is a choice; agents/point_names.py:61 already has the Otsu split (bone_gaps.npz `threshold`) - use that one list everywhere |
| agents/point_names.py:61 | `np.log(np.maximum(gp, 1e-6))` | floor on relative gap before the Otsu split | not just a safety epsilon: it places all zero-gap pairs at log 1e-6 and so shapes the split. Floor at the atlas mesh edge / bone size (measured resolution) |
| agents/point_names.py:59 | `lo[a_] - m_ > hi[b_]` | "nearby" pairs = boxes within one smaller-bone size; defines the pool the Otsu split sees | ? self-relative; state it, or take all pairs with a bounding-sphere test as skeleton_fit.py:61 |
| parts/material_regions.py:21, 20 | `range(2, 17)`, `min(50000, ...)` | colour clusters 2..16 by BIC on a 50k sample | BIC until it rises (stop by its own test, no upper bound); sample size is compute but changes BIC - use all or state as memory limit |
| agents/segment_label.py:174, 36 | `colour_pieces(idx, 6)`, `min(30000, ...)` | "mixed" re-split 1..6 clusters by BIC | same: stop when BIC rises |
| parts/contacts_patches.py:24 | `query(V[S], k=9)` ... `repeat(..., 8)` | contact vertices joined by their 8 nearest neighbours | mesh edges among contact vertices plus the partner links (the mesh is there), or radius = contact split `thr` |
| parts/masks_vote.py:31; parts/body_vote.py:36 | `S.max() > med[c]` | a view's top mask counts when above the pool's MEDIAN | the Otsu split already computed; or ask Jev on the weak view |
| rig/joints3d.py:35, 50 | `for _ in range(2)` (twice) | fixed two re-weighting passes / two flip passes | iterate until `keep` / `flips` stop changing |
| rig/joints3d.py:39 | `res <= max(otsu(res), np.median(res))` | outlier cut: larger of Otsu and median | Otsu alone with a minimum of 2 views kept, or residual vs detector noise (selftest `detected_vs_true_landmark_mm`) |
| rig/body_views.py:20 | `range(0, 360, 45)` | 8 views; drives triangulation, votes, LLM count | decision value, not a render setting: add views until the share of unseen vertices (vis.npz) stops falling |
| anatomy/planes.py:15 | `TH = 72` | 5-degree bins for every slice outline (also za_slices.py:60) | bin width = mesh edge / section radius |
| anatomy/planes.py:32 | `np.linspace(0, 1, 9)` | 9 samples per cut segment | segment length / bin arc |
| anatomy/fit_slices.py:62 | `max(101, len(tl))` | 101 plane levels | spacing = mesh edge along the path |
| anatomy/fit_slices.py:72 | `2 * TH` | observation bins twice the atlas's | as planes TH |
| anatomy/kp_frames.py:26 | `1 / 4, 2 / 4, 3 / 4` | spine helper joints at even quarters | atlas vertebra-level fractions (segments.Atlas has the vertebrae) - the docstring promises "atlas's own fractions" |
| anatomy/kp_frames.py:32, 33, 37, 39, 41 | `0.5` | helper joints (clavicle, wrist twist, foot, pinky0, transverse tarsal) at the middle | ? midpoints, but they decide which skeleton bone owns surface in fit_slices `segdist`; atlas fractions where the atlas has the bone |
| anatomy/toe_lines.py:33; face/eye_geo.py:14 | `near - 2 * edge` | visible = within TWO edges of the front surface | philtrum_area.py:41 and skull_line_fit.py:43 use ONE edge; use body/vis.npz-style visibility or one edge everywhere |
| face/philtrum_area.py:36 | `w_ / 2`, `h_ / 2` | next crop = box plus half its size each side | pad by Jev's own box uncertainty (1 - confidence) or ask Jev for the crop |
| anatomy/skull_line_fit.py:39 | `im.min(2) < 250` | figure bounding box by "not white" | alpha / the projected head vertices (cameras.json is loaded) |
| anatomy/fat_layer.py:22, 24 | `max valence * 3`, `rank <= 3` | search 3 rings, average the 3 nearest | comparison only; one nearest same-facing vertex, or barycentric on the nearest face |
| anatomy/fat_layer.py:15 | default `'female'` | which MRI table | ask Jev / the LLM once and pass it |
| anatomy/atlas_warp.py:107 | `0.5 * np.clip(1 - dist / rj, 0, 1)` | linear hand-over within ONE limb radius | ? 0.5 is the midpoint; zone width 1 x radius and the linear shape are choices - harmonic weights along the bone would have none |
| anatomy/muscle_inside.py:80 | regex + drop nose cartilages | hand rule on the result | decide from the measured outside-depth after repair |

---

## 3. LOW importance violations

| file:line | fragment | note |
|---|---|---|
| solids.py:6; part_bone_fit.py:83; skull_line_fit.py:84; body_assembly.py:109, 147; muscle_carry.py:89 | `4 * step` grid padding | also a latent bug: dilation runs to 16-32 cells but the grid has 4 cells of margin; pad by the seal bound |
| solids.py:7; part_bone_fit.py:84; skull_line_fit.py:85; body_assembly.py:109, 148; muscle_carry.py:90 | `/ (step / 2)`, `clip(..., 1, 64)` (256 at 148) | half-cell sampling is a definition; the 64 cap leaves holes in triangles longer than 32 cells; caps differ |
| skeleton_fit.py:34; part_bone_fit.py:106; skull_line_fit.py:101; skull_stage2.py:247; body_assembly.py:122 | `len(ii) >= 8` | hull only for 8+ vertices; the try/except already covers degenerate cases |
| part_bone_fit.py:118; skull_stage2.py:86, 176; skeleton_fit.py:51, 74, 132; fit_slices.py:186; flesh_islands.py:17 etc. | `> 2`, `> 3`, `< 4`, `m < 3` | minimum counts; most are the algebraic minimum (4 points for an affine map) - fine; `> 2` for a spread is a choice |
| solids.py:32, 46; body_assembly.py:133; muscle_carry.py:55; muscle_depth.py:66; muscle_inside.py:40 | `np.round(V, 6)` / `7` | weld tolerance in metres (two different values); flesh_islands.py:34 welds exactly with `np.unique` - use that |
| skull_line_fit.py:50; skull_stage2.py:207; agents/landmark_lines.py:29 | `max(8, int(L / step) + 1)` | at least 8 samples per line |
| skull_line_fit.py:124 | `np.percentile(g0, 10)` | neck-gap check only (report) |
| skeleton_fit.py:167, 180 | `< 3 * S['d']`, `ev > 1.0` | checks only (bone-in-bone prefilter, "pinned by data") |
| body_assembly.py:215 | `np.ptp(Pk[:, 2]) / 2`, median | "front half of the leg" window |
| body_assembly.py:66 | `eps * max * 8` | mirror-pair tolerance (numerical) |
| body_assembly.py:339 | `spread / 8`, `max_nfev=400` | BA_PROBE debugging branch only |
| anatomy/planes.py:43 | `0.99` | near-parallel guard |
| anatomy/segments.py:135 | knee front = sum of two unit directions | equal weights |
| parts/sam3_masks.py:28; parts/part_concept_masks.py:19 | `mask_threshold=0.5` | SAM 3 pixel threshold (library default / probability midpoint); `threshold=0.0` is fine |
| parts/part_concept_masks.py:22-23 | `o_[:8]`, `S[k] < 0.3` | overlay sheet only |
| face/eye_geo.py:14 | `np.argsort(-S)[:2]` | two best instances = two eyes; Otsu on scores as toe_lines does |
| face/eye_geo.py:19-20 | `range(50)`, `1e-7`, knee at 1 sigma | absolute tolerance in metres; use relative to the mesh edge |
| parts/flesh_islands.py:50, 68; material_regions.py:18, 26 | `* 4`, `(n, 4)` | four materials hard-wired as a count (structure, breaks if the list changes) - use `len(MN)` |
| parts/body_vote.py:60 | `'mean_share': 1.0`, `'tripo_part': 0` | fixed confidence for geometry details |
| agents/segment_label.py:90, 100, 141; contacts_patches.py:43, 46; contacts_split.py:51 | top 3 views / hints, `* 1.5` zoom | what the LLM is shown - affects answers slightly |
| skull_stage2.py:242, 314 | `S_['sig'] < SIG_MED` | ? hard vs soft sites split at the median spread: data-derived, but "median" is the rule; an Otsu split of the spreads would be self-chosen |
| skull_stage2.py:330 | `prev_u - u < 1` | gain under one standard = no gain (unit is data-derived) |
| symmetry.py:25 | `max_iterations=100` | ICP cap (trimesh default threshold 1e-5 ends it) - safety, OK |

---

## 4. CITED values / methods
| file:line | what | source in code |
|---|---|---|
| body_assembly.py:178-180 | patellar ligament length / kneecap size about 1 | Insall and Salvati 1971 (the code then uses the atlas's own measured length, so no literal depends on it) |
| symmetry.py:5 | rigid ICP | Besl and McKay 1992 |
| part_bone_fit.py:6; body_assembly.py:6; skeleton_fit.py:15 | closed-form similarity | Umeyama 1991 |
| body_assembly.py:326 | Levenberg-Marquardt | Marquardt 1963 (method cited; the constants are not - see Medium) |
| muscle_carry.py:8, 10 | harmonic coordinates; dual-quaternion blend | Joshi et al. 2007; Kavan et al. 2008 |
| muscle_inside.py:3 | Laplacian mesh editing | Sorkine et al. 2004 |
| agents/point_names.py:5; joints3d.py:22 | 133-point scheme and indices | COCO-WholeBody, Jin et al. 2020 |
| skull_stage2.py:16, 58, 67, 251 | tissue depth medians / spreads, skull proportion spread, nose feature spread | CT study files (388 people): summary.json, skull_shape_spread.json, face_skull_model.npz |
| fat_layer.py:1-2 | fat layer share of height | HIT MRI statistics (241 women, 157 men) |
| skull_stage2.py:282 | detector noise | atlas selftest.json (measured) |
| all `1.4826` | MAD to sigma | mathematical constant |
| flesh_islands.py:97 | 0.04045, 12.92, 0.055, 1.055, 2.4 | sRGB transfer function (standard, not cited in code) |
| muscle_inside.py:27-29 | skin 2 mm | NOT cited - listed as violation H4 |

---

## 5. Fixed vocabularies, regular expressions, model names (hand-written knowledge)

| file:line | content | could an LLM / Jev supply it? |
|---|---|---|
| parts/sam3_masks.py:11-17 | `CONCEPTS` per band (face parts; 21 clothing/body words for 'tex'; 'button', 'badge', 'cuff stripe', ... for 'torso') | Yes - ask Jev per view what is visible and use its words (loose_names.py already works that way). The 'torso' list is specific to one model |
| parts/loose_names.py:13-15, 17 | `NAMES` = cape, cloak, robe, coat, long skirt, scarf, veil; prompt says "standing with arms out" | Yes - free-text answer instead of a choice list; the pose sentence is an assumption about the input |
| material_regions.py:17; contacts_combine.py:15; segment_label.py:27; concept_material.py:13; material_fix.py:17 | `['flesh', 'cloth', 'hair', 'prop']` (+ 'mixed') | This is the pipeline's own schema (5 copies); keep, but define once |
| agents/joint_names.py:17 | `KINDS` (fused, fibrous, ... none) | schema of the question; fine |
| agents/point_names.py:29-30; part_bone_fit.py:45, 57; body_assembly.py:71 | BODY / HAND point names, finger names, `ORD` first..fifth, `CO` joint list | detector's published point list - OK; the mapping point -> bone already comes from the LLM |
| anatomy/segments.py:20-71 | all segment definitions: bone name lists, skin-region regexes, roll landmarks, model joint ownership | Partly: bones per segment and skin regions could be asked once against the atlas's name list (as point_names.py / joint_names.py do) and cached with hard-rule checks. Largest block of hand anatomy in the repo |
| part_bone_fit.py:40 | `PAT` hand / foot bone regex | Yes - from points_named.json bones plus bone_gaps contacts, or one LLM call |
| skull_line_fit.py:75; skull_stage2.py:60 | head bone regex + `cartilage|Incus|Malleus|Stapes|Sinus|concha`, exclusion `rib|Thyroid|Cricoid|...` | Yes - same one-call naming against the atlas list |
| skull_stage2.py:62, 69, 71; muscle_inside.py:80 | `nasal|alar`, `Mandible` / `Lower `, tooth regex, nose-cartilage regex | Yes |
| skull_line_fit.py:48-49; skull_stage2.py:208-209 | `RUNS` / `SOFT` / `WHOLE`: which 68-point runs lie over bone | Yes - agents/landmark_lines.py asks exactly this, but its answer (lines/answers.json) is not read here; the docstring of skull_line_fit says "named by a language model" while the dict is hard-coded |
| skull_stage2.py:221-222 | `SITES`: CT site code <-> landmark index (n, g, mso, sn, mp, ls, li, gn) | Yes, once, with a rule check; or a table shipped with the CT study |
| skull_stage2.py:67 | nose features `{'prn', 'acL', 'acR', 'sn'}` | from the CT model's own feature labels |
| agents/landmark_lines.py:20-24 | run descriptions in words | index ranges are the published 68-point scheme; descriptions are prompt text |
| atlas_warp.py:16-17; muscle_carry.py:31 | `keep_sys`, name suffixes `.g`, `.j` | atlas file convention - OK |
| body_assembly.py:187, 216, 245; solids.py:171; muscle_depth.py:47, 103 | 'Patella', 'Tibia', 'Femur', 'Hip bone', 'Patellar ligament'; material names 'hair', 'prop', 'flesh', 'cloth' | atlas names for specific rules; could come from points_named.json |
| toe_lines.py:28; body_assembly.py:205; eye_geo.py:23 | SAM 3 concepts 'toe', 'knee', 'eye', 'eyeball' | fixed prompts, one per rule |
| toe_lines.py:46 | exactly five toes | anatomy, fails loudly - fine |
| model names | `jev-1.14.0` (loose_names.py:11, philtrum_area.py:13, segment_label.py:119 via env), `facebook/sam3` (sam3_masks.py:18, part_concept_masks.py:10), RTMW `rtmw-dw-x-l_simcc-cocktail14_270e-256x192` (skull_line_fit.py:34), `Wholebody(mode='balanced')` (joints3d.py:8) | configuration; put in one place (env / config file) |
| prompts | concept_material.py:7-10, material_fix.py:10-14, subcontact_label.py:12-20, segment_label.py:104-115, joint_names.py:19-23, point_names.py:34-44, philtrum_area.py:11 | prompt text - inherent |

---

## 6. OK counts per file (approximate; indices, math constants, epsilons, render settings, safety caps, chunk sizes, timeouts, paths)
symmetry 14, solids 22, body_assembly 70, part_bone_fit 40, skull_line_fit 35, skull_stage2 75, skeleton_fit 45, muscle_carry 30, muscle_inside 15, muscle_depth 22, fat_layer 12, toe_lines 30, atlas_warp 18, fit_slices 45, kp_frames 30 (COCO indices), planes 15, segments 5,
loose_names 8, loose_vote 10, flesh_islands 14, masks_vote 14, body_vote 12, material_regions 6, contacts_patches 18, contacts_combine 6, contacts_split 14, sam3_masks 3, part_concept_masks 12,
philtrum_area 14, eye_geo 30 (mostly plotting), joints3d 22, body_views 8,
segment_label 22, concept_material 2, joint_names 4, point_names 6, landmark_lines 12, material_fix 2, subcontact_label 3, pipeline 1 (budget default), ct_fit_all 2.

Files with no violation at all: loose_vote.py, contacts_combine.py, contacts_split.py, concept_material.py, joint_names.py, material_fix.py, subcontact_label.py, pipeline.py, ct_fit_all.py, segments.py (numbers; it is all vocabulary).

---

## 7. Not in the audit list but on the run path (only grepped, not read)
- anatomy/za_slices.py (called by ct_fit_all.py): `TH = 72` (60), `np.linspace(t_lo, t_hi, 101)` (68), `np.linspace(0, 1, 9)` (63, 104) - same three numbers as planes.py / fit_slices.py; how `nlev` and the critical levels are chosen was not checked.
- pipeline.py also runs, unaudited: parts/views_vis.py, part_colours.py, part_normals.py, contacts_find.py, show_subcontacts.py, geo_views.py, geo_vote.py, geo_paint.py; rig/fit_mhr.py, rig/skeleton.py; agents/geo_colour_parts.py (from segment_label.py:161).
- pipeline.py:38-39 still runs the MHR fit and stick skeleton; kp_frames.py exists to replace it but is not a pipeline step.
- pipeline.py:41, 70: the step marker `tex_a315.npz` is tied to the 8-view / 45-degree choice in body_views.py:20.
