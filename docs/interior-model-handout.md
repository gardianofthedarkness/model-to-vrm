# Interior model: mathematical construction and research handout (version 5.1)

**Status: a proposal, not a built system; the design is frozen through experiment T1b and measurement has started.** Nothing here has been trained or tested. Version 5.1 folds in the seventh review round (four corrections, listed at the end of section 1). Version 5 followed six rounds of outside review, a census of the data we can obtain, and a report on skin-attachment mechanics supplied by the user. Section 1 records what changed; section 13 holds the open positions; section 15 lists what nobody has verified.

## 0. Purpose

We fit an anatomy atlas (bones, muscles) inside a generated 3D character by least squares and convex optimisation. The optimiser knows hard rules (bones inside the body, seams closed, left and right alike) but little about where tissue lies under a given surface. We want a learned model of how bodies are built, **with every part in the context of all the others**, with calibrated uncertainty, in a form the optimiser uses directly, and ending in a figure that can be animated.

The user's requirements: cross-sections ("critical slices") at important places are the working currency; all predictions for one person share context; confident predictions guide the optimiser; a transformer with diffusion is the preferred family; the input character is clothed, with cloth kept close to the body where the user can arrange it; the result must serve as a rig.

## 1. What changed from version 4, and why

| # | Point | Verdict | Where it lands |
|---|---|---|---|
| 1 | The garment equation was stated ($q = $ skin $+\,c\,\nu$) but not enforced: the objective penalised a free clearance $c$, so the solver could choose a comfortable gap without moving the skin to the cloth. | **Accepted: the main error of version 4.** | Sections 3.3 and 7.2: the clearance is computed from the geometry, not a free variable. |
| 2 | Adipose thickness was defined from CT, but the state held only two thicknesses, so fat could not be read out at inference. | **Accepted.** | Section 3.1: an adipose fraction. |
| 3 | "Reference" meant supine-as-scanned in training and unclothed-unloaded at inference. Those are different targets, and one clothed observation cannot separate reference thickness, posture and clothing. | **Accepted.** | Section 3.1: one declared reference posture; standing is a separate experiment. |
| 4 | A minimiser is one estimate, not a posterior. Coverage tests need an explicit uncertainty calculation. | **Accepted.** | Section 8. |
| 5 | The transfer length does not set how strongly the mechanics competes with the data; the normal-to-tangential ratio is separate; the membrane term charges rigid rotation. | **Accepted.** | Section 3.2. |
| 6 | Normal-offset layers keep their order along each line, but neighbouring lines can cross. | **Accepted as a check.** | Section 3.1. |
| 7 | The reviewer now prefers construction (I), prior plus one likelihood, because with explicit layers one can inspect why a sleeve changed the femur. | **Agreed; no longer a disagreement.** | Section 6. |
| 8 | Acceptance test: the exterior fixes only total soft-tissue thickness; revealing one fascia contour should narrow the muscle-fat split without moving the femur. | **Adopted.** | Section 11, T1b. |
| 9 | Feasibility depends on how many people have a usable thigh with femur, skin and enough visible fascia. | **Open fact; a count is needed.** | Section 12. |
| 10 | The user's question: once bone-to-skin connection and tissue physics exist, is the figure rigged for animation? | **Yes, with conditions.** | Section 10 (new). |
| 11 | User decisions: anonymised research CT kept private, and face renders inside private jobs, are acceptable (as I read the user's reply "Acceptable to me"). | Recorded. | Section 14. |

**Version 5.1 (seventh round).** (a) Section 8.1 overstated one result: the surface adds no *precision* along the muscle-fat split, but the split's *variance* can still shrink through the prior's correlation with total thickness; the fascia test rewards such narrowing when it is accurate, and scores femur change instead of forbidding it. (b) Adipose thickness on an arbitrary ray is an integral through the subcutaneous volume, not $f^{A} t^{F}$. (c) The surface likelihood must not grow with mesh density. (d) The exact benchmark is run with the Gaussian prior first; the Huber tail comes after. Also: a one-leg rig audit moves ahead of the CT experiments (T-rig), and the swimsuit-and-outfit pair is a repeatability test only. Working defaults adopted for the pilot: the reference skin is the persistent rig asset, the current skin is the per-frame output, and joints are driven by clips.

Kept from version 4: the three-part state; training against measurements; the three-part oracle test; clothing as one-sided observations calibrated on paired data; Gaussian, mixture and diffusion compared on equal terms; body first because faces are scarce.

## 2. Objects and notation

Person or figure $p$; observed outer surface $\mathcal O_p$ with material labels; person facts $g_p$ with known/unknown flags; lengths as proportions of a surface-measurable size $S_p$. An atlas template with bones, muscles and three nested surfaces per region: bone $\Gamma^{B}$, deep fascia $\Gamma^{F}$, skin $\Gamma^{S}$; template skin points $X \in \Gamma^{S}_0$ with unit normals $n(X)$. Slice planes $\Pi_k$ from the outer surface only.

Three depths are distinct and are never substituted for one another: skin to fascia $d^{SF}$, skin to bone $d^{SB}$, and adipose thickness $d^{\text{fat}}$ (the length of a ray, between skin and fascia, occupied by fat).

## 3. The state

$$ z = \big( z^{\text{ref}},\; u \big), \qquad \text{with the garment clearance a function of } z \text{ and the observation (3.3).} $$

### 3.1 Reference anatomy $z^{\text{ref}}$

**Bone** $z^{B}$: the fit's deformation parameters in canonical form (global pose and size removed, minimum-norm representative, area-weighted displacement norm).

**Layer thickness fields** on the template skin: lean thickness $t^{L}(X) \ge 0$ (bone-carried support to fascia) and subcutaneous thickness $t^{F}(X) \ge 0$ (fascia to skin):
$$ \Gamma^{F}_{\text{ref}}(X) = b(X; z^{B}) + t^{L}(X)\, n(X), \qquad \Gamma^{S}_{\text{ref}}(X) = \Gamma^{F}_{\text{ref}}(X) + t^{F}(X)\, n(X). $$
Positive offsets keep the order bone, fascia, skin along each line; they do not prevent neighbouring lines from crossing, so reconstructed layers are checked for crossings and inverted elements. Independently deformable fascia and skin surfaces replace the offsets only if the oracle test shows a real failure.

**Adipose fraction** $f^{A}(X) \in [0,1]$: the share of the subcutaneous layer that is fat, so that the model's own readout is
$$ d^{\text{fat}}(X) = f^{A}(X)\, t^{F}(X) \quad \text{along the template's own thickness column at } X, $$
and on an arbitrary ray $r$ (an oblique slice passes through more of the layer than its normal thickness)
$$ d^{\text{fat}}(r) = \int_{r\, \cap\, \Omega_{\text{subcutaneous}}} f^{A}(x)\, d\ell , $$
with $f^{A}$ extended through each column's subcutaneous volume (piecewise constant per column is enough for the pilot), so that axial and oblique readouts refer to the same anatomy. It is supervised by the CT-measured $d^{\text{fat}}$ on valid rays, or by a regional integral where CT supports only a regional measurement. If the usable scans support neither, $t^{F}$ is reported as "subcutaneous-layer thickness" and no fat claim is made.

**Reference posture.** One declared posture for the pilot: **supine, unclothed, as scanned**, used in training and in evaluation alike. Transfer to a standing character is a separate experiment (section 11, T7), not an assumption.

**Reduced coordinates.** $z^{\text{ref}} = \bar z + W a + \eta$ with one joint basis over bone parameters, log-thicknesses and the logit of $f^{A}$.

**What the exterior cannot know.** Locally the surface observes $b + t^{L} + t^{F}$. The direction $(\delta t^{L}, \delta t^{F}) = (+\epsilon, -\epsilon)$ leaves the surface unchanged: the split between muscle and fat is not observed from outside. What the model may still know about it comes only from how the split varies with total thickness and with the rest of the body in the population (section 8.1 makes this exact).

### 3.2 Tissue displacement $u$

$u_s(X)$: displacement of current skin from reference skin; $u_d(X)$: displacement of the deeper support; jump $g_\Gamma = u_s - u_d$. From the user's report, in reduced form:
$$ E_\Gamma = \frac{\alpha}{2} \int_{\Gamma_0} \Big[ \big\| (I - n n^\top) g_\Gamma \big\|^2 + r_n \big( n^\top g_\Gamma \big)^2 \Big] dA, \qquad E_S = \frac{\alpha\, \ell_s^2}{2} \int_{\Gamma_0} \| \nabla_\Gamma \tilde u_s \|^2\, dA , $$
written so that its three independent unknowns are visible: the overall strength $\alpha$ (how hard the mechanics competes with the data), the normal-to-tangential ratio $r_n = \kappa_n / \kappa_t$, and the transfer length $\ell_s = \sqrt{S/\kappa_t}$ (how far a tangential pull spreads, $u(x) = u(0)\, e^{-x/\ell_s}$ on a strip). A displacement profile identifies $\ell_s$ only; $\alpha$ and $r_n$ need force or paired-load data. None is identifiable from one static scan, and none is set by hand: each carries a prior and enters the sensitivity analysis separately.

$\tilde u_s$ is $u_s$ with its best-fitting rigid motion removed, for a fixed pose. For pose transfer the membrane term must be co-rotated or written in surface strain; otherwise a rotation is charged as stretch.

For fat under compression, a tracked patch with reference thickness $h_0$ and area ratio $J_A$ obeys $h\, J_A \approx h_0$; this is a volume condition over a 3D region, read off on slices afterwards.

**Scope in the pilot.** $u = 0$ in T0, T1 and T1b. $u$ is free only in T2 (indentation), with a prescribed displacement or paired geometry.

### 3.3 Garment clearance, computed, not free

For an observed cloth point $q$ with correspondence $X_q$ and unit direction $\nu_q$, the current skin is $\Gamma^{S}_{\text{cur}}(X; z) = \Gamma^{S}_{\text{ref}}(X; z^{\text{ref}}) + u_s(X)$ and
$$ c_q(z) = \nu_q^\top \big( q - \Gamma^{S}_{\text{cur}}(X_q; z) \big), \qquad \varrho_q(z) = \big( I - \nu_q \nu_q^\top \big) \big( q - \Gamma^{S}_{\text{cur}}(X_q; z) \big), $$
the clearance and the tangential residual. Both are affine in $z$ once $X_q$ and $\nu_q$ are fixed. When the observation is a ray intersection instead of a matched point, $c_q$ is the distance along that ray and there is no tangential residual. A waistband pressing in is $u_s < 0$ with $c \approx 0$; a loose panel is $u_s = 0$ with $c > 0$.

## 4. Mechanics adopted from the report, and what is left out

Adopted: the split between how a person is built and how tissue has moved; the reduced interface and membrane energies; the transfer length; anchors that follow the tissue they attach to (fascia following muscle) rather than the nearest bone; the warnings against counting one stiffness twice and against reading stiffness from shape alone.

Left out of the static pilot and needed for animation (section 10): discrete cables, hyperelastic laws, contact with friction, inertia and damping, viscoelasticity.

## 5. Clothing: what the cloth does and does not say

- Bare skin: equality up to surface noise. Cloth: $c \ge 0$, one-sided. Hair and props: $c \ge 0$ with no useful upper limit.
- The step across a garment's edge is an *effective edge offset* $\tau_e$ (fabric plus hem, seam, gap or indentation), a feature of the garment.
- For tight cloth near a closing of the body by a ball of radius $R$, the cloth's largest concave curvature satisfies $\kappa_{\max} \le 1/R$: it gives an upper bound on $R$.
- A flat panel may lie on flat skin or span a hollow of any depth; the cloth alone does not decide it.
- **Clearance model.** $p(c \mid \text{features})$ with features $\tau_e$, local curvature, distance to the garment's edge, material. A first family: a Gaussian truncated to $c \ge 0$, location $m$ and spread $s$ fitted separately for tight contact, loose panels and uncertain correspondence, on paired body and garment geometry. Its negative logarithm on the support, $(c - m)^2 / (2 s^2)$, is convex.
- **Sources of paired data:** the user's characters where the body under an outfit is known; simulated garments on CT bodies; an independent cloth-simulation dataset (named in review, unverified). A retained *pre-compression* body mesh gives the clearance only where the garment does not press the skin in; it cannot say how far a waistband pushed.
- **Both skins are predicted**: the reference skin (which the anatomy belongs to) and the current skin (which the cloth touches).
- **The swimsuit base (the user's answer to U1).** An unclothed body cannot be generated (the image service flags it). The nearest obtainable thing is the same character generated in a swimsuit as the base model. That gives bare skin, an equality, over the limbs, shoulders, back and most of the trunk, and leaves tight cloth over the pelvis and chest only. Two consequences. First, the anatomy is fitted on the swimsuit base, where almost everything is observed; the remaining cloth is tight and small, but it covers exactly the regions that bridge (between the breasts, the crotch, under the buttocks), so those stay inequalities. Second, a clothed version of the same character is a second, independent generation: it is not the same geometry with cloth added, so the difference between the two after registration mixes garment clearance with generation drift. The drift must be measured on regions bare in both (it is the noise floor of any clearance read from the pair) before the pair is used to calibrate $p(c \mid \text{features})$.

## 6. Inference: prior plus one likelihood

$$ \hat z = \arg\min_z \; \underbrace{-\log p_\vartheta(z^{\text{ref}} \mid g)}_{\text{learned prior}} \;+\; \underbrace{-\log p(\mathcal O \mid z)}_{\text{surface, used once}} \;+\; \underbrace{E_\Gamma(u) + E_S(u)}_{\text{mechanics}} \;+\; \text{(seams, joints)} \quad \text{s.t. hard rules}. $$

Context comes from the prior's covariance. The exterior is not also an input to the prior. A conditional network $p(z^{\text{ref}} \mid \mathcal O, g)$ may initialise the solver; its output is not multiplied into the objective. Construction (II), conditional predictor plus feasibility, is kept only as a comparison (T3).

**The learned model** is a generative prior over anatomies: person-fact tokens, global coefficient tokens for $a$, structure-query tokens anchored on the atlas; attention relations: same segment, adjacent, mirror, same garment. Under this construction a low-rank Gaussian over a joint skin, fascia and bone basis is the classical statistical shape model; a transformer with diffusion earns its place only if the distribution is not well described by that (T4).

## 7. The optimiser's terms

### 7.1 Prior

Gauge directions removed; the learned subspace penalised by a Huber function of the Mahalanobis norm, $\psi(\| \Sigma_a^{-1/2}(a - \mu_a) \|)$; the remainder keeps the fit's existing stiffness. For one coordinate in prior standard deviations, with the surface asking $k$ at relative precision $\lambda$: quadratic gives $x^\ast = \lambda k/(1+\lambda)$, Huber beyond its transition $c_H$ gives $x^\ast = k - c_H/\lambda$. $c_H$ is estimated on validation data.

### 7.2 Surface, with the garment equation closed

$$ -\log p(\mathcal O \mid z) = \sum_{q \in \text{skin}} \frac{\| q - \Gamma^{S}_{\text{cur}}(X_q; z) \|^2}{2 \sigma_{\text{surf}}^2} \;+\; \sum_{q \in \text{cloth}} \Big[ \varphi\big( c_q(z) \big) + \frac{\| \varrho_q(z) \|^2}{2 \sigma_{\tan}^2} \Big] \;+\; \sum_{q \in \text{hair, prop}} \iota_{\ge 0}\big( c_q(z) \big), $$
with $c_q(z)$ and $\varrho_q(z)$ from 3.3 and $\varphi = -\log p(c \mid \text{features})$, infinite for $c < 0$. The skin is now pulled to where the cloth says it can be; there is no free clearance to absorb the misfit.

**Mesh density.** The sums are over observations of a surface, not over vertices: each term is weighted by the area its point represents, with $\sigma_{\text{surf}}$ and the clearance scale calibrated per unit area (or the surface is resampled at a fixed physical spacing). Subdividing the same mesh must leave the inferred anatomy and its uncertainty unchanged; the fit is run on two tessellations of one geometry to check it.

### 7.3 Hard rules

$t^{L} \ge 0$, $t^{F} \ge 0$, $0 \le f^{A} \le 1$; $c_q(z) \ge 0$; seams and joints as now.

### 7.4 What is and is not convex

With correspondences, normals and ray assignments fixed, the surface and mechanics terms are convex in the skin positions; they are affine in the coefficients $a$ only for the deformation graph, and nonlinear in $a$ wherever the decoder uses log-thicknesses or bone rotations. Joint rotations, log-thickness and logit maps, and moving intersections are not; they are linearised each step. The solver is successive convex steps; after the last step, intersections, clearances and feasibility are recomputed on the decoded meshes with the exact geometry.

## 8. Uncertainty is a separate calculation

### 8.1 The exact benchmark

For a local linear observation $y = H a + \varepsilon$, prior $a \sim \mathcal N(\mu, \Sigma)$, noise $\varepsilon \sim \mathcal N(0, R)$:
$$ \Lambda_{\text{post}} = \Sigma^{-1} + H^\top R^{-1} H, \qquad \mu_{\text{post}} = \mu + \Lambda_{\text{post}}^{-1} H^\top R^{-1} (y - H\mu). $$
Three checks with known answers:

1. A correctly learned conditional model (construction II) reproduces this update under the same assumptions.
2. Using the exterior twice adds $H^\top R^{-1} H$ twice: the overconfidence of version 3, in one line.
3. With $H = [\,1 \;\; 1\,]$ on $(t^{L}, t^{F})$ the data add precision only along $(1, 1)$. Writing $s = t^{L} + t^{F}$, $d = t^{L} - t^{F}$ and observing $y = s + \epsilon$ with noise variance $R$,
$$ \mathrm{Var}(d \mid y) = \mathrm{Var}(d) - \frac{\mathrm{Cov}(d, s)^2}{\mathrm{Var}(s) + R} . $$
The likelihood adds nothing along the split, yet the split's uncertainty shrinks by exactly as much as the population correlates it with total thickness. That narrowing is legitimate context; whether it is *right* is what coverage on held-out people tests.

The benchmark is first reproduced end to end with the plain Gaussian prior: posterior mean, covariance and coverage. The Huber prior of 7.1 is a different distribution (its scale matrix is no longer its covariance) and is introduced afterwards as a separate robust fit.

### 8.2 Beyond the benchmark

With clothing inequalities, Huber tails or several modes there is no closed form. Candidates: a Laplace approximation at the minimiser (curvature of the final convex step, with active inequalities handled as one-sided); sampling the posterior (guided sampling with a diffusion prior, or a mixture of fits); each checked against **held-out empirical coverage**, which is the only test that counts.

## 9. Training

**Targets are measurements** (per CT person): skin surface points; bone surface points from intensity; adipose tissue from its intensity range; fascia where visible or labelled, with its uncertainty recorded; named-ray depths; regional fat area where fascia is ambiguous.

**Gaussian head.** $y_p \mid g_p \sim \mathcal N\big( y^0_p + H_p(\mu_a - a^0_p),\; H_p \Sigma_a H_p^\top + R_p \big)$. A covariance direction $\Delta$ is unlearnable when $H_p \Delta H_p^\top = 0$ for all scans.

**Diffusion or flow head.** Denoising on $a$ where clean latents are credible; otherwise trained and compared through the measurement function with a proper sample score (energy score), measurement noise added, normalisation fixed on the training split, sampling cost counted.

**Point predictions** use squared geometric error; never a random draw against the truth.

**Cover.** Simulated garments and hair as connected 3D regions on CT bodies, plus the calibration sources of section 5.

## 10. From the fitted anatomy to an animated rig

The user asked whether, once the connection from bone to skin and the tissue physics exist, the figure is ready for animation. Yes: bones driven through a motion clip, with attachments carrying that motion to muscle, fat and skin, is a physics-based deformation rig. The conditions:

### 10.1 What is frozen and what is driven

| Part | During animation |
|---|---|
| Character anatomy: bone shapes and lengths, reference thicknesses and fat fraction, attachment locations | **Frozen** after the fit. Re-solving a femur's length against the silhouette each frame would let the anatomy drift. |
| Articulated skeleton: joint centres, local axes, parent-child order, bind pose | Defined once. A clip supplies joint coordinates $\theta_t$; each bone's rigid transform $(R_b(\theta_t), c_b(\theta_t))$ follows. |
| Muscle, fascia, fat, skin | **Solved**: anchors move with their bones, $a_j(t) = R_b(\theta_t) A_j + c_b(\theta_t)$ for an anchor at the fixed material point $A_j$ of bone $b$; the tissue solver finds the deformation. Optional muscle activation is an extra input. |
| Visible mesh | A persistent binding from each render vertex to the simulated tissue (or the simulated skin is the render mesh). |

### 10.2 Two solves

*Quasi-static* (first milestone): for each frame,
$$ x^\ast(\theta_t) \in \arg\min_{x \in \mathcal A} \; E_{\text{int}}\big(x;\, \text{anchors}(\theta_t)\big), $$
with $E_{\text{int}}$ the tissue, attachment and contact energies and $\mathcal A$ the non-penetration set. *Dynamic* (when fast motion shows the need): implicit Euler as an incremental minimisation,
$$ x^{n+1} = \arg\min_x \; \frac{\| x - \hat x \|_M^2}{2 \Delta t^2} + \frac{\| x - x^n \|_D^2}{2 \Delta t} + E_{\text{int}}(x; t_{n+1}), \qquad \hat x = x^n + \Delta t\, v^n , $$
which adds lag and settling. Here the mechanics left out of the static pilot (section 4) come back, with parameters that need loading or motion data.

### 10.3 What the current pipeline has and lacks

Has: fitted bones with closed joints; joint centres; for every muscle vertex a share of each bone it follows (harmonic weights, dual-quaternion blending), which is already a kinematic rig for the muscles; a repaired, non-protruding rest state.

Lacks: joint axes and ranges of motion; a binding of the visible skin to the tissue; muscles that do not overlap bone or each other (16% of muscle points overlap bone in the current result); attachment areas for most muscles (129 of 684 muscle items have both ends identified); any calibrated tissue parameters.

### 10.4 Milestone and finish line

Milestone: the thigh pilot extended with pelvis and lower leg, driven through a slow hip and knee bend, pose by pose. Check that the femur keeps its fitted shape, attachments stay on their material points, the layers keep their order, and the visible thigh bends continuously. Then the same motion fast.

Finish line: **fit once, bind once, play a motion clip not seen before without rerunning the anatomy inference.**

Who drives the bones matters. For an animator or a motion-capture clip, $\theta_t$ is given and the above suffices. A character that moves by its own muscles also needs a controller, articulated dynamics, balance and ground contact; that is outside this handout.

## 11. Experiments, in order

**T0a Measurement.** Skin, fascia and femur boundaries on prescribed rays of reviewed scans, repeated under perturbed landmarks and planes. Gives $R_p$ and the count of usable thighs.

**T0b Oracle fit.** Femur fixed at its measured boundary: can the layered state fit skin and fascia? Then all free. If the skin fits only when the femur leaves its boundary, the state lacks soft-tissue freedom. Check layers for crossings.

**T0c Compression.** The same with the reduced basis; skin, fascia, bone errors separately.

**T1 One body, changing cover** ($u = 0$). Bare skin, then tight cover, then bridged or loose cover on the same held-out body. Score reference skin, femur, fascia, interval coverage; baseline with material labels only. Expected: changing the outfit mainly changes uncertainty about hidden skin.

**T1b Reveal the fascia, in sequence, on one held-out thigh.**

| Evidence supplied | Scored, each on its own |
|---|---|
| Exterior only | error and interval width for total thickness, the layer split, femur position |
| Plus one fascia contour | change in split error and interval width on **other, withheld planes** (near and far); change in femur error |
| Plus a reviewed adipose measurement, where one exists | change in adipose-thickness error on withheld rays or regions |

A fascia contour fixes where the two layers meet; it says nothing directly about the fat fraction of the outer layer, which needs its own evidence. Scoring on withheld planes, near and far, tests the original aim: does evidence on one slice improve the same person's anatomy elsewhere, and does the uncertainty know how far that reaches? Scoring only the contour just supplied would test obedience to a constraint. A femur estimate may move when fascia is revealed; it is scored on whether it improves.

**T-rig One-leg rig audit** (no CT needed; run now). For one side of the fitted figure: list the muscles crossing the hip or knee; check each one's attachment areas against the fitted bones; measure initial muscle-bone and muscle-muscle intersections; bind a patch of visible skin; drive a slow hip and knee bend from prescribed joint angles. Compare the existing blended-muscle result with the reduced tissue solve under the same bone transforms. Look for attachment drift, skin sliding from a broken binding, and tissue pushed out by an intersecting rest state. These faults have different fixes from an inaccurate anatomy prior.

*Build order: T-rig and T0a now; then T0b (fixed femur), T0c, T1 (bare and loose), T1b. Indentation and diffusion wait until this sequence works.*

**T2 Indentation.** Prescribed local indentation with known skin displacement; sensitivity to $\alpha$, $r_n$, $\ell_s$ varied separately.

**T3 Constructions.** (I), (II), and (II) plus a surface penalty, against the benchmark of 8.1.

**T4 Heads.** Gaussian, mixture, diffusion.

**T5 Paradigms.** Per-slice (A), shared state with slice supervision (B), plus off-plane supervision (C).

**T6 Outfits on one known body.** On CT bodies with simulated garments the body is known exactly. On characters, the swimsuit base stands in for the known body: fit the anatomy on the base, then on the clothed generation of the same character, and compare, after measuring generation drift on regions bare in both. This is a repeatability test only: drift measured on arms bare in both generations estimates disagreement on those arms, and does not establish the error under a covered pelvis. Accuracy under cover is tested on CT bodies with simulated garments (T1).

**T7 Posture.** Supine reference to a standing character: as a known load producing $u$, or as a conditioned prior.

**T8 Slow bend** (section 10.4).

**Then the face**, separately: independent CT depth at the upper lip as the acceptance test, conflicting atlas-depth terms and the hand-written philtrum rule removed.

## 11b. Results so far: T0a (measurement), 2026-11-22

Measured on the private CT set; numbers only, no images. Lengths in millimetres at the scanned person's own size.

**Thigh.** Only 29 whole femurs from 17 adults exist in the set. Scored slice by slice (a slice counts when its section is clean), 201 of 348 slices are usable. Median across thighs: skin to bone 48.5 (middle half 44.3 to 60.6), skin to fascia 10.5, fat 8.2, dermis 1.5. Repeatability (20 re-extractions with the femur label moved by one voxel): 0.1 to 0.35 for skin, fascia and fat, 1.3 for bone (the label's own resolution). The fascia stand-in is clear on 76% of rays. Too few people for a population model; the thigh is a method check.

**Pelvis (chosen pilot region).** 408 adults with a whole pelvis; nine landmark slices each (iliac crests, top of sacrum, top of femoral heads, bottom of sacrum, ischial tuberosities, and midpoints); 3,274 of 3,672 slices usable (the rest hold metal implants or arms). Rays: 264,240; 0.2% dropped for an arm against the side, 8.5% for reaching the image edge.

| Quantity (median across people) | Front | Side | Back |
|---|---|---|---|
| Dermis | 1.1 to 1.9 | 1.5 to 2.2 | 3.0 to 3.8 |
| Subcutaneous fat | 18.8 to 28.9 | 16.7 to 19.9 | 14.2 to 21.2 |
| Skin to fascia | 20.2 to 33.0 | 18.8 to 23.4 | 18.4 to 25.1 |
| Fascia stand-in clear | 87 to 92% | 88 to 92% | 81 to 86% |
| Repeat spread, soft layers | 0.3 to 1.3 | 0.3 to 0.6 | 0.0 to 1.0 |
| Repeat spread, bone | 1.3 to 2.5 | 1.3 to 1.6 | 1.2 to 2.0 |

Checks: (a) at the back, where the gluteal and back muscles are labelled, the fascia stand-in lies 1 to 4 mm outside the labelled muscle, an independent confirmation; at the front and side the first labelled muscle is deep (the abdominal wall muscles are not labelled), so that check does not apply there. (b) Subcutaneous fat at the femoral-head level is thicker in women (median 21.8, middle half 16.5 to 30.9; 171 people) than in men (15.0, 11.2 to 20.2; 236 people), as expected. (c) Dermis reads thicker at the back; back skin is thicker, but these people lie on their backs, so compression on the table may contribute (open: 12.6). (d) The pelvic tilt summary failed for some people (a missing value; to fix).

What this establishes: the soft layers of the pelvis can be measured on hundreds of people with sub-millimetre repeatability; the fascia boundary needed by T1b is visible on most rays; measurement noise $R_p$ per quantity is now known. Next is T0b: fitting the layered atlas into these scans.

## 12. Data

Reported by the data session, not re-checked by me: one whole-body CT set of 1,228 people held privately (chest about 580, pelvis about 580, legs 148; 38 full faces, 110 at most); a 42-person head set and two MRI sets being fetched; a further head set withheld because its source moved to controlled access; large face collections need an institutional application. Labels: each bone except forearm, hand, shin and foot; about ten named muscles; **no skin or fascia labels**; fat from intensity or a separately licensed model.

**The number that decides the pilot** is how many people have the same usable thigh segment with reviewable femur, skin and enough fascia to score the split. It is not known. T0a produces it.

## 13. Open positions

**13.1 How much mechanics.** $\alpha$, $r_n$ and $\ell_s$ are unidentifiable from what we hold and may not be set by hand. To settle: which paired data identify them per region, and whether cloth-simulation or published tissue tests can provide priors in proportions.

**13.2 Stylisation.** How bone follows a stylised surface is a policy (fully, partly, not at all), to be tested on characters with known construction; paired warps test recovery of the policy that made them, not which is right.

**13.3 Gaussian or diffusion; convex fit or posterior sampling.** Decided by T4 and by coverage (8.2).

**13.4 Fixed points.** Remove the terms that pulled the wrong way in the skull fit before adding any firm rule.

**13.5 Fat readout.** Whether the adipose fraction is supportable from the scans, or the pilot reports subcutaneous-layer thickness only.

**13.6 Attachments for animation.** Most muscles lack identified attachment areas; whether they can be completed from the atlas's geometry or need another source.

## 14. Decisions

Answered (as I read the user's reply "Acceptable to me" to the two yes-or-no questions): anonymised research CT may be used, kept in private storage; CT faces may be rendered inside private jobs, with nothing leaving them.

Open:
- **U1 (answered).** No unclothed body is available. The base model can be generated in a swimsuit; outfits are separate generations of the same character (section 5).
- **U2, U5 (working defaults for the pilot, open to change).** The reference skin is the persistent rig asset and the current skin is the per-frame output; joints are driven by clips. Muscle-driven control stays possible later.

## 15. Not verified

The reference world model's architecture; every work cited by the reviewer (including the physics-based character work cited for section 10 and the thigh fascia studies); the references of the skin-attachment report (the report was read, not its sources); the data census; the intensity range for fat in CT; that keypoints and minimal-area planes work on clinical CT; that garment edges on generated characters are clean enough to measure an edge offset.

**Constraints on any proposal:** no hand-set decision numbers; proportions, not millimetres; a pilot within a few dollars; CT of real people stays private.

## 16. Research questions for the next round

1. **Layer parametrisation:** where normal-offset lines cross (inner thigh, armpit, trunk), and the minimal alternative.
2. **The clearance family:** beyond the truncated Gaussian, a family that stays convex and handles uncertain correspondence; the estimator of $R$ from curvature and its bias; and how to separate garment clearance from generation drift when the base and the clothed character are two independent generations.
3. **Posterior under inequalities:** a calibrated approximation for 8.2, with its expected failure cases.
4. **The muscle-fat split:** which measurements or person facts reduce the $(1,-1)$ uncertainty, and how T1b should be scored.
5. **Mechanics priors:** what published tissue tests support for $\alpha$, $r_n$, $\ell_s$ per region, in proportions; the co-rotated membrane term.
6. **Constructions (I) and (II):** formal equivalence conditions, using 8.1 as the base case.
7. **Training the diffusion head by a sample score:** consistency, variance, cost at our scale.
8. **Covariance identifiability** for the scan types we hold.
9. **From fit to rig:** joint axes and ranges from fitted bone geometry (articular surfaces), completion of muscle attachments, and the minimal tissue model that passes T8.
10. **Stylisation policies:** a parametrised family and how to infer the policy.
11. **Verification** of section 15 from primary sources.

## 17. What to send back

The clearance family and posterior approximation (2, 3); the layer parametrisation (1); the scoring of T1b (4); mechanics priors (5); the formal comparison of constructions (6); the diffusion objective (7); the rig requirements (9); a position on each item of section 13; verified accounts of the cited works.
