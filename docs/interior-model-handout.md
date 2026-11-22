# Interior model: mathematical construction and research handout (version 4)

**Status: a proposal, not a built system.** Nothing here has been trained or tested. Version 4 follows five rounds of outside review, a census of the data we can obtain, and a report on skin-attachment mechanics supplied by the user. Section 1 records what changed; section 12 holds the open disagreements; section 14 lists what nobody has verified.

## 0. Purpose

We fit an anatomy atlas (bones, muscles) inside a generated 3D character by least squares and convex optimisation. The optimiser knows hard rules (bones inside the body, seams closed, left and right alike) but little about where tissue lies under a given surface. We want a learned model of how bodies are built, **with every part in the context of all the others**, with calibrated uncertainty, in a form the optimiser uses directly.

The user's requirements: cross-sections ("critical slices") at important places are the working currency; all predictions for one person share context; confident predictions guide the optimiser; a transformer with diffusion is the preferred family; the input character is clothed, with cloth kept close to the body where the user can arrange it.

## 1. What changed from version 3, and why

| # | Point | Verdict | Where it lands |
|---|---|---|---|
| 1 | Version 3 used the outer surface twice: once to predict $p(z \mid \mathcal O)$ and again as a likelihood $p(\mathcal O \mid z)$ in the optimiser. A tight sleeve would narrow the arm twice and the result would be overconfident. | **Accepted: the main error of version 3.** | Section 6: one construction, the exterior used once. |
| 2 | The body's state was bones only. A thicker thigh could then only be explained by a larger femur. | **Accepted.** | Section 3: layered state (bone, deep fascia, skin). |
| 3 | "How much tissue is there" and "how tissue has moved" are different. So are "skin to first lean tissue", "skin to fascia" and "fat thickness". | **Accepted.** | Sections 3 and 4. |
| 4 | Clothing claims overstated: a hem step is an edge offset, not fabric thickness; a flat panel may cover flat skin or a hollow of any depth; the curvature bound was stated backwards; a waistband can press skin in at zero gap. | **Accepted, including the reversed bound (my error).** | Section 5. |
| 5 | Two outfits giving one skeleton shows repeatability, not accuracy. The strong test puts two outfits on one known body. | **Accepted.** | Section 10. |
| 6 | Convexity holds only with fixed normals and correspondences. | **Accepted.** | Section 7.4. |
| 7 | A diffusion head cannot be trained "through $H_p$" when the clean latent is unknown. A direction's variance can be learned while its correlation with another cannot. | **Accepted.** | Sections 9.2 and 9.3. |
| 8 | A surface stylisation warp does not determine how bone follows it. | **Accepted**: stylisation becomes an explicit policy (section 12.3). | |
| 9 | The handout asked for keypoints on renders of CT skin and also forbade rendering faces. | **A real contradiction; the user's decision.** | Section 13, question U3. |
| 10 | Skin-attachment mechanics (the user's report): an interface energy between skin and deeper tissue, a membrane energy in the skin, and a transfer length. | **Adopted in reduced form**, with its own warning that the parameters are not identifiable from one static scan. | Section 4. |
| 11 | Start with one thigh region. | **Accepted.** | Section 10. |

Kept from version 3: one shared anatomical state per person; every slice cut from it; planes from the outer surface only; training against measurements, not against our own fits; the three-part oracle test; Gaussian, mixture and diffusion compared on equal terms; body first because faces are scarce.

## 2. Objects and notation

**Person and size.** Person or figure $p$; observed outer surface $\mathcal O_p$ with material labels (skin, cloth, hair, prop); person facts $g_p$ (sex, age, size, each with a known/unknown flag). Lengths are proportions of a size $S_p$ measurable from the outer surface.

**Atlas.** A template with bones, muscles, and three nested surfaces per region: bone $\Gamma^{B}$, deep fascia $\Gamma^{F}$ (the envelope of the muscle compartment), skin $\Gamma^{S}$. Points of the template skin are $X \in \Gamma^{S}_0$ with unit normal $n(X)$.

**Planes and readouts.** Planes $\Pi_k$ from the outer surface only. On a plane and along each ray: the observed radius $\rho^{\text{obs}}$; the model's current skin radius; the fascia and bone radii; and three distinct depths: skin to fascia $d^{SF}$, skin to bone $d^{SB}$, and adipose thickness
$$ d^{\text{fat}} = \int_{\text{skin}}^{\text{deep fascia}} P(\text{adipose at } x \mid \text{CT})\, d\ell , $$
the length of the ray occupied by fat between skin and fascia. Fat deeper than the fascia is a separate compartment.

## 3. The state: three parts

$$ z = \big( z^{\text{ref}},\; u,\; c \big). $$

### 3.1 Reference anatomy $z^{\text{ref}}$: how this person is built

- **Bone**: the deformation parameters of the fit (deformation graph for the skull, jointed bones with size and stretches for the body), in canonical form (global pose and size removed, minimum-norm representative, area-weighted displacement norm).
- **Soft-tissue thickness fields on the template skin**: lean thickness $t^{L}(X) \ge 0$ from bone-carried support to fascia and subcutaneous thickness $t^{F}(X) \ge 0$ from fascia to skin, so that
  $$ \Gamma^{F}_{\text{ref}}(X) = b(X; z^{B}) + t^{L}(X)\, n(X), \qquad \Gamma^{S}_{\text{ref}}(X) = \Gamma^{F}_{\text{ref}}(X) + t^{F}(X)\, n(X), $$
  with $b(X; z^{B})$ the point the bone state carries. Two people with the same femur and different thighs differ in $t^{L}, t^{F}$, not in $z^{B}$.
- **Reduced coordinates**: $z^{\text{ref}} = \bar z + W a + \eta$ with one joint basis over bone parameters and (log) thickness fields, so that their covariance is learned: a thick thigh informs the femur only as far as the population says it does. The surface alone often cannot tell more muscle from more fat; that ambiguity is a direction of $a$ with large conditional variance and must survive into the result.

The reference configuration is **the body as scanned** (supine, unclothed) for training, and **the character's own unclothed, unloaded body** at inference. A static scan cannot separate a person's unloaded shape from pre-stress; the reference is a convention, not a stress-free state.

### 3.2 Tissue displacement $u$: how the tissue has moved from the reference

$u_s(X)$ is the displacement of the current skin from the reference skin and $u_d(X)$ that of the deeper support (fascia), under posture, load or a tight band. With the jump $g_\Gamma = u_s - u_d$, the user's report gives the reduced interface energy
$$ E_\Gamma = \frac12 \int_{\Gamma_0} \Big[ \kappa_t \big\| (I - n n^\top)\, g_\Gamma \big\|^2 + \kappa_n \big(n^\top g_\Gamma\big)^2 \Big]\, dA , $$
separate costs for sliding and for normal separation, and a membrane energy for the skin itself,
$$ E_S = \frac12 \int_{\Gamma_0} S\, \| \nabla_\Gamma u_s \|^2\, dA . $$
For a strip pulled at one end the displacement decays as $u(x) = u(0)\, e^{-x/\ell_s}$ with transfer length
$$ \ell_s = \sqrt{S / \kappa_t} . $$
Three properties matter here. Skin and deeper tissue can move together at no cost. The quadratic forms are convex in $u$ for fixed normals. And a displacement profile identifies only the ratio $S/\kappa_t$, never the two stiffnesses separately; a single static scan identifies neither. The report's numbers are illustrative calculations, not measurements, and are **not** used.

For fat under compression, a tracked patch of reference thickness $h_0$ and area ratio $J_A$ satisfies $h\, J_A \approx h_0$ (near-incompressibility): a band that thins the layer by 20% spreads it by 25%. This is enforced, if at all, as a volume condition over a 3D region; thickness on a fixed slice is read off afterwards, since tissue also moves through the plane.

**Scope in the pilot.** On CT, $u = 0$ by definition of the reference. $u$ is non-zero only where the input shows deformation that the reference cannot explain: a groove at an elastic edge, an indentation. There $u$ is regularised by $E_\Gamma + E_S$ with the single unknown $\ell_s$ per region, which stays an uncertain quantity until paired loading data exist (12.2).

### 3.3 Garment clearance $c$: the gap between current skin and cloth

$$ q = \Gamma^{S}_{\text{ref}}(X) + u_s(X) + c(X)\, \nu, \qquad c(X) \ge 0 , $$
for an observed cloth point $q$ with correspondence $X$ and outward direction $\nu$. A waistband pressing the skin in is $u_s < 0$ with $c \approx 0$; a loose waistband is $u_s = 0$ with $c > 0$. One gap variable could not express both.

## 4. Mechanics adopted from the report, and what is left out

Adopted: the three-way split of section 3; the reduced interface and membrane energies as the regulariser for $u$; the transfer length as the physical meaning of "how far along the skin a local change is felt"; the statement that anchors follow the tissue they attach to (fascia following muscle), not the nearest bone; the warning against counting one stiffness twice (membrane and interface), and against reading stiffness from shape alone.

Left out of the pilot: discrete cable networks and slack lengths, hyperelastic matrix and collagen laws, contact with friction, dynamics, viscoelasticity, damage. They need loading or motion data that we do not have; they become relevant when the figure is animated (the project's later physics stage), not for inferring a static interior.

Consequence for the attention design: the shared skin surface already carries neighbour-to-neighbour mechanical coupling through $E_S$. Whole-person attention is for the other kind of context, how a visible region informs reference anatomy elsewhere. The report's example transfer lengths are not an attention radius.

## 5. Clothing: what the observed cloth does and does not say

### 5.1 Observation types

Bare skin: $c = 0$, an equality up to surface noise. Cloth: $c \ge 0$, one-sided. Hair, props: $c \ge 0$ with no useful upper limit.

### 5.2 What can be estimated from the cloth alone, stated carefully

- **Edge offset.** The step from skin to cloth across a garment's edge is an *effective edge offset* $\tau_e$: fabric thickness plus whatever a rolled hem, seam, gap or indentation adds. It is a feature of the garment, not a measurement of fabric thickness.
- **Bridging radius.** If tight cloth lies near a closing of the body by a ball of radius $R$, the cloth's concave curvature satisfies $\kappa_{\max} \le 1/R$. So the largest concave curvature observed on the cloth gives a **lower** bound on $1/R$, an upper bound on $R$. (Version 3 stated this the wrong way round.)
- **Bridged or not.** A flat or ruled panel may lie on flat skin or span a hollow of any depth. The cloth alone does not decide it. Evidence comes from elsewhere: the mirror side, the garment's edges, bare skin nearby, and the population prior over $z^{\text{ref}}$.

### 5.3 The clearance distribution

$p(c \mid \text{cloth features})$ with features $\tau_e$, local curvature, distance to the garment's edge and material. It is **calibrated on paired body and garment geometry**, never assumed: (a) the user's own characters where the unclothed body under an outfit is known (question U1); (b) simulated garments on CT bodies; (c) an independent cloth-simulation dataset with varying tightness, fabric and body shape, so that the model is not only tested on its own simulator's assumptions (candidate named in review: CLOTH3D; unverified).

### 5.4 Which skin is predicted

Both, and they are different variables: the reference skin $\Gamma^{S}_{\text{ref}}$ (the unclothed, unloaded body, which is what the anatomy belongs to) and the current skin $\Gamma^{S}_{\text{ref}} + u_s$ (the body as worn, which is what the cloth touches).

### 5.5 What the user's arrangement buys

Cloth kept close to the body makes $c$ small and tightly distributed on convex regions, which is nearly an equality there. It does not remove bridging over hollows, compression at elastic edges, or anything about hair. If the outfit is fitted onto a known body mesh and that mesh is kept, clothing stops being inference: $c$ and the contact regions are known.

## 6. Inference: the exterior is used once

Two coherent constructions; version 4 chooses the first and keeps the second as the comparison.

**(I) Prior plus one likelihood (chosen).** Learn the population's anatomy $p_\vartheta(z^{\text{ref}} \mid g)$ jointly over bone and soft tissue, and apply the character's surface once, in the optimiser:
$$ \hat z = \arg\min_z \; \underbrace{-\log p_\vartheta(z^{\text{ref}} \mid g)}_{\text{learned prior}} \;+\; \underbrace{-\log p(\mathcal O \mid z)}_{\text{surface, once}} \;+\; \underbrace{E_\Gamma(u) + E_S(u)}_{\text{mechanics}} \;+\; \text{(seams, joints)} \quad \text{s.t. hard rules}. $$
Context comes from the prior's covariance: seeing a bare forearm moves the estimate of the covered thigh because the population says they vary together. Reasons for choosing it: the surface geometry is trustworthy on any body, including a stylised one, while a network's guess is not; clothing inequalities live naturally in the likelihood; it is how our fit already works.

**(II) Conditional predictor plus feasibility.** Learn $p_\vartheta(z^{\text{ref}} \mid \mathcal O, g)$ and use the optimiser only for hard rules and for genuinely new observations (an artist's correction given afterwards). The reviewer prefers this for the pilot.

A conditional network is still useful under (I) as an initialiser or proposal for the solver; its output is then not multiplied into the objective.

**What the transformer-diffusion model is, under (I).** A generative prior over anatomies: tokens for the person facts, for global coefficients $a$, and structure-query tokens anchored on the atlas; attention with the relations of version 3 (same segment, adjacent, mirror, same garment). Sampling from the posterior is either the convex fit with a Gaussian or Huber form of the prior, or guided sampling with the surface likelihood (posterior sampling with a diffusion prior; open, 12.4).

## 7. The optimiser's terms

### 7.1 Prior

Three kinds of direction: gauge (removed); the learned subspace, penalised by $\psi(\| \Sigma_a^{-1/2}(a - \mu_a) \|)$; the remainder, which keeps the fit's existing stiffness. The penalty $\psi$ is Huber: for a coordinate $x$ in prior standard deviations with the surface asking for $k$ at relative precision $\lambda$, a quadratic prior gives $x^\ast = \lambda k / (1 + \lambda)$ (a fixed fraction lost), a Huber prior with transition $c_H$ gives $x^\ast = k - c_H/\lambda$ beyond it (a bounded pull). $c_H$ is estimated on validation data. A result in the tail is feasible and unusual; its accuracy there is unknown.

### 7.2 Surface

$$ -\log p(\mathcal O \mid z) = \sum_{\text{skin}} \frac{\| q - (\Gamma^{S}_{\text{ref}} + u_s)(X_q) \|^2}{2\sigma_{\text{surf}}^2} \;+\; \sum_{\text{cloth}} \varphi\big(c(X_q)\big) \;+\; \sum_{\text{hair, prop}} \iota_{\ge 0}\big(c(X_q)\big), $$
with $\varphi = -\log p(c \mid \text{features})$ from 5.3, required to be convex and infinite for $c < 0$.

### 7.3 Hard rules

Bone inside fascia inside reference skin ($t^{L} \ge 0$, $t^{F} \ge 0$ by construction); current skin inside or on the cloth ($c \ge 0$); seams and joints as now.

### 7.4 Convexity

With correspondences $X_q$ and normals fixed, every term is convex in the variables. Reassigning correspondences or moving the surfaces the rules refer to changes the problem, so the solver is successive convex steps (as now), and feasibility is reported on the final meshes, not only in the last linearised step.

### 7.5 Bounds and fixed points

No learned bound is added on top of the prior unless it improves held-out fits; equality only for observed or artist-fixed points.

## 8. Symmetries and inputs

Unchanged from version 3: planes and frames from the outer surface only; mirror as one condition on the state; roll landmark kept, no Fourier-phase canonicalisation; relative plane geometry $d_{ij} = R_i^\top(o_j - o_i)/S$, $Q_{ij} = R_i^\top R_j$ in attention where planes are tokens (construction II and the amortised initialiser).

## 9. Training

### 9.1 Targets are measurements

Per CT person: skin surface points (threshold); bone surface points (bone is the highest-contrast tissue and is found from intensity even without per-bone labels); adipose tissue from its intensity range (a standard convention; the exact range is a cited value to be confirmed, 14); fascia where it is visible or labelled, with its uncertainty recorded where it is not; named-ray depths $d^{SB}$, $d^{SF}$, $d^{\text{fat}}$; regional fat area on a slice where the fascia is ambiguous.

### 9.2 Gaussian head: likelihood in measurement space

With $y_p$ the stacked measurements, $H_p$ their sensitivity to $a$ around a reference fit and $R_p$ their noise,
$$ y_p \mid g_p \sim \mathcal N\big( y^0_p + H_p(\mu_a - a^0_p),\; H_p \Sigma_a H_p^\top + R_p \big). $$
Identifiability of the covariance is decided by which symmetric perturbations $\Delta$ satisfy $H_p \Delta H_p^\top = 0$ for all scans: if half the scans measure only $a_1$ and half only $a_2$, both variances are learned and their correlation is not. The intersection of null spaces does not reveal this; the condition on $\Delta$ does.

### 9.3 Diffusion or flow head

Where clean latent targets are credible (scans with enough measurements to pin $a$), ordinary denoising on $a$. Otherwise the head is trained and compared by a proper sample score through the measurement function: draw several anatomies, map each through the scan's measurement operator, add modelled measurement noise, and score the samples against the observed measurements with the energy score, normalisation fixed on the training split. This needs a differentiable sampler and its cost is counted.

### 9.4 Point predictions

Squared geometric error is for a point prediction only; never between a random draw and the truth.

### 9.5 Cover

Simulated garments and hair on CT bodies as connected 3D regions, giving $(\mathcal O, c)$ pairs with known skin; plus the calibration sources of 5.3.

## 10. Experiments, in order: the thigh pilot

Region chosen because it has a clear bone boundary, substantial soft tissue, and both tight and loose clothing in practice; the data we hold covers it (pelvis about 580 scans, legs 148; exact census running).

**T0a Measurement.** On reviewed scans: skin, fascia and femur boundaries on prescribed rays, repeated under perturbed landmarks and planes. Gives $R_p$.

**T0b Oracle fit, twice.** (i) Femur fixed at its measured boundary; can the layered state fit the skin and fascia? (ii) All free. If the skin fits only when the femur leaves its measured boundary, the state lacks soft-tissue freedom.

**T0c Compression.** The same with the reduced basis; skin, fascia and bone errors reported separately.

**T1 One body, three observations.** Each held-out body fixed; supply bare skin, tight cover, and bridged or loose cover in turn. Score reference-skin error, femur error, fascia and fat error, and interval coverage. Baseline: a model given material labels only. *The decisive reading:* does changing the clothing on a fixed body change the inferred femur, the hidden skin, or only the uncertainty?

**T2 Indentation.** A prescribed local indentation with known skin displacement on a known thigh. Does the model move skin and thin the fat locally while keeping the femur and the reference anatomy? With a simulated indentation this tests the implementation only; stiffness calibration stays open.

**T3 Constructions.** (I) against (II) against (II) plus a surface penalty, on the same bodies. This tests directly whether using the exterior twice improves accuracy or only narrows uncertainty.

**T4 Heads.** Gaussian, mixture, diffusion over the same state: energy score, coverage, error after the fit.

**T5 Paradigms** (kept from version 3): per-slice prediction (A), shared state with slice supervision only (B), plus off-plane supervision (C), with one frozen basis.

**T6 Outfits on one known body.** Two outfits fitted on the same known mesh, the mesh kept for scoring: accuracy, not only repeatability.

**Then the face**, as a separate pilot: independent CT depth at the upper lip as the acceptance test; the conflicting atlas-depth terms removed from the skull fit; the hand-written philtrum rule removed. The thigh validates the machinery; it says nothing about facial precision, and a thigh fat model cannot say how a facial depth error divides between fat, muscle and bone position.

## 11. Data

Reported by the data session, not re-checked by me: TotalSegmentator CT (1,228 people; 38 full faces, 110 at most; chest about 580, pelvis about 580, legs 148) held privately; open head sets small (HaN-Seg 42; one further set withheld because its source moved to controlled access); MRI sets being fetched; large face collections need an institutional application. Labels: each bone except forearm, hand, shin, foot; about ten named muscles; no skin or fascia labels; fat through intensity or a separately licensed model. The face-model budget is to be based on reviewed, usable faces actually in hand.

CT bodies are supine; characters stand. This shifts soft tissue, not bone, and belongs to $u$ (12.6).

## 12. Disagreements and open positions

**12.1 Construction (I) or (II).** *Reviewer:* (II) for the pilot. *Mine:* (I), for the reasons in section 6. T3 decides. Note for the user: under (I) the learned model is a population prior, and a low-rank Gaussian over a joint skin, fascia and bone basis is a classical statistical shape model. A transformer with diffusion earns its place only if the anatomy distribution is not well described by that (several modes, strong dependence on person facts). T4 measures it.

**12.2 How much mechanics.** *Reviewer:* add reference skin, a reduced displacement field and a separate garment model. *Agreed on structure.* *Mine:* the stiffness ratio per region is not identifiable from anything we hold, and the user's rule forbids setting it by hand. Either it is learned from paired data we do not yet have (the same body bare and compressed), or $u$ is restricted to cases where the surface itself shows deformation and $\ell_s$ carries a wide prior. To settle: what paired data would identify $\ell_s$ per region, and whether cloth-simulation data can stand in.

**12.3 Stylisation.** A stylised character's surface differs from people systematically. How bone follows is a **policy**, not a fact: a wider thigh may be more soft tissue, a broader femur, or both. Policies (bone follows the surface warp fully, partly, or not at all) are explicit options to test on characters with known construction. Paired warps of CT test recovery of the policy that made them, not which policy is right for an independently designed character.

**12.4 Sampling the posterior.** Convex fit around a Gaussian or Huber prior, or guided diffusion sampling with the surface likelihood. The second is exact in principle and costly; whether it is needed depends on T4.

**12.5 Fixed points.** The hand-written philtrum rule compensated for terms that pulled the wrong way. First remove those terms and apply a prior of the right strength; add firm rules only if still needed.

**12.6 Supine to standing.** Treat posture as a known load producing $u$, or condition the prior on posture. No data on hand distinguishes them.

## 13. Decisions needed from the user

- **U1.** Is the unclothed body mesh under a character's outfit available, or can outfits be fitted onto a known body? If yes, clearance is known rather than inferred, and T6 becomes possible.
- **U2.** Which skin is the deliverable where they differ: the unclothed reference (default here) or the body as worn?
- **U3.** May faces from CT be rendered inside the private job, with nothing leaving it, so that the same keypoint detector runs on CT and on the character? Otherwise a surface-native landmark method is needed for the face pilot.
- **U4.** Does CT of real, anonymised people fit the consent rule that excluded a model built from one person's photo?

## 14. Not verified

The reference world model's architecture; every work cited by the reviewer (surface-to-CT torso generation; head-to-skull regression errors; relative-transform attention; segmentation-conditioned CT generation; a medical volume diffusion model; signed-distance shape representations; the cloth-simulation dataset; thigh CT studies using fascia to separate subcutaneous from intermuscular fat; the thigh-compression and gluteal-fat mechanics studies); the references of the skin-attachment report (I read the report, not its sources); the data census; the intensity range used to find fat in CT; that keypoints and minimal-area planes work on clinical CT; that garment edges on generated characters are clean enough to measure an edge offset.

**Constraints on any proposal:** no hand-set decision numbers; proportions, not millimetres; a pilot within a few dollars; CT of real people stays private.

## 15. Research questions for the next round

1. **The layered state.** Is the normal-offset parametrisation of fascia and skin (3.1) adequate where the limb is not star-shaped about its bone (inner thigh, armpit, trunk)? Propose the alternative if not.
2. **Identifiability of fat against muscle from the exterior**, given bone: which measurements or person facts reduce it, and what the posterior should look like.
3. **The clearance model** $p(c \mid \text{features})$: a family that is convex in $c$ after the negative logarithm, supports $c \ge 0$, and is calibrated on paired data; the corrected estimator of $R$ from curvature and its bias.
4. **Skin under a bridged panel:** with the corrected claims of 5.2, how wide is the posterior of the hollow's depth given the edges, the mirror side and the prior?
5. **Mechanics as a prior for $u$:** with only $\ell_s$ identifiable in principle, what prior over $\ell_s$ follows from the cited tissue studies, expressed in proportions, and how sensitive are T1 and T2 to it?
6. **Constructions (I) and (II):** show formally when they coincide, and quantify the overconfidence of using both.
7. **Training the diffusion head by a sample score:** consistency, variance, and cost for a few dozen coefficients and a few hundred people.
8. **Covariance identifiability:** apply the $H_p \Delta H_p^\top = 0$ criterion to the scan types we hold and list what cannot be learned.
9. **Stylisation policies:** a parametrised family between "bone follows the surface" and "bone stays human", and a way to infer the policy from characters with known construction.
10. **Volume preservation of fat on a fixed slice:** what a slice can and cannot say about a 3D redistribution.
11. **Verification** of section 14 from primary sources.

## 16. What to send back

The clearance family and corrected estimators (3, 4); the layered-state parametrisation (1) and the fat-muscle posterior (2); the formal comparison of the two constructions (6); the diffusion training objective (7); the list of unlearnable covariances (8); a position on each item of section 12; verified accounts of the cited works.
