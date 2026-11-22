# Interior model: mathematical construction and research handout (version 3)

**Status: a proposal, not a built system.** Nothing here has been trained or tested. Version 3 follows three rounds of outside review and one census of the data actually available. Section 1 records what changed, section 4 is new (clothing), section 11 holds the open disagreements, section 13 lists what nobody has verified.

## 0. Purpose

We fit an anatomy atlas (bones, muscles) inside a generated 3D character by least squares and convex optimisation. The optimiser knows hard rules (bones inside the body, seams closed, left and right alike) but little about where tissue lies under a given surface. Sparse hand-fed numbers lost against the other terms: the bone under the upper lip stayed 3 mm too deep through five runs, until a rule was written by hand. We want a learned model that, from the outer surface alone, predicts the interior **with every part in the context of all the others**, with calibrated uncertainty, in a form the optimiser uses directly.

The user's requirements: cross-sections ("critical slices") at important places are the working currency; all predictions for one person share context (neighbouring slices, overall appearance, age, the 3D structure around each slice); confident predictions guide the optimiser; a transformer with diffusion is the preferred family; **the input character wears clothes**, kept close to the body where the user can arrange it, and that part must be handled carefully.

## 1. What changed, and why

### 1.1 From version 2 to version 3 (third review round, data census, clothing)

| # | Point | Verdict | Where it lands |
|---|---|---|---|
| 1 | Training targets are atlas fits to CT. Where the labels leave a region underdetermined, our own stiffness and shape rules supply the answer, and the model learns our bias, not anatomy. | **Accepted: the main risk.** We know the bias exists (the philtrum). | Sections 8.1, 9 (Test 0). Targets are measurements, not fitted parameters. |
| 2 | With global shape coefficients, "missing labels" are not missing coordinates: a partial view constrains combinations of coefficients. | **Accepted: an error in version 2.** | Section 8.2: likelihood in measurement space. |
| 3 | "Score a draw or the mean" mixes two objectives; squared error on a random draw punishes spread and collapses the distribution. | **Accepted: an error in version 2.** | Section 8.3. |
| 4 | Runs B and C were confounded: a basis learned from full 3D fits already gives B off-plane information. | **Accepted.** | Section 9, Test 2. |
| 5 | A covariance pseudo-inverse gives zero penalty in directions the data never saw; a singular Gaussian gives zero probability there. With about 1,400 graph parameters and a few hundred people, most directions are unseen. | **Accepted.** | Section 7.1: three kinds of direction, each with its own treatment. |
| 6 | A Gaussian prior gives back a fixed fraction of whatever the surface asks; a Huber-type penalty pulls back by a bounded amount. | **Accepted** as the answer to version 2's question on tails. | Section 7.2. |
| 7 | A bound derived from the same model as the prior counts that model twice. | **Accepted**, with a note on why our hand rule was needed (section 11.4). | Section 7.3. |
| 8 | A global coefficient has no single place on the body; do not split coefficients by structure. | **Accepted.** | Section 6.1. |
| 9 | Fix the gauge by removing global pose and scale, area-weighting displacements, and taking the minimum-norm representative. | **Accepted.** | Section 5.2. |
| 10 | Split the oracle test into measurement, full fit, and compression. | **Accepted.** | Section 9, Test 0. |
| 11 | **Data census:** the CT set we hold has 1,228 people but only 38 full faces (110 at most). Open sets could bring faces to about 130 to 200. Large pools need an institution. | Fact (reported by the data session, not re-checked by me). | Section 10: body first; the face model is small and statistical for now. |
| 12 | **Clothing** (the user): characters are clothed; cloth is kept close to the body where possible. | New requirement. | Section 4. |
| 13 | Stylisation is systematic, not random (my addition). | Open. | Section 11.2. |

### 1.2 Kept from version 2

One shared anatomical state per person, in the optimiser's own deformation parameters; every slice cut from it; planes from the outer surface only; relative plane geometry in attention; Gaussian, mixture and diffusion heads compared on equal terms; slices as inputs, supervision and readouts; no autoregressive next-slice objective; observed or artist-approved slices as extra conditions.

## 2. Objects and notation

**Person and size.** Person or figure $p$; observed outer surface $\mathcal O_p$ (what a camera or scanner sees: skin, cloth, hair); true skin $\mathcal S_p$. All lengths are proportions of a size $S_p$ measurable from the outer surface on CT and on the figure alike (open: 12.3).

**Materials.** Each point $q \in \mathcal O_p$ has a label $m(q) \in \{\text{skin}, \text{cloth}, \text{hair}, \text{prop}\}$ (for the figure: from the segmentation and labelling steps; for CT: skin, unless cover is simulated).

**Atlas and state.** A template with vertices $x \in \mathcal A$, each in a structure $c(x)$ (a bone, a muscle, a cartilage, **the skin**). An individual is a parameter vector $z$:

- deformation graph (skull): $D_z(x) = x + \sum_j w_j(x)\,[(A_j - I)(x - g_j) + t_j]$, so $D_z(x) = x + B(x)\, z$, linear in $z$;
- jointed bones (body): $T_b(x) = c_b + e^{s_b} \mathrm{Rot}(r_b)\big(I + (e^{e_b}-1) a_b a_b^\top + (e^{f_b}-1) a'_b a'^{\top}_b\big)(x - c_b) + t_b$, linearised at each step.

**The shared anatomical state is $z$**, and it includes the skin: $\mathcal S(z) = D_z(\mathcal A_{\text{skin}})$ is the model's own statement of where the bare body surface is.

**Planes and readouts.** Plane $\Pi_k$ with origin $o_k$ and frame $R_k = [e_1^k, e_2^k, n_k]$, from the outer surface only. On it: the observed outline $\rho^{\text{obs}}_k(\theta_j)$, the model's skin outline $\rho^{\text{skin}}_k(\theta_j; z)$, the section of each structure $\Omega_{k,c}(z)$, and depths along each ray (skin to lean tissue $d_k$, skin to bone $\beta_k$) with a validity flag where a ray meets no bone.

**Person facts.** $g_p = (\text{sex}, \text{age}, \text{size}, \dots)$, each with a known/unknown flag.

## 3. Inputs

### 3.1 No interior information in the inputs

The pipeline that produces inputs for a CT person is the figure's pipeline run on the CT's outer surface: keypoints on renders, axes from keypoints, planes from the surface. Bone labels make targets only. Bone-derived planes are an oracle condition.

### 3.2 Three sources of context

1. **Whole-person appearance:** the outer geometry as a point set $\{(q_i, \nu_i, m_i)\}$ (position, normal, material) in proportions, plus $g$.
2. **Local surface around each plane:** nearby points in the plane's frame with signed height, the outline $\rho^{\text{obs}}_k$, materials around the ring, tilt.
3. **Other planes**, through attention with relative geometry (section 6.2); and observed or artist-approved interior slices when supplied.

### 3.3 Symmetries

Roll of a plane frame: $\rho(\theta) \mapsto \rho(\theta - \delta)$; the anatomical roll landmark is kept, and slices are not canonicalised by the phase of the first Fourier term. Mirror: one condition on the state, $p(\mathsf M z \mid \mathsf M \mathcal O, g) = p(z \mid \mathcal O, g)$. Size: removed by proportions.

## 4. Clothing: the observed surface is not the skin

This section is new. It is where a clothed character differs from every CT body, and it interacts with the hard rules of the fit: "bones and muscles inside the body" must mean inside the **skin**, which under cloth is not observed.

### 4.1 The observation model

For an outer-surface point $q$ with outward direction $\nu$,
$$ q = s(q) + \delta(q)\, \nu, \qquad s(q) \in \mathcal S, \quad \delta(q) \ge 0 , $$
where $\delta$ is the gap between skin and the outside of the cover.

- Bare skin: $\delta = 0$ up to surface noise, $q - s \sim \mathcal N(0, \sigma_{\text{surf}}^2)$.
- Cloth: $\delta \ge 0$ always (cloth is never inside the body). This is a **one-sided** observation: the skin is at or under the cloth, never outside it.
- Hair and props: $\delta \ge 0$ with no useful upper limit; they say almost nothing about the skin beneath.

So a clothed character gives, for the skin, equalities on bare patches and inequalities elsewhere:
$$ \rho^{\text{skin}}_k(\theta; z) = \rho^{\text{obs}}_k(\theta) \ \text{(skin)}, \qquad \rho^{\text{skin}}_k(\theta; z) \le \rho^{\text{obs}}_k(\theta) \ \text{(cloth, hair, prop)}. $$

### 4.2 What "close to the body" buys, and its three components

For tight cloth write
$$ \delta(q) = \tau_{\text{fabric}} + \delta_{\text{bridge}}(q) + \delta_{\text{fold}}(q) . $$

- $\tau_{\text{fabric}}$: the fabric's own thickness, roughly constant over a garment. **Measurable on the figure itself**: wherever a garment ends on bare skin (hem, neckline, cuff) the step from skin to cloth across the edge is $\tau_{\text{fabric}}$ plus noise. Collect these steps per garment; their distribution is that garment's thickness prior. No number is assumed.
- $\delta_{\text{bridge}}$: cloth under tension does not follow concavities (between the breasts, the small of the back, the crotch, the armpit, under the buttocks). There the gap equals the depth of the concavity that the cloth spans. Geometrically, tight cloth lies near the surface obtained by rolling a ball of some radius $R$ over the body (a morphological closing), so
  $$ \delta_{\text{bridge}}(q) \approx \mathrm{close}_R(\mathcal S)(q) - \mathcal S(q) \ge 0 , $$
  zero on convex regions and positive in hollows. $R$ is a property of the garment's tension; it is estimable from the observed cloth alone, since a closed surface cannot have concavities sharper than $1/R$: the largest concave curvature present on the cloth bounds $1/R$ from above.
- $\delta_{\text{fold}}$: wrinkles and loose panels; non-negative, spatially sparse, large where present.

Two consequences. First, on convex, tight regions (thigh front, upper arm, chest above the bust) the cloth outline is the skin outline plus a nearly constant, measurable offset: almost as good as bare skin. Second, in spanned hollows the cloth is uninformative about depth: the skin could be anywhere between the cloth and a deep hollow. The model must know which case it is in. That is decidable from the cloth's own geometry: a region where the cloth is flat or ruled while the body, by symmetry or by the neighbouring bare skin, should be concave, is a bridged region.

### 4.3 Compression

Tight cloth also pushes soft tissue in (waistbands, straps, elastic edges), so the skin under cloth is not the skin the same body would have unclothed. The atlas and CT statistics describe uncompressed bodies. For the interior this matters little for bone and much for fat. We treat compression as part of $\delta$ being small rather than model it; a visible groove at an elastic edge is a local reduction of fat depth, not of bone depth (open: 12.9).

### 4.4 How it enters the model

- **Inputs.** Each surface token carries its material, and for cloth the garment's estimated $\tau_{\text{fabric}}$ distribution, its estimated $R$, and a local bridging indicator $\kappa(q)$ computed from the cloth's curvature relative to its surroundings.
- **Outputs.** The skin is part of the state, so the model predicts the bare body under the clothes together with the interior, and they are consistent by construction.
- **Likelihood of the outer surface given the state** (used in training with simulated cover, and in the optimiser):
  $$ -\log p(\mathcal O \mid z) = \sum_{q:\,\text{skin}} \frac{\|q - s_z(q)\|^2}{2\sigma_{\text{surf}}^2} \;+\; \sum_{q:\,\text{cloth}} \phi\big(\delta_z(q);\ \tau_{\text{fabric}}, \kappa(q)\big) \;+\; \sum_{q:\,\text{hair, prop}} \iota_{\ge 0}\big(\delta_z(q)\big) , $$
  with $\delta_z(q) = \langle q - s_z(q), \nu \rangle$, $\iota_{\ge 0}$ the indicator of $\delta \ge 0$, and $\phi$ a convex one-sided penalty: infinite for $\delta < 0$, minimal near $\delta = \tau_{\text{fabric}}$ on convex tight regions, and flat (uninformative) for $\delta > \tau_{\text{fabric}}$ where $\kappa$ says the cloth is bridging. Each term is convex in $\delta$, and $\delta_z$ is linear in $z$ for the deformation graph, so the whole expression stays inside a convex fit.
- **Hard rules.** "Inside the body" becomes: bones and muscles inside $\mathcal S(z)$, the model's own skin, with the skin itself constrained by 4.1. This replaces testing against the observed surface, which under hair and loose cloth is not the body (the current pipeline handles that by hand-built rules: limb pieces, boxes from a decision model, the mirror image).

### 4.5 Borrowing across the body

Under cover the skin is inferred from: (a) bare skin elsewhere on the same person; (b) the mirror side if it is less covered; (c) continuity across a garment's edge, where skin and its slope are seen on one side; (d) the population prior over $z$; (e) the interior itself, since skin, fat and bone are jointly distributed. All of these are what the shared state and the attention relations of section 6 are for; nothing extra is needed beyond the materials on the tokens.

### 4.6 Training for clothing with unclothed CT

CT bodies are bare (or under a thin gown and a blanket, which are separate from the body and can be removed as low-density sheets; unverified, 13). Clothed inputs must be simulated:

1. choose a garment region as a connected patch of the CT skin (a shorts, top, sleeve or stocking pattern), not independent arcs;
2. build the cover surface as $\mathrm{close}_R(\mathcal S)$ over the patch, offset outward by $\tau_{\text{fabric}}$, with $R$ and $\tau$ drawn from distributions fitted to real garments on real characters (edge steps for $\tau$, maximal concave curvature for $R$);
3. optionally add folds as sparse outward bumps along lines;
4. label the patch as cloth and hand the model only the outer surface.

The true skin is known, so the skin-under-cloth error is measured exactly. Hair is simulated as thick, irregular cover with no upper bound on $\delta$.

### 4.7 Tests specific to clothing

- **Synthetic:** skin-under-cloth error and interior error as functions of covered fraction, $R$ and $\tau$; separately on convex regions and in hollows.
- **Outfit invariance on real characters:** the same character generated in two different outfits should yield the same skeleton and the same bare-skin estimate where both are covered. No ground truth is needed; the disagreement between the two fits is the error measure. A bare-limbed and a covered version of the same limb is the cleanest case.
- **Edge consistency:** the predicted skin under a garment must meet the observed skin at the garment's edge in position and slope.
- **Ablation:** materials only, against materials plus $\tau$, $R$ and $\kappa$.

### 4.8 What the user controls

Keeping cloth close to the body shrinks $\delta_{\text{fold}}$ and makes $\tau_{\text{fabric}}$ the dominant term, which is the favourable case of 4.2. What it cannot remove is bridging over hollows and all of the hair. A character whose important regions (face, hands, the limbs' convex surfaces) are bare or tightly covered gives the model equalities or near-equalities there; skirts, capes, long hair over the back and loose sleeves give only inequalities.

## 5. The model's output: a distribution over the shared state

### 5.1 The distribution

$$ p_\vartheta\big(z \mid \mathcal O_p,\, g_p\big). $$
One draw is one complete body: skin under the clothes, bones, muscles. Every slice or ray requested later is cut from the same draw.

### 5.2 Canonical form and reduced coordinates

Before any statistics: remove the global pose and size from $z$ in one fixed way; among parameter vectors giving the same vertex positions take the minimum-norm one; measure a displacement by its area-weighted vertex norm, $\|u\|_{\mathcal A}^2 = \sum_x \omega_x \|B(x) u\|^2$ with $\omega_x$ the vertex areas, so that dense tessellation does not dominate. Then
$$ z = \bar z + W a + \eta, \qquad a \in \mathbb R^r , $$
with $W$ the leading directions in the $\|\cdot\|_{\mathcal A}$ geometry and $\eta$ what the basis cannot express. Optional regional residual modes (for example for the upper jaw) are added only if Test 0 shows the global basis loses them.

## 6. The network

### 6.1 Tokens

Person token; surface-patch tokens from a set encoder over the outer point set with materials; plane tokens; **global coefficient tokens** for the noisy $a$ (not assigned to structures); **structure-query tokens** anchored at each structure's place on the reference atlas, which read out local quantities (depths, key lines) for supervision.

### 6.2 Attention with relative geometry

$$ A_{ij} = \frac{\langle W_Q h_i, W_K h_j\rangle}{\sqrt d} + b_{\mathrm{rel}(i,j)}\big(\phi(d_{ij}, Q_{ij})\big), \qquad d_{ij} = R_i^\top (o_j - o_i)/S, \quad Q_{ij} = R_i^\top R_j , $$
with relations: same segment, adjacent segment, mirror partner, **same garment**, and other. The garment relation lets covered points share their $\tau$ and $R$.

### 6.3 Heads over the same state

Gaussian with low rank plus diagonal, $a = \mu + L\xi + D^{1/2}\epsilon$; a mixture of such; a diffusion or rectified-flow head over $a$. Same encoder, basis, splits and evaluation.

## 7. Into the optimiser

### 7.1 The prior, with three kinds of direction

Split parameter space into: gauge directions (no change of shape; removed by 5.2); the learned subspace $\mathrm{span}(W)$, where the model supplies $\mu_a, \Sigma_a$; and the remainder, which the data never resolved. For the remainder neither "free" nor "forbidden" is right: it keeps the fit's existing stiffness terms, scaled as now. The model's term is
$$ \Psi(z) = \psi\!\Big( \big\| \Sigma_a^{-1/2}\, (W^{+}(z - \bar z) - \mu_a) \big\| \Big), $$
added to the existing objective, with $W^{+}$ the projection onto the basis in the $\|\cdot\|_{\mathcal A}$ geometry.

### 7.2 The tail $\psi$

Take one coordinate: $x$ the upper-jaw displacement in prior standard deviations, $k$ the displacement the surface terms prefer, $\lambda$ their precision relative to the prior. With $\psi(t) = t^2/2$ the solution is $x^\ast = \lambda k/(1+\lambda)$: a fixed fraction of what is asked is lost at every $k$. With a Huber $\psi$ of transition $c$, once $x^\ast > c$ the solution is $x^\ast = k - c/\lambda$: the prior pulls back by a bounded amount. A stylised character lives in that tail. $c$ is estimated on validation data. A fit that satisfies the hard rules and lands in the tail is *feasible and unusual*; its accuracy there is unknown and must be reported as such.

### 7.3 Bounds and fixed points

A learned quantity enters through the prior only. A calibrated bound $|\ell(z) - \mu_\ell| \le \kappa \sigma_\ell$ from the same model is added only where it improves the final fit on held-out CT, since it counts the model twice. Exact equality is for observed or artist-fixed points.

### 7.4 The surface term

The fit's surface data term is the likelihood of section 4.4, replacing the current tests against the observed surface.

### 7.5 Several modes

Fit each mode separately; compare fit cost plus log-probability.

## 8. Training

### 8.1 Targets are measurements

For CT person $p$ the training target is not a fitted $z_p$ but what the scan shows: points on the skin surface; points on bone surfaces (bone is the highest-contrast tissue in CT and is found from intensity even where no per-bone label exists); labelled structure surfaces where labels exist; skin-to-bone depth along prescribed rays. A reference fit $z^0_p$ is used only as the point of linearisation.

### 8.2 Likelihood in measurement space

Let $y_p$ stack person $p$'s measurements. Around $z^0_p$,
$$ y_p \approx y^0_p + H_p\,(a - a^0_p) + \varepsilon_p, \qquad \varepsilon_p \sim \mathcal N(0, R_p), $$
with $H_p$ the sensitivity of the measurements to the coefficients (from $B(x)$ at the measured points) and $R_p$ the measurement covariance (repeat-extraction and label noise). For the Gaussian head,
$$ y_p \mid \mathcal O_p, g_p \;\sim\; \mathcal N\!\big( y^0_p + H_p(\mu_a - a^0_p),\; H_p \Sigma_a H_p^\top + R_p \big). $$
A scan that shows only part of the anatomy constrains only the combinations $H_p a$; a direction $u$ with $H_p u = 0$ for every scan is unidentifiable and stays with the atlas's stiffness. A table of which labels co-occur is a first audit; the null spaces of the $H_p$ are the real answer.

### 8.3 Which loss for which purpose

Squared geometric error for a point prediction (the mean); the likelihood of 8.2 or a proper sample score (energy score) for the distribution; the diffusion or flow objective on $a$ only where clean latent targets are well identified, otherwise on measurements through $H_p$. Never squared error between a random draw and the truth.

### 8.4 Three kinds of "missing"

No label for a structure (handled by 8.2); a ray that meets no bone (an observation, with a validity target); skin hidden by cover (an input condition, section 4).

### 8.5 Supervision at several scales

Points through the volume (inside or outside each structure); points on and near surfaces, both sides; critical planes, random oblique planes and named rays. Weights are inverse measurement variances.

### 8.6 Conditions seen in training

Random dropping of person facts; hidden bands of planes; simulated garments and hair (4.6); with and without observed interior slices.

## 9. Experiments, in order

**Test 0a: can the target be measured?** On reviewed scans, the skin point and the bone boundary on a prescribed ray (upper lip, chin, mid-shin, greater trochanter), repeated under perturbed landmarks and planes. The spread is $R_p$.

**Test 0b: can the full fit recover it?** Fit the full deformation graph with the labels normally available; compare its depth with the independent measurement; repeat with the local bone boundary supplied. Improvement on supplying the boundary means missing evidence; persistent error means graph resolution, correspondence or the ray definition.

**Test 0c: does compression lose it?** The same with the reduced basis. The difference is the cost of the basis.

**Test 1: how much the exterior knows.** Fixed Gaussian head; inputs from local outline only, to all outlines, to oriented surface neighbourhoods, to the whole body, to known age. Held-out gain at each named site.

**Test 2: paradigms**, with one frozen basis learned on training people:

| Run | Interior | Training supervision |
|---|---|---|
| A | joint per-slice predictions (version 1) | critical-slice measurements |
| B | shared state, slices cut from it | critical-slice measurements only |
| C | as B | plus off-plane measurements |

All three are scored after the actual atlas fit, against independent CT measurements on unseen people, including oblique sections. Basis rank and loss weights are chosen on a separate validation set.

**Test 3: heads.** Gaussian, mixture, diffusion; energy score, coverage and width, error after the fit.

**Test 4: clothing** (section 4.7).

**Test 5: the use.** The skull fit with the hand-written philtrum rule removed and the model's prior in its place.

**Test 6: stylisation.** Paired smooth warps of CT skin and bone; then the real figure, judged by its distance into the prior's tail and by the hard rules.

**Confounding.** Known, withheld and within-source-shuffled age.

## 10. Data: what exists, and the order it forces

Reported by the data session; not re-checked by me.

| Source | People | Full faces | Access |
|---|---|---|---|
| TotalSegmentator CT (held, private) | 1,228 | 38 (110 at most) | open, CC BY 4.0 |
| HaN-Seg | 42 | 42 | open, non-commercial, no derivatives |
| PDDCA-Lite | 48 | 48 | open copy exists; terms to be confirmed |
| Head-and-neck cancer collections | about 4,000 | intact | controlled access, institutional application |
| A decedent CT database | about 15,000 | intact | institutional agreement; no redistribution or derivative works |

Regions well covered now: chest about 580, pelvis about 580, legs 148 (from scan-type metadata; an exact census is being run). Labels: every bone separately except forearm, hand, shin and foot; about ten named muscles; no skin label (threshold); fat only through a separately licensed model.

Consequences:

1. **Body first.** The method (surface-only planes, shared state, measurement likelihood, clothing, the link to the optimiser) is tested on trunk, pelvis and thigh, where hundreds of people exist.
2. **The face model is small.** With 130 to 200 faces a low-rank statistical model is the realistic choice; a published estimate quoted to us is 1.5 to 4 mm error at this scale, with learned models ahead of a linear one by about 0.1 mm at 81 scans (unverified). The existing tissue-depth table of 388 people is a check, not training data.
3. **Supine bodies.** CT subjects lie on a table with arms raised or at the sides; soft tissue of the trunk and breasts sits differently from a standing character. This is a domain gap in the skin, not in bone (open: 12.10).
4. **Privacy.** A skin surface rendered from head CT is a recognisable face. Scans stay in private storage, no face is rendered, nothing leaves that storage. Whether such data fits the user's consent rule is the user's decision and is not assumed here.

## 11. Disagreements and open positions

**11.1 Where the shared anatomy lives.** *Review:* the optimiser's deformation graph is a good interface if the full-graph oracle succeeds. *Agreed*, conditional on Test 0b. If it fails with the local boundary supplied, the graph (node placement, weights) is the next thing to fix, before any learning.

**11.2 A non-human figure.** *Review:* heavier tails let the fit leave the human prior. *Mine:* her differences from people are systematic (small jaw, flat nose root, long legs, large head), the same across a style. A low-dimensional *stylisation map* $\sigma$ estimated from the surface (a smooth warp taking a stylised surface to its nearest human-proportioned one), with the human prior applied to the de-stylised body and the interior carried back through $\sigma$, may fit better than tails alone: $p(z \mid \mathcal O) = p_{\text{human}}(\sigma^{-1} z \mid \sigma^{-1}\mathcal O)$. Open: whether the interior should follow the surface warp affinely (bone stretches with the face) or only partly (a larger head on a normal skull thickness). Neither position has been worked through.

**11.3 Gaussian or diffusion.** *Review:* Gaussian first. *User:* prefers diffusion. With hundreds of people and a few dozen coefficients they may be indistinguishable; build both (Test 3).

**11.4 Fixed points.** *User:* confident slices fix the optimiser. *Review:* equality only for observed or artist-fixed points; a bound from the model counts it twice. *Mine:* the hand-written philtrum rule was needed because other terms of the fit pulled the wrong way (line targets using the atlas's own depths), not because the prior was weak. The lesson may be to replace those terms by the model's prior rather than to add firm rules. To settle: rerun the skull fit with the conflicting terms removed and a prior of the right strength, and see whether any firm rule is still needed.

**11.5 Slices or volume.** Slices remain inputs, readouts and supervision; the predicted object is 3D. Run A keeps the per-slice design under test.

**11.6 How much to trust cloth.** New. The observation model of section 4 treats tight cloth on convex regions as nearly as good as skin. That rests on the user's arrangement holding and on $\tau_{\text{fabric}}$ being measurable at edges. If a garment has no visible edge on bare skin (a full bodysuit), $\tau$ has no measurement and must come from a prior over garments. How informative is such cloth then?

## 12. Research questions for the next round

**12.1 Clothing as inequalities.** With the skin part of the state and cloth giving $\delta \ge 0$, characterise the posterior over the skin under a garment: where is it determined by the edges and the prior, and where does it remain wide? Give the convex $\phi$ of 4.4 explicitly as a function of $\tau_{\text{fabric}}$ and the bridging indicator, and show the fit stays convex.

**12.2 Estimating $R$ and $\tau$ from the cloth alone.** Prove or refute: the maximal concave curvature of a tensioned cloth surface bounds $1/R$; the step at a garment edge estimates $\tau$ without bias in the presence of a rolled hem. Propose estimators and their variances.

**12.3 The size $S_p$.** A surface-only size stable on partial CT, whole CT and a stylised figure.

**12.4 Bridging and identifiability.** In a spanned hollow the cloth is flat while the skin is concave by an unknown depth. Which other observations (the mirror side, the edges, the interior prior) identify that depth, and with what variance?

**12.5 Measurement-space training.** For the Gaussian and the diffusion head, write the objective of 8.2 exactly, including the linearisation point's effect, and state which directions of $a$ are identifiable from the scan types we hold.

**12.6 The stylisation map.** Define $\sigma$ (11.2) so that it is identifiable from the surface, low-dimensional, and composes with the mirror and with pose. How should bone follow it?

**12.7 Tails.** Extend 7.2 from one coordinate to the full problem: with a Huber penalty on the Mahalanobis norm, which combinations of coordinates enter the tail together, and is a per-direction tail better?

**12.8 Deformation-graph reach.** What local shape variation can a graph with nodes placed by bone size not express; where must nodes be added?

**12.9 Compression under tight cloth.** A model for how fat depth changes under an elastic edge, and whether it matters for the fit.

**12.10 Supine to standing.** How should skin and fat measured on a supine body be mapped to a standing one, or should the model be conditioned on posture instead?

**12.11 Attention symmetry.** $\phi(d_{ij}, Q_{ij})$ exactly equivariant to rigid motion and mirror.

**12.12 Diffusion at this scale.** Schedule, parametrisation and sample complexity for a few dozen coefficients and a few hundred people.

**12.13 Verification.** Primary-source accounts of the cited works (section 13).

## 13. Not verified

- The reference world model's architecture (reported: a multimodal autoregressive transformer with explicit camera poses, a shared 3D spatial context and rectified flow). I have not read it.
- The works cited in the reviews: surface-to-CT torso generation conditioned on demographics; head-to-skull regression with mean skull-shape errors of 1.67 to 3.99 mm on 209 subjects; relative-transform attention for multi-view reconstruction; CT generation conditioned on segmentations; a patch-and-volume medical diffusion model; continuous signed-distance shape representations.
- The data census figures of section 10 and the error estimates quoted there.
- That keypoint detection works on a surface rendered from CT; that minimal-area planes are stable on clinical CT; that gowns and blankets can be removed from CT as separate sheets.
- That garment edges on generated characters are clean enough to measure $\tau_{\text{fabric}}$.

**Constraints on any proposal:** no hand-set decision numbers; proportions, not millimetres; a pilot must fit a budget of a few dollars; CT of real people stays private.

## 14. What to send back

1. The clothing mathematics: $\phi$, the estimators of $\tau$ and $R$, and the identifiability of skin in bridged regions (12.1, 12.2, 12.4).
2. The measurement-space objective for both heads (12.5).
3. A worked proposal, or a refutation, of the stylisation map (12.6), and the multi-coordinate tail (12.7).
4. Expected outcomes of Tests 0a to 0c, and what each would force us to change.
5. A position on each item of section 11.
6. Verified accounts of the cited works.
