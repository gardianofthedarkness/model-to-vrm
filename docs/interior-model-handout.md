# Interior model: mathematical construction and research handout (version 2)

**Status: a proposal, not a built system.** Nothing here has been trained or tested. Version 2 replaces the per-slice design of version 1 as the main proposal after one round of outside review. Section 1 records what changed and why, section 9 holds the disagreements that the next rounds of research should settle, and section 11 lists what nobody has verified yet.

## 0. Purpose

We fit an anatomy atlas (bones, muscles) inside a generated 3D character with least-squares and convex optimisation. The optimiser knows hard rules (bones inside the body, seams closed, left and right alike) but knows very little about where tissue lies under a given surface: nine CT tissue-depth points on the face and the atlas's own proportions elsewhere. Those sparse numbers, each predicted alone, lost against the other terms (the bone under the upper lip stayed 3 mm too deep through five runs, until a rule was written by hand). We want a learned model that, from the outer surface alone, predicts the interior **with every part in the context of all the others**, with a calibrated uncertainty, in a form the optimiser can use directly.

The user's requirements: cross-sections ("critical slices") at important places are the working currency; all predictions for one person share context (neighbouring slices, overall appearance, age and other basic facts, the 3D structure around each slice); confident predictions act as fixed points for the optimiser; a transformer with diffusion is the preferred family.

## 1. What changed from version 1, and why

Version 1 predicted each slice's interior as its own token, jointly through attention. A review (by another model, given version 1) raised the points below. I have not verified the papers it cited; its reasoning stands on its own where I adopt it.

| # | Point raised | Verdict | Where it lands in this version |
|---|---|---|---|
| 1 | Version 1 cut the training planes from CT **bone** labels, while the figure's planes come from surface keypoints. The plane then tells the model about the bone it must predict. | **Accepted: an error in version 1.** | Section 3: planes from the skin only, by the same code on both sides. Bone-derived planes kept only as an oracle comparison. |
| 2 | Version 1 said the regression baseline gives no joint samples. A Gaussian with shared factors does. | **Accepted: an error in version 1.** | Section 5: low-rank-plus-diagonal Gaussian is the first head, not a straw man. |
| 3 | Slice tokens that attend to each other can still draw one bone in incompatible places. Predict one anatomy per person and cut every slice from it. | **Accepted, with a change** (section 9.1): the anatomy is expressed in the optimiser's own deformation parameters. | Sections 4 and 6. |
| 4 | Train on the person's 3D anatomy; use slices as one of several kinds of supervision. CT has evidence between the critical planes that a slice-only loss ignores. | **Accepted.** | Section 7. |
| 5 | Predicting "the next slice from previous slices" is a weak objective here: a generated slice adds no new evidence about the hidden anatomy, unlike a new photograph of a scene. | **Accepted.** | No autoregressive slice objective. Observed or artist-approved slices may still be given as conditions (section 5.4). |
| 6 | Encode how each plane sits relative to the others (position and orientation), not only distance along the body. | **Accepted.** | Section 5.2. |
| 7 | Do not canonicalise a slice by the phase of its first Fourier term; near a centred ring the phase is unstable. | **Accepted.** | Section 3.3. |
| 8 | Masked loss with noise in unlabelled slots does not identify correlations between structures that are never labelled together. | **Accepted.** | Section 7.4. |
| 9 | Simulate hair and clothing as connected 3D regions across planes, not as independent offsets per ring. | **Accepted.** | Section 7.5. |
| 10 | Age can stand in for the scan's source (hospital, protocol). | **Accepted.** | Section 8. |
| 11 | Use the full cross-slice covariance in the optimiser, or nearby predictions count as independent votes. | **Accepted; automatic in this version** (one covariance over the shared state). | Section 6. |
| 12 | Exact equality only for observed or artist-fixed points; learned anchors get calibrated bounds. | **Accepted in part** (section 9.5 keeps the user's "fixed points" as an open design question). | Section 6.3. |
| 13 | Start with a Gaussian head; promote diffusion only if it wins a fair comparison. | **Kept as the comparison protocol, not as the decision** (section 9.4). The user prefers diffusion. | Section 5.3, section 8. |
| 14 | Fit a statistical shape basis to human skulls and sample from it. | **Disputed** (section 9.2): the figure is far outside human skull variation. | The prediction is a prior the optimiser may leave, never the final anatomy. |

## 2. Objects and notation

**Person and size.** Person or figure $p$, surface $\mathcal S_p \subset \mathbb R^3$. All lengths are proportions of a size $S_p$ that can be measured on both a CT body and the figure from the surface alone (open: stature is not available for a head-only CT; see 10.3).

**Atlas.** A fixed template: vertices $x \in \mathcal A$, each belonging to a structure $c(x)$ (a bone, a muscle, a cartilage), with skin, bones and muscles in one coordinate frame.

**Deformation state.** The optimiser already describes an individual anatomy by a parameter vector $z \in \mathbb R^{n}$ acting on the atlas. Two forms exist in code:

- *Deformation graph* (skull): nodes $g_j$ on the bones, each with a $3 \times 3$ matrix $A_j$ and a shift $t_j$; a point moves with the nodes of its own bone,
  $$ D_z(x) = x + \sum_j w_j(x) \big[ (A_j - I)(x - g_j) + t_j \big], \qquad z = \big(\mathrm{vec}(A_j - I),\, t_j\big)_j \;\oplus\; (\text{whole-skull pose and size}). $$
  $D_z(x)$ is **linear in $z$**: $D_z(x) = x + B(x)\, z$ with a sparse matrix $B(x)$.
- *Jointed bones* (body): per bone $b$ a turn $r_b$, shift $t_b$, log-size $s_b$ and two log-stretches $e_b, f_b$ along the bone's own axes $a_b, a'_b$,
  $$ T_b(x) = c_b + e^{s_b}\, \mathrm{Rot}(r_b) \big( I + (e^{e_b} - 1)\, a_b a_b^\top + (e^{f_b} - 1)\, a'_b a'^{\top}_b \big)(x - c_b) + t_b , $$
  nonlinear in $z$, linearised at each step of the fit.

**The shared anatomical state is $z$.** One value of $z$ is one complete interior. Everything the model outputs is a statement about $z$.

**Slice planes.** For segment $s$ with axis path $\gamma_s$ and level $t \in [0,1]$, the plane $\Pi_k$, $k = (s,t)$, through $o_k = \gamma_s(t)$ with frame $R_k = [e_1^k, e_2^k, n_k]$: the plane of least cut area of the segment's surface, $e_1^k$ toward a roll landmark. **Axis paths, levels and landmarks are computed from the outer surface only** (section 3.1).

**Slice readouts.** On plane $k$, with angles $\theta_j = 2\pi j / J$:

- skin outline $\rho_k(\theta_j)$: first surface point of the segment along the ray, over $S_p$;
- for a state $z$, the section of each structure, $\Omega_{k,c}(z) = D_z(\mathcal A_c) \cap \Pi_k$, and from it the key lines $\partial\Omega_{k,c}(z)$, the skin-to-bone depth $\beta_k(\theta_j; z)$ and the skin-to-lean depth $d_k(\theta_j; z)$ along each ray, with a validity mask where a ray meets no bone.

A section may consist of several loops; nothing here assumes one star-shaped outline.

**Person facts.** $g_p = (\text{sex}, \text{age}, \text{size}, \dots)$, each entry possibly unknown, with its own known/unknown flag.

## 3. Inputs: what the model sees

### 3.1 No interior information in the inputs

For a CT person the pipeline that produces the inputs must be exactly the figure's pipeline, run on the CT **skin**: keypoints detected on renders of the skin, joints and axes from those keypoints, planes from the skin. Bone labels are used only to make targets. Planes built from bone labels are kept as an *oracle* condition; the gap between oracle and skin-only results measures how much accuracy depends on knowing the skeleton in advance.

### 3.2 Three sources of context

1. **Whole-person appearance:** the visible outer geometry, including the surface between critical planes, as a decimated point set $\{(q_i, \nu_i, m_i)\}$ (position, normal, seen/covered flag) in proportions; plus $g$.
2. **Local surface around each plane:** nearby surface points in the plane's own frame, with their signed height above or below the plane; the angular outline $\rho_k$, the visibility mask $m_k$, the plane's tilt.
3. **Other planes:** through attention with the relative geometry of section 5.2; and, when supplied, observed or artist-approved interior slices (section 5.4).

### 3.3 Symmetries

- **Roll of a plane frame by $\delta$:** $\rho(\theta) \mapsto \rho(\theta - \delta)$; in Fourier coefficients $\hat\rho[n] \mapsto e^{-in\delta} \hat\rho[n]$; for an in-plane map $U$, contour points $u \mapsto U u$ and a covariance $C \mapsto U C U^\top$. The anatomical roll landmark is kept. Slices are **not** canonicalised by $\arg \hat\rho[1]$, which is unstable when $\hat\rho[1] \approx 0$.
- **Mirror:** the reflection $\mathsf M$ swaps left and right segments and structures and maps $\theta \mapsto -\theta$. Because the state is $z$, mirror equivariance is the single condition $p(\mathsf M z \mid \mathsf M \mathcal S, g) = p(z \mid \mathcal S, g)$ with $\mathsf M$ acting on atlas parameters (for the deformation graph: $A_j \mapsto M A_{\bar j} M$, $t_j \mapsto M t_{\bar j}$ with $\bar j$ the mirror node and $M = \mathrm{diag}(-1,1,1)$).
- **Size:** removed by proportions.
- **Gauge:** $z$ is not unique for a given shape (the whole-skull pose can be absorbed into node transforms; neighbouring nodes can trade stretch for shift). Targets must be put in a canonical form before learning (open: 10.2).

## 4. The model's output: a distribution over the shared state

$$ p_\vartheta\big(z \mid \mathcal S_p,\, g_p\big). $$

One draw $z^{(i)}$ is one interior; every slice, ray or oblique section requested later is cut from the same draw and is therefore consistent with every other. Requesting an axial slice today and an oblique slice through the same place tomorrow gives the same bone where they meet, provided the draw is kept.

**Reduced coordinates.** $z$ is high-dimensional (a skull graph has roughly 120 nodes times 12 numbers) and gauge-redundant. Let $\{z_p\}$ be the canonical states obtained by fitting the atlas to the training CTs with full interior labels (section 7.1), and
$$ z = \bar z + W a + \eta, \qquad a \in \mathbb R^{r}, $$
with $W$ the leading directions of the training fits **measured in geometry, not in parameters**: principal components of the stacked vertex displacements $B z_p$ (so that a direction's weight is the movement it causes), mapped back to $z$. The model predicts $a$. $\eta$ is what the basis cannot express; its size on held-out people is the *oracle representation error* and must be measured before anything else (section 8, test 0).

## 5. The network

### 5.1 Tokens

- person token: $h_0 = E_g(g)$ with known/unknown flags;
- surface tokens: a learned set encoder over the point set of 3.2(1), giving $n_s$ tokens (patches of the outer surface);
- plane tokens: for each critical plane $k$, $E_x(\hat\rho_k[0..N], m_k, \text{tilt}_k, \text{local points})$;
- state tokens (the output side): the current noisy estimate of $a$, split per structure group so that a structure can attend to the planes and surface near it.

### 5.2 Attention with relative geometry

Pre-norm transformer blocks with logits
$$ A_{ij} = \frac{\langle W_Q h_i, W_K h_j\rangle}{\sqrt d} + b_{\mathrm{rel}(i,j)}\big(\phi(d_{ij}, Q_{ij})\big), $$
where for two plane tokens
$$ d_{ij} = R_i^\top (o_j - o_i)/S, \qquad Q_{ij} = R_i^\top R_j , $$
the position and orientation of plane $j$ seen from plane $i$'s frame, and analogous relative positions for surface and state tokens (a state token sits at its structure's atlas centre carried by the current estimate). $\mathrm{rel}$ still distinguishes same segment, adjacent segment, mirror partner and other, because a hand resting against the trunk is close in space and far in anatomy. With 8 to 12 head planes and at most a few hundred tokens, full attention is cheap; the geometry is there for statistical efficiency.

### 5.3 Heads over the same state (to be compared on equal terms)

- **Gaussian, low rank plus diagonal:**
  $$ a = \mu(\cdot) + L(\cdot)\,\xi + D(\cdot)^{1/2} \epsilon, \qquad \xi \sim \mathcal N(0, I_q),\ \epsilon \sim \mathcal N(0, I_r), $$
  trained by negative log-likelihood. It gives joint samples and a full covariance $\Sigma_a = L L^\top + D$.
- **Mixture of such Gaussians** (a cheap test for several plausible interiors).
- **Diffusion or rectified flow over $a$:** $a^\tau = \alpha_\tau a^0 + \sigma_\tau \varepsilon$, a velocity network $f_\vartheta(a^\tau, \tau, \text{tokens})$, loss $\mathbb E \| v - f_\vartheta \|^2$ weighted as in 7.3; samples by a deterministic solver; moments $\mu_a, \Sigma_a$ from the samples.

The encoder, planes, basis, splits and evaluation are identical across heads. This is the user's preferred family (transformer plus diffusion) with the simplest alternative alongside, so that the gain from diffusion is measured rather than assumed.

### 5.4 Observed interior as a condition

If some interior is genuinely known (a real scan slice, an artist-approved key line), it enters as extra tokens with their plane pose and an "observed" flag, and training includes examples with none, one or several such slices. A generated slice is never fed back as if it were evidence.

## 6. Into the optimiser

### 6.1 The prior is in the optimiser's own variables

$$ \min_z \;\; \| A z - b \|_2^2 \;+\; \big(z - \mu_z\big)^{\!\top} \Sigma_z^{+} \big(z - \mu_z\big) \quad \text{s.t.} \quad G z \le h,\;\; E z = e , $$
with $\mu_z = \bar z + W \mu_a$, $\Sigma_z = W \Sigma_a W^\top + \Sigma_\eta$ ($\Sigma_\eta$ the measured covariance of what the basis misses) and $\Sigma_z^{+}$ its (regularised) inverse. $\|Az - b\|^2$ holds the existing terms, $Gz \le h$ the linearised inside rule, $Ez = e$ exact ties. Because the prior is one quadratic form over one state, ten nearby readouts of the same uncertain displacement count once, not ten times. No per-slice linearisation is needed for the prior itself.

### 6.2 Slices as readouts and as supervision

Critical slices remain the working currency in three places: as inputs (outlines), as supervision during training (section 7.2), and as the view in which a person checks or corrects the result. They are no longer separate predictions.

### 6.3 Fixed points

A learned quantity $\ell(z)$ (for example the bone depth on a named ray) with predicted mean $\mu_\ell$ and spread $\sigma_\ell$ can enter as (a) nothing beyond the prior, (b) a calibrated bound $|\ell(z) - \mu_\ell| \le \kappa\, \sigma_\ell$, or (c) an equality. Version 2 proposes (b) for learned quantities and (c) only for observed or artist-fixed points; whether a sufficiently confident prediction should be allowed to become (c) is open (section 9.5).

### 6.4 Several modes

If $p(z \mid \cdot)$ has well-separated modes, the quadratic around the mean is wrong. Fit each mode separately and compare the fit cost plus the mode's log-probability.

## 7. Training

### 7.1 Targets: fitting the atlas into each CT

For each training CT with interior labels, run the atlas fit with the labels as data (bone surfaces, and muscle surfaces where labelled) to obtain a canonical $z_p$. This is the expensive part of data preparation and the place where label coverage matters: a structure with no label in any scan gets its $z$ only from the atlas prior and must not be counted as learned.

### 7.2 Supervision at several scales, from the same person

Encode the exterior once, then score a draw or the mean against queries taken from that person's CT:

| Query | Loss | What it teaches |
|---|---|---|
| The state itself | $\| a - a_p \|^2_{\Lambda}$, or the head's likelihood | The distribution over anatomies. |
| Points on and near labelled surfaces | distance from $D_z(x)$ to the labelled surface, both sides | Fine boundary position, including between critical planes. |
| Points spread through the volume | inside/outside agreement of each structure | Overall placement and extent. |
| Critical planes, random oblique planes, named depth rays | contour distance and depth error on the plane | The quantities the optimiser and the user consume; oblique planes expose errors the usual orientation hides. |

Because $D_z$ is linear in $z$ for the deformation graph, point and plane losses are quadratic in $z$ to first order, with matrices $B(x)$.

### 7.3 Weights from data

Each query's weight is the inverse of its own measurement variance: repeat-extraction variance for planes (perturb landmarks, crops and the surface threshold within their measured repeatability), label noise per structure, and the area or length the query represents.

### 7.4 Three different kinds of "missing"

- *No label for this structure in this scan* (annotation missing): excluded from the likelihood by marginalising, i.e. the likelihood of the observed block only, with covariance $\Sigma_{OO}$.
- *The ray meets no bone* (anatomical absence): an observation in its own right, with a validity flag as a target.
- *Skin hidden by cloth or hair* (cover): an input condition, not a missing target.

Before choosing joint targets, tabulate which structures are ever labelled together. Correlations between structures that never share a scan cannot be learned from the data and come only from the atlas.

### 7.5 Cover

Hair and clothing are simulated as connected 3D regions over the CT skin (a garment or a hair mass covering a patch across several planes), producing a covering surface outside the skin and a covered flag. Two variants are compared: the model sees only the flag, or it also sees the covering geometry. The test set uses real character hair and clothing masks.

### 7.6 Training conditions

Random dropping of $g$ entries; random hiding of contiguous bands of planes; with and without observed interior slices (5.4), evaluated separately.

## 8. Experiments, in order

**Test 0, before any training: oracle representation error.** Fit the basis directly to held-out CT labels (the answer supplied) and measure the residual at the upper lip, the jaw and the skull base. If the best case already misses, a better predictor cannot help; add local freedom to the basis first.

**Test 1: how much the exterior knows.** With the head fixed (Gaussian), compare inputs: local outline only; all outlines; all outlines plus oriented surface neighbourhoods; whole head; plus known age. Report the held-out gain at each named site. There is no defensible numerical ceiling from published whole-skull results; the irreducible error of a depth $Y$ is $\mathbb E[\mathrm{Var}(Y \mid \mathcal S, g)]$, and this ladder measures how close each input gets.

**Test 2: three training paradigms**, same held-out people, same inputs, comparable compute:

| Run | Interior representation | Supervision |
|---|---|---|
| A (version 1) | joint per-slice predictions | critical slices |
| B | one shared state $z$, slices cut from it | the same critical slices |
| C | as B | critical slices plus off-plane points and surfaces |

A to B measures the value of a shared anatomy; B to C the value of the CT between the slices.

**Test 3: heads.** Gaussian, mixture, diffusion or flow over the same $a$; scored by a joint sample score (energy score), interval coverage and width, and error after the actual atlas fit.

**Test 4: the use.** Run the skull fit with the hand-written philtrum rule removed and the model's prior in its place. The acceptance test is the bone depth under the upper lip after the fit, with the jaw and skull base no worse.

**Test 5: stylisation.** Smooth paired warps of CT skin and bone together (so the truth is known), planes re-extracted afterwards; report how error and overconfidence grow with the warp. Then the real figure, where there is no truth, judged by the Mahalanobis distance of the fitted $z$ from the prior and by the hard rules.

**Checks on confounding.** Known age, withheld age, and age shuffled within scan source; results by age range and by source.

## 9. Disagreements and open positions for the next rounds

Each item states the review's position, mine, and what would settle it.

**9.1 Where the shared anatomy lives.** *Review:* a statistical shape basis of skull and mandible, or an implicit field per structure; cut slices from it. *Mine:* use the optimiser's own deformation parameters, so the prediction is a prior in the variables being solved for and needs no conversion, linearisation or second fit. *Risk in mine:* $z$ is gauge-redundant and was designed for optimisation, not for statistics; its principal directions may be dominated by parametrisation artefacts. *To settle:* the canonical form of $z$ (10.2), and Test 0 run for both representations.

**9.2 Human shape models and a non-human figure.** *Review:* sample one plausible anatomy from a basis learned on people; measure the basis's error on held-out people; stress-test with warps. *Mine:* the figure needed a 17% stretch of the upper jaw to follow her face, where human skull proportions vary by about 3.5%. A draw from a human basis is therefore wrong for her by construction, however good the basis is on people. The prediction must be a prior with honest tails that the optimiser can leave where the surface insists, and the mathematics of leaving it must be worked out: a Gaussian prior penalises a 5-sigma deviation as $25$, which the data terms may or may not overcome. *To settle:* the right tail (Gaussian, Student-t, Huber on the Mahalanobis distance, or a learned out-of-distribution score that inflates $\Sigma$), and whether the model can be made to *know* it is extrapolating (10.5).

**9.3 The cost that the review did not count.** Its plan needs the maxilla and mandible as 3D labels in every scan. As reported so far (unverified), the available CT set labels the skull coarsely and individual face bones not at all. So the targets $z_p$ require fitting our atlas into every scan from coarse labels plus intensity, which is the larger part of the work and has its own error. *To settle:* an estimate of that fit's accuracy at the upper jaw from coarse skull labels (can it resolve what we want to learn?), and whether a subset with fine labels is needed.

**9.4 Gaussian first, or diffusion.** *Review:* start Gaussian; promote diffusion only if it wins. *User:* prefers a transformer-diffusion model. *Mine:* with 100 to 300 heads a Gaussian over a few dozen coefficients is likely to match diffusion, but this is a guess; the question is empirical and cheap, since the head is a small part of the network. Build both from the start (Test 3) rather than argue. *To settle:* an analysis of when diffusion over a low-dimensional, strongly conditioned state can beat a Gaussian at this sample size, and what the test should be.

**9.5 Fixed points.** *User:* the model fixes important critical slices and those guide the optimiser. *Review:* equality only for observed or artist-fixed points. *Mine:* a hand-written firm rule is what finally fixed the philtrum, so firm constraints have a real role; but an equality from an uncalibrated model is dangerous. *To settle:* a rule, derived from calibration data, for when a bound may harden (for example, when the bound's width falls below the optimiser's own resolution), and whether hard and soft forms differ in outcome once the covariance is right.

**9.6 Slices or volume.** *User:* 2D models on important slices are enough to guide the optimiser. *Review:* train on the 3D anatomy, keep slices as queries. *This version:* slices stay as inputs, readouts and supervision; the predicted object is 3D. Run A is kept so that the user's original design is tested, not dropped.

**9.7 An autoregressive, pose-conditioned generator in the style of the reference world model.** The review says the transferable idea is a shared, spatially grounded context and that next-slice prediction is a weak fit. I agree with the argument but have not read the reference model's architecture (section 11). *To settle:* a primary-source account of that architecture and a direct statement of which mechanisms apply here.

## 10. Research questions for the next round

**10.1 Test 0 on paper.** For a deformation graph with $M$ nodes, what local shape variation can it not express (a bound in terms of node spacing and the weight functions $w_j$)? Is the upper jaw's alveolar process within reach with nodes placed by bone size, or must nodes be placed where the data ask?

**10.2 Canonical form of $z$.** Define a gauge fixing for the deformation graph (and for the jointed bones) under which equal shapes have equal $z$, the mirror action is linear, and principal components are meaningful. Is it better to do statistics on $B z$ (vertex displacements) and map back by a pseudo-inverse?

**10.3 The size $S_p$.** Which surface-only size is stable on head-only CT, whole-body CT and a stylised figure with non-human proportions (for the head: a robust distance between surface landmarks, the cube root of the head's volume)? How does the choice bias depth proportions on a figure with a large head?

**10.4 Tails and leaving the prior.** Derive the optimiser's solution as a function of the prior's tail for a single conflicting constraint (the philtrum case: a surface that demands a displacement $k$ spreads from the mean). Which convex penalties keep the problem convex and let the data win at large $k$? What is the correct term when the model itself reports extrapolation?

**10.5 Knowing when it extrapolates.** Give a principled out-of-distribution measure on the inputs (surface proportions outside the training hull) and a way to turn it into inflated uncertainty, with a test on warped CT.

**10.6 Marginal likelihood with structured missingness.** Write the training objective for the Gaussian and for the diffusion head when each scan labels a different subset of structures, and state exactly which cross-structure correlations are identifiable from a given co-occurrence table.

**10.7 Losses in geometry.** Using linearity $D_z(x) = x + B(x) z$, derive the quadratic forms of the point, surface and plane losses in $z$ and in $a$, their natural weights, and the relation between the energy score on $a$ and on readouts.

**10.8 Relative geometry in attention.** Specify $\phi(d_{ij}, Q_{ij})$ so that the network is exactly equivariant to a global rigid motion and to the mirror, and say whether invariance to a plane's roll should be exact or only encouraged.

**10.9 Diffusion at this scale.** For $r$ of a few dozen and $N$ of a few hundred, what noise schedule, parametrisation and regularisation make a diffusion head competitive with a Gaussian, and what does the theory of score estimation say about sample complexity here?

**10.10 Plane stability.** How sensitive are minimal-area planes to landmark and surface noise, analytically, for a near-cylindrical limb and for the face; and how does plane jitter propagate into depth targets at the philtrum?

**10.11 Verification.** Establish from primary sources what the cited works actually show (section 11), and correct this handout where they do not support what is attributed to them.

## 11. Not verified

- The architecture of the reference world model (reported as World Labs' Atlas: a multimodal autoregressive transformer with explicit camera poses, a shared 3D spatial context and rectified flow). I have not read it.
- The works the review cited and the figures attributed to them: surface-to-CT torso generation conditioned on demographics; head-to-skull regression with mean skull-shape errors of 1.67 to 3.99 mm on 209 subjects; relative-transform attention for multi-view reconstruction; CT generation conditioned on segmentations; a patch-and-volume medical diffusion model; continuous signed-distance shape representations. None of these has been checked by me.
- The exact labels in the CT download (a job is fetching it).
- That keypoint detection works on skin rendered from CT, which section 3.1 depends on.
- That minimal-area planes are stable on clinical CT (arms raised, cropped fields of view, tables and blankets).

**Constraints on any proposal:** no hand-set decision numbers (thresholds from data, a statistical model or the optimisation); proportions, not millimetres; a head-only pilot must fit a budget of a few dollars of GPU time; CT of real people is stored privately and no scan or rendering of one leaves that storage.

## 12. What to send back

1. A canonical form for $z$ and the basis construction (10.1, 10.2), with Test 0's expected outcome.
2. The tail and out-of-distribution treatment that lets the optimiser leave a human prior (10.4, 10.5), worked through for the philtrum case.
3. The training objective under structured missing labels (10.6) and the geometric losses (10.7).
4. The attention specification with its symmetries (10.8).
5. A reasoned position on Gaussian against diffusion at this scale (10.9), stated as what the experiment should show for each to be chosen.
6. A position on each disagreement in section 9, with the argument, including where this version is wrong.
7. Verified accounts of the cited works (10.11).
