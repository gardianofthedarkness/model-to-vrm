# Interior slice model: mathematical construction and research handout

**Status: a proposal, not a built system.** Nothing here has been trained or tested. Section 9 lists what is decided, what is assumed, and what I could not verify. The handout is written so that a mathematics agent can work on the architecture questions in section 8 before any training starts.

## 0. Purpose in one paragraph

We fit an anatomy atlas (bones, muscles) inside a generated 3D character with least-squares and convex optimisation. The optimiser knows hard rules (bones inside the body, seams closed, left and right alike) but has very little knowledge of where tissue lies under a given surface: nine CT tissue-depth points on the face and the atlas's own proportions elsewhere. Those sparse numbers, each predicted alone, lost against the other terms (the bone under the upper lip stayed 3 mm too deep through five runs). We want a learned model that, from the body's outer surface alone, predicts the interior on a set of 2D cross-sections ("critical slices"), **jointly over all slices of one person**, with an uncertainty for every predicted number. Its output becomes targets and fixed points for the optimiser.

## 1. Objects and notation

**Person and size.** A person (or figure) $p$ has a surface $\mathcal S_p \subset \mathbb R^3$ and a size $H_p$ (stature). All lengths are used as proportions, $\tilde\ell = \ell / H_p$, so that the model applies to a body of any size.

**Segments and axes.** The body is divided into segments $s \in \mathcal G$ (16 in our code: pelvis, trunk, neck, head, and left/right upper arm, forearm, hand, thigh, calf, foot). Each segment has an axis path $\gamma_s : [0,1] \to \mathbb R^3$ between its two end joints, and a roll landmark (thumb, front of the knee, the body's front, ...).

**Slice planes.** At level $t \in [0,1]$ the slice plane $\Pi_{s,t}$ passes through $\gamma_s(t)$ and is the plane of least cut area of the segment's surface (searched over two tilt angles, started from the neighbouring level, with the turn between neighbours bounded so that planes do not cross inside the body). It carries an orthonormal frame $(e_1, e_2, n)$ with $e_1$ toward the roll landmark. This construction exists in code (`planes.py`) and is the same for a CT body and for the figure.

**Skin outline.** In $\Pi_{s,t}$, with angles $\theta_j = 2\pi j / J$, $j = 0,\dots,J-1$ (currently $J = 72$),

$$ r_{s,t}(\theta_j) = \text{distance from } \gamma_s(t) \text{ to the first surface point of the segment along } \cos\theta_j\, e_1 + \sin\theta_j\, e_2 . $$

We write $\rho_k = r_k / H_p \in \mathbb R^J$ for slice $k = (s,t)$.

**Visibility.** $m_k \in \{0,1\}^J$: $m_k(\theta_j) = 1$ where the outline point is bare skin, $0$ where it is cloth, hair or a prop (for the figure this comes from the material labels; in CT it is 1 everywhere unless we simulate cover).

**Interior of a slice.** Let $\mathcal C_s$ be the structures that can appear in segment $s$ (bones; muscles or muscle groups; optionally fat). For slice $k$ and structure $c$ let $\Omega_{k,c} \subset \Pi_k$ be its section. We consider three target representations.

1. *Depth fields* (dense, no template):
   $$ d_k(\theta_j) = r_k(\theta_j) - r^{\text{lean}}_k(\theta_j), \qquad \beta_k(\theta_j) = r_k(\theta_j) - r^{\text{bone}}_k(\theta_j), $$
   where $r^{\text{lean}}_k$ is the radius of the outermost non-fat tissue along the ray and $r^{\text{bone}}_k$ the radius of the first bone hit (undefined on rays that hit no bone; a validity mask $u_k(\theta_j)$ records this). $d$ is the subcutaneous depth; $\beta$ the skin-to-bone depth, the quantity the CT face study measures at single points.
2. *Template placement* (what our slice fit already solves): an atlas outline $\partial\Omega^{A}_{k,c}$ placed by five numbers
   $$ q_{k,c} = (c_x, c_y, \varphi, s_x, s_y), \qquad x \mapsto \begin{pmatrix} c_x \\ c_y \end{pmatrix} + R(\varphi) \begin{pmatrix} s_x & 0 \\ 0 & s_y \end{pmatrix} x , \quad x \in \partial\Omega^{A}_{k,c}. $$
3. *Free outlines*: a polar function about the structure's own centre, $a_{k,c}(\theta)$, for star-shaped sections.

Everything is in proportions: lengths divided by $H_p$, angles as $(\cos\varphi, \sin\varphi)$, scales as $\log s_x, \log s_y$.

**Target vector.** $y_k \in \mathbb R^{D_s}$ concatenates the chosen representation for slice $k$, with a mask $M_k \in \{0,1\}^{D_s}$ marking entries that have a label for this person (labels are missing for some structures in every dataset).

**Person facts.** $g_p = (\text{sex}, \text{age}, H_p, \dots)$, each entry possibly unknown. For the figure all may be unknown.

## 2. Slice tokens

A slice enters the model as
$$ x_k = \big(\rho_k,\; m_k,\; s_k,\; t_k,\; \tau_k,\; \text{side}_k \big), $$
with $\tau_k$ the plane's tilt from the axis direction. Two properties of the outline matter for the encoding.

**Roll.** Turning the slice frame by $\delta$ shifts the outline, $\rho(\theta) \mapsto \rho(\theta - \delta)$. In the discrete Fourier coefficients
$$ \hat\rho[n] = \frac1J \sum_{j=0}^{J-1} \rho(\theta_j)\, e^{-i n \theta_j}, \qquad n = 0, \dots, N, $$
it is a phase: $\hat\rho[n] \mapsto e^{-i n \delta} \hat\rho[n]$. The roll landmark fixes $\delta$ only as well as the landmark is found, so the model should be equivariant, or at least robust, to small $\delta$.

**Mirror.** The left-right reflection maps $\theta \mapsto -\theta$ (so $\hat\rho[n] \mapsto \overline{\hat\rho[n]}$), swaps left and right segments, and maps a placement $q = (c_x, c_y, \varphi, s_x, s_y) \mapsto (c_x, -c_y, -\varphi, s_x, s_y)$ in a frame whose $e_2$ is the mirrored direction.

The low Fourier terms already have meaning in our code: $n=0$ is the mean radius, $n=1$ the off-centre shift, $n=2$ the flattening; the "critical levels" are the extrema of these along the segment.

## 3. The probabilistic model

Let $X = (x_1, \dots, x_K)$ and $Y = (y_1, \dots, y_K)$ be all slices of one person. We want samples from, and moments of,
$$ p_\vartheta(Y \mid X, g). $$
The key requirement is the joint: $y_k$ must depend on every $x_{k'}$ and be consistent with every $y_{k'}$.

**Diffusion form.** Standardise each target entry to zero mean and unit variance over the training set, $y \leftarrow (y - \bar y)/\mathrm{sd}(y)$. With a noise schedule $(\alpha_\tau, \sigma_\tau)$, $\alpha_\tau^2 + \sigma_\tau^2 = 1$, $\tau \in [0,1]$,
$$ Y^\tau = \alpha_\tau Y^0 + \sigma_\tau \varepsilon, \qquad \varepsilon \sim \mathcal N(0, I). $$
A network $f_\vartheta$ predicts the velocity $v = \alpha_\tau \varepsilon - \sigma_\tau Y^0$:
$$ \mathcal L(\vartheta) = \mathbb E_{p,\, \tau,\, \varepsilon} \Big\| M \odot \big( v - f_\vartheta(Y^\tau, \tau, X, g) \big) \Big\|_2^2 . $$
The mask $M$ removes unlabelled entries from the loss; at those entries the input $Y^\tau$ carries pure noise at every $\tau$, so the network learns to ignore them (open question 8.6 asks whether this is the right treatment).

**Sampling and moments.** A deterministic sampler (DDIM or the probability-flow ODE) gives samples $Y^{(1)}, \dots, Y^{(S)}$. For slice $k$,
$$ \mu_k = \frac1S \sum_i y_k^{(i)}, \qquad \Sigma_k = \frac{1}{S-1} \sum_i \big(y_k^{(i)} - \mu_k\big)\big(y_k^{(i)} - \mu_k\big)^{\!\top}, $$
and cross-slice covariances $\Sigma_{kk'}$ likewise.

**Baseline to beat.** A heteroscedastic regression with the same network body, outputs $(\mu_k, \log \sigma_k^2)$ and Gaussian negative log-likelihood
$$ \mathcal L_{\text{reg}} = \mathbb E \sum_{k,i} M_{k,i} \Big[ \frac{(y_{k,i} - \mu_{k,i})^2}{2\sigma_{k,i}^2} + \tfrac12 \log \sigma_{k,i}^2 \Big]. $$
It gives per-entry spreads but no joint samples and no multimodality. The existing ridge regression (12 face points to 41 depths) is the simplest baseline.

**Conditioning dropout.** During training, with some probability the person token $g$ is replaced by a learned "unknown" token, individual entries of $g$ likewise, and whole slices have their outline replaced by a "covered" token. This trains the model for the figure's case (no person facts, parts of the skin hidden) and allows classifier-free guidance.

**Skin under cover.** Where $m_k = 0$ the observed outline is cloth or hair, not skin. We add the bare-skin outline $\rho^{\text{skin}}_k$ to the targets, so the model also predicts the body under clothing. CT bodies are mostly unclothed, so cover must be simulated: on a random arc set $m = 0$ and replace $\rho$ by $\rho + \delta\rho$ with $\delta\rho \ge 0$ drawn from a model of cloth offsets (open question 8.7).

## 4. The network

**Sequence.** One token for the person, one per slice:
$$ h^0_0 = E_g(g), \qquad h^0_k = E_x(x_k) + E_y(y^\tau_k) + E_{\text{pos}}(s_k, t_k) + E_\tau(\tau), \quad k = 1..K . $$
$E_x$ acts on the Fourier coefficients $(\hat\rho[0..N])$, the mask, tilt and side; $E_y$ is a linear map of the noisy target (per segment type, since $D_s$ differs); $E_{\text{pos}}$ embeds the segment identity and level; $E_\tau$ the diffusion time.

**Blocks.** $L$ pre-norm transformer blocks,
$$ h^{l+\frac12} = h^l + \mathrm{Attn}\big(\mathrm{LN}(h^l);\, B\big), \qquad h^{l+1} = h^{l+\frac12} + \mathrm{MLP}\big(\mathrm{LN}(h^{l+\frac12})\big), $$
with attention logits
$$ A_{kk'} = \frac{\langle W_Q h_k, W_K h_{k'} \rangle}{\sqrt d} + B_{kk'}, \qquad B_{kk'} = b_{\mathrm{rel}(k,k')}\big( \Delta(k,k') \big). $$
$\Delta(k,k')$ is the distance between the slices along the body's segment tree, in proportions: within a segment $|t_k - t_{k'}|\, L_s / H$, across segments the path length through the joints. $\mathrm{rel}(k,k') \in \{\text{same segment}, \text{adjacent}, \text{mirror partner}, \text{other}\}$ selects a learned scalar function $b$. The mirror-partner relation lets the model use the other side directly, which is how we recover a back hidden by hair today (by hand).

**Head.** $\hat v_k = W_{o,s_k}\, h^L_k$.

**Size.** With $d = 256$, $L = 8$, 8 heads: about 6 to 8 million parameters; $K \approx 50$ to $100$ tokens per person (8 to 12 for a head-only pilot). These are starting values, not conclusions.

**Symmetries to respect.** (a) Mirror: $f(\mathsf M X) = \mathsf M f(X)$ with $\mathsf M$ acting as in section 2. (b) Roll: equivariance in the Fourier phases, or robustness by augmentation. (c) Size: built in by proportions. (d) Slice order: none assumed beyond the tree distances (the model is a set model with relational bias).

## 5. From predictions to the optimiser

Our fits minimise a sum of squared residuals over parameters $z$ (bone poses and sizes, or deformation-graph node transforms) under hard constraints, by successive convex problems in which every term is linearised at the current iterate.

For slice $k$ let $\ell_k(z) \in \mathbb R^{D_s}$ be the slice quantities of the *fitted atlas* (depths along the rays, placements of each structure's section). Linearised at $z_0$: $\ell_k(z) \approx L_k z + c_k$. The model contributes
$$ \min_z \;\; \| A z - b \|_2^2 \;+\; \sum_k \big(L_k z + c_k - \mu_k\big)^{\!\top} \Sigma_k^{-1} \big(L_k z + c_k - \mu_k\big) \quad \text{s.t.} \quad G z \le h, \;\; E z = e, $$
where $\|Az-b\|^2$ holds the existing terms (seams, joints, stiffness, mirror), $Gz \le h$ the linearised inside rule, $Ez = e$ exact ties. With cross-slice covariance the sum becomes one quadratic form in the stacked residual with the block matrix $(\Sigma_{kk'})^{-1}$.

**Fixed points.** The user's intent is that confident slices pin the solution. Three candidate forms, to be compared (question 8.5):

- soft only: the quadratic above (a small $\sigma$ is already nearly hard);
- chance bounds: $|L_{k,i} z + c_{k,i} - \mu_{k,i}| \le \kappa\, \sigma_{k,i}$ as linear constraints, $\kappa$ from the required coverage and the calibrated error distribution;
- equalities $L_{k,i} z + c_{k,i} = \mu_{k,i}$ for entries whose spread is below a data-derived level.

The current skull fit uses a mixture decided by hand (hard one-sided bounds at "steadier than the median site", soft elsewhere), which is one of the things this model should replace.

**Realisability.** The slice predictions are 2D and independent of $z$; nothing forces them to be the sections of one rigid skeleton. The optimiser projects them onto what the atlas can realise. The Mahalanobis residual $\chi_k^2 = (L_k z^\ast + c_k - \mu_k)^\top \Sigma_k^{-1} (L_k z^\ast + c_k - \mu_k)$ at the solution is the diagnostic: large values mean either a bad prediction or a body the atlas cannot follow.

**Tighter coupling (optional).** Instead of sample-then-fit, guide the sampler toward realisable interiors: at each sampling step, replace the current estimate $\hat Y^0$ by a step toward its projection onto $\{\ell(z) : Gz \le h, Ez = e\}$ (a convex problem per step). Whether this is sound and worth its cost is question 8.8.

## 6. Training data

For each CT scan $p$ with per-voxel labels: the skin surface (threshold or a body model), bone labels (per bone), muscle labels (few named muscles shipped; more by atlas registration or a segmentation model), fat where available.

1. Joints and axes $\gamma_s$ from the bone labels (centres of facing bone ends), as in our atlas code.
2. Planes $\Pi_{s,t}$ by the minimal-area construction on the CT skin; levels at the critical levels (extrema of Fourier terms 0, 1, 2 of the outline along the axis) plus, optionally, a regular grid.
3. Per slice: $\rho_k$, the sections $\Omega_{k,c}$, then the targets of section 1.
4. Person facts from the scan's metadata where present.

Splits are by person. Sizes: about 1,200 whole-body CT scans times 50 to 100 slices, about $10^5$ slices; a head-only pilot uses 100 to 300 heads times 8 to 12 slices.

## 7. Evaluation

- **Accuracy:** per entry, error in proportions and in millimetres at a reference size; for the face, at the CT study's named sites, against the ridge baseline and the "mean proportion" baseline.
- **Calibration:** coverage of $\mu \pm \kappa\sigma$ intervals; the distribution of $\chi_k^2$ against its nominal law; probability integral transform histograms.
- **Use:** run the skull fit with the hand-written philtrum rule removed and the model's targets in its place; compare the bone depth under the upper lip and the other sites with the current result (upper-lip bone 7.7 mm against a CT median of 7.0 at the figure's size; philtrum area 6.0 to 7.8 mm).
- **Out of distribution:** stylised proportions (large head, long legs, small jaw). Test with synthetic warps of CT bodies (section 8.9).

## 8. Research questions for the mathematics agent

Each question asks for a derivation or a reasoned recommendation with its assumptions stated, not for code.

**8.1 What should be diffused?** Compare the three target representations of section 1 (depth fields, template placements, free outlines) and mixtures. Which has the best-conditioned distribution given the outline (unimodal, near-Gaussian after standardisation)? Template placements have a rotation ambiguity for near-circular sections ($s_x \approx s_y$ makes $\varphi$ unidentifiable): propose a parametrisation without that singularity (for example the symmetric matrix $R(\varphi)\,\mathrm{diag}(s_x^2, s_y^2)\,R(\varphi)^\top$, or its matrix logarithm).

**8.2 Is diffusion needed at all?** State conditions under which $p(Y \mid X)$ is close enough to Gaussian that the heteroscedastic regression with a low-rank cross-slice covariance is sufficient, and a test on data that decides it (for example, normality of standardised residuals, or the gain in held-out log-likelihood of a mixture over a Gaussian). Give the expected sample complexity of each for $N \approx 300$ and $N \approx 1200$ persons.

**8.3 How much can the outline know?** Give an information-theoretic or statistical bound on predicting bone position from the skin outline: for example the fraction of variance of $\beta_k$ explained by $\rho$ over all slices of the person, as a function of fat-layer variability. The CT face study gives a reference point: ridge regression from 12 face points explains about 40 to 60% of the variance of skull measurements, but lowers the relative tissue-depth error at the nose and lip sites only from about 14-17% to 13-15%. What ceiling should we expect from full-outline context?

**8.4 Equivariance.** Derive the action of roll and mirror on every token and target of sections 1 and 2, and design $E_x$, $E_y$ and the head so that $f$ is exactly mirror-equivariant and roll-equivariant (or show that exact roll equivariance conflicts with the absolute roll landmark and propose the right compromise). Should the Fourier phases be canonicalised (rotate each slice so that $\arg\hat\rho[1] = 0$) and the removed angle passed as a feature?

**8.5 From samples to constraints.** Given calibrated $(\mu, \Sigma)$, or samples from a non-Gaussian $p(Y\mid X)$, which convex term is the correct one for the fit: Gaussian quadratic, Huber, quantile bounds, or a scenario (sample-average) formulation? When is an equality "fixed point" justified, and how should its threshold follow from the data rather than be chosen? If $p(Y \mid X)$ is multimodal, the quadratic around the mean is wrong: propose a treatment (mode selection, mixture of convex problems).

**8.6 Missing labels.** Muscles are labelled for some structures and some datasets only. Is masking the loss and feeding noise at unlabelled entries a consistent estimator of the conditional for the labelled ones? Compare with marginalising architectures (targets as a variable set of tokens, one per structure, absent tokens simply not present).

**8.7 Cover model.** Specify a generative model of clothing and hair offsets $\delta\rho(\theta) \ge 0$ for augmenting bare CT outlines, and say what the model can and cannot learn from simulated cover. How should the visibility mask enter attention so that covered slices borrow from visible neighbours and from the mirror partner?

**8.8 Coupling.** Analyse the alternating scheme of section 5 (guide the sampler by projection onto the realisable set). Is it a valid posterior sampler for $p(Y \mid X, Y \in \text{realisable})$, under what assumptions, and how does its cost compare with sample-then-fit using cross-slice covariance?

**8.9 Extrapolation to stylised bodies.** The figure is outside the CT distribution (proportions of a stylised character). Which parametrisation extrapolates most safely: targets relative to the local outline size (depth over local mean radius) rather than to stature; conditioning on local Fourier terms only; a hierarchical model (segment-level proportions, then slice detail)? Propose a family of synthetic warps of CT bodies (affine per segment, smooth radial scalings) under which the right answer is known, to measure extrapolation error.

**8.10 Attention structure.** Justify or replace the relational bias $B_{kk'}$ of section 4. Is a tree-distance bias enough, or should tokens be organised hierarchically (person, segment, slice) with cross-attention? For $K \le 100$ full attention is cheap, so the question is statistical efficiency, not cost.

**8.11 The reference architecture.** The user pointed to a transformer-diffusion world model from Fei-Fei Li's group; a research session identified it as World Labs' "Atlas" (September 2026). I have not read it and cannot describe it. Please establish what its architecture actually is from primary sources and state which of its ideas transfer to this problem (set-structured latents, conditioning scheme, sampling) and which do not.

**8.12 Loss weighting and scales.** Derive entry weights for $\mathcal L$ from the data (inverse label noise per structure, slice area) instead of hand weights, and a noise schedule suited to low-dimensional, strongly conditioned targets.

## 9. Decided, assumed, unverified

**Decided by the user:** 2D slices at important places rather than a full 3D model; slices grouped per person so each has the context of its neighbours and the person's basic facts; the model's confident slices act as fixed points for the optimiser; a transformer with diffusion is the preferred family.

**Assumed by me (open to change):** the slice planes and outline of our existing code; proportions by stature; $J = 72$ angles; the token layout and attention bias of section 4; velocity prediction; the sizes in section 4.

**Not verified:** the architecture of the reference model (8.11); the exact labels in the CT download (a job is fetching it now); that our face-point detector works on skin rendered from CT; that the minimal-area planes are stable on clinical CT (arms raised, cropped fields of view, tables and blankets in the scan).

**Constraints on any proposal:** no hand-set decision numbers (thresholds must come from data, a statistical model, or the optimisation); proportions, not millimetres; training must fit a small budget (a head-only pilot at a few dollars of GPU time); CT of real people is stored privately and no scan or rendering of one leaves that storage.

## 10. What to send back

1. A recommended target representation with its symmetry actions written out (8.1, 8.4).
2. A recommendation between diffusion and Gaussian regression for the pilot, with the deciding test (8.2), and the expected ceiling (8.3).
3. The convex term(s) for the optimiser, including the rule for fixed points (8.5).
4. Treatment of missing labels and of cover (8.6, 8.7).
5. An extrapolation test design (8.9).
6. A corrected version of sections 3 and 4 where the analysis shows they are wrong.
7. A short list of anything that should be measured on the data before the architecture is frozen.
