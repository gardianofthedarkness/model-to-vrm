# Interior model: training and decoding mathematics

Handout for review, 2026-11-24. It states what the code does today (`research/interior_model/flow_f1.py`, `flow_sample.py`, `face_compress.py`, `elf_shell.py`), the fault found in decoding and its fix, and the changes proposed next. Design background is in [interior-model-plan-v8.md](interior-model-plan-v8.md). Section 9 lists the points where I am least sure and would most like to be checked.

## 1. Task

Given only the outer surface of a body part, produce a label for every point inside it: one of $K$ tissue or structure classes. The output must be a plausible human interior that fits this surface; it is not unique, so the model is generative.

Two parts so far, each in its own frame:

| Part | Label voxel $m$ | Classes $K$ | Token edge $F m$ | Token grid $G$ | Training people |
|---|---|---|---|---|---|
| Face | 1 mm | 74 | 8 mm ($F = 8$) | $28 \times 40 \times 40$ | 76 (152 with mirror images) |
| Pelvis | 3 mm | 20 | 12 mm ($F = 4$) | $47 \times 45 \times 48$ | 345 |

A label volume is $L : \Omega \to \{0, \dots, K-1\} \cup \{u\}$ on the part's voxel window $\Omega$, with $0$ = outside the body and $u$ = unknown (outside the scan). The body is $B = \{x : L(x) \notin \{0, u\}\}$.

## 2. Label autoencoder (frozen)

A 3D convolutional autoencoder per part, trained before the generator and then frozen:

$$ E : L \mapsto Z \in \mathbb R^{C \times G}, \qquad D : Z \mapsto \ell \in \mathbb R^{K \times \Omega}, \qquad C = 8 . $$

- $E$: class embedding (16 numbers, with one extra entry for $u$), then $\log_2 F$ stride-2 stages of two $3^3$ convolutions each, then a $1^3$ convolution to $C$ channels. One code vector $Z_g \in \mathbb R^8$ per token cell $g$.
- $D$: two $3^3$ convolutions at token resolution, then $\log_2 F$ stages of trilinear upsampling and two $3^3$ convolutions, then a $1^3$ convolution to $K$ logits per voxel.
- Loss: cross-entropy on the voxels, weighted per class by $(\text{voxel share})^{-1/2}$, unknown voxels ignored. No regulariser on $Z$ (it is not a variational autoencoder; the codes have no imposed prior).
- Sizes: face 7.1 M parameters, pelvis 1.8 M.

**$D$ is not local.** A decoded voxel depends on the codes of every token within about three cells of its own. This matters in section 6.

**Ceiling.** Encoding and decoding the true labels gives a median Dice over classes of 0.97 (pelvis) and 0.75 (face). The generator cannot score above this.

## 3. Tokens and inputs

**Body tokens.** $\mathcal T = \{ g : \text{cell } g \text{ contains a body voxel} \}$. About 7,000–12,000 per part and person. Target for token $g$: the true code $Z_g$.

**Shell tokens.** $\mathcal S = \{ g \in \mathcal T : \text{some cell in the } 3^3 \text{ neighbourhood of } g \text{ is not in } \mathcal T \}$, about a third of $\mathcal T$. They are the reference the model reads.

**Shell feature** of cell $g$, $f_g \in \mathbb R^{65}$: the signed distance to the body surface, $\phi(x) = d(x, \partial B)$ positive inside, in units of 10 mm, average-pooled to a $4^3$ grid over the cell (64 numbers), plus the share of the cell's voxels that are not unknown.

**Position.** $p_g = (g + \tfrac12) F m / 100\ \text{mm}$, in the part frame. Encoding: with a fixed random matrix $W \in \mathbb R^{3 \times 16}$, $W_{ij} \sim \mathcal N(0, 2^2)$,
$$ \gamma(p) = \mathrm{Lin}\big([\sin(pW),\ \cos(pW),\ p]\big) \in \mathbb R^{d} . $$
The same $\gamma$ is used for shell and body tokens: a shell token and a body token of the same cell carry the same position. That shared coordinate is the only link between the geometry and the anatomy code.

**Facts.** Three tokens: age (a number with a known/unknown flag), sex (unknown, male, female), and the list of classes the person's source draws (a $K$-vector of zeros and ones). In training, age and sex are each hidden with probability 0.1.

**Frames (training data).** Head: origin at the midpoint of the two ear canals, $x$ from left to right canal, $y$ towards the midpoint of the eyes made orthogonal to $x$, $z = x \times y$. Pelvis: origin at the midpoint of the femoral heads, $x$ left to right, $z$ towards the centroid of L4–L5 made orthogonal to $x$. Coordinates are real millimetres. Both frames are built from internal anatomy, which a surface-only input does not have (section 7).

## 4. Training: flow matching

**Normalisation.** Per part and channel, $\mu_c, \sigma_c$ are the mean and standard deviation of the true codes over all training tokens; $z_1 = (Z - \mu) / \sigma$ on the body tokens. (Face $\sigma$: 0.83–1.42; pelvis: 0.27–0.70.)

**Path.** For one person, with $N = |\mathcal T|$:
$$ z_0 \sim \mathcal N(0, I_{N \times C}), \qquad t = \mathrm{sigmoid}(n),\ n \sim \mathcal N(0, 1), \qquad z_t = (1 - t)\, z_0 + t\, z_1 . $$

**Partial reveal.** With probability $\tfrac12$ nothing is revealed. Otherwise a share $r \sim \mathcal U(0, 0.9)$ is drawn and each token is revealed independently with probability $r$; $R \subset \mathcal T$ is the revealed set. A revealed token enters with its clean code through its own projection and without the time embedding.

**Network.** One sequence of fact, shell and body tokens through a bidirectional transformer $v_\theta$ (8 layers, width $d = 384$, 6 heads, pre-norm, 14.6 M parameters):
$$ h^{\text{body}}_g = \gamma(p_g) + e_{\text{part}} + e_{\text{body}} + \begin{cases} \mathrm{Lin}_r(z_{1,g}) & g \in R \\ \mathrm{Lin}_z(z_{t,g}) + \tau(t) & g \notin R \end{cases} \qquad h^{\text{shell}}_g = \mathrm{MLP}(f_g) + \gamma(p_g) + e_{\text{part}} + e_{\text{shell}} $$
with $\tau$ a sinusoidal time embedding through an MLP. The output layer is a linear map to $\mathbb R^C$ on the body tokens, initialised at zero.

**Loss.** Plain velocity regression on the tokens that were not revealed, averaged within the person:
$$ \mathcal L(\theta) = \mathbb E_{\text{person},\, t,\, z_0,\, R}\ \frac{1}{|\mathcal T \setminus R|\, C} \sum_{g \notin R} \big\| v_\theta(z_t, t \mid \text{shell}, \text{facts}, z_{1,R})_g - (z_{1,g} - z_{0,g}) \big\|^2 . $$
No decoder and no labels are used in training.

**Optimisation.** One person per step (uniform over training people, so each part is sampled in proportion to its number of people), AdamW, learning rate $3 \times 10^{-4}$ on a one-cycle schedule with 5% warm-up, weight decay 0.05, gradient norm clipped at 1, 16-bit mixed precision, 20,000 steps. An exponential moving average of the weights, $\bar\theta \leftarrow a\, \bar\theta + (1 - a)\, \theta$ with $a = \min(0.999, \tfrac{i+1}{i+10})$, is used for validation and sampling.

**Validation loss** (fixed noise, $t \in \{\tfrac18, \tfrac38, \tfrac58, \tfrac78\}$, nothing revealed) fell from 1.99 to 0.527; a constant-zero output gives 2.

## 5. Sampling

Nothing revealed, facts given, fixed-step Euler from noise:
$$ z^{(0)} \sim \mathcal N(0, I), \qquad z^{(k+1)} = z^{(k)} + \tfrac1n\, v_{\bar\theta}\big(z^{(k)}, \tfrac kn\big), \quad k = 0, \dots, n - 1, \qquad \hat Z = \sigma\, z^{(n)} + \mu . $$
No guidance. $n$ is picked among $\{4, 8, 16, 32\}$ by the median Dice of one draw on validation people. $n = 4$ won every time (0.419 against 0.411 at $n = 32$).

## 6. Decoding, the fault and the fix

The decoder needs a code at every cell of the grid, but the model only produces codes on $\mathcal T$.

**As first written (wrong).**
$$ \tilde Z_g = \begin{cases} \hat Z_g & g \in \mathcal T \\ 0 & g \notin \mathcal T \end{cases} \qquad \hat L = \arg\max_k D(\tilde Z)_k \ \text{ on } B, \quad 0 \text{ elsewhere.} $$
The zero vector is not the code of "outside" or of "unknown"; it is not a code $E$ ever produces there. Because $D$ reads neighbours up to about three cells away, every body voxel near the body's edge or near the scan's edge was decoded next to codes $D$ had never seen.

Evidence, one test head, with the **true** codes $Z$ on $\mathcal T$:

| Codes outside $\mathcal T$ | Right parotid volume / true | Right eye Dice | Median Dice |
|---|---|---|---|
| true codes (the autoencoder as trained) | 1.0 | 0.88 | 0.814 |
| zeros | 10.5 | 0.27 | 0.692 |
| shell-volume codes (the fix) | 1.0 | 0.91 | 0.815 |

**Fix.** Build the shell-only volume $S$ from the input: $S = 0$ outside the body, $u$ where unknown, and one generic tissue class inside $B$. It needs no anatomy. Then
$$ \tilde Z_g = \begin{cases} \hat Z_g & g \in \mathcal T \\ E(S)_g & g \notin \mathcal T \end{cases} $$
With the generated codes on the same head: right eye Dice 0.22 → 0.85, right parotid 0.05 → 0.64. Using the true codes outside $\mathcal T$ instead of $E(S)$ gives the same scores (0.349 against 0.350), so $E(S)$ loses nothing there.

The fault was in test-time decoding only. It depressed every earlier score of all three models. Corrected results (55 test people, median Dice over classes):

| Part | Flow, one draw | Flow, majority of 3 draws | One-shot regression, same size | Nearest training person |
|---|---|---|---|---|
| Pelvis | 0.583 | 0.636 | 0.528 | 0.544 |
| Face | 0.339 | 0.393 | 0.365 | 0.310 |

**Majority over draws.** Per voxel, the most frequent label among the draws. It raises Dice but breaks structures: on the pelvis 96% of structures in a single draw have a number of connected pieces inside the range of training people, against 67% for the majority.

## 7. A surface without anatomy: frame and size

For a generated character there are no ear canals, femoral heads or millimetres. What is used now:

**Solid.** Inside test by the generalised winding number of the mesh, $w(x) > \tfrac12$.

**Head frame.** $x$ = the normal of the mesh's fitted left-right symmetry plane; origin = the midpoint of the detected ears projected onto that plane; $y$ towards the midpoint of the detected eyes, made orthogonal to $x$; $z = x \times y$.

**Size from bare skin.** The mesh has no unit. With $\{x_i\}$ the bare-skin vertices (hair and loose cloth excluded) in the frame, and $\phi_p$ the signed distance to the body surface of training person $p$,
$$ s^\ast = \arg\min_s\ \min_p\ \operatorname{median}_i\ \big| \phi_p(s\, x_i) \big| , $$
searched in one dimension. Points where person $p$ has no data are dropped; a person counts only if more than half of the points remain.

Sizing by overlap of the whole solid (hair included) gave a head 19% too small and eyes of 0.2 and 1.0 ml. With $s^\ast$ and the symmetry-plane frame the same character gets eyes of 4–8 ml at the positions real eyes have. Her hair still counts as body, so the skull is too large.

## 8. Proposed changes

**P2. One normalisation for training and inference.** Define each part's frame and size from the surface alone (upright axis, symmetry plane, a size functional such as $s^\ast$), apply it to CT people and to characters alike, and retrain. Today training uses anatomy-based frames and inference uses surface-based estimates of them.

**P3. Jitter.** Draw a similarity transform $A = (s, Q, b)$ per training sample, with $s$ near 1, $Q$ a small rotation and $b$ a small shift, and train on $L \circ A^{-1}$: the labels are resampled, re-encoded by $E$, and the shell features and tokens recomputed. The model then sees the errors that P2 will make. The ranges come from a sensitivity measurement on real heads (running).

**P4. Covers (hair, loose cloth).** For a training person, add a cover $H \subset \Omega \setminus B$ adjacent to the surface and present the shell of $B' = B \cup H$. Tokens are taken on $B'$. Each shell token also carries $c_g \in [0, 1]$, the share of its surface that is cover (for a character: the share of its surface marked hair or loose cloth). The target is unchanged: the true codes $E(L)$ on the enlarged token set, where the cover's cells decode to "outside". The loss is the same. At inference, the cells the model decodes as outside are removed from the body.

**F7. Continuous decoder.** Replace $D$ by $D_c : (Z, x) \mapsto \pi(x) \in \Delta^{K-1}$ for any point $x$, trained with cross-entropy at sampled points; the encoder, the codes and the generator are unchanged. A label is still $\arg\max_k \pi_k(x)$, so regions cannot overlap or leave gaps; meshes are the boundaries between regions.

**F3 (already planned). Voxel-level terms.** With the one-step estimate $\hat z_1 = z_t + (1 - t)\, v_\theta$, decode a random crop and add weighted cross-entropy and a Dice term. Not used so far.

## 9. Where I would most like to be checked

1. **The codes have no prior.** $E$ was trained without any regulariser, and flow matching runs on per-channel standardised codes. Is the 4-step optimum a sign that the learned field is close to a straight line to a conditional mean (so sampling adds little beyond regression), or just that Dice rewards smooth outputs? The single-draw gain over one-shot regression is 0.055 on the pelvis and negative on the face (0.339 against 0.365).
2. **Loss weighting.** The loss is a mean over tokens within a person. Deep tokens far from any surface and thin structures count the same. Should tokens be weighted (by distance to the surface, by label entropy inside the cell), or is that better left to voxel-level terms (F3)?
3. **Body tokens that are mostly outside.** A cell with a single body voxel is a body token; its true code mostly encodes "outside". The model predicts these codes too. Is there a cleaner split between what the model generates and what $E(S)$ supplies, for example letting $E(S)$ enter every body token as an input feature?
4. **The fix is a boundary condition, not a guarantee.** $E(S)$ is exact outside the body only because outside and unknown do not depend on anatomy. Inside $\mathcal T$ nothing constrains a generated code to be one that $E$ could produce. Is a projection or a decoder fine-tuned on generated codes worth it?
5. **Time sampling and solver.** Logit-normal $t$ and Euler steps were taken from image models. With $n = 4$ best, is a different time distribution or a consistency-style objective more suitable?
6. **Size functional.** $s^\ast$ takes the best-fitting single person. It is not smooth in $s$ and depends on which people exist. Would a fit to a low-dimensional shape model of the training surfaces be sounder?
7. **Covers.** In P4 the target under a cover is "outside". Thick covers make the true surface ambiguous; the flow should then produce a spread of surfaces. Is a per-token cover share enough conditioning, or does the cover need its own tokens?
8. **Class conventions.** Half of the training heads come from a source that never draws the brain, and its region carries a generic class in the target codes. Conditioning on the list of drawn classes did not make the model produce a brain on request (a character's head got 96 ml, later 635–867 ml once sized correctly, against about 1,400 ml in real heads). Is conditioning the wrong tool, and repairing the labels at the source the only sound one?
