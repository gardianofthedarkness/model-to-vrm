# Han et al. 2024: what we can copy

Desk notes, 2026-11-24. Nothing installed, no datasets fetched.

**Sources read**
- **[D]** Y. Han, *Novel Methods for Efficient Musculoskeletal-Driven Skin Deformation of Animated Characters*, PhD thesis, UCLA
  Mathematics, 2024 ([eScholarship](https://escholarship.org/uc/item/4fk0x7n8)), 184 pp. Read in full for Ch. 1–3 and §4.2, §5.1,
  §6.8; page numbers below are the printed page numbers. Ch. 3 "is based on [HCO24]", the TOG paper (D p.11).
- **[P]** Han, Chen, Ong, Chen, Hicks, Teran, *A Neural Network Model for Efficient Musculoskeletal-driven Skin Deformation*,
  ACM TOG 43(4), 12 pp., doi 10.1145/3658135. **Not read:** the ACM PDF and full-text pages return 403. Only the abstract (Crossref,
  SIGGRAPH archive) was read. The paper cites a "Supplementary Technical Document" with no public URL. Every number below comes from
  [D] Ch. 3. Whether the paper's numbers are the same is *unverified*.
- No project page, code, talk video or dataset was found.

Some equations came out of the PDF garbled (Greek letters, fractions). Where I rebuilt one, it is marked *reconstructed*.

---

## 1. Geometry

- **Source of anatomy:** surface meshes of muscles, bones and skin "created from MRI scans" of one subject. They are asymmetric, so
  the results are asymmetric too (D §3.6 p.36, p.38). Region: the upper body, with 46 muscles, both shoulders, scapulae, clavicles and
  elbows (D §3.4 p.30, §3.5 p.32).
- **Four layers** (D §3.2 p.21):
  1. kinematic bones;
  2. muscle and tendon in **one tetrahedral mesh per muscle**. Tendon is part of the same mesh, with its own Young's modulus;
  3. **fascia**, a thin triangle membrane wrapped around the muscles;
  4. **fat**, a tet layer between the fascia and the outer skin.
- **Muscle meshes:** volumetric tets. All muscles together have about 182 k vertices in the network data (D Table 3.2 p.42) and
  534 966 output values, which is 178 k vertices × 3 (Fig. 3.4 p.31). The PBNG version of the same musculature has 284 k particles
  and 1 097 k elements (D §6.8.2.2 p.118). The mesher used for the muscles is not stated (*unverified*). The thesis's own
  hex-grid embedding mesher for self-intersecting surfaces (Ch. 5) is shown on heads and test shapes, not said to be used on the
  muscles (D §5.8).
- **Fascia:** a triangle mesh of about 145 k vertices (D Table 3.2). How it is built is not described (*unverified*).
- **Fat/skin:** a thin tet layer whose **inner boundary (fascia) and outer boundary (skin) share one triangle topology**, about
  145 k vertices per surface (D §3.3.3 p.29, Table 3.2). The skin output is 868 332 values, which is 289 k vertices × 3 (Fig. 3.4).
  Because the topology matches, **body-fat percentage is changed by moving skin vertices toward their fascia twins** (D p.29–30,
  Fig. 3.7).
- **Rest state:** the A-pose geometry is the undeformed state for every layer. The authors say this is "clearly inaccurate" (D Fig.
  3.1 b1–b3, §3.7 p.43).
- **Muscle–bone attachment:** muscle vertices that lie **inside bones in the A-pose** are Dirichlet nodes that move rigidly with that
  bone (D §3.3.1 p.24). Origin and insertion vertices are picked by hand, as in Teran 2005 (D p.25).
- **Muscle–muscle:**
  - permanent **"fascia" weak constraints** tie a boundary point of one muscle to its closest point on a *different* muscle's
    surface, using barycentric weights (D §3.3.1.2 p.26–27, Fig. 3.1 a3);
  - plus **contact constraints**, switched on each step only where a point has penetrated another muscle, tested with the dot
    product against the normal at the closest boundary point (D p.27).
- **Patch for gaps:** at the scapula, latissimus and trapezius drifted apart, so a **connective-tissue membrane** (triangles) was
  added between them. It is held in tension by `Σ k̂/2·|F_cod|²·A` and its border is tied to the nearest muscle surfaces (D §3.3.1.3
  p.28).
- **Self-intersection cleanup:** none is described for the anatomy. Ch. 5 (Gagniere et al., CGF 2024) embeds self-intersecting
  surfaces in a hex grid without repairing them, then converts to BCC tets. It fails near elbows and shoulders (D §1.3 p.5–8, §5.9).

## 2. Physics

- **Constitutive model:** **isotropic fixed corotated** (Stomakhin 2012 volume fix) for muscle, tendon and fat (D eq. 2.7 p.14,
  §3.3.1 p.23, §3.3.3 p.29). Neo-Hookean and stable Neo-Hookean are supported in PBNG but not used here (D §2.2).
- **Parameters** (D §3.6.3 p.40), all chosen by the authors, **with no tissue citation**:

  | tissue | E | ν |
  |---|---|---|
  | muscle | 1e5 Pa | 0.3 |
  | fat | 1e5 Pa | 0.3 |
  | tendon | 5e5 Pa | 0.3 |
  | membrane | 1e2 Pa | 0.3 |

  - ν = 0.3 because "closer to 0.5 … burdens the solver".
  - Fascia (tie) constraint stiffness = 2e6 × the weighted vertex mass; isotropic.
  - Contact stiffness = 1e8 × the weighted vertex mass along the normal and **0 tangentially**, so contact is frictionless.
  - In the PBNG muscle example: density 1000 kg/m³, k = 1e8 (D §6.8.2.2 p.118).
- **Active model.** Activation does not enter as a stress. It changes the **rest shape**: `F̃ = F · U Σ(a) Uᵀ`, with
  `U = [D, D⊥1, D⊥2]` and Σ diagonal (D §3.3.1.1 p.24–25).
  - Σ is *reconstructed* as Σ₁₁ = (1+a)/(1+γa) and Σ₂₂ = Σ₃₃ = ((1+γa)/(1+a))^(α/2). The order of numerator and denominator is
    garbled in the text. This form gives a fibre stretch of (1+γa)/(1+a) at equilibrium and preserves volume when α = 1.
  - γ = 0.4 and α = 1 were picked "for desirable … visual bulging" (Fig. 3.2 shows a sweep).
  - Activation does **not** change stiffness. The authors chose this because isotropic models converge better (after Li et al.
    2013).
  - Activations go up to **3**, an analogue of scaling the physiological cross-section by 2–3× (D §3.4 p.30).
- **Fibre directions:** solve a Laplace problem with FEM on each muscle mesh, with potential 0 at the origin and 1 at the insertion
  (Inouye 2015). The per-tet gradient is the fibre direction. Then 2 624 **streamlines** on 46 muscles, one seeded at random in each
  origin tet. All of this is precomputed in the A-pose and carried barycentrically (D p.25, Fig. 3.3).
- **Solve order** (D §3.3 p.22–30):
  1. **Muscle + tendon + membrane, solved together.** Dirichlet nodes ride the bones; the energy includes the tie and contact terms
     (eq. 3.1). Contact covers muscle/muscle, muscle/bone and muscle/membrane.
  2. **Fascia.** Vertices within a threshold of the muscle in the A-pose (90 %) are pinned barycentrically to the muscle surface. The
     other 10 % are a tensioned membrane that collides with the static muscles and bones.
  3. **Fat/skin.** Inner vertices are fixed to the fascia result; minimise corotated energy for the rest.
  - The coupling is strictly one-way, outward. **Fascia slides on muscle and bone; fat does not slide on fascia.** The authors
    list fat sliding as future work (D §3.7 p.43).
- **Solver:** quasi-static energy minimisation. Ch. 3 cites Teran et al. 2005 (Newton with a definiteness fix) for the nonlinear
  solve (D p.24). Ch. 6's **PBNG** (position-based nonlinear Gauss-Seidel) was written for this kind of training data. It does
  per-node 3×3 Newton steps with an SPD-projected Hessian, graph colouring, and SOR or Chebyshev acceleration. On the 284 k-particle
  musculature it takes **67 s/frame against 430 s for Newton**, on 8 threads of a Threadripper PRO 3995WX (D §6.8.2.2 p.118,
  §1.4 p.8–10). Which solver produced the Ch. 3 data is not stated (*unverified*).
- **Timing per frame** (D Table 3.2 p.42), CPU only, Ryzen PRO 3995WX with 128 threads: **283 s** muscle, **25 s** fascia,
  **39 s** fat/skin. There is no GPU anywhere.
- **No dynamics** in Ch. 3. Ch. 4 (Jin et al. 2022) adds jiggle with per-vertex **zero-rest-length springs**, learned from a few
  backward-Euler clips and integrated analytically (D §1.2 p.4, §4.5).

## 3. Activations

- **Musculoskeletal model:** not OpenSim's. They build their own:
  - the **lines of action are the 2 624 FEM fibre streamlines**, deformed by the passive muscle network PNN_m (D §3.2 p.22,
    §3.5.2 p.33–34);
  - peak isometric force per muscle from the **Seth et al. 2018 shoulder model**, divided by the number of streamlines (D p.34).
- **Fibre force** (eq. 3.3): `f = σ_max (f_p(l) + a f_a(l)) n`, with
  - f_p = e^(l−1) − 1 for l > 1;
  - f_a = l(2 − l) on [0, 2], deliberately wider than the Hill curve;
  - no force–velocity term, an inextensible tendon, l normalised to the A-pose length (D p.34).
- **Inverse problem:** quasi-static torque balance (virtual work) on **N_C = 20 DOF**: clavicle, shoulder, scapula and elbow,
  both sides (D §3.5 p.32, App. A).
  - Gravity on each muscle's vertices is assigned to one bone.
  - External loads such as dumbbells enter here (D p.35, Fig. 3.8).
  - Objective: min ½ aᵀ σ_max a. This is squared activation weighted by strength, with a 1e10 penalty on negative a.
  - Solved as a KKT system in 3–5 iterations with an N_C × N_C QR (eq. 3.4, D p.35–36).
  - Per-fibre activations are averaged into one value per muscle for the active network.
  - Cost: 180 ms/frame (Table 3.2).

## 4. Network

- **Form** (D §3.2 p.21): `s(X) = LBS(X + PNN(θ) + ANN(a); θ)`. Both networks output A-pose displacements, and the ANN output is
  skinned forward. Muscle (PNN_m, ANN_m) and fat/skin (PNN_s, ANN_s) each get their own pair (D §3.4).
- **Inputs:**
  - PNN: 23 joint rotations as 3×3 matrices, so **207 values**;
  - ANN: **46 activations**;
  - **pose and activation are decoupled**: the ANN has no pose input (D p.30, Fig. 3.4).
- **Architecture** (after Jin et al. 2020, D Fig. 3.4 p.31):
  - FC 2000 → FC 1500, each with BatchNorm and ReLU;
  - a fixed **PCA output layer** with **20** components for the PNN and **92** for the ANN;
  - outputs are vertex displacements: 289 k skin vertices and 178 k muscle vertices.
- **Data:**
  - PNN: **about 6 000 passive frames** over ranges of motion;
  - ANN: **920 active frames**, 46 muscles × 20 activations from 0 to 3, one muscle at a time, presumably in the A-pose
    (*unverified*);
  - targets by inverse LBS;
  - L2 loss on vertices, 80/20 split, 1 000 epochs of SGD (D p.30–31).
- **Accuracy:** per-vertex mean squared error (units not stated), training / evaluation (D Table 3.1 p.38):

  | network | training | evaluation |
  |---|---|---|
  | PNN_s | 1.02e-2 | 9.8e-3 |
  | ANN_s | 2.0e-4 | 1.0e-4 |
  | PNN_m | 6.2e-3 | 6.5e-3 |
  | ANN_m | 6.7e-5 | 5.8e-5 |

  - Training took 4.5–8.5 h per network on the CPU.
  - Otherwise the checks are visual: Fig. 3.5 (unseen poses) and Fig. 3.6 (a biceps curl not in training).
- **Speed:** **36 ms** for the muscle networks and **52 ms** for the skin networks, plus 180 ms for the inverse activation, all on the
  CPU. That is ">1000× faster than simulation" and "real time if slightly fewer curves and mesh vertices are used" (D p.40, Table 3.2).

## 5. Validation

- **Activations against EMG:** shoulder flexion and abduction EMG from Seth et al. 2019, compared with their OpenSim CMC estimates
  (D §3.6.2 p.38–39, Fig. 3.10):
  - the results are qualitative curve overlays for four muscles (superior trapezius, middle and posterior deltoid, biceps);
  - they are "similar and in some cases improved" (biceps late in the task);
  - the posterior deltoid takes too much load in flexion (D p.43).
- **Deformation:**
  - only network against its own simulation (MSE and pictures) and against plain LBS;
  - **no comparison with measured tissue or skin shape** (MRI, scans, ultrasound) and no material validation (D §3.6.1).

## 6. Copy / adapt / skip for our plan

| module | verdict | what | why |
|---|---|---|---|
| **E-PACK** | **copy**: fat/skin as one tet layer whose inner (fascia) and outer (skin) surfaces share topology; Dirichlet "nodes inside bone" for attachments; Laplace fibre field (0 at origin, 1 at insertion) and streamlines seeded per origin tet. **Adapt**: fascia as a membrane pinned barycentrically to muscle where close, tensioned elsewhere | D p.25, 28–30 | Cheap and proven. Shared topology gives fat depth per vertex, which maps onto `skin_fat.npz`, and a body-fat control for free. Streamlines become our E-ACT lines of action. Our deep fascia is a stiff anisotropic shell (§5.2), not their 1e2 Pa membrane. Their mesher does not repair self-intersections, and libuipc needs intersection-free surfaces with a gap, so our T1 repair stays. |
| **E-JOINT** | **skip** | — | Their rig is plain LBS joints and they call the scapula joint inaccurate (D p.28, p.43). Nothing on the patella or sesamoids. |
| **E-ACT** | **adapt**: the idea of taking **lines of action from the simulated or learned muscle geometry** and checking the moment arms of MyoSuite paths against it; adding load (dumbbell, seat) through the static torque balance. **Skip**: their own solver | D §3.5 | MyoSuite static optimisation already does the minimum-effort QP. Their force–length curve is widened for animation, and they have no force–velocity term and a rigid tendon, which breaks our "calibrated against Hill" rule. Their activation > 1 (PCSA fudge) is a hand-set number and is not allowed. |
| **E-MAT** | **skip the values** | D p.40 | E = 1e5 Pa for muscle and fat, ν = 0.3, and stiffnesses scaled by mass are uncited and tuned for the solver. They are 20–100× stiffer than our §5.2 living-tissue values and not near incompressible. Only useful as a "what converges" reference. |
| **E-TIS** | **adapt**: the **staged solve** (muscle → fascia → fat/skin) as a cheap mode and as a warm start for our coupled solve; **tie constraints** between neighbouring muscles plus normal-only contact switched on by penetration, which matches our "no separation, sliding" rule; fat tied to fascia, which matches §5.2. **Skip**: the rest-shape active model with isotropic material (no stiffening, which fails T4 and lesson 4 of §3); A-pose as stress-free | D §3.3 | Their one-way layering ignores fat and skin pushing back on muscle, so the chair (T7) and self-contact (T6) need two-way coupling. PBNG is the same family as Newton VBD (per-vertex Gauss-Seidel with Newton steps and colouring), which supports our VBD choice. Their 283 s/frame on 128 CPU threads is a warning for 4 GB. |
| **E-RUN** | **copy** | D §3.2, §3.4, Ch. 4 | `LBS(X + PNN(θ) + ANN(a))`; a decoupled passive and active network; a PCA output layer (20 / 92); about 6 k passive + about 1 k one-muscle-at-a-time active frames as a starting data budget; Ch. 4 zero-rest-length springs for jiggle (T8). **Add** what they lack: contact and chair inputs (§4.1 E-RUN), and pose × activation cross terms (the ANN has no pose input, so bulging at a flexed joint is extrapolated). |
| **E-TEST** | **adapt**: EMG overlay (Seth 2019 shoulder) as an E-ACT sanity check when we reach the arm; network-vs-simulation error on held-out clips (T10) | D §3.6 | Their validation has no measured-shape check. Our T5–T9 against MRI and ultrasound data go beyond it. |

**Bottom line.** Han 2024 is the right template for **E-RUN** and for the **fat/skin geometry**. It is not a source for materials,
the active law or two-way coupling, which our plan must do better. The TOG paper and its supplement are still unread; get them
through a library login before quoting any number as "the paper's".
