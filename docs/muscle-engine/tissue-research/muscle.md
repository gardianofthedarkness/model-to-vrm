# Skeletal muscle: mechanical behaviour in living people (desk research)

Scope: what to put into E-MAT for muscle, and what to check it against. Sources were web-searched and, where marked **[read]**, the full text or a primary table was read. **[abstract]** means only the abstract or index record was read. *unverified* means the value comes from memory or a secondary quote and still needs checking against the primary source. No datasets were downloaded.

Units: 1 N/cm² = 10 kPa. "Shear modulus" from elastography is μ = ρ·c² with ρ = 1000 kg/m³ and a linear-elastic assumption. It is a small-strain, roughly 100–500 Hz value, not a large-strain hyperelastic constant.

---

## 1. Passive behaviour

### 1a. Measured values

| Quantity | Value | Method / subject | Source |
|---|---|---|---|
| Shear modulus **along** fibres, relaxed, elbow 90° | 5.40 ± 0.25 kPa | SSI shear-wave elastography, human brachialis in vivo | Gennisson et al. 2010, Ultrasound Med Biol 36:789, doi:10.1016/j.ultrasmedbio.2010.02.013 **[read]**, Table 3 |
| Same, muscle passively stretched (elbow 115°, 140°, 165°) | 8.46, 15.8, 29.5 kPa | same | same |
| Shear modulus **across** fibres, relaxed | 0.05–1.03 kPa, almost unchanged by stretch | same | same |
| Shear viscosity (Voigt) along / across | 0.6–1.6 / 0.5–1.5 Pa·s | same | same |
| Shear stiffness at rest, vastus medialis | 3.69 ± 0.80 kPa (1D MRE); 4.36 ± 0.98 kPa (2D MRE) | MR elastography, human in vivo | Bensamoun et al. 2008, https://ww2.odu.edu/~sringleb/publications/pub_PDFs/Bensamoun_1DMRE.pdf **[abstract]** |
| Passive-tension stiffening | +14.5 ± 1.77 kPa per kg load | MRE, human in vivo | Dresner et al. 2001, J Magn Reson Imaging 13:269 **[abstract]** |
| Transverse compression, Ogden fit | μ = 15.6 ± 5.4 kPa, α = 21.4 ± 5.7 | Indentation of rat tibialis anterior **in vivo**, inverse FEM | Bosboom et al. 2001, J Biomech 34:1365, doi:10.1016/S0021-9290(01)00083-5 **[abstract]** |
| Transverse indentation shear modulus, short term / long term | 8.5 / 0.61 kPa (not preconditioned); 5.7 / 0.81 kPa (preconditioned) | Rapid indentation of porcine gluteus, fresh, in vitro; plateau after about 20 s | Palevski et al. 2006, J Biomech Eng 128:782 **[abstract]** |
| Tension along vs across fibres | along ≈ 447 kPa, across ≈ 22 kPa (linear-region moduli) *unverified* | Uniaxial tension, rabbit tibialis anterior, ex vivo | Morrow et al. 2010, J Mech Behav Biomed Mater 3:124, doi:10.1016/j.jmbbm.2009.03.005 (numbers *unverified*) |
| Compression, cross-fibre vs fibre | cross-fibre about 1.2× stiffer at 2 % cycles | Unconfined compression, fresh porcine, up to 30 % | Van Loocke et al. 2009, J Biomech 42:1038, doi:10.1016/j.jbiomech.2009.02.022 **[read]** |
| Compression stress level | about 1–3 kPa at 25–30 % compression (slow) | same | same, Figs 2–5 |
| SWE validity | SWE shear modulus tracks tensile Young's modulus linearly | Porcine muscle, SWE vs materials testing | Eby et al. 2013, J Biomech 46:2381, https://pmc.ncbi.nlm.nih.gov/articles/PMC3818126 **[abstract]** |

What this means:

- **Anisotropy depends on the load direction.** In tension and in shear waves, along-fibre is much stiffer: about 5× (relaxed) to 30× (stretched) in Gennisson's in-vivo shear data, and about 20× in Morrow's tension data (*unverified*). In compression the two directions are close, and cross-fibre is slightly stiffer (Van Loocke 2009). A transversely isotropic model with a fibre term that acts **only in fibre tension** fits both observations. Blemker's model does this.
- **Stiffening with stretch is strong.** The along-fibre shear modulus rose about 5.5× over 75° of elbow extension (Gennisson 2010). The across-fibre value stayed flat.
- **In-vivo compression gives a higher modulus than ex-vivo relaxed indentation.** Bosboom's μ of about 16 kPa was measured in a perfused, toned, living muscle. Palevski's long-term 0.6–0.8 kPa comes from dead tissue after relaxation. The current engine value, G = 500 Pa, sits at the ex-vivo, fully relaxed end.

### 1b. Constitutive fits used in FEM

| Model | Constants | Source |
|---|---|---|
| Blemker–Delp, transversely isotropic: Ψ = G1·B̃1² + G2·B̃2² + F_fibre(λ) + U(J) | G1 = 500 Pa (along-fibre shear), G2 = 500 Pa (cross-fibre shear), P1 = 0.05, P2 = 6.6, σ_max = 0.3 MPa, λ_ofl = 1.4 (Blemker); FEBio example uses Lofl = 1.07 and k = 1 MPa | Blemker, Pinsky & Delp 2005, J Biomech 38:657; Blemker & Delp 2005, Ann Biomed Eng 33:661, doi:10.1007/s10439-005-1433-7; FEBio manual https://febiosoftware.github.io/febio-feature-manual/features/solid_material_muscle_material **[read]**. γ1 = 0.05, γ2 = 6.6, λ0 = 1.4 confirmed in https://pmc.ncbi.nlm.nih.gov/articles/PMC2857665 **[abstract]** |
| First-order Ogden, transverse compression | μ = 15.6 kPa, α = 21.4 | Bosboom 2001 (above) |
| Neo-Hookean (sitting models) | muscle instantaneous G ≈ 8.5 kPa from Palevski *unverified* (as used by Gefen's group) | Linder-Ganz et al. 2007, J Biomech 40:1443 *unverified* |
| Strain-dependent Young's moduli (SYM), transversely isotropic, quasi-static compression | fitted per direction, R² = 0.99 up to 30 % | Van Loocke et al. 2006, J Biomech 39:2999 **[cited in Van Loocke 2009, read]** |

Note that G1 = G2 = 500 Pa in Blemker's model came from matching whole-muscle behaviour, not from tissue tests. It is about 10× below the in-vivo elastography values. The cross-fibre SWE value (≈ 1 kPa) is the closest match.

### 1c. Bulk vs shear

Muscle is about 80 % water (Van Loocke 2009 **[read]**). The bulk modulus is therefore close to that of water, about 2 GPa *unverified*. FEM practice treats muscle as nearly incompressible:

- FEBio's example uses k = 1 MPa with G = 500 Pa, so K/G = 2000.
- Volume change during contraction is reported as below 0.1 % *unverified* (Baskin & Paolini 1967).

For VBD, K/G is a stability choice, not a material fact. Use K/G ≥ 100 (Poisson's ratio ≥ 0.495) and report the measured volume drift as the check. This matches the T2 requirement "volume ≈ 1".

---

## 2. Active behaviour

### 2a. Stiffness vs activation

| Finding | Numbers | Method | Source |
|---|---|---|---|
| SSI shear modulus vs joint torque | linear, R² = 0.982 (ramp); 11 subjects; pennation did not change the accuracy | SSI, abductor digiti minimi | Bouillard, Nordez & Hug 2011, PLoS ONE 6:e29261, doi:10.1371/journal.pone.0029261 **[abstract]** |
| Shear modulus vs EMG | R² ≈ 0.94 *unverified*, linear up to about 60 % MVC *unverified* (biceps) | SSI | Nordez & Hug 2010, J Appl Physiol 108:1389, https://hal.archives-ouvertes.fr/hal-03409603 **[abstract]** |
| Along-fibre stiffening with load | about 20× from 0 to 4 kg of isometric load; across-fibre **constant**; viscosity ×10 along, ×3 across | SSI, brachialis | Gennisson 2010 **[read]**, discussion |
| MRE, rest → 20 % MVC | 3.69 → 9.52 kPa (1D); 4.36 → 9.22 kPa (2D), vastus medialis | MRE | Bensamoun 2008 **[abstract]** |
| MRE quadriceps contraction | +1.10 ± 0.34 kPa | multifrequency MRE | Klatt et al., ISMRM 2009 **[abstract]** |

**Consequence for the engine:** activation stiffness belongs on the **fibre direction**. The current model adds isotropic activation-proportional stiffness (g_max 0.321). The data show cross-fibre stiffness almost unchanged (Gennisson 2010). Shear-wave speed along the fibres reflects fibre tension, so the stiffness increase in the data comes from the fibre stress σ_max·a·f_L(λ) itself. A Blemker-type active fibre term reproduces it without a separate isotropic g term. T4 can test exactly this: indent along vs across the fibres at several values of a.

### 2b. Hill relations and active stress

- **Active force–length (FEBio/Blemker form) [read]:**
  - f = 1 − 4(1 − λ/λ_ofl)² for 0.6 < λ/λ_ofl < 1.4
  - f = 9(λ/λ_ofl − 0.4)² below 0.6
  - f = 9(λ/λ_ofl − 1.6)² above 1.4
- **Passive fibre law:** f_p = P1·(e^{P2(λ/λ_ofl − 1)} − 1) above λ_ofl, continued linearly beyond λ*.
- **Force–velocity:** not part of Blemker's static model. Use Zajac 1989 or Millard et al. 2013, doi:10.1115/1.4023390. Both use v_max ≈ 10 L_opt/s *unverified*, concentric curvature about 0.25 *unverified*, and eccentric plateau about 1.4–1.8 × F_iso *unverified*. MyoSuite's MuJoCo muscles already carry such curves, so take the curve from MyoSuite to stay consistent.
- **Peak isometric stress (specific tension):**
  - Human in-vivo estimates span 2–73 N/cm² (29 studies, 95 values). Persad et al. 2024, J Appl Physiol 137:945, doi:10.1152/japplphysiol.00296.2024 **[abstract]**. The review recommends 26.8 N/cm² (0.27 MPa); that figure is *unverified* because only a secondary quote was seen.
  - A direct intraoperative measurement in human gracilis gave 171 ± 84 kPa, n = 12 **[abstract]** (https://researchdiscovery.drexel.edu/esploro/outputs/journalArticle/Direct-measurement-of-human-skeletal-muscle/991021855575804721).
  - The current σ_max = 0.3 MPa is inside the cited band, close to the Persad recommendation.

### 2c. Bulging and cross-section change

- Ultrasound thickness and pennation change steeply up to about 30 % MVC, then barely change. Thickness change is detectable from about 4 % MVC in biceps and 5 % in brachialis. Hodges et al. 2003, Muscle Nerve 27:682, doi:10.1002/mus.10375 **[abstract]**. The per-muscle % values are *unverified*.
- Vastus medialis thickness changes by about 17–20 % between rest and contraction at 90° knee flexion, https://pmc.ncbi.nlm.nih.gov/articles/PMC3872419 **[abstract]**. The exact contraction level is *unverified*.
- No vastus lateralis rest-to-MVC % with a clear source was found. This is an open item for T5.

The direction for T5 is clear: under load the muscle shortens, thickens and stiffens along its fibres. Magnitude: about 15–20 % thickness gain and 2–20× along-fibre shear stiffness, depending on level and method.

---

## 3. Viscoelasticity and damping

| Item | Value | Source |
|---|---|---|
| Viscous share of stress | ≈ 50 % of total stress already at 0.5 %/s compression; peak stress ×5 from 0.05 to 10 %/s at 30 % | Van Loocke et al. 2008, J Biomech 41:1555, doi:10.1016/j.jbiomech.2008.02.007 *DOI unverified*; numbers confirmed in Van Loocke 2009 **[read]** |
| Prony time constants (fitted, NLV-SYM, 0.4 Hz cycles) | τ = 0.6, 6, 30, 60, 300 s. Fibre weights p = 0.374, 0.285, 0.077, 0.078, 0.101. Cross-fibre weights p = 0.097, 0.532, 0.072, 0.000, 0.086. Elastic k1, k2, k3 = 1.005, 2.989, 10.985 kPa (fibre) and 1.293, 0.911, 10.869 kPa (cross-fibre) | Van Loocke 2009, Table 2 **[read]** |
| Frequency dependence | Cyclic stress amplitude rises ≈ 4–5× from 0.22 to 80 Hz at 10 % amplitude (25 % mean strain); quasi-linear viscoelastic (QLV) models miss the stiffening above 5 Hz | Van Loocke 2009 **[read]** |
| Single-τ in-vivo fit | δ = 0.549 ± 0.056, τ = 6.01 ± 0.42 s (rat TA, in-vivo indentation) | Bosboom 2001 **[abstract]** |
| Short/long-term modulus ratio | ≈ 10:1, plateau ≈ 20 s | Palevski 2006 **[abstract]** |
| Shear viscosity at audio-range shear waves | ≈ 0.5–1.6 Pa·s relaxed; ×10 along fibres when loaded | Gennisson 2010 **[read]** |
| Wobbling mass, in vivo | Soft-tissue oscillation 3–55 Hz, excursion 3 mm to 4 cm relative to bone in running/landing; thigh dissipates more energy horizontally | Schmitt & Günther 2011, Arch Appl Mech, doi:10.1007/s00419-010-0458-z **[abstract]** |
| Muscle activity raises damping | Vibration frequency and damping rise with muscle force and shortening velocity (skin-mounted accelerometers on quadriceps and triceps surae, heel impacts) | Wakeling & Nigg 2001, J Appl Physiol 90:412; J Biomech 34:539 **[abstract]**. Damping-coefficient values *unverified*, e.g. thigh ≈ 10–20 Hz, damping ratio ≈ 0.2–0.4 |

For T8 (drop and settle): a relaxed thigh should ring at roughly 10–20 Hz *unverified* and decay within a few cycles. With activation, the frequency should rise and the decay should be faster.

---

## 4. Muscle pressed together / external load

| Item | Value | Source |
|---|---|---|
| Sitting, gluteus under ischium | Peak principal compressive strain ≈ 74 %, peak compressive stress ≈ 32 kPa (healthy). Values *unverified*: the extracted abstract text was garbled | Linder-Ganz et al. 2007, J Biomech 40:1443, doi:10.1016/j.jbiomech.2006.06.020 (DOI *unverified*). MRI of the undeformed and sitting buttock plus a subject-specific FE model |
| Paraplegics vs healthy | Compression, tension, von Mises and shear strain 1.2×, 3.1×, 1.4× and 1.4× higher | Linder-Ganz et al. 2008, J Biomech 41:567 **[abstract]** |
| High-fidelity seated thigh/buttock FE model | 28 muscles plus fat and skin; muscle/fat parameters from heuristic optimisation against seated MRI (gravity loading). The constitutive form and constants are *unverified*, so read Section 2 of the paper | Al-Dirini et al. 2016, Ann Biomed Eng 44:2805, doi:10.1007/s10439-016-1560-3 **[abstract]** |
| In-vivo 3D strain, sitting | 10 subjects, MR digital volume correlation, unloaded / partial / full load. Gluteus maximus slides about 13.7 mm away from the ischial tuberosity; muscle strains more than fat | Zappalà et al. 2024, J Biomech 163:111913, doi:10.1016/j.jbiomech.2023.111913, open access **[abstract]** |
| Transverse load reduces muscle force | about 13 % force loss (rat gastrocnemius, Siebert 2014a); up to 15 % force and 35 % rate of force development loss in isolated muscle. Depends on transverse force, tested at 1.1–3.2 N/cm² | Siebert et al. 2016, J Biomech, PubMed 26976226 **[abstract]** |

For T7 (paired chair test): the in-vivo targets are the 13.7 mm gluteal slide and the direction of displacement (down and away from the ischium) from Zappalà 2024. The two independent sources agree that muscle strains more than fat. For T5/T9: a compressed muscle produces slightly less force, so this is a second-order term.

---

## 5. Variation and what to use for the small figure

- **In vivo vs ex vivo.** In-vivo (toned, perfused) values are about 10× above ex-vivo relaxed long-term values (Bosboom 16 kPa vs Palevski 0.6–0.8 kPa). Van Loocke tested within 2 h post mortem to limit drift.
- **Joint angle and muscle length** dominate passive SWE values (5 → 30 kPa in Gennisson 2010). Biceps femoris SWE repeatability work also stresses joint position, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6716782/ **[abstract]**.
- **Muscle group.** MRE databases exist for biceps, flexor digitorum profundus, trapezius, tibialis anterior, gastrocnemius and soleus (Ringleb et al. 2007, review) **[abstract]**. The spread is roughly 2–15 kPa relaxed *unverified*.
- **Age and sex.** Effects are small next to length and activation. Eby et al. 2015 (Clin Biomech) report effects of sex and age on passive SWE stiffness. Direction and size are *unverified*.
- **Small stylised figure (0.98 m).** Moduli, σ_max and time constants are intensive: keep them at human values.
  - Gravity strain scales as ρ·g·L/G, so a half-size body sags about half as much.
  - Wobble frequency scales as about 1/L, so a half-size thigh rings about 2× higher. This follows from dimensional analysis, not a source.
  - Do not retune materials to make the figure look softer. Any style choice must sit outside E-MAT.

---

## 6. Datasets to verify a simulation

Nothing was downloaded. "Licence" means what the landing page states.

| Dataset | Contents | Size / format | Access / licence | URL | Plan test |
|---|---|---|---|---|---|
| **van Riel et al. 2026, in-vivo validation set for dynamic volumetric MRI** | 9 volunteers; thigh muscles deformed by a pneumatic cuff. Validation scans step 0→80→0 mmHg in 20 mmHg steps, 22 s each; dynamic cycles; anatomical reference, Dixon images, muscle segmentation masks | ISMRMRD .mrd (HDF5 k-space) + NIfTI; total size not stated | Zenodo, public; dataset licence not stated on arXiv (paper CC BY 4.0) | https://doi.org/10.5281/zenodo.18459637 ; paper https://arxiv.org/abs/2602.08918 | **T5/T7**: cuff = known external pressure on a thigh, so compare simulated vs MRI displacement field |
| **Zappalà et al. 2024, buttock MR-DVC** | 10 healthy subjects, semi-recumbent, unloaded / partial / loaded buttock; 3D displacement and strain fields | not stated | "released as a public dataset"; link in article header; paper CC BY | https://orca.cardiff.ac.uk/165053 ; doi:10.1016/j.jbiomech.2023.111913 | **T7**, also T9 |
| **Al-Dirini seated MRI** (same group's earlier data) | seated, gravity-loaded buttock/thigh MRI of one subject | — | *unverified* whether open; request from authors | doi:10.1007/s10439-016-1560-3 | T7 |
| **OpenArm (SimTK) Multisensor 1.0 / 2.0** | Brachioradialis cross-section ultrasound under elbow load + sEMG + force (+ acoustic myography); 2.0 adds trajectories | not stated | SimTK project; free account; licence on project page | https://simtk.org/projects/openarm | **T4/T5** analogue: thickness/shape vs force and EMG |
| **Zenodo 232032**, EMG + M-mode ultrasound | 10 men, isometric elbow flexion 20–70 % MVC; EMG, torque, M-mode | not stated | Zenodo, licence on record | https://doi.org/10.5281/zenodo.232032 | T5 (thickness vs force) |
| **Zenodo 7081639** | 2-D ultrasound videos of distal biceps myotendinous junction vs elbow angle and tendon load | not stated | Zenodo | https://zenodo.org/records/7081639 | T3 (tendon/muscle length split) |
| **UMUD** catalogue | 14 datasets, 76 124 images / 2 674 videos, 1 901 participants; filter by muscle (VL fascicle sets, DeepACSA, DL_Track_US) | various | per-dataset licences | https://eprints.glos.ac.uk/15848/ (paper); DL_Track_US: https://zenodo.org/records/7866598 ; DeepACSA: https://data.mendeley.com/datasets/cjztczmd4w | T5: VL architecture at rest/contraction |
| **Leg-3D-US (TUM)** | 3D ultrasound volumes of calf muscles | — | GPL stated on page | https://www.cs.cit.tum.de/camp/publications/leg-3d-us-dataset | T1 geometry sanity |
| **FALLMUD (CEA)** | 812 lower-leg ultrasound images with fascicle/aponeurosis masks | — | licence not shown | https://kalisteo.cea.fr/?p=3568 | T1/T3 orientation check |
| **K2MUSE** | lower-limb walking: kinematics, kinetics, A-mode ultrasound, sEMG | — | see paper | https://arxiv.org/abs/2504.14602 | T9 (motion + activation + muscle shape) |
| **Van Loocke 2006/2008/2009 compression data** | fibre / cross / 45° / 60° ramp, relaxation and 0.22–80 Hz cyclic tests, porcine | — | authors stated data "will be made freely available" (2009); location *unverified*; supplementary doi:10.1016/j.jbiomech.2008.10.038 | — | **T2/T8** material-point check |
| Elastography (Gennisson 2010; Bouillard 2011) | tables of μ vs angle / torque only | in papers | published tables | above | **T4** |

No open in-vivo muscle indentation dataset was found. Bosboom 2001 and Palevski 2006 are paper-only.

---

## 7. Recommended E-MAT muscle row (cited)

| Parameter | Value | Why / source |
|---|---|---|
| Model | Blemker-type transversely isotropic (B̃1, B̃2 shear terms + fibre term that acts in tension only) | Blemker & Delp 2005; FEBio "muscle material" |
| Density | 1000 kg/m³ (SWE convention; muscle ≈ 1060 *unverified*) | Bouillard 2011 |
| Cross-fibre shear G2 (long-term) | 0.8 kPa | Palevski 2006 preconditioned G_L; matches Gennisson ⊥ ≈ 1 kPa |
| Along-fibre shear G1 (relaxed, rest length) | 5 kPa | Gennisson 2010, elbow 90° |
| Instantaneous / long-term ratio | ≈ 10 (G_short 5.7 kPa) | Palevski 2006 |
| Prony series | τ = 0.6, 6, 30 s with Van Loocke fibre weights 0.374, 0.285, 0.077 (drop 60/300 s for animation) | Van Loocke 2009 Table 2; Bosboom τ = 6 s agrees |
| Fibre passive | P1 = 0.05, P2 = 6.6 | Blemker 2005 / FEBio |
| λ_ofl | 1.4 (Blemker) — check against FEBio's 1.07 before use | open question |
| σ_max | 0.27–0.30 MPa | Persad 2024 (26.8 N/cm², *unverified*); Blemker 0.3 MPa |
| Active force–length | Blemker quadratic (Section 2b) | FEBio |
| Force–velocity | take from MyoSuite muscle model | consistency with activations |
| Activation stiffness | **none extra**: stiffening comes from the active fibre stress; cross-fibre stays at G2 | Gennisson 2010 (⊥ constant, ∥ ×20) |
| Bulk | K/G ≥ 100 for stability; target volume change < 1 % | FEBio example K/G = 2000 |
| Transverse-load force loss | optional, about −13 % at 1–3 N/cm² | Siebert 2014/2016 |

**Change from the current engine.** Keep σ_max = 0.3 MPa. Raise the passive G from 500 Pa to about 0.8 kPa cross-fibre / 5 kPa along fibre, and add viscosity. Remove the isotropic g_max activation term, or limit it to the fibre direction, and test it in T4.

## 8. Which dataset checks which scene

- **T2, T3**: Van Loocke material-point curves; force–length from FEBio/Blemker; Zenodo 7081639 for muscle–tendon length split.
- **T4**: Gennisson 2010 ∥/⊥ table (anisotropy of activation stiffening); Bouillard 2011 linear μ–torque; Bensamoun 2008 rest → 20 % MVC ratio ≈ 2.5.
- **T5**: van Riel cuff MRI (external load); OpenArm and Zenodo 232032 (thickness vs force); Hodges 2003 shape (saturation above 30 % MVC).
- **T7**: Zappalà 2024 DVC (13.7 mm gluteal slide, strain maps); Linder-Ganz 2007 peak values; Al-Dirini 2016.
- **T8**: Van Loocke 2009 frequency stiffening (×4–5 from 0.2 to 80 Hz); Schmitt & Günther 2011 wobble range (3–55 Hz); Wakeling & Nigg 2001 (damping rises with activation).
- **T9**: K2MUSE (motion + EMG + ultrasound).

## 9. Open questions

1. Primary numbers for Morrow 2010 (tension anisotropy), Linder-Ganz 2007 (74 % / 32 kPa) and Al-Dirini 2016 (muscle constants) need a full-text read.
2. Blemker λ_ofl: 1.4 (papers) vs 1.07 (FEBio example). Which one fits the Z-Anatomy rest lengths?
3. Vastus lateralis thickness change rest → MVC under load: no clean cited % found yet. Look at Werkhausen et al. 2023 (PeerJ 15194) or the UMUD VL sets.
4. Thigh wobble damping ratio with a primary number (Wakeling & Nigg 2001 full text).
5. Whether the Zappalà DVC data and the Van Loocke data are actually downloadable, and their licences.
6. Elastography μ is a small-strain value at about 100–500 Hz. How much of it should go into a large-strain hyperelastic G? This needs a T4-style fit, not a hand choice.
7. Van Loocke's data are porcine and post-mortem. No in-vivo human compression-relaxation time constants were found.
