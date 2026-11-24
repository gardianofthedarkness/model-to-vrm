# Plan: an anatomy-based muscle engine

Drafted 2026-11-24, for review. Status: **draft, nothing here is decided until the user signs off the decisions in §9.**
It builds on [sit-physics-research-handoff.md](sit-physics-research-handoff.md) (the experiments so far, Tests 1–5) and takes its
input from the interior model ([interior-model-plan-v7.md](../interior-model-plan-v7.md), milestone 21 *Rig*).

## 1. Why a plan now

We have been running one-off experiments: one muscle tube, one flesh mesh, one knee, one stance. Each answered a question, and
together they show that the questions are **not separable**:
- An isolated muscle sags 40–70 mm under gravity (Test 5), so its shape under load cannot be measured without the neighbours, the
  fascia, the fat and the skin around it.
- The knee crease inverts tets (Test 3) because the patella is wired to the wrong bone and there is no crease contact.
- Two silent geometry bugs (tet winding, bone winding) invalidated earlier results for days (Tests 1 and 4), because no step
  checked its own inputs.
- The activation (MyoSuite) and the tissue (Newton) use two different bodies and two meanings of "activation".

So the next step is an engine with fixed interfaces, a test harness and one asset format, not another experiment.

## 2. The product

**Input:**
1. an **anatomy pack** for one character: bones, muscles with tendons and fibres, fascia, fat, skin, joints, all in one frame
   (today: the Z-Anatomy atlas warped into the elf; later: milestone 21 of the interior model);
2. a **motion** (ARDY clip mapped to the skeleton);
3. an **environment**: floor, chair, other bodies, gravity, carried weights.

**Output:**
1. per frame, the **deformed tissue**: every muscle, the fat and the skin surface;
2. per frame, the **mechanical state**: activations, muscle forces, contact pressures, joint loads;
3. a **run-time model** small enough for a game or the VRM export, trained on 1 and 2.

### 2.1 Subject profile (decided by the user, 2026-11-24)

The elf is an elf only as a character. Physically she is a **mature adult woman**. Tissue properties and their layout change with
age, sex and body type, so every test uses **one fixed profile: a healthy woman of about 28** (source populations of women aged
roughly 20–35 preferred).

Rules that follow from it:
- **Every E-MAT value records its source population:** sex, age range, body build or BMI, and whether it was measured in vivo or
  ex vivo.
- **Priority:** women about 20–35 in vivo; then mixed-sex young adults in vivo; then any adult in vivo; ex-vivo or animal data
  last. A value from a weaker source is flagged. If the literature gives a sex or age effect, that effect is applied and cited;
  otherwise the gap is stated, never guessed.
- **Geometry comes from her own body,** not from population averages: fat depth from her surface (`skin_fat.npz`), muscle and
  bone shapes from the warped atlas. Known mismatch: the atlas (BodyParts3D) and myoLegs (Rajagopal 2016) are both **adult
  male**. Their muscle volumes, attachment sites and fat layout need a female correction; until then this is stated in every
  result.
- **Body build** follows her mesh: her BMI-equivalent is computed from her own volume and height, and source data are chosen to
  match it.
- Changing the profile later (another age, sex or build) means swapping the E-MAT population, not editing numbers.

**What it must reproduce** (each becomes a test in §6):

| phenomenon | example |
|---|---|
| active muscle is stiff and bulges | same pose, the leg that carries the body has a harder, thicker quadriceps than the hanging leg |
| muscles slide on each other and on bone | the rectus stays in its groove while the knee bends |
| sesamoid bones ride on tendons | the patella tracks the femoral groove |
| skin, upper fat and superficial fascia move as one; the muscle slides under the deep fascia | under 1 mm slip skin↔superficial fascia, 5–6 mm muscle↔fascia; skin moves 8–70 mm against bone (§5.2) |
| skin is a stretch-stiffening membrane | no skin flips at the hip and knee creases |
| contact with the world | the buttock and thigh flatten on the chair, displaced down and along the thigh |
| self-contact | calf against thigh at deep knee bend, crease folds instead of inverting |
| inertia and damping | flesh jiggles after a landing and settles within a cited time |
| loads are consistent | seat and feet reactions in the solve match the loads the activations were computed for |

## 3. Lessons we already paid for

| lesson | from | consequence for the engine |
|---|---|---|
| Newton drops tets with negative volume silently | Test 1 | the pack compiler normalises winding and every solver build asserts its element count |
| atlas bones are wound inward | Test 4 | the same: one orientation check for every closed surface |
| a muscle alone sags; neighbours and fascia hold it | Tests 2, 5 | the smallest unit we simulate is a **compartment** (all muscles of a region inside their fascia), never one muscle |
| active strain alone keeps muscle soft | Test 5 | activation must change stiffness (`ACTIVE=stiff`, or Blemker's active fibre stress) |
| activation ≠ shortening | review point §5 | the active law is calibrated against Hill force–length, not set as a·g_max |
| the patella must follow its own kinematics | Test 3 | joints in the pack include coupled sesamoid motion |
| quasi-static by zeroing velocity converges slowly | Tests 2, 5 | a real static solver (or damping) is needed for comparison poses |
| contact is weaker than active muscle | Test 5 (up to 4.9 mm into bone) | contact stiffness is set from the material stiffness, and penetration is a test metric |
| MyoSuite's body is not the elf | handoff §4 | the activation model is built from the same pack as the tissue model |

## 4. Architecture

```
        ┌──────────────────────── anatomy pack (one file per character) ────────────────────────┐
        │ bones · joints (+ patella coupling) · muscles (tets, fibres, attachments, tendon) ·    │
        │ fascia shells · fat/skin tets · material ids · provenance of every number             │
        └───────────────┬───────────────────────────────────────────────────┬───────────────────┘
                        │                                                   │
   E-ACT  activation model (line-of-action, CPU)             E-TIS  tissue solver (FEM, GPU)
   MuJoCo/MyoSuite rebuilt from the pack;                    Newton VBD, layered compartments,
   static opt → CMC → RL;  contacts from the environment    active stiffness, contacts, damping,
        │  a_i(t), joint loads, contact loads                      static + dynamic modes
        └──────────────────────────────► coupling ◄───────────────────┘
                                   (one way first; seat/foot loads checked both ways)
                                            │
                               E-RUN run-time model (subspace / neural), VRM export
   E-TEST harness: unit scenes, metrics, reference data, run on every change
```

### 4.1 Modules

**E-PACK, the anatomy pack compiler.** Turns the warped atlas (later the generated labels) into a simulation asset.
- Selects the structures that matter for the region, with the rest left to the passive fat.
- Repairs and checks every closed surface: orientation, self-intersection, overlaps between neighbours, volumes against the
  surfaces (Sartorius failed this: 22.6 against 14.0 cm³).
- Meshes each compartment conformally: muscles touching each other share no nodes but have matched contact surfaces; fat fills
  the space to the skin.
- Fibres (harmonic field, later Blemker & Delp templates), attachment and tendon patches, aponeuroses.
- Fascia (fascia lata, crural fascia, intermuscular septa) as thin shells or a stiff boundary layer.
- Writes one versioned file with a manifest of where every number came from.

**E-JOINT, skeleton and joints.** Joint centres and axes from the bone fits; the patella and other sesamoids coupled to the knee
angle (Rajagopal/Arnold coupling) or solved by contact plus a ligament length; ligaments as tension-only springs only where cited.

**E-ACT, activations.** The MyoSuite/MuJoCo model rebuilt from the pack (scale segments, move attachment sites to the atlas
patches), with the environment's contacts. In order of cost:
1. static optimisation (works now, per frame);
2. computed muscle control or tracking with activation dynamics (CPU, minutes);
3. RL imitation (HF Job, asked first).
Reserves are reported, never hidden; a large reserve means a missing muscle group.

**E-MAT, material library.** One table: muscle passive, muscle active (σ_max, g_max, force–length), tendon, aponeurosis, fascia,
fat, skin, damping, friction between layers. Every value has a citation and a range; nothing is tuned. The calibration of the
active law against Hill force is a test, not a fit by hand.

**E-TIS, the tissue solver.** Newton VBD on the GPU.
- Layers: muscles (active, anisotropic), fascia, fat, skin membrane.
- Interfaces, each chosen per pair: glued, tied, sliding with friction, or contact only.
- Contacts: bone, neighbour, self (creases), world (chair, floor).
- Modes: dynamic (with cited damping) and **static** (a converged equilibrium for comparison poses).
- Budget: 4 GB VRAM. A thigh compartment with all its layers is about 250 k tets (7 muscles measured at 15–23 k each).

**E-RUN, the run-time model.** Learns flesh = skin(bones) + correction(pose, activation, contact, recent history) in bone-local
coordinates, by region. Chair and load are inputs, so the same pose gives different shapes. Han et al. 2024 is the template.

**E-TEST, the harness.** Every scene in §6 as a script with a pass threshold, results written to json and a picture, run before
any commit that touches the engine. Viewers and notebooks read the same outputs.

## 5. Reuse before building

Desk survey, 2026-11-24. Nothing was installed. Anything marked *unverified* must be checked before we rely on it.

| need | best existing piece | status for us | use |
|---|---|---|---|
| reference design for the layer stack | **Ziva VFX**: muscles with fibres and excitation, fascia cloth shrink-wrapped over them, fat volume to the skin, sliding attachments ([80.lv](https://80.lv/articles/how-to-animate-a-realistic-3d-creature-with-ziva-vfx/)) | discontinued April 2024; DNEG holds it in-house ([cgchannel](https://www.cgchannel.com/2024/04/unity-discontinues-the-ziva-products/)) | **copy the design, not the code**. Houdini Muscles is the same design, commercial |
| sim-then-learn for run time | **Han et al. 2024** (layered quasi-static muscle → fascia → fat, then a network) ([eScholarship](https://escholarship.org/uc/item/4fk0x7n8)); **UE Chaos Flesh + ML Deformer** (about 7 000 simulated frames in Epic's sample) | Han: no code found. UE: source under the UE EULA | Han is the template for E-RUN; reimplement from the paper and dissertation |
| elastic + active muscle energy | **Newton VBD** (Apache-2.0) | works now; no anisotropic or active material built in, so our active term stays custom | keep for E-TIS |
| robust contact between layers, crease, chair | **libuipc** (Apache-2.0, GPU IPC, from GIPC / StiffGIPC; [GitHub](https://github.com/spiriMirror/libuipc)) | guarantees no intersections; anisotropic muscle would need a custom energy; 100 k+ tets in 4 GB *unverified* | **bake-off against Newton VBD in Phase 0** |
| material check | **FEBio** (transversely isotropic muscle, active contraction; [manual](https://febiosoftware.github.io/febio-feature-manual/features/solid_materialprop_active_contraction)) | CPU, slow, accurate; licence *unverified* | small reference scenes for T2–T4 |
| cheap fibre + fascia model | Romeo et al. 2020, *Muscle and Fascia Simulation with XPBD* ([CGF](https://diglib.eg.org/handle/10.1111/cgf13734)) | no code | fall-back if VBD is too slow; Newton has XPBD |
| anatomy fitting to a body | Saito 2015, Kadleček 2016 | no code | our warp and the interior model already do this |
| activations | **MyoSuite / MuJoCo** (Apache-2.0); **MASS** (open, DART, RL over about 300 muscles; [GitHub](https://github.com/lsw9021/MASS)) | MyoSuite works now | MyoSuite first; MASS if we go to learned control |
| patella and knee | co-simulation: the musculoskeletal model drives the patella, FE does the soft tissue (Müller et al. 2020; Open Knee(s) templates) | whether myoLegs already couples the patella to knee angle is *unverified* | read the myoLegs XML in Phase 0; drive the patella kinematically |
| tendons stiff next to soft muscle | EMU (code "coming soon"), StiffGIPC | — | tendon as stiff elements in the same solve; watch convergence |
| rejected | PhysX 5 FEM (isotropic only), PolyFEM (lacks biomech materials), Taichi (no muscle code), PhysX/Weta in-house tools | — | — |

**Conclusion.** No open engine does the whole stack. We build the integration (E-PACK, E-JOINT, E-MAT, E-TEST) and reuse:
- MyoSuite for activations;
- Newton VBD for the elastic and active energy, with libuipc as the challenger for contact;
- Ziva's layer design and Han 2024's layered solve and network.

### 5.2 Tissue research (2026-11-24)

Six desk reports in [tissue-research/](tissue-research/): [muscle](tissue-research/muscle.md), [fat](tissue-research/fat.md),
[skin](tissue-research/skin.md), [muscle↔bone and muscle↔muscle](tissue-research/link-muscle-bone-muscle.md),
[muscle↔fat through the deep fascia](tissue-research/link-muscle-fat-fascia.md), [fat↔skin](tissue-research/link-fat-skin.md).
Values marked *unverified* there must be checked before they enter E-MAT. What they change in this plan:

| finding | source report | change to the engine |
|---|---|---|
| living resting muscle is about 1 kPa across the fibres and 5 kPa along them; our 500 Pa is a value for tissue tested after death | muscle | E-MAT: transversely isotropic passive muscle |
| activation stiffens muscle only **along** the fibres (about 20×), linearly with force | muscle | replace the isotropic `ACTIVE=stiff` term with an active fibre stress; test T4 |
| about half of muscle stress is viscous; relaxation times 0.6–300 s | muscle | cited damping for T8 |
| fat about 1.2 kPa in living buttocks, near incompressible, rate-independent below about 10 s⁻¹ | fat | quasi-static solve is valid for sitting |
| skin cannot take compression, it folds; thin tets invert | skin | skin as a triangle shell (membrane + bending) over the fat, not a tet layer; check that Newton can couple shells to tets |
| deep fascia is a stiff anisotropic sheet (56–192 MPa along the limb, 5–19 MPa across) | fascia link | fascia as an anisotropic shell |
| muscle slides under the deep fascia; fat, superficial fascia and skin move with the fascia | fascia link, fat↔skin link | muscle↔fascia: sliding contact that cannot open; fascia↔fat and skin↔fat tied; tethers in the deep fat |
| neighbouring muscles slide a few mm, never separate; Teran 2005 saw our sag without fascia | muscle↔muscle link | muscle↔muscle: no-separation sliding plus a weak tangential spring, fitted to measured sliding |
| the patella is held by the quadriceps and patellar tendons, not by a joint | muscle↔bone link | patella as a free body on tendon springs in frictionless contact with the femur (Lenhart 2015) |
| muscle ends attach over large areas (about 500 mm² on the patella); tendon about 1.1 GPa | muscle↔bone link | tie over the full atlas footprint; tendon as stiff elements |
| under the sitting bone the gluteus slides about 14 mm away and fat carries the seat | muscle, fat, fascia link | T7 checks that slide, not only compression |

**Ground-truth data found** (none downloaded; each licence checked and the user asked before any download):
- Rupani 2025 skin stretch over the bending knee, 15 people, Zenodo (T6);
- Zappalà 2024 seated buttock MRI displacement fields, 10 people (T7);
- van Riel 2026 thigh MRI under a pressure cuff, 9 people (T5);
- Open Knee(s), CC BY 4.0 (T6);
- skin-marker against bone-pin or X-ray-video sets: Cereatti 2017, CAMS-Knee (T6, T9);
- Dyna and Dynamic FAUST 4D scans, research licence (T8);
- MULTiS, CC BY 4.0: fat and muscle thickness and indentation, 100 people.

**Gaps nobody has measured:** stiffness and spacing of the fibrous septa in fat; friction between muscle sheaths; how far the
thigh muscles slide against each other; skin-on-skin friction behind the knee; damping of fat alone. Where a value is
missing, it is fitted to one of the datasets above and the fit is recorded in E-MAT, never set by hand.

## 6. Test scenes (the harness)

Each scene has a pass rule from published data or a conservation law. Ordered from cheapest.

| # | scene | checks | reference |
|---|---|---|---|
| T1 | pack integrity | orientation, volumes, overlaps, element counts | geometry only |
| T2 | block contraction, both ends fixed | tension rises with a, volume ≈ 1 | analytic |
| T3 | isometric force–length | end force matches a·f_L(λ)·F_max of the line model | Hill model of E-ACT |
| T4 | activation stiffness | indentation stiffness rises with a | elastography (Bouillard et al. 2011, to verify) |
| T5 | loaded vs hanging (repeat of handoff Test 5) | vasti and hamstrings thicker and harder when loaded; drift during the hold < 2 mm | direction; magnitudes from ultrasound studies (to find) |
| T6 | knee flexion 0–150° | patella in the groove, rectus in its groove, crease folds, inverted tets < 0.1 % | patellar tracking data (to find) |
| T7 | paired chair test | displacement down and along the thigh under the ischium | Linder-Ganz 2007, Al-Dirini 2016 |
| T8 | drop and settle | thigh jiggle decays in the cited time | Van Loocke 2008 damping; video-based studies (to find) |
| T9 | sit clip, full leg and pelvis | all of the above over the motion; seat/feet load share | Roebroeck 1994 |
| T10 | run-time model | error on held-out chair heights and clips; frame time | E-TIS output |

## 7. Phases

Each phase ends with its tests passing and a viewer. Costs are laptop time unless marked.

**Phase 0, foundations.**
- E-PACK v0 for the left thigh: femur, patella, tibia head, the 8 thigh muscles, fascia lata, fat and skin.
- E-MAT table for the subject profile (§2.1): every value used so far and the ones Phase 1 needs, each with its citation and
  source population, starting from the six tissue reports.
- E-TEST runner with T1–T4; the existing block and isometric scripts become T2 and T3.
- Read Han et al. 2024 and decide what to copy (layering, solve order, network).
- **Solver bake-off:** the same two-muscle-plus-fascia scene in Newton VBD and in libuipc. Measure penetration, sliding, time
  and VRAM on the 1650.
  - *2026-11-24:* pyuipc 0.0.29 installed in `E:\splat\.venv-uipc` (own venv); its doctor passes on the 1650 (sm_75).
  - Smoke test (one muscle dropped on a floor, placeholder material, 30 frames at 1/30 s):

    | muscle | tets | median s/frame | worst frame | peak VRAM |
    |---|---|---|---|---|
    | vastus lateralis | 21.0 k | 0.75 s | 15 s (impact) | 1.2 GB |
    | vastus medialis | 15.1 k | 0.61 s | 144 s (impact) | 1.0 GB |

  - **libuipc rejects 5 of the 8 muscle meshes** (self-intersecting surfaces, 1–392 triangles), and the biceps femoris passes
    the check but has surface parts at zero distance, so its contact kernel asserts. Newton never reported either. **E-PACK
    must output surfaces that are intersection-free with a gap above the contact distance**; this is now a T1 check.
- Read the myoLegs XML for patella coupling.
- *Pass:* T1–T4, and a solver decision made from measurements.

**Phase 0 status (2026-11-24):**
- E-MAT v1 done ([e-mat.md](e-mat.md)).
- Han 2024 read from the dissertation (the TOG paper itself was not reachable): [han2024-notes.md](han2024-notes.md). Copy their
  run-time network form and fat-layer geometry; do not copy their materials (uncited, 20–100× stiffer than living tissue) or
  their active law (it changes rest shape only, so it fails T4).
- Patella: myoLegs hangs the patella from the femur and drives its two translations and its tilt by 4th-order polynomials of
  the knee angle (Rajagopal coupling, `myolegs_assets.xml` equality constraints). E-JOINT v0 reuses them, scaled to the elf.
- **The warped atlas surfaces overlap** (`research/muscle_engine/thigh_overlap_survey.py`): every thigh surface is closed
  and manifold, but 97 muscle–muscle and muscle–bone pairs overlap, up to 28 % of a muscle's vertices inside another structure
  (adductors inside the femur, sartorius inside vastus medialis). So muscles cannot be meshed one by one; E-PACK must first
  decide who owns each point, as a **labelled volume** — the same form the interior model will output.
- Pack format v0 and the T1 check: `research/muscle_engine/pack_format.md`, `t1_pack_check.py`. On the old per-muscle
  meshes T1 fails for four reasons: non-manifold boundary edges (1 700–2 300, from the Delaunay fill), self-intersection,
  surface parts closer than the contact distance, and tets with a zero fibre. Newton keeps every tet (rule 6 passes).
- Proposed mesher: wildmeshing 2.0 `simwild` (MIT, Windows wheel): one tet mesh from many closed surfaces, a region per
  input by winding number, `resolve_overlaps` gives each overlap to the nearer structure, a sizing field per region, and
  offsets for bands (fascia, skin). Needs the user's OK to install.

**Phase 1, the thigh compartment.**
- All thigh muscles together, sliding contact, fascia lata, fat, skin; patella coupling.
- Static solver mode.
- *Pass:* T6 (knee flexion), and a passive hold under gravity with drift < 2 mm (the sag of Test 5 must be gone).

**Phase 2, the buttock and the chair.**
- Pelvis, gluteals, hip rotators and adductors added; chair contact on the skin.
- *Pass:* T7.

**Phase 2b, loaded vs hanging leg.**
- The Phase 1 thigh with the Phase 2 hip, in the single-stance pose: the same bones, carrying the body vs hanging, with the
  MyoSuite activations of each case.
- *Pass:* T5. This repeats handoff Test 5 with the container in place.

**Phase 3, dynamics and the whole leg.**
- Shank and foot compartments; damping; the sit clip.
- *Pass:* T8, T9. Decide whether a whole leg fits in 4 GB, or the full run goes to an HF Job (cost table first).

**Phase 4, the elf's own activation model.**
- E-ACT rebuilt from the pack; CMC with the same environment contacts; seat and foot loads checked between E-ACT and E-TIS.
- *Pass:* reserves small against muscle torques; load share consistent between the two models.

**Phase 5, run-time.**
- Generate training data with E-TIS over clips × chairs × loads (HF Job, asked first); train E-RUN; export.
- *Pass:* T10.

**Phase 6, the rest of the body.** Trunk, arms, then the generated characters from the interior model instead of the atlas.

## 8. Risks

- **Many touching bodies in VBD** may be slow or unstable. Fall-back: fuse a compartment into one mesh with per-tet fibres and
  activations (Lee et al. 2009; Kadleček et al. 2016).
- **4 GB VRAM.** A whole leg with all layers may not fit. Fall-back: solve compartments in sequence with frozen neighbours, or run
  on an HF Job.
- **The atlas is one adult male (BodyParts3D),** warped to a 0.98 m elf. Proportions, force scaling and fat distribution are
  extrapolated; the interior model is the long-term fix.
- **Reference data for flesh shape under load is thin.** Some tests can only check direction, not magnitude; that is stated per
  test.
- **The solver choice may be wrong.** E-TIS keeps the solver behind one interface so it can be swapped (see §5).

## 9. Decisions for the user

1. **Scope of the first proof — decided 2026-11-24:** the thigh first (Phase 1), then the buttock on the chair (Phase 2), then
   the loaded vs hanging leg (Phase 2b).
2. **Solver:** stay with Newton VBD alone, or run the Phase 0 bake-off against libuipc (an install, needs your OK).
3. **Where heavy runs go — decided 2026-11-24:** heavy runs go to HF Jobs from now on, each one asked first with a cost table.
   The laptop keeps quick smoke tests and viewers; small cards (t4/l4/a10g-small) for smoke and development need no ask.
   Both solvers have Linux builds (pyuipc ships manylinux wheels).
4. **Stop the one-off experiments** until Phase 0 is done, except anything Phase 0 needs.
