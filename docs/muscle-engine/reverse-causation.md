# Flesh when the bones are given: the current engine, and how effort can act on flesh

**Status:** 2026-11-25. This is a design note, not a result. Claims marked *(to check)* have not been verified against the source.

## 1. The problem

In a real body the chain of cause runs one way:

> nerve drive → muscle force → tendon → joint torque → bones move (and flesh moves with them)

In this project the bones are an **input**: ARDY writes the skeleton's motion. Nothing we simulate is allowed to change it. So the chain runs backwards:

> bones move (ARDY) → muscles and fat are dragged along → what does effort still change?

The first teacher run with effort (`act1`, 2026-11-25) showed what goes wrong when the real-life chain is copied as it is:
- Fibre tension was applied as a force (up to 268 kPa on 0.8 kPa tissue).
- The belly bunches up and the skin above it wrinkles like plastic.
- *Corrected after review (§6):* the earlier claim "the bones absorb nothing" is wrong. A prescribed bone can carry a reaction force, as a fixed knee does when the quadriceps tense. The real fault is the load path. The lattice has no tendons, so the fibre tension is spread over the whole belly and never reaches an attachment as a tendon force. It is balanced inside the soft tissue instead, and that is what bunches it.

The run without effort (`jog1`) stays smooth. In a real leg, that force is exactly what moves the bones. Here the motion is already decided, so the force has no job.

The question is therefore not "how do muscles move the leg" but:

> **Given the bones' motion, and the effort R2 says was needed to make it, what shape and stiffness does the flesh have?**

## 2. The current engine

### R0: the runtime (0.3 ms per frame on the GTX 1650)

- **Skin:** dual-quaternion skinning with bone-heat weights (Baran & Popović 2007).
- **Bulge:**
  - Passive part: volume kept as a muscle shortens, r·(√(l0/l) − 1).
  - Active part: cited thickening × effort.
  - Both are spread over the skin by fat depth.
- **Jiggle:** one damped oscillator per muscle compartment.
  - Forced by gravity change m(g − R(t)g) and bone acceleration.
  - Its frequency moves with effort.
  - Frequencies and damping come from published measurements (`r0_params.json`).

R0 already uses reversed causation: effort only changes the surface (thickening, stiffer wobble), never the bones.

### R2: effort from motion (validated)

- **Ground force:** found from the motion (Ren, Jones & Howard 2008), split over the planted heel and toe points.
- **Muscle effort:** MyoSuite static optimisation (least effort that reproduces the joint torques), then MuJoCo activation dynamics.
- **Validation:** against three women running at 3 m/s.
  - Vertical ground force r 0.95–0.98.
  - Soleus, vastii and gluteus medius r ≥ 0.92.
  - Tibialis anterior and rectus femoris are wrong: minimum-effort optimisation never co-contracts.

R2 is the inverse-dynamics step: motion in, effort out. It is where the real-life chain is solved backwards correctly.

### R3: the teacher (libuipc, offline)

- **Lattice:** 4.6 mm cubes, 6 tets each, tissue fractions per tet (McAdams et al. 2011). 24k nodes, 123k tets for the thigh.
- **Material:** stable neo-Hookean per tet.
  - Fat 1.17 kPa, muscle 0.8 kPa across the fibre.
  - Skin layer 201 kPa × thickness / cube size.
  - Fascia 0.79 MPa × thickness / cube size where the tissue model marks it.
- **Bones:** soft position constraints pull the bone nodes onto ARDY's pose (one way: flesh cannot push them).
- **Gravity:** applied as its change from the scanned rest pose.
- **Damping:** BDF1's own damping. At 300 Hz it gives ζ ≈ 0.15 at 14 Hz, the measured thigh value (Zhang 2019).
- **Effort, two versions:**
  - `act1`: fibre tension as forces. Wrinkles, as above.
  - `EFFORT=clip` (new, running now): each muscle is held stiffer by its mean effort over the run.
    - Linear law: Bouillard et al. 2011.
    - Slope: ×2.58 at 20 % MVC (Bensamoun 2008).
    - libuipc only reads stiffness at start-up, so effort is fixed per run, and effort levels are swept across runs.
- **Cost:** about 4 s per step, about 20 minutes per simulated second.

## 3. What effort can change when the bones are given

With the motion fixed, a muscle's **length** is already decided by the bones. Its **force** is decided by R2. Effort can only show on the surface in three ways:

1. **Stiffness.** Working muscle is 2–4× stiffer, so it wobbles faster and less. (Done: R0 oscillator, R3 `EFFORT`.)
2. **Shape at a given length.**
   - A working muscle–tendon unit of fixed length has a shorter belly, because the force stretches the tendon and aponeurosis.
   - A shorter belly at constant volume is a thicker one.
   - This is the visible "flex" (a calf tensing in stance), and it needs no bone motion at all.
3. **Damping and timing.** Effort rises and falls within a stride. Stiffness changing over time shifts when the wobble stops.

Fibre *force* as such is not one of them. Its only real job, moving the bones, is already done by ARDY.

## 4. Ideas

### A. Fixed-effort stiffness (now)

- **What:** each muscle's stiffness is set from effort for the whole run (§2), and effort is swept across runs.
- **Pros:** stable, every number cited, matches the planned effort sweep.
- **Cons:** no shape change; effort cannot vary within a run.

### B. Active strain from the muscle–tendon balance (recommended next)

Turn effort into a **change of rest shape** instead of a force.

**Step 1: belly length from the bones and the force.**

L_mt(t) = L_tendon(F) + L_fibre · cos α

- **Muscle–tendon length** L_mt comes from the bones, through the musculotendon path length (MuJoCo `actuator_length`, in metres).
  - *Corrected (§6):* `fiber_norm` is **not** this length. In MuJoCo's muscle model it is (actuator_length − L_T)/L0 with a rigid tendon (`r0_clip_activations.py` saves exactly that), so it has no force-dependent tendon stretch. Export path length, force, activation and the length parameters separately.
- **Force** F comes from R2.
- **Tendon stretch** L_tendon(F) comes from a standard tendon curve.
  - Strain is about 3–5 % at peak isometric force (Zajac 1989) *(to check)*.

So the belly length L_fibre follows.

**Step 2: impose that length as active strain.**

- Write the deformation as F = F_e · F_a.
- F_a shortens the fibre by γ and widens across it by 1/√(1 − γ), so volume is kept.
- In heart mechanics this "active strain" form is known to be more stable than "active stress" (an added force). That is the failure we hit: Ambrosi & Pezzuto 2012; Rossi et al. 2012 *(to check)*.
- The solver then finds the shape where the passive tissue balances the new rest shape.
- No force is left over to bunch the belly.

**Why it fits.** It is the real-life chain solved backwards: bones and force known, so length is known, so shape is known. That is the same direction as R2. Every input already exists (ARDY, R2 effort, `fiber_norm`).

**Unknowns:**
- Whether libuipc can change rest shape mid-run. Its stiffness cannot change; rest shape is untested. A five-minute test, like the earlier `uipc_mu_change.py`, will tell.
- If it cannot, fall back to one of:
  - per-run constant γ (as in A);
  - a small custom quasi-static solve for the muscles only (see C).

**Cheap check:** hold the knee fixed and switch the vastus lateralis from 0 to 50 % effort. Compare its thickness change with ultrasound measurements of the vastus lateralis contracting *(source to find)*.

### C. Split: muscles quasi-static, fat and skin dynamic

Muscle shape follows the bones and effort almost at once. The visible wobble lives in fat and skin, which hold most of the soft mass.

- **Each step:**
  1. Solve the muscles' shape quasi-statically from B: no inertia, fast.
  2. Use that shape as a moving inner boundary.
  3. Simulate only fat, skin and fascia dynamically on top.
- **Pros:** fewer stiff degrees of freedom in the dynamic solve, and a clean match to what the learned runtime should do.
- **Cons:** muscle mass also jiggles (calf, quadriceps), so the muscle needs at least a reduced dynamic mode. R0's per-compartment oscillator could serve as that mode.

### D. Two-way bones that track ARDY

- **What:**
  - Bones become rigid bodies pulled to ARDY's pose by a stiff spring and damper, instead of being pinned.
  - Muscle forces, with tendons, push back on them.
- **Pros:**
  - The size of the spring torque needed to stay on ARDY is a free consistency check of R2: near zero if the effort is right.
  - It brings back real-life causation inside the teacher.
- **Cons:**
  - It needs tendons and attachments, which the lattice does not have yet.
  - libuipc's affine bodies could carry the bones *(to check)*.
  - Biggest build of the four. It is the faithful version of "keep fibre tension".

### E. Existing work to read before building B or C

- **Ziva-style animation-driven muscles:** a commercial muscle system for film, driven by keyframed bones with fibre "excitation". Its muscle model is close to B *(to check)*.
- **Modi et al. 2021, EMU:** efficient muscle simulation in deformation space *(to check how activation is set)*.
- **Fan, Litven & Pai 2014:** active volumetric musculoskeletal systems.
- **Sifakis et al. 2005:** facial muscle activations found from motion (the inverse direction).
- **OpenSim's equilibrium muscle model:** the tendon–fibre length split of B, already solved for gait. Its fibre length could be read directly instead of re-deriving it.

## 5. Recommendation

1. **Now:** A. It is running as `eff1`, at the effort of the same strides as `act1`, for a side-by-side.
2. **Next, small and cheap:** test whether libuipc can change rest shape during a run. Then try B on one muscle with the knee held fixed, and compare against a published thickness change.
3. **For the data runs** (after the tissue model, ~2026-11-27/28):
   - A + B: stiffness and shape from effort, effort swept across runs.
   - C only if the cost per simulated second has to fall.
   - D stays a research option. It is the only route that checks R2 inside the teacher, but nothing in the runtime needs it.

## 6. Outside review (2026-11-25) and what changes

An outside review of this note was received on 2026-11-25. It was checked against the code here, and its points are taken as follows.

**Accepted:**
- **Prescribed bones carry reactions.** The soft position constraints already give the attachment force (spring × target error). Record it, and the attachment drift, in every teacher run. Summed as moments about the knee and hip, it checks R2's joint torques without letting the flesh move ARDY's bones. Idea D is therefore not needed for this check.
- **`act1` shows a broken load path, not that force-based effort cannot work.** Check first:
  - Does the integrated fibre stress give the expected tendon force in newtons?
  - Does that force reach an attachment?
- **`fiber_norm` is the wrong input for B.** Confirmed in `r0_clip_activations.py`: it is MuJoCo's normalised fibre length with a rigid tendon. See the correction in §4 B.
- **Corrected B for the next run:**
  - A 1-D compliant-tendon model (as in OpenSim's Millard 2012 equilibrium muscle) gives a mean fascicle-length target:
    - from path length and R2 force;
    - L_T = T⁻¹(force), with a slack branch;
    - L_mt = L_T + L_f cos α.
  - The target is applied as a **soft target**, and the surrounding volume is solved for shape.
  - Full R2 tension is **not** added on top: that would count the same contraction twice.
- **The scope of ×2.58 is narrow.** It comes from one muscle (vastus medialis), knee at 30°, 90 Hz MR elastography, and it is ×2.11 with the 2-D method. `eff1` is kept as a stiffness sensitivity experiment, not as the effort model. Later A+B runs measure the activated teacher's own tangent stiffness before any separate stiffness gain is added.
- **Damping check.** Backward Euler's amplitude factor per step is 1/√(1 + (ω dt)²). At 14 Hz and 300 Hz that gives ζ ≈ 0.14, matching the measured 0.12–0.16; at 600 Hz it falls to ≈ 0.07. So run a ring-down at both rates before any damping value is moved into R0.
- **First experiment: fixed-knee vastus lateralis contraction.**
  - Compare against ultrasound fascicle length, pennation and thickness at 10° and 100° knee flexion, at rest and at 40 % MVC.
  - Runs: passive, A, corrected B, A+B, all at the same bone pose.
- **Precedents to read:**
  - Murai et al., Disney Research: joint angles → muscle length and tension → quasi-static muscle shape, plus soft-tissue dynamics. This is closest to this project.
  - Blemker et al. 2005: prescribed hip motion as a boundary condition in a 3-D muscle simulation.
- **Runtime target.** R3 teaches R0, per compartment:
  - the **equilibrium deformation** as a function of pose, estimated loading and contact;
  - the **dynamic response around it**.

  That is the reduced form of C.

**Open, not settled by the review:** whether libuipc can take a fascicle-length target during a run. Rest shape cannot be changed mid-run (untested); a penalty spring on mean fibre length, applied as nodal force, is the fallback. Unlike `act1`'s fibre tension, that force stops once the target length is reached.
