# Live flesh at minimum compute — plan (2026-11-24)

**Goal.** Flesh that moves like real flesh, live, beside a motion model such as ARDY, in games and on VRChat/VRM avatars,
covering action and breathing. Today's flesh looks fake because the jiggle settings never change, muscles never bulge, skin
never slides over muscle, nothing squashes on contact and nothing breathes.

**Mindset.** We know the bones and muscles under each character. Most of the interior never shows on the outside, so the
anatomy is used **offline, once per character**, to work out a few facts per patch of skin. The live part stays tiny.

What the outside needs, per patch of skin:

| fact | where it comes from |
|---|---|
| fat depth, soft mass under the skin | the character's interior (atlas fit + fat) |
| which muscle is under it, how hard it works this frame | ARDY motion → joint torques → muscle activation |
| bulge direction and size when that muscle works | offline simulation |
| wobble frequency and damping, relaxed vs tensed | offline simulation, checked against human measurements |
| squash on contact | offline simulation (sitting, pressing) |

## Runtime (live, per frame)

The live inputs are the bone motion ARDY generates and **the shape, position and velocity of every outside object the body
touches** (floor, chair, another body, a hand), plus the character's fixed offline data. Everything else is computed
inside the runtime. Objects matter twice: they squash the flesh (step 4) and they carry weight (sitting puts the body's
weight through the seat, not the feet, which changes every muscle's effort in step 1).

1. **Muscle activation**, computed from the motion, not supplied: the ground forces are estimated from the motion and the
   feet's contact (the whole body's acceleration fixes their total; the split between two feet is part of the next step);
   any other touching object takes its share the same way, from its contact patch;
   inverse dynamics gives each joint's torque; a small per-frame optimisation shares each torque among the muscles that
   cross the joint (about 40 per leg), by least effort, or a small learned stand-in for it.
   Limit: tensing without moving (bracing, flexing in place) does not show in the motion. Least effort assumes none; an
   optional "tension" input could add it later.
2. **Bulge**: corrective shapes driven by joint angles and activation (SMPL pose correctives, SoftSMPL).
3. **Jiggle**: a few damped wobble modes per body region, driven by bone acceleration; stiffness rises with activation
   (reduced dynamics: Barbič & James; Dyna/DMPL).
4. **Contact**: a local correction from the touching object's shape and motion, computed only where and while something
   touches (learned contact corrections with general collider descriptors, Romero et al. 2021). Training in R3/R6 uses
   varied object shapes, not one chair, so new objects work.
5. **Breathing**: rib bones and belly shape driven by a breathing signal tied to exertion.

Target: well under 1 ms per character per frame. Delivery: a small runtime (CPU or one compute pass) that game mods can
load, so it may run any code; VRM and VRChat avatars are out of scope for now (user, 2026-11-24).

## Phases

| | phase | what it does | done when | compute |
|---|---|---|---|---|
| R0 | **proof** | one ARDY clip on the elf, legs and hips: A plain skinning, B small runtime (bulge + jiggle that stiffens with activation) with settings from cited human measurements | B clearly looks more real, under 1 ms/frame on the GTX 1650; side-by-side video | local, hours |
| R1 | mod runtime shape | the runtime as a small library with one call per frame (bone transforms + activation in, vertex offsets out), so a game mod can host it | runs standalone with a test harness; frame time logged | local |
| R2 | activation from motion | ARDY motion → torques → muscle activation each frame, with the elf's own masses and the muscles' moment arms | activation patterns match published gait EMG timing | local |
| R3 | offline teacher, per character | coarse layered simulation of the character (bones, muscle volumes, fat, skin; Newton VBD; layers tied, sliding only where it shows). Training motions cover **local states, not a list of actions**: broad random sampling from ARDY itself, plus targeted excitations (sudden stops, impacts, holds at the ends of each joint's range, activation sweeps from relaxed to full) | stable runs; per region, the training covers the range of joint angles, bone accelerations and activations seen in ARDY | local first; HF Jobs with a cost table if slow |
| R4 | fit the runtime | a learned flesh **response** model: input = bone motion over the last fraction of a second (activation is computed from it inside, step 1 of the runtime), output = surface offsets. Physics-shaped (bulge shapes + damped wobble modes whose stiffness follows activation), with a small learned correction | error measured on **held-out ARDY motions never used in training**, per region | local |

The runtime must answer any motion ARDY can generate, so R3's clips are training data, not a lookup table. It can
generalise because flesh responds locally: a patch of thigh only feels the nearby joints, the leg's acceleration and a few
muscles' activation over roughly the last half second. That space is far smaller than the space of whole actions, and it
is what R3 has to cover. Unlike ARDY, this model is not generative: given the motion, the flesh is set by physics, so a
small deterministic model is enough.
| R5 | breathing | rib and belly motion from a breathing signal; rate tied to exertion | visible, plausible breathing at rest and after a run | local |
| R6 | contact | sitting and pressing corrections (the old T7 buttock-on-chair scene) | squash matches the teacher | local / HF |
| R7 | live test | ARDY live + runtime on the elf; frame time; side-by-side against plain skinning; whole body | the user judges it real; frame budget met | local |

Every phase gets a notebook in `notebooks/` and pictures as it goes.

## What happens to the earlier engine plan (plan.md)

- It becomes the **offline teacher** (R3, R6) and runs at coarse resolution. The 0.77 mm thigh pack is not needed for that.
- Kept: the E-MAT tissue parameters (woman about 28), the tissue research, the stance findings, and the GPU label grid +
  surface nets route (2.5 min instead of 68 min).
- Not needed for now: a converged fine-mesh anatomy sim with sliding contact everywhere; it is far beyond 60 Hz and most
  of it never shows.
- Later option: run the teacher once over many generated bodies and learn the settings, instead of once per character.
