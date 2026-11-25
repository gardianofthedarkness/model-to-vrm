# The elf's flesh setup as of 2026-11-25

This is what the flesh work runs on today. Every number here is read from the files the code uses: the K6 lattice, `e-mat.json`, `r0_params.json`, the R3 teacher and the 1-D quadriceps model.

## Subject

- **The model:** the elf, 0.98 m tall, physically an adult woman of about 28.
- **What is simulated:** only the **left thigh and the top of the shank**, z 0.18–0.59 m above the floor.
  - The lower shin and foot are outside the region and follow plain skinning.
  - The hip and buttock are cut where the region ends. They wait for the tissue model's body under the garment.

## Geometry (R3 lattice `r3_lattice_K6.npz`)

### Inputs

| Part | Source |
|---|---|
| Skin | The elf's own mesh, 60 texture patches joined into one surface: 46,957 vertices, 92,783 triangles |
| Bones and muscles | Anatomical atlas warped into her body (`anatomy_dense`: `atlas_warp.npz`, `muscle_repair.npz`), turned into a 0.77 mm label grid |
| Fat | Everything inside the skin that is not bone or muscle. No separate fat model yet: the tissue model replaces this (~2026-11-27/28) |
| Fascia | None yet. The code takes it as labels of kind `fascia` from the tissue model |

### Lattice

| Quantity | Value |
|---|---|
| Cube size | 4.6 mm |
| Nodes | 24,180 |
| Tets | 122,550 (6 per cube) |
| Driven nodes | 1,935: pelvis 19, thigh 1,252, shank 642 (422 of them cut-face nodes) |

### Tissue volumes inside the lattice

| Tissue | Volume (cm³) |
|---|---|
| Fat | 892 |
| Muscle (37 muscles) | 678 |
| Bone (femur 99, tibia 31, fibula 9, patella 4, hip 0.5) | 143 |

### Largest muscles (cm³)

| Muscle | Volume | Muscle | Volume |
|---|---|---|---|
| Vastus lateralis | 83.5 | Rectus femoris | 61.3 |
| Biceps femoris, long head | 64.7 | Semimembranosus | 54.5 |
| Adductor magnus | 64.7 | Vastus medialis | 53.0 |
| Vastus intermedius | 45.9 | Gastrocnemius, medial | 44.8 |
| Gastrocnemius, lateral | 26.4 | Semitendinosus | 22.3 |
| Gluteus maximus (cut) | 20.0 | | |

### Fibres

- Each muscle's fibres run along its centre line.
- There is no pennation and no tendon or aponeurosis geometry yet.

## Materials (R3 teacher, `e-mat.json`)

| Tissue | Value | Source |
|---|---|---|
| Fat | shear 1,167 Pa, density 950 kg/m³ | Al-Dirini 2016; Woodard & White 1986 |
| Muscle at rest | shear across fibres 800 Pa, density 1,050 kg/m³ | Gennisson 2010; Woodard & White 1986 |
| Muscle, working | stiffness × (1 + 1.58 × effort / 0.2), i.e. ×2.58 at 20 % of maximum | Bensamoun 2008, linear by Bouillard 2011 (one muscle, one knee angle: see reverse-causation.md §6) |
| Muscle, active fibre stress (only in the old `act1` run and the 1-D model) | maximum 268 kPa, force–length half-width 0.5 | Persad 2024; Blemker 2005 |
| Bulk / shear (near-incompressible) | 100 | e-mat solver choice (FEBio uses 2,000) |
| Skin | 201.4 kPa × 1.68 mm / 4.6 mm = 73 kPa in the outer layer of cubes | Ni Annaidh 2012; thickness Wang 2025. **Flagged:** in-vivo skin is about 10× softer (~20 kPa, Hendriks) |
| Fascia lata (when the tissue model supplies it) | 2 C1 = 0.79 MPa × 0.94 mm / 4.6 mm = 162 kPa | Bonaldi 2023 |
| Mass density used | 993 kg/m³ (fat–muscle mix; libuipc takes one per object) | from the two above |

**Solver:** libuipc, stable neo-Hookean, BDF1 at 300 Hz, MAS preconditioner, self-contact on (contact distance 0.92 mm). BDF1's own damping gives ζ ≈ 0.14 at 14 Hz, the measured thigh value (Zhang 2019).

## Bones and loads

- **Bone motion:** from ARDY, through soft position constraints (strength 100) on the bone and cut-face nodes.
  - Pelvis comes from the hips; thigh and shank from the calibrated leg transforms.
  - The new held-knee mode turns the shank about the knee axis instead (axes from the femur and tibia shapes).
- **Gravity:** applied as its change from the scanned pose, m(g − R(t)g), per bone group.
- **Muscle effort:** R2.
  - myoLegs (80 muscles, an adult-size model, not scaled to the elf) driven by ARDY.
  - Static optimisation with least effort, then activation dynamics (τ 10/40 ms).
  - Validated on running women: vertical ground force r 0.95–0.98; soleus, vasti and gluteus medius r ≥ 0.92. Tibialis anterior and rectus femoris are unreliable.
- **Effort in the teacher today:** stiffness only, held fixed per run (`EFFORT`).
- **Being tested:** a soft fibre-length target from the 1-D muscle–tendon model (`LENTARGET`). The tendon law is Thelen 2003 with 4.9 % strain at maximum force (Millard 2012 default).

## R0 runtime (the game side, 0.3 ms per frame)

- **Skinning:** dual-quaternion skinning with bone-heat weights, 4 bone groups (pelvis, thigh, shank, foot).
- **Bulge from effort:** front-thigh thickness +20 % at full effort. One value for all muscles.

**Wobble per compartment** (`r0_params.json`):

| Compartment | Relaxed f (Hz) | Tensed f (Hz) | ζ relaxed | ζ tensed | Source |
|---|---|---|---|---|---|
| Quadriceps | 9.5 | 13.3 | 0.14 | 0.18 | Wakeling & Nigg 2001; Zhang 2019 |
| Hamstrings | 12.9 | 18.0 | 0.095 | 0.12 | Zhang 2019 |
| Calf | 10.5 | 14.7 | 0.15 | 0.20 | Behling 2021 (damping uncertain) |
| Adductors, gluteals, other | quadriceps values | | | | stand-in: no data found |

Tensed = relaxed × 1.4 (Christensen 2023); tensed damping × 1.3 (Zhang 2019).

## Known gaps

1. Fat is "whatever is left over", and there is no fascia. The tissue model fixes both.
2. The hip and buttock are cut. The body under the garment comes from the tissue model.
3. The skin stiffness is from a tensile test on excised skin, about 10× the in-vivo value. This is unresolved.
4. There are no tendons or aponeuroses in the lattice, so effort cannot be carried by tendon force (see `act1`).
5. The calf rings at about 30 Hz in the teacher, against about 10 Hz measured.
6. R2 uses an adult-size leg model. Effort is a fraction, so it carries over, but forces in newtons do not.
7. The knee crease and the contact between calf and hamstrings belong to R4/R6.
