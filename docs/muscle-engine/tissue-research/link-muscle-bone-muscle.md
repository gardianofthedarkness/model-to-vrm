# How muscles attach to bone and to each other (living people) and how simulators model it

Desk research, 2026-11-24. Every number has a source. Anything I could not check against a source is marked *unverified*. Nothing was downloaded or installed. The datasets are only listed.

**Why this matters for us.** Our isolated muscle sags 40–70 mm. It is glued at its attachment patches and is otherwise in frictionless contact that can separate. Our patella, pinned to the shank, lifts off the femur. Section 5 shows that both are known failure modes with known fixes (Teran 2005, Fig. 10).

---

## 1. Muscle → bone

### 1.1 Tendon and aponeurosis stiffness (in vivo, adults)

| Quantity | Value | Method | Source |
|---|---|---|---|
| Patellar tendon stiffness | 4334 ± 562 N/mm (SEM) | B-mode ultrasound of tibia and patella displacement during maximal 10 s isometric ramps. Tibial motion was 45 % of total and was corrected for. | Hansen et al. 2006, Clin Biomech, [10.1016/j.clinbiomech.2005.07.008](https://doi.org/10.1016/j.clinbiomech.2005.07.008) |
| Patellar tendon strain at MVC | 6.9 ± 0.6 % | same | same |
| Patellar tendon Young's modulus | 1.09 ± 0.12 GPa | same | same |
| Reproducibility | typical error: stiffness 9.9 %, strain 5.5 %, modulus 9.4 % | same | same |
| Gastrocnemius free tendon (Achilles) | stiffness 759 ± 132 N/mm, E = 788 ± 181 MPa, strain 8.0 ± 1.2 % at 2641 N, CSA 73 mm² | ultrasound, n = 5 | Magnusson et al. 2003, Acta Physiol Scand, [10.1046/j.1365-201x.2003.01048.x](https://doi.org/10.1046/j.1365-201x.2003.01048.x) |
| Gastrocnemius distal aponeurosis | strain **1.4 ± 0.4 %** at the same force (5.7× less than the free tendon). Deformation 2.12 mm over 145 mm. | same | same |
| Aponeurosis transverse strain | about 4× the longitudinal strain during active force (turkey) | sonomicrometry | Azizi & Roberts 2009, [PMC2754367](https://pmc.ncbi.nlm.nih.gov/articles/PMC2754367) |
| Measurement caveat | If distal (tibial) displacement is ignored, strain is underestimated and stiffness overestimated. | review | Onambele-Pearson et al. 2007 (cited in [MMU e-space](https://e-space.mmu.ac.uk/86598)); Hansen 2006 measured the 45 % tibial share |

What this means for the model:
- The free tendon is about 1 GPa, so it is effectively inextensible next to muscle (kPa to MPa).
- Inside the muscle, the aponeurosis stretches far less than the free tendon, but it stretches about 4× more across the fibres. It should be anisotropic and must not be welded rigid sideways.

### 1.2 Attachment footprint (entheses)

| Item | Value | Source |
|---|---|---|
| Quadriceps tendon footprint on the patella | mean 502.5 mm² (range 336–611). Dome-shaped, with no distinct bony landmarks. | Cadaver 3D CT/MRI, Knee 2023 (via [BVS mdl-37399632](https://pesquisa.bvsalud.org/ses/resource/pt/mdl-37399632)) |
| Quadriceps tendon width at the patella | 42.4 mm (33.8–50.5) | Asian Biomed 2019, [10.1515/abm-2019-0058](https://reference-global.com/download/article/10.1515/abm-2019-0058.pdf) |
| Quadriceps tendon graft CSA (not the footprint) | 91.2 ± 10 mm² | Arthroscopy 2016, [PubMed 26382635](https://pubmed.ncbi.nlm.nih.gov/26382635/) |
| Patellar tendon tibial insertion | crescent shape; bony apex at 35.5 ± 2.8 % of tendon width (32 knees); 50.7 % (33–63 %) of the footprint lies medial to the tubercle (56 knees) | Oikawa et al. 2019, Knee, [10.1016/j.knee.2018.12.002](https://doi.org/10.1016/j.knee.2018.12.002); Eseonu et al. 2022, [PMC8536615](https://pmc.ncbi.nlm.nih.gov/articles/PMC8536615). Area in mm² not found. |
| Enthesis types | **Fibrocartilaginous** entheses (tendon → uncalcified → calcified fibrocartilage → bone, a graded stiffness) occur at epiphyses and apophyses. **Fibrous** (periosteal) entheses occur along diaphyses, where tendon blends into the periosteum. The "enthesis organ" spreads stress over neighbouring fat pads, bursae and sesamoid fibrocartilage. | Benjamin et al. 2006, J Anat, "Where tendons and ligaments meet bone", [PMC2100202](https://pmc.ncbi.nlm.nih.gov/articles/PMC2100202) |

**Fleshy versus tendinous attachments.** Many muscles (vasti, tibialis anterior, brachialis) attach "fleshy" along a long strip of periosteum or intermuscular septum rather than at a point. The vastus medialis is an example: it rises from a broad "hammock" and inserts along a long line into the patella, the rectus femoris tendon and the vastus intermedius aponeurosis (Grob et al. 2017/18, KSSTA, [doc.rero.ch 280234](https://doc.rero.ch/record/280234/)). The quadriceps tendon is layered: rectus is superficial, and an intermediate two-layer sheet is formed by the VI, the tensor VI and the VL aponeuroses (Grob 2016, [10.1186/s40634-016-0068-y](https://doi.org/10.1186/s40634-016-0068-y)). For us, a fleshy origin is a **long glued strip**, not a small patch. The atlas origin area should come from Z-Anatomy, not from a radius we pick.

### 1.3 Sesamoid: the patella

- The patella sits inside the quadriceps/patellar tendon chain. It is held against the trochlea by the **tension in that chain**, not by a joint. OpenSim gait models therefore often reduce it to a frictionless pulley (as described in [PMC4886716](https://pmc.ncbi.nlm.nih.gov/articles/PMC4886716)). A patella rigidly pinned to the shank has no reason to stay in the groove. This is the lift-off we see.
- **Tracking in vivo**, weight-bearing flexion, dual fluoroscopy plus MRI, n = 8 (Nha et al. 2008, J Orthop Res, [10.1002/jor.20554](https://doi.org/10.1002/jor.20554)):
  - mean shift stays within 3 mm over the whole range
  - tilt stays within 6°: lateral from 0–75°, medial beyond 75°
  - total rotation excursion about 8°
  - lateral displacement within 6 mm
- Patellar flexion correlates strongly with knee flexion, R² = 0.91 (Li et al. 2007, J Biomech Eng, [10.1115/1.2803267](https://doi.org/10.1115/1.2803267)). The slope is **not** in the abstract. A value of about 0.6–0.7° per degree is often quoted (*unverified*).
- At 90° squat in healthy knees: medial spin 4.71 ± 1.17°, translation −4.93 ± 3.93 mm (Wilson et al. 2009, JBJS Am, [10.2106/JBJS.G.00572](https://doi.org/10.2106/JBJS.G.00572)).
- Supine unloaded tasks give more lateral translation at 0–5° than weight-bearing, so loading state matters ([s41598-020-72332-9](https://doi.org/10.1038/s41598-020-72332-9)).

## 2. Muscle sliding on bone and periosteum; bursae

- The deep surfaces of the vasti lie on the femur. Fleshy parts are fixed, and the free parts glide over periosteum and fat pads. I found **no** measured friction coefficient for muscle on periosteum.
- **Bursae** are synovial sacs placed where tendon or muscle glides over bone or skin:
  - The **suprapatellar bursa** lies under the distal quadriceps, superficial to the prefemoral fat pad. It joins the knee joint in about 85 % of people ([Radiopaedia](https://radiopaedia.org/articles/suprapatellar-bursa)).
  - The **prepatellar bursa** measures 39.7 × 40.5 × 3.2 mm (craniocaudal × lateromedial × AP). It is usually trilaminar (cadaver MRI, 9 knees; [radsource](https://radsource.us/knee-bursae/), primary *unverified*).
- **Lubrication.** A hyaluronan film about 0.05 mm thick lies between the fascia and the muscle (Stecco, cited in [PMC6131966](https://pmc.ncbi.nlm.nih.gov/articles/PMC6131966)). Epimysial fascia contains far less HA (about 6 µg/g) than retinacula.
- **Friction.** I found no friction coefficient for bursa or fascia. Cartilage-on-cartilage µ is about 0.001–0.01 (*unverified* here). **Frictionless sliding is a defensible model over bursae and the articular surface. It is not defensible as a model of the whole muscle-to-muscle interface** (Section 3).

## 3. Muscle ↔ muscle

### 3.1 Anatomy

Each muscle sits in its own **epimysium**. Neighbours are linked by:
- loose areolar tissue
- shared **neurovascular tracts**
- **intermuscular septa**, which are fused to bone

Maas & Sandercock separate two pathways (2010, [PMC2853902](https://pmc.ncbi.nlm.nih.gov/articles/PMC2853902)):
- **intermuscular** transmission, belly to belly
- **extramuscular** transmission, belly to septum, bone or the neurovascular tract

Areolar layers are "compliant interfaces that allow adjacent dense connective tissue layers to move past one another". They are softer and more viscous than dense fascia by rheometry ([PMC3414643](https://pmc.ncbi.nlm.nih.gov/articles/PMC3414643)). Their absolute modulus was not in the abstracts. For scale, subcutaneous loose tissue has a tensile modulus of **4.77 kPa (instantaneous) and 2.75 kPa (equilibrium)** ([Mount Sinai abstract](https://scholars.mssm.edu/en/publications/subcutaneous-tissue-mechanical-behavior-is-linear-and-viscoelasti-2/)).

### 3.2 Force transmission (Huijing, Maas)

| Finding | Setting | Source |
|---|---|---|
| Proximal and distal EDL forces are **not equal** at most lengths. Freeing EDL from its neighbours removed the active difference, but a passive difference remained. | rat, in situ | Huijing & Baan 2001, Acta Physiol Scand, [10.1046/j.1365-201X.2001.00911.x](https://doi.org/10.1046/j.1365-201X.2001.00911.x) |
| Lengthening TA+EHL distally raised proximal EDL force and lowered distal EDL force. | rat | Maas, Baan & Huijing 2001, J Biomech ([VU record](https://research.vu.nl/en/publications/intermuscular-interaction-via-myofascial-force-transmission-effec/)) |
| The linkage acts as a lumped **nonlinear spring**. The neurovascular tract contributes a share; scar mesh changes it. | rat | Bernabei, Maas & van Dieën 2016, BMMB, [10.1007/s10237-016-0795-0](https://doi.org/10.1007/s10237-016-0795-0) (N/mm values in full text, not checked) |
| In humans, evidence is mostly indirect. Linkages are "an extra force pathway". | review | Finni, de Brito Fontana & Maas 2023, J Biomech, [10.1016/j.jbiomech.2023.111575](https://doi.org/10.1016/j.jbiomech.2023.111575) |

Magnitude: the proximo-distal differences are a small fraction of tetanic force, roughly ≤ 10–20 % (*unverified*). They are larger under supraphysiological relative positions.

### 3.3 How much neighbours slide in vivo

| Measurement | Value | Method | Source |
|---|---|---|---|
| MG vs soleus aponeurosis displacement at MVC, knee extended | MG 12.6 ± 1.7 mm, Sol 8.9 ± 1.5 mm, so **≈ 3.7 mm relative** | ultrasound, n = 8 | Bojsen-Møller et al. 2004, J Appl Physiol, [10.1152/japplphysiol.00084.2004](https://doi.org/10.1152/japplphysiol.00084.2004) |
| same, knee flexed (125°) | MG 7.9 ± 1.2, Sol 9.6 ± 1.0, so **≈ −1.7 mm relative** (sign reverses) | same | same (differences derived from the means) |
| Soleus strains when only the knee moves (173° → 150°) although its MTU length is constant | substantial local strains (values in full text) | MRI deformation analysis, healthy volunteers plus cadavers | Huijing et al. 2011, Surg Radiol Anat, [10.1007/s00276-011-0863-1](https://doi.org/10.1007/s00276-011-0863-1) |
| Soleus shortens more than gastrocnemius during moment generation | 78 % more | dual-probe ultrasound | Clark & Franz 2018, PeerJ, [10.7717/peerj.5182](https://doi.org/10.7717/peerj.5182) |
| Rat SO vs LG distal tendons during locomotion | max length-change difference 2.0 mm | in vivo, rat | Bernabei et al. 2017, J Exp Biol ([VU pdf](https://research.vu.nl/ws/files/42485130/Bernabei_JExpBiol_2017.pdf)) |
| Thoracolumbar fascia shear strain over 15° passive trunk flexion | 70.2 ± 3.6 % (no LBP) vs 56.4 ± 3.1 % (LBP) | ultrasound cross-correlation | Langevin et al. 2011, BMC MSD, [10.1186/1471-2474-12-203](https://doi.org/10.1186/1471-2474-12-203) |
| Iliotibial tract deep vs superficial layer relative displacement | 4.8 mm controls, 2.9 mm knee pain, 1.3 mm hEDS | ultrasound | search snippet; primary citation *unverified* |
| Subcutaneous tissue vs vastus lateralis gliding | Thicker dense connective tissue and thicker subcutaneous tissue reduce gliding. Speed matters; limb position does not. | ultrasound PIV | Kawanishi & Kudo 2020, J Bodyw Mov Ther ([summary](https://fasciaguide.com/research/quantitative-analysis-of-gliding-between-subcutaneous-tissue-and-the-vastus)) |

**Gap.** I found **no** paper that gives mm of sliding between rectus femoris and the vasti, or between the hamstrings, during knee flexion. Candidates: Affagard et al., single-subject thigh ultrasound DIC ([HAL](https://hal.utc.fr/hal-03893831v1/document)), and the 2025 MRI velocity strain maps of the thigh (Frontiers Physiol, [10.3389/fphys.2025.1583024](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2025.1583024/pdf)). Both are unread in full.

Order of magnitude so far: neighbouring human muscles slide **a few mm (about 2–5 mm) per contraction or joint excursion**. Sliding is not zero, and it is not free. The interface also carries enough shear to strain a passive neighbour (Huijing 2011).

## 4. Pressed together, or one active next to one passive

- When one muscle is active and its neighbour passive, the **passive neighbour deforms**. Soleus strains with knee-only motion (Huijing 2011). Proximal EDL force follows the TA position (Maas 2001). The interface therefore transmits shear. A sliding-only model predicts zero coupling, which is wrong in direction.
- Relative displacement depends on **joint angle**, and its sign can reverse (Bojsen-Møller 2004). The linkage is a compliant spring, not a glue.
- Under compression (sitting), muscles stay in contact. Inside an intact compartment there is no open gap, because the fascia and loose tissue hold surfaces together, so normal separation is resisted. Teran et al. 2005 note this directly. Without a fascia model, forearm muscles "wobble around, unnaturally separating from the bone" (Fig. 10).
- Contraction stiffens the active side: active cat muscle is 4.0 ± 0.3 MPa by shear-wave velocity (Bernabei 2020, [10.1152/japplphysiol.00112.2019](https://doi.org/10.1152/japplphysiol.00112.2019)). The interface sees a large stiffness jump. That argues for a compliant interface layer rather than a direct weld.

## 5. How simulators model these links

| System | Muscle → bone | Muscle ↔ muscle | Note |
|---|---|---|---|
| **Teran et al. 2005** (TVCG, [10.1109/TVCG.2005.42](https://doi.org/10.1109/TVCG.2005.42), [pdf](https://math.ucdavis.edu/~jteran/papers/TSBNLF05.pdf)) | Origins and insertions are projected from high-res geometry onto the BCC mesh as boundary conditions. Tendon tetrahedra use the same law but are stiffer (Mooney–Rivlin c1 = 60 kPa vs 30 kPa for muscle; bulk modulus 80 vs 60 kPa). | **Fascia "links"**: each embedded surface node is linked to its closest-point anchor on every neighbour within a distance. Each step the closest link corrects position and velocity **along the surface normal only**, giving "frictionless contact" that *does not separate*. Relaxation α = 0.1 (position), β = 0.5 (velocity), with attenuation for long links. Also an extra linear elongation stiffness on muscle boundaries. | Fig. 10 shows our sag symptom, and this method fixes it. Note that α and β were hand-tuned there. |
| **Lee, Sifakis & Terzopoulos 2009** (TOG, [10.1145/1559755.1559756](https://doi.org/10.1145/1559755.1559756)) | 814 Hill-type line muscles plus a quasistatic FEM flesh volume attached to the bones (*unverified* details) | Muscles are lines, so there is no muscle–muscle contact; the flesh is one continuum. | |
| **Ziva VFX** | zAttachment, **Fixed** mode, with per-attachment strength. Damping was added in v2.1 ([Unity blog](https://blog.unity.com/en/engine-platform/improve-simulations-with-ziva-vfx-2-1)). | **Sliding** zAttachment between muscles, so they glide but stay bound (tutorial: [Rookies](https://discover.therookies.co/2020/08/02/a-beginners-guide-to-ziva-dynamics/)). Fascia is a zCloth shrink-wrap. | Solver maths not public. Houdini's equivalent is a springy sliding attachment with distance limits ([SideFX docs](https://sidefx.com/docs/houdini/muscles/attachments.html)). |
| **Han et al. 2024** (TOG, [10.1145/3658135](https://doi.org/10.1145/3658135)) | bones drive muscles | Layered quasistatic sim (muscle, then fascia, then fat) of "hyperelastic soft tissues in close contact". Position-based nonlinear Gauss–Seidel; a neural net is trained on it. | Contact details *unverified* |
| **FEBio** | `tied-elastic`: displacement continuity across the interface, penalty or augmented Lagrangian ([manual](https://febiosoftware.github.io/febio-feature-manual/features/solid_surfaceinteraction_tied-elastic)) | `sliding-elastic` with **`tension = true`**: surfaces may slide but cannot separate normally; optional `fric_coeff`; `auto_penalty` from element size and stiffness ([manual](https://febiosoftware.github.io/febio-feature-manual/features/solid_surfaceinteraction_sliding-elastic)); Zimmerman & Ateshian ([PMC6056201](https://pmc.ncbi.nlm.nih.gov/articles/PMC6056201)) | Forum reports: tied contact is fragile on curved surfaces |
| **OpenSim–knee co-simulation** | Thelen, Choi & Schmitz 2014 co-simulate neuromuscular dynamics and knee contact (J Biomech Eng, DOI *unverified*). Lenhart et al. 2015 ([10.1007/s10439-015-1326-3](https://doi.org/10.1007/s10439-015-1326-3)): 6-DoF patellofemoral and tibiofemoral joints, 14 ligament bundles as nonlinear springs, elastic-foundation cartilage contact, validated against dynamic MRI. COMAK and OpenSim-JAM (Smith 2017 thesis). | n/a | Patella is a free body pulled by tendon springs into contact. This is the right fix for our lift-off. |
| **Open Knee(s)** ([simtk](https://simtk.org/home/openknee)) | FE knee models with tied and sliding contact built from 8 specimens | n/a | Chokhandre et al. 2023, [10.1007/s10439-022-03074-0](https://doi.org/10.1007/s10439-022-03074-0) |

## 6. Datasets to check a simulation (listed only; nothing downloaded)

| Dataset | Contents | Size | Access / licence | Use |
|---|---|---|---|---|
| **Open Knee(s)** | 8 cadaver knees: MRI (NIfTI), CT; 6-DoF robot tibiofemoral **and patellofemoral** mechanics; FE models | 8 specimens; size *unverified* | CC BY 4.0 per project page. Data DOI [10.18735/4e78-1311](https://doi.org/10.18735/4e78-1311), models [10.18735/b0zv-n395](https://doi.org/10.18735/b0zv-n395). Data paper: Data in Brief 2021, [10.1016/j.dib.2021.106824](https://doi.org/10.1016/j.dib.2021.106824). | T6 (patellar kinematics under quad load); patellar-tendon link |
| **UMUD** (index of musculoskeletal ultrasound sets) | 14 datasets, 76,124 images: fascicles, aponeuroses, CSA | 76k images | Per-dataset licences on OSF/Zenodo; paper CC BY 4.0 ([pdf](https://eprints.glos.ac.uk/15848/1/15848%20Ritsche%20et%20al%20%282026%29%20UMUD%20a%20web%20application.pdf)) | T4, T5 (thickness, aponeurosis motion) |
| **DL_Track_US** data | Aponeurosis and fascicle networks plus example videos | *unverified* | [Zenodo 7866598](https://zenodo.org/records/7866598); newest on OSF; licence to check | T5 (aponeurosis displacement under load) |
| **FALLMUD** | 812 lower-leg ultrasound images with fascicle and aponeurosis masks | 812 images | Licence *unverified* ([arXiv 2009.04790](https://arxiv.org/pdf/2009.04790) lineage) | T5 |
| Gastrocnemius fascicle-tracking trials (Penn) | Experimental data, tracking code, tracked trials | *unverified* | [Zenodo 2598553](https://zenodo.org/record/2598553); licence to check | MG/Sol sliding analogue for the interface law |
| **Thelen lab** dynamic MRI knee (Lenhart 2015, Westphal 2013) | Cine-PC MRI of tibiofemoral and patellofemoral motion under elastic or inertial quad loads, n = 8 | — | **Request from the authors**; no repository listed (Europe PMC hasData = N). JAM model on GitHub/SimTK, licence *unverified*. | T6 |
| **Sheehan (NIH)** cine-PC MRI cohort | 79 healthy controls plus 41 PFP patients; 3D patellofemoral kinematics; accuracy < 0.33 mm | — | Not public; collaboration request only (*unverified*). Fick 2022, [10.1016/j.jbiomech.2021.110819](https://doi.org/10.1016/j.jbiomech.2021.110819) | T6 (best reference for patellar tracking against angle) |
| simtk **kneemodel** | Freely downloadable OpenSim knee model | — | [simtk.org/home/kneemodel](https://simtk.org/home/kneemodel/); licence *unverified* | T6 geometry check |

**No open dataset of intermuscular sliding in the thigh was found.** The best numeric targets are in papers: Bojsen-Møller 2004 (calf, ± 2–4 mm), Huijing 2011 (MRI strain) and Langevin 2011 (shear strain).

## 7. Recommended interface model for our engine

1. **Muscle → bone: tied along the full atlas footprint.**
   - Make it a strip for fleshy origins and a patch for tendinous ends. Take the area from Z-Anatomy, with 500 mm² for the adult quadriceps on the patella as the check (Knee 2023).
   - Tendons are explicit stiff tets or 1D springs with **E ≈ 1.1 GPa** (Hansen 2006). Scale stiffness as k = E·A/L from the elf's geometry. The 4300 N/mm adult value is a check, not an input.
2. **Aponeurosis: anisotropic.**
   - Longitudinal strain about 1.4 % at high force (Magnusson 2003); transverse strain about 4× larger (Azizi 2009).
3. **Patella: free rigid body.**
   - Joined by quadriceps and patellar tendon springs, with frictionless contact on the femoral trochlea. Do not pin it.
   - Seating comes from quadriceps tension, including passive tension (Lenhart 2015; OpenSim pulley).
   - This removes the lift-off. Pass for T6: shift ≤ 3 mm, tilt ≤ 6°, rotation ≈ 8° over flexion (Nha 2008).
4. **Muscle ↔ muscle and muscle ↔ septum: "bound sliding" plus a weak tangential spring.**
   - Normal direction: **bilateral**. Contact may not open, which is FEBio `tension = true` or Teran's normal-only fascia links. This alone should remove most of the 40–70 mm sag, which comes from unilateral contact letting the belly fall away.
   - Tangential direction: a compliant cohesive spring with shear stiffness per unit area k_t. It stands for areolar tissue and neurovascular tracts. Over bursae and articular surfaces, use frictionless contact with no tangential spring.
   - **k_t is not picked by hand.** Identify it by inverse fit to: (a) Bojsen-Møller 2004 relative displacement (+3.7 / −1.7 mm, sign flip with knee angle); (b) Huijing 2011 soleus strain at constant MTU length; (c) the 2016 lumped-stiffness curve once read.
   - Seed range for the fit: modulus of loose tissue (≈ 3–5 kPa) divided by layer thickness (≈ 0.05–1 mm). That gives k_t ≈ 3–100 kPa/mm (*derived, unverified*).
5. **Newton/VBD implementation.**
   - Per surface vertex, store a closest-point anchor on each neighbour surface and re-project it every step, as in Teran 2005.
   - Add a VBD energy term ½k_n(d·n)² + ½k_t|d_t|² per anchor. Set k_n large, limited by stability, and take k_t from the fit.
   - This fits VBD's per-vertex block solve, so no global constraint solve is needed.

**Which dataset checks which scene**

| Scene | Check |
|---|---|
| T1 | Footprint areas: Knee 2023, Grob |
| T4/T5 | UMUD, DL_Track and FALLMUD ultrasound for aponeurosis displacement and thickness under load; Bojsen-Møller and Magnusson numbers |
| T6 | Open Knee(s) robot patellofemoral data; Nha 2008 and Wilson 2009 tracking numbers; Thelen and Sheehan MRI on request |
| T7/T9 | Interface law validated in T5 must not let the vasti and hamstrings drift under the ischium. Thigh sliding target missing (gap). |
| T8 | k_t damping affects jiggle decay. No interface damping data found. |

## 8. Open questions

1. What is the in vivo **mm of sliding between rectus femoris, the vasti and the hamstrings in knee flexion**? Read Affagard (HAL) and the 2025 Frontiers MRI strain paper in full.
2. Get the numeric N/mm lumped stiffness from Bernabei 2016 and the soleus strain values from Huijing 2011 (full text).
3. What are the **patellar flexion slope** (Li 2007 full text) and the superior shift against angle (Sheehan group tables)?
4. Is there a friction or shear-stiffness value for epimysium on epimysium (tribometry)? None was found. Is the interface viscous (rate-dependent, per Kawanishi 2020)?
5. How should adult tendon and footprint values scale to a 0.98 m stylised body? Modulus is intrinsic; areas and lengths come from our geometry. Confirm with the user.
6. Licences of the DL_Track, FALLMUD, Zenodo 2598553 and JAM repositories need checking on their record pages before use.
