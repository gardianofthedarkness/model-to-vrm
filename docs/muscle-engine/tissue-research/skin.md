# Skin (epidermis + dermis): mechanics in living people

Desk research for E-MAT (`docs/muscle-engine/plan.md`) and the knee-crease flip in
`docs/muscle-engine/sit-physics-research-handoff.md` §6b. Nothing was downloaded or installed.
Every number has a source. *unverified* means I saw the figure only in a secondary
source or from memory and did not read it in the primary paper. A gap means I found no
source for it.

Main point: **in vivo skin is soft at small strain (about 10–300 kPa) and stiff at
large strain (tens of MPa). It is pre-tensioned by about 30–90 kPa, and it does not
compress: it folds.** Over the knee it stretches by more than 60 %. Behind the knee it
contracts and folds.

---

## 1. Stretch

### 1.1 J-curve, anisotropy, Langer lines (excised human skin)

| quantity | value | method | source |
|---|---|---|---|
| initial (toe) slope | 1.18 ± 0.88 MPa | uniaxial tension, 0.012 s⁻¹, back skin, 56 samples, 7 donors aged 81–97 | Ní Annaidh et al. 2012, JMBBM, doi:10.1016/j.jmbbm.2011.08.016 (arXiv 1302.3022) |
| linear-region modulus | 83.3 ± 34.9 MPa (all samples) | same | same |
| modulus ∥ / ⊥ Langer lines (mid-back) | 112.5 ± 36.5 / 63.8 ± 24.6 MPa | same | same, Table 3 |
| UTS ∥ / ⊥ | 28.6 ± 9.0 / 16.5 ± 5.7 MPa | same | same |
| failure stretch ∥ / ⊥ | 1.46 / 1.52 | same | same |
| size change after excision | 73 % of samples grew 5.1 ± 3.4 %, 27 % shrank 3 ± 2.9 % (the paper cites a maximum shrinkage of 20 % from the literature) | calliper before/after cutting | same |
| tensile maximum slope across studies | 4–140 MPa, depending on strain rate | review | Kalra, Lowe, Al-Jumaily 2016, J Mater Sci Eng 5:254 ([PDF](https://www.researchgate.net/publication/307522356_Mechanical_Behaviour_of_Skin_A_Review)) |
| Young's modulus across all methods | 5 kPa – 140 MPa | review | same |

Langer lines follow the main direction of collagen fibres in the dermis, and Ní Annaidh
et al. confirmed this with histology. In the linear region, skin is about 1.7× stiffer
along the lines than across them.

### 1.2 Hyperelastic fits with constants

| model | constants | data | source |
|---|---|---|---|
| **GOH** (Gasser–Ogden–Holzapfel, neo-Hookean matrix plus two dispersed fibre families) | μ = 0.2014 MPa, k₁ = 243.6 (MPa), k₂ = 0.1327, κ = 0.1404 ± 0.0028 (from histology); mean fibre angle γ ≈ 41° to the Langer line *unverified* | excised back skin, uniaxial | Ní Annaidh et al. 2012, Ann Biomed Eng 40:1666, doi:10.1007/s10439-012-0542-3 |
| **Ogden + QLV**, and **Tong & Fung + QLV**, with an orthotropic pre-stress | Pre-stress **28–92 kPa**. Fit error 13.7–21.5 %. Per-volunteer constants are in the full text (not read). | **in vivo**: 3-D force micro-robot on the anterior forearm and posterior upper arm | Flynn, Taberner, Nielsen 2011, Ann Biomed Eng 39:1935, doi:10.1007/s10439-011-0292-7 |
| same micro-robot | in vivo maximum tension 6.2 N/m (lower forearm), 11.4 N/m (upper forearm), 5.6 N/m (upper arm) | in vivo | Flynn et al. 2011, Biomech Model Mechanobiol 10:27, doi:10.1007/s10237-010-0216-8 |
| **Extended Mooney**, two-layer | upper layer C₁₀ = 0.11 kPa, reticular dermis C₁₀ = 0.16 MPa (one subject) | in vivo suction with OCT and ultrasound, volar forearm | Hendriks et al. 2006, Med Eng Phys (values via search, *unverified*); 2004 paper doi:10.1111/j.1600-0846.2004.00077.x |
| Veronda–Westmann matrix + Weiss fibres | constants in paper (not read) | excised human and mouse skin | Groves et al. 2013, JMBBM 18:167 ([ORCA](https://orca.cardiff.ac.uk/47152)) |
| **Tong & Fung** exponential | constants are for **rabbit** abdominal skin, not human. Not read. | biaxial tests | Tong & Fung 1976, J Biomech 9:649 |

Two notes on these fits:

- The out-of-plane response (indentation, suction) is predicted poorly when a model is
  fitted to in-plane data only (Flynn 2011). Parameters from tension tests alone do not
  transfer to pressing.
- The GOH k₁ value is fitted to the stiff linear region of very old skin from donors.
  Treat it as an upper bound for a young character.

### 1.3 In vivo devices: small-strain stiffness

| device / principle | value | site, subjects | source |
|---|---|---|---|
| suction (Cutometer family) | E ≈ 25–260 kPa, as quoted in a secondary source *unverified* | various | arXiv 2010.13849 citing Kalra 2016 |
| suction + ultrasound (taut-membrane model) | E and σ₀ fitted. The values I saw, 129 ± 88 kPa and 13.5 ± 5 kPa, are *unverified*. | volar forearm, 10 men | Diridollou et al. 2000, Skin Res Technol (PMID 11428960) |
| torsion | modulus doubles with age; values of about 0.42 MPa (< 30 y) and 0.85 MPa (> 30 y) are *unverified* | 138 people aged 3–89 | Agache et al. 1980, Arch Dermatol Res 269:221, doi:10.1007/BF00406415 |
| dynamic indentation, Kelvin–Voigt | complex modulus at 10 Hz: 10.7 ± 2.6 kPa (young) to 7.2 ± 2.1 kPa (old) | 46 subjects aged 18–70 | Boyer et al. 2009, Skin Res Technol, doi:10.1111/j.1600-0846.2008.00324.x |
| quasi-static indentation (Hertz) | 7–33 kPa | forearm and face, 10 people | Skin Res Technol 2007, cited in search *unverified* |
| suction vs shear vs indentation compared | the three principles rank skin stiffness differently | volunteers at several sites | Junker et al. 2023, doi:10.1007/s42558-023-00053-w |
| Cutometer convention | elastic and viscous parts split at 0.1 s; typical runs of 450–500 mbar, 2–5 s on / 2–5 s off | — | Qu & Seehra, [SCC](https://library.scconline.org/v067n01/38) |

Low values (7–30 kPa) come from indentation, which also loads the fat underneath.
Higher values (100–800 kPa) come from suction and torsion, which mostly measure the
dermis in-plane. For an FEM skin layer, use the dermis-only values.

---

## 2. Thickness, membrane vs volume, bending, folding

### 2.1 Thickness

| site | value | method | source |
|---|---|---|---|
| thigh, epidermis + dermis | ≈ 1.9–2.1 mm | ultrasound, review table | J Anat 2021, [PMC7930759](https://pmc.ncbi.nlm.nih.gov/articles/PMC7930759/) (table 2) |
| thigh dermis, men / women | 1.89 / 1.65 mm | ultrasound, adults with diabetes | PLoS ONE 2014 ([DOAJ](https://doaj.org/article/6c3f8e6e30364517b830f039743851bc)) |
| epidermis, six other sites | 77–267 µm | histology, 180 samples | Oltulu et al. 2018, Turk J Plast Surg ([DOAJ](https://doaj.org/article/b54969bf2a214274ad1c8a0cff19b0e0)) |
| dermis, six other sites | 2.1–5.9 mm | same | same |
| buttock, knee, hip | **gap**: no value found. Gluteal subcutaneous fat is about 3 cm. | — | [MDPI review](https://www.mdpi.com/2673-4095/7/1/3) |

A 2022 meta-analysis found that high-frequency ultrasound gives higher epidermis values
than histology or OCT (PMID 35366353).

### 2.2 Membrane or volume? Bending

- Skin is 1.5–2 mm thick on a leg whose radius is centimetres, so thickness/radius is
  below 0.05. That is shell territory.
- "Thin Skin Elastodynamics" (Li, Sueda, Neog, Pai 2013, ACM TOG 32(4):49,
  [project page](https://www.cs.ubc.ca/research/thinskinelastodynamics)) models skin as
  a 2-D hyperelastic membrane that slides over the body.
- **Bending stiffness:** I found no direct measurement. *Derived, not measured:*
  D = E t³ / (12(1−ν²)), with t = 1.5 mm and E = 0.1–1 MPa, gives
  **D ≈ 4·10⁻⁵ – 4·10⁻⁴ N·m**.
- **Wrinkling:** wrinkles form because a stiffer top layer (the stratum corneum or
  epidermis) sits on a softer substrate. With a stiffness ratio of about 10 or more,
  period-doubling appears. Above about 100, the microrelief stops controlling the
  wrinkles (Limbert & Kuhl 2018, Soft Matter (issue 8),
  [eprint](https://eprints.soton.ac.uk/416791)).

Consequence for joint creases: skin does not take compressive stress. It buckles at a
wavelength set by D and the stiffness of the fat below. A model that lets skin compress
in volume instead of folding is wrong behind the knee.

---

## 3. Viscoelasticity, damping, relaxation

| quantity | value | source |
|---|---|---|
| model used for in vivo fits | QLV (Flynn 2011). Kelvin–Voigt fits dynamic indentation (Boyer 2009). | above |
| Kelvin–Voigt stiffness / damping (dynamic indentation) | 47–128 N/m and 0.08–0.12 N·s/m (complex modulus 13–33 kPa) | Boyer et al. 2007 conference paper, via search *unverified* |
| QLV τ₁ | 0.19 ± 0.10 s, on **excised** thin specimens 211–671 µm thick (species not checked) | [PMC4372409](https://pmc.ncbi.nlm.nih.gov/articles/PMC4372409) *unverified* |
| strain-rate effect | failure stress and strain energy rise between 0.012 and 29 s⁻¹ | Ní Annaidh/Ottenio et al. 2014, conf. paper ([Galway repo](https://researchrepository.universityofgalway.ie/entities/publication/7b2fe824-8857-4640-abd7-b6343318c3b9)) *unverified* |
| age | the viscous share (Uv/Ue) rises with age | Cutometer studies, PubMed 2221979 |

**Gap:** I found no in vivo Prony series (gᵢ, τᵢ) for skin on the thigh or buttock. The
FEMTO-ST CutiScan dataset (§6) has the raw suction curves needed to fit one.

---

## 4. Pressed together: friction and seated load

| pair | μ | conditions | source |
|---|---|---|---|
| dry skin against various materials | > 0.2, typically ≈ 0.5, independent of pressure | review of the literature | Derler & Gerhardt 2012, Tribol Lett 45:1, doi:10.1007/s11249-011-9854-y |
| moist or wet skin | typically **> 1**, rising as contact pressure falls | same | same |
| skin against textile, static / dynamic | chest 0.90 / 0.79, dorsal forearm 0.31 / 0.25 | handheld device | Tribol Lett 2021, doi:10.1007/s11249-021-01560-5 |
| skin against aluminium | rises with age and ambient temperature | 3 sites, large sample | Veijgen et al. 2013, JMBBM, doi:10.1016/j.jmbbm.2013.02.009 |
| skin against nylon, silicone, cotton, Pelite, aluminium | values in paper (not read) | in vivo | Zhang & Mak 1999, Prosthet Orthot Int 23:135 |
| trousers against seat | vinyl gives the highest μ, two-layer nylon the lowest; sled and human results within 10 % | seated people and a sled | Lamsal 2023 MSU thesis ([etd](https://d.lib.msu.edu/etd/51266)); Exp Mech 2023, doi:10.1007/s11340-023-01022-0 |
| **skin on skin (back of knee)** | **gap**: no direct measurement found. Bracket it with dry ≈ 0.5 and sweaty > 1 from Derler. | — | — |

Seated load: on a flat hard seat the peak pressure lies under the ischial tuberosities,
and men's peaks are about 10 kPa higher than women's
([TU Delft](https://repository.tudelft.nl/file/File_3bd35d13-ec53-44f5-85d7-c90c6707fab4)).
Leaning back or sliding the pelvis forward raises shear on the skin
([Kobara 2007](https://kwmw.repo.nii.ac.jp/record/13572/files/Kobara%20K%282007%2012%201%29.pdf)).
I found no data on how much of the seated deformation the skin itself takes up, as
opposed to the fat and muscle. Under ultrasound, the dermis is the layer that resists
suction, not the fat (Diridollou 1998).

---

## 5. Skin strain over joints during motion (ground truth)

| study | method | result |
|---|---|---|
| **Rupani, Cleland, Saal 2025**, J R Soc Interface 22:20240794, doi:10.1098/rsif.2024.0794 (CC BY) | 4-camera 3-D DIC on stamped speckle, 15 participants, about 1 mm² resolution, error about 0.3–0.9 % | At maximal knee flexion the principal stretch is **> 60 % at the knee** and **10–20 % on the upper thigh**. It stays above 10 % even 15–20 cm from the knee. Medial stretch is larger than lateral. Stretch rises with angle (20–70°) and saturates near the joint. Skin area expands near the knee by up to about 20–25 % (read from the figure); away from the knee the area barely changes, because stretch along one axis is balanced by compression across it. |
| Ringenbach et al. (PolyU 2024), [PDF](https://ira.lib.polyu.edu.hk/handle/10397/108077) | 3-D scans, 50 people, knee 0–90° | over the patella: vertical strain **19 %**, horizontal **7 %** |
| Wang & Liu (4-D scan while running), via search *unverified* | 4-D scan | elongation at the front of the knee and the back of the hip; **contraction at the back of the knee and the front of the hip** |
| Wessendorf & Newman 2012, IEEE TBME 59:3432, doi:10.1109/TBME.2012.2215859 | motion capture markers, squat | strain maps and lines of non-extension around the knee. Strains were large enough to need second-order Lagrangian strain. Values in paper (not read). |
| Choi & Hong 2015; Choi & Ashdown | 3-D skin-length change, knee flexion, sitting at 90° (25 women) | lines of non-extension and maximum stretch on the hip and leg. Values in papers (not read). |
| Pierrat et al. 2015, Exp Mech 55:349, doi:10.1007/s11340-014-9947-2 | DIC, 11 subjects, knee flexion | strong effect of sex on knee skin strain |
| Lines of non-extension | Iberall, Newman | [overview](https://en.wikipedia.org/wiki/Lines_of_non-extension). Near the joint itself, true lines of non-extension may not exist (patent US10028697). |

---

## 6. Datasets for checking the simulation

| dataset | contents, size | access, licence | URL |
|---|---|---|---|
| **Rupani 2025 skin-stretch DIC** | processed DIC (3-D meshes, about 9 300 triangles per patch, stretch maps) and anonymised reference images. Knee and thigh at 20–70°, plus wrist, neck, waist, back, face. 15 participants. | **open on Zenodo** (ref. 71 of the paper). Paper is CC BY; the data licence is *unverified*. | via doi:10.1098/rsif.2024.0794 |
| **FEMTO-ST in vivo skin anisotropy (CutiScan)** | AVI videos plus CSV 2-D DIC displacements. 30 repeats × 21 suction set-points (100–500 mbar, steps of 20 mbar). Anterior forearm, one 28-year-old man. | public repository. Licence not stated *unverified*. | https://dmarc.femto-st.fr/fichiers/686/arborescence ; code https://dmarc.femto-st.fr/fichiers/687 |
| **DFAUST** | more than 40 000 raw 4-D scans at 60 fps, registered meshes, and masks of the reliable areas. 10 subjects. | registration required; PS:License 1.0, research only | https://dfaust.is.tue.mpg.de |
| **Dyna** | more than 40 000 registered meshes of soft-tissue motion, 10 subjects, one shared topology | PS:License 1.0, research only | https://ps.is.tuebingen.mpg.de/code/dyna |
| **3DCP / TUCH** | SMPL-X fits to scans of 5 subjects in self-contact poses, plus AMASS meshes optimised for self-contact | PS:License 1.0, research only | https://ps.is.mpg.de/code/3dcp-3d-contact-pose |
| Derler & Gerhardt friction tables | μ by site, counter-material and hydration (from the literature) | open PDF | https://doc.rero.ch/record/315296 |
| Lamsal 2023 seat friction | μ for 7 trouser fabrics × 2 seat covers, human and sled | open thesis | https://d.lib.msu.edu/etd/51266 |
| Junker 2023 device comparison | suction, shear and indentation on volunteers and on a phantom | paper open; raw-data availability *unverified* | doi:10.1007/s42558-023-00053-w |
| Wessendorf & Newman, Pierrat, Ringenbach, Flynn micro-robot | knee strain fields, in vivo force–displacement | not released; ask the authors | DOIs above |

None of these involve pictures of real people being used by us. Rupani's reference
images are skin patches; use only the processed meshes.

---

## 7. Recommended E-MAT entry (skin)

| parameter | value | source / note |
|---|---|---|
| model | GOH: neo-Hookean matrix plus 2 fibre families along and across Langer lines | Ní Annaidh 2012 |
| matrix μ | **0.20 MPa** | Ní Annaidh 2012 (GOH). It lies inside the in vivo suction range of 25–260 kPa. |
| k₁, k₂, κ | 243.6 MPa, 0.1327, 0.14 | same. This is old, excised skin; mark k₁ as an upper bound. |
| fibre direction | Langer-line field on the mesh | Langer lines; the direction comes from data, not tuning |
| in vivo pre-stress | **28–92 kPa**, orthotropic. Start at the median, about 60 kPa *derived*. | Flynn 2011 |
| thickness | thigh about **2.0 mm** (total), dermis 1.65–1.9 mm; scaling to 0.98 m is an open question | PMC7930759; PLoS ONE 2014 |
| bending | from the shell formula, D ≈ 4·10⁻⁵ – 4·10⁻⁴ N·m *derived* | §2.2 |
| viscosity | Kelvin–Voigt / QLV. τ₁ ≈ 0.2 s *unverified*. Fit a Prony series to the FEMTO-ST curves. | §3 |
| density | 1100 kg/m³ *unverified, no source read* | — |
| friction, skin–skin | dry 0.5, moist > 1 | Derler 2012 |
| friction, skin–seat | 0.3–0.9 against textile | Tribol Lett 2021 |
| limits to check | failure stretch 1.46–1.52; flag any element stretched beyond 1.45 | Ní Annaidh 2012 |

### Membrane or tet layer

Use a **triangle shell (membrane plus bending) attached to the outer surface of the fat
tets**, not a separate tet layer:

1. Skin about 2 mm thick, cut into tets, gives slivers. Thin tets under compression are
   exactly what flips at the knee crease (handoff §6b).
2. Real skin folds instead of compressing (§2.2). A shell with a bending term buckles
   naturally. Thin tets either lock or invert.
3. GOH is a membrane-friendly law, and published skin simulation in graphics is
   membrane-based (Li et al. 2013).
4. Pre-stress is easy to apply in-plane: set a rest shape smaller than the mesh.

Whether Newton's VBD solver couples cloth-style triangles to tets in one solve is
*unverified*; check the Newton docs before choosing this path. If it does not, the
fallback is one prism layer, but expect the same crease inversions.

### Which scene each dataset checks

| scene | check | data |
|---|---|---|
| T1 | skin thickness and Langer-direction field present | thickness table §2.1 |
| T4 | indentation and suction response of skin over fat | FEMTO-ST CutiScan, Boyer 2009, Junker 2023 |
| **T6** (knee 0–150°) | stretch above the knee > 60 % and 10–20 % on the upper thigh; 19 % / 7 % over the patella at 90°; contraction behind the knee with folds, not inverted tets | Rupani 2025 (Zenodo), Ringenbach 2024, 3DCP for crease self-contact |
| T7 / T9 (chair) | interface μ and skin shear under the ischium | Derler 2012, Lamsal 2023, TU Delft pressure |
| T8 (drop and settle) | surface jiggle and decay | DFAUST / Dyna (research licence) |
| T10 | error on held-out motion | DFAUST sequences |

---

## 8. Open questions

1. There is **no in vivo GOH or Prony fit for thigh or buttock skin**. Every constant
   comes from the forearm, the back, or donors aged 81–97. Fitting to FEMTO-ST and
   Rupani data would avoid hand values.
2. **Skin-on-skin friction behind the knee** is unmeasured, and hydration moves μ from
   0.5 to above 1.
3. **Buttock, knee and hip skin thickness**: I found no source.
4. **Scaling to a 0.98 m body**: do thickness and pre-stress scale with size, or should
   adult values be kept? Child-skin data are needed.
5. **Bending stiffness** has not been measured directly. Mine is derived.
6. **Does Newton VBD support coupling a shell to tets?** This is the deciding fact for
   the membrane recommendation.
7. **Licences**: the Rupani Zenodo licence and the FEMTO-ST licence need checking
   before the project uses them in public code.
8. The low suction values (25–260 kPa) and the 1.18 MPa tensile toe slope are hard to
   reconcile. The suction values include pre-stress and boundary effects (Flynn 2011
   warns about this), so treat the gap as a reason to fit in vivo data rather than
   pick one number.
