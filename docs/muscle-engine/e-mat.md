# E-MAT v0.1: material table for the subject profile

Built 2026-11-24 for [plan.md](plan.md) §4.1 (E-MAT), from the six reports in [tissue-research/](tissue-research/) plus a targeted
search for sex and age data. Machine-readable copy: [e-mat.json](e-mat.json) (same rows, SI units, generated from one list).
Nothing here is tuned. A value with no measured source is left empty and marked **FIT** with the dataset it must be fitted to.

## 1. Subject profile

A healthy woman of about 28, normal build (plan §2.1). Her BMI-equivalent comes from her own mesh; fat depth from her surface;
muscle and bone shapes from the warped atlas, which is **male** (section 5).

Source tiers, in order of preference: **A** women about 20–35 in vivo · **B** mixed-sex young adults in vivo · **C** any adult in vivo ·
**D** ex vivo human or animal · **M** model constant (no tissue population) · **N** no measured source.

## 2. Summary

| count | |
|---|---|
| parameters (rows) | 99 |
| tier A (women 20–35, in vivo) | 6 |
| tier B / C / D / M / N | 1 / 30 / 34 / 8 / 20 |
| numerically adjusted for sex or age | 0 (see below) |
| direct female source used | 8 |
| sex effect checked, none found (value kept) | 1 |
| sex/age effect published but not transferable (direction stated) | 5 |
| no sex/age data found (flagged) | 85 |
| status *unverified* | 14 |
| must be fitted to a dataset (**FIT**) | 22 |
| flagged (weaker source than tier A/B, or a gap) | 85 |

**Why no numeric adjustment.** Where a young-female in vivo value existed it was used directly (muscle along-fibre shear,
patellar tendon, thigh skin thickness). Where a sex study existed for a mixed source it showed no difference (specific tension,
O'Brien 2010). The remaining sex or age effects (skin stiffness with age, Agache 1980; friction by sex, Veijgen 2013; fascia
thickness young vs old women) are from other methods or have no extractable factor, so applying them would be a guess. Their
direction is written in the row.

Columns: *value* is the central value with the cited range in brackets; a range alone means the source gives no single value.
*Tier* is the source population class. *Tests* are the scenes in plan §6 that check the row.

## 3. Table

### Muscle (passive, active, damping)

| id | parameter | value | unit | model | source | population | tier | adjustment for the profile | status | tests |
|---|---|---|---|---|---|---|---|---|---|---|
| `muscle.passive.mu_along` | shear modulus along fibres, relaxed, short length (G1 target) | 4160 (3190–5130) | Pa | Blemker-Delp transversely isotropic (G1); small-strain SWE value, mapping to the large-strain constant is a T4 fit | Maeda et al. 2021, Int J Environ Res Public Health 18:8947, doi:10.3390/ijerph18178947 | 9 women 30-39 y (34.8 +/- 3.0), BMI 20.3, Japanese, not regularly exercising; in vivo SWE; vastus lateralis, knee full extension | A | no adjustment: direct female 30-39 value; same study found no sex effect at full extension (Gennisson 2010 brachialis 5.4 kPa agrees) | read in full | T4, T5 |
| `muscle.passive.mu_along_knee90` | shear modulus along fibres, moderately stretched (knee 90 deg) | 8630 (6810–10450) | Pa | Blemker-Delp passive fibre curve (check of P1, P2, lambda_ofl) | Maeda et al. 2021, Int J Environ Res Public Health 18:8947, doi:10.3390/ijerph18178947 | 9 women 30-39 y, BMI 20.3; in vivo SWE; vastus lateralis, knee 90 deg flexion | A | no adjustment: direct female value. Akagi 2019 (18 men, 23 women, 21-22 y, VL/VM/RF at 90 deg) found men > women (p<0.001) and supports a female-specific value; its kPa values are only in a figure | read in full | T3, T4 |
| `muscle.passive.mu_along_fullflex` | shear modulus along fibres, strongly stretched (knee full flexion) | 43020 (33580–52460) | Pa | Blemker-Delp passive fibre curve (check) | Maeda et al. 2021, Int J Environ Res Public Health 18:8947, doi:10.3390/ijerph18178947 | 9 women 30-39 y, BMI 20.3; in vivo SWE; vastus lateralis, knee full flexion | A | no adjustment: direct female value; in this study women were lower than men at full flexion (p=0.001; men 30-39: 56.7 kPa) | read in full | T3, T6 |
| `muscle.passive.mu_cross` | shear modulus across fibres, relaxed (G2) | 800 (50–1030) | Pa | Blemker-Delp transversely isotropic (G2) | Gennisson et al. 2010, Ultrasound Med Biol 36:789, doi:10.1016/j.ultrasmedbio.2010.02.013; Palevski et al. 2006, J Biomech Eng 128:782 | Gennisson: healthy volunteers, in vivo SSI, brachialis (n, sex, age not in abstract); Palevski: porcine gluteus, fresh, ex vivo, preconditioned long-term 0.81 kPa | C | no adjustment: no sex or age data for cross-fibre shear found; value set at Palevski's long-term figure inside Gennisson's in vivo band; flagged | read in full (Gennisson, per tissue report); abstract (Palevski) | T4 |
| `muscle.passive.fibre_P1` | passive fibre stress coefficient P1 | **FIT** 0.05 | 1 (fraction of sigma_max; normalisation unverified) | Blemker-Delp passive fibre law f_p = P1(exp(P2(l/l_ofl-1))-1) | Blemker, Pinsky & Delp 2005, J Biomech 38:657; Blemker & Delp 2005, Ann Biomed Eng 33:661, doi:10.1007/s10439-005-1433-7; FEBio muscle material https://febiosoftware.github.io/febio-feature-manual/features/solid_material_muscle_material | model constant matched to whole-muscle behaviour, no tissue population | M | no adjustment: model constant; refit to the female passive SWE angle series | read in full (FEBio manual, per tissue report) | T3 |
| `muscle.passive.fibre_P2` | passive fibre exponent P2 | **FIT** 6.6 | 1 | Blemker-Delp passive fibre law | Blemker, Pinsky & Delp 2005, J Biomech 38:657; Blemker & Delp 2005, Ann Biomed Eng 33:661, doi:10.1007/s10439-005-1433-7; FEBio muscle material https://febiosoftware.github.io/febio-feature-manual/features/solid_material_muscle_material | model constant, no tissue population | M | no adjustment: model constant; refit as for P1 | read in full (FEBio manual, per tissue report) | T3 |
| `muscle.passive.lambda_ofl` | fibre stretch at optimal length relative to mesh rest | **FIT** 1.4 (1.07–1.4) | 1 | Blemker-Delp (1.4 in papers; 1.07 in the FEBio example) | Blemker, Pinsky & Delp 2005, J Biomech 38:657; Blemker & Delp 2005, Ann Biomed Eng 33:661, doi:10.1007/s10439-005-1433-7; FEBio muscle material https://febiosoftware.github.io/febio-feature-manual/features/solid_material_muscle_material | model constant; really a property of the atlas rest pose | M | no adjustment: must be computed per muscle from myoLegs optimal fibre length against the warped atlas fibre length (both male, see conflicts) | abstract (PMC2857665 confirms 1.4) | T3 |
| `muscle.passive.bulk_physical` | physical bulk modulus | 2e9 | Pa | volumetric term U(J); physical value, too stiff for VBD | muscle is about 80 % water (Van Loocke 2009); K of water used as the estimate | none (water) | N | no adjustment: physical estimate, not a measurement of muscle | *unverified* | T2 |
| `muscle.solver.K_over_G` | bulk-to-shear ratio used by the solver | 100 (100–2000) | 1 | volumetric penalty; stability choice, not a material fact | FEBio muscle example uses K/G = 2000 (k = 1 MPa, G = 500 Pa) | solver setting | M | no adjustment: numerical setting, judged only by the T2 volume check | read in full (FEBio manual, per tissue report) | T2 |
| `muscle.check.volume_change` | volume change during contraction (check) | 0.001 | 1 (upper bound) | check target | Baskin & Paolini 1967 (as quoted in the muscle report) | not read | N | no adjustment | *unverified* | T2 |
| `muscle.density` | density | 1050 (1040–1060) | kg/m^3 | inertia | Woodard & White 1986, Br J Radiol 59:1209 (values as tabulated against ICRU 44 in PMC3396587, Table 10); ICRU 44 (1989) 1040 | reference adult tissue composition, sex not stated | C | no adjustment: no sex effect on muscle density found | abstract (comparison table seen via search, paper not read) | T8, T9 |
| `muscle.active.sigma_max` | peak isometric fibre stress (specific tension) | 2.68e5 (1.71e5–5.73e5) | Pa | Blemker-Delp active fibre stress sigma_max * a * f_L(lambda) | Persad et al. 2024, J Appl Physiol 137:945, doi:10.1152/japplphysiol.00296.2024; range ends: gracilis direct 171 kPa (Drexel 2023), O'Brien et al. 2010, Exp Physiol 95:202, doi:10.1113/expphysiol.2009.048967 | systematic review: 29 in vivo human studies, 95 values (2-73 N/cm^2), mixed sex | B | no adjustment: O'Brien 2010 found no sex difference in in vivo quadriceps specific tension (women 57.3 +/- 13, men 55 +/- 11 N/cm^2, 20 adults); women differ in PCSA, not stress | abstract | T3, T5 |
| `muscle.active.fl_halfwidth` | active force-length: half-width of the central parabola | 0.5 | 1 (of lambda/lambda_ofl) | f = 1-4(1-x)^2 for 0.6<x<1.4; 9(x-0.4)^2 below; 9(x-1.6)^2 above; x = lambda/lambda_ofl | Blemker, Pinsky & Delp 2005, J Biomech 38:657; Blemker & Delp 2005, Ann Biomed Eng 33:661, doi:10.1007/s10439-005-1433-7; FEBio muscle material https://febiosoftware.github.io/febio-feature-manual/features/solid_material_muscle_material | model curve (sarcomere-based), no tissue population | M | no adjustment: no sex effect on the force-length shape was searched for or found; flagged | read in full (FEBio manual, per tissue report) | T3 |
| `muscle.active.fv_vmax` | max shortening velocity | 10 | L_opt/s | Hill force-velocity; take the curve from MyoSuite to stay consistent with E-ACT | Zajac 1989; Millard et al. 2013, doi:10.1115/1.4023390 | model constant | M | no adjustment: use MyoSuite's curve | *unverified* | T9 |
| `muscle.active.fv_ecc_plateau` | eccentric force plateau | 1.4–1.8 | 1 (of F_iso) | Hill force-velocity (MyoSuite curve) | Zajac 1989; Millard et al. 2013 | model constant | M | no adjustment: use MyoSuite's curve | *unverified* | T9 |
| `muscle.active.stiffening_direction` | how stiffness rises with activation | - | - | form: stiffening only along fibres, produced by the active fibre stress itself; no extra isotropic g_max term; cross-fibre stays at G2 | Gennisson et al. 2010, Ultrasound Med Biol 36:789, doi:10.1016/j.ultrasmedbio.2010.02.013 | in vivo brachialis, healthy volunteers (n, sex not in abstract): along-fibre x20 from 0 to 4 kg, cross-fibre constant | C | no adjustment: no sex data on activation stiffening found; flagged | read in full (per tissue report) | T4 |
| `muscle.active.mu_ratio_20pctMVC` | shear stiffness ratio, rest to 20 % MVC (check) | 2.58 (2.11–2.58) | 1 | check of the active fibre term | Bensamoun et al. 2008 (MRE, vastus medialis; 3.69->9.52 kPa 1D, 4.36->9.22 kPa 2D) | in vivo MRE; n, sex, age not read | C | no adjustment: no sex data found; flagged | abstract | T4 |
| `muscle.active.mu_torque_linearity` | linearity of shear modulus vs torque (check, R^2) | 0.982 | 1 | check: indentation/SWE stiffness must rise linearly with force | Bouillard, Nordez & Hug 2011, PLoS ONE 6:e29261, doi:10.1371/journal.pone.0029261 | 11 subjects, in vivo SSI, hand muscles (sex, age not in abstract) | C | no adjustment | abstract | T4 |
| `muscle.active.thickness_gain` | muscle thickness gain rest to contraction (check) | 0.17–0.2 | 1 | check for T5 bulging | vastus medialis at 90 deg knee, https://pmc.ncbi.nlm.nih.gov/articles/PMC3872419 ; saturation above ~30 % MVC: Hodges et al. 2003, doi:10.1002/mus.10375 | in vivo ultrasound; population and contraction level not read | C | no adjustment | abstract | T5 |
| `muscle.active.transverse_load_loss` | force loss under transverse load 1.1-3.2 N/cm^2 (optional) | 0.13 (0.13–0.15) | 1 | optional second-order term | Siebert et al. 2016, J Biomech (PubMed 26976226); Siebert 2014 | rat gastrocnemius, in situ / isolated | D | no adjustment: animal only | abstract | T5, T7 |
| `muscle.visc.eta_along` | shear viscosity along fibres (relaxed) | 0.6–1.6 | Pa*s | Voigt shear viscosity (high-frequency) | Gennisson et al. 2010, Ultrasound Med Biol 36:789, doi:10.1016/j.ultrasmedbio.2010.02.013 | in vivo brachialis (n, sex not in abstract) | C | no adjustment: no sex data; x10 when loaded | read in full (per tissue report) | T8 |
| `muscle.visc.eta_cross` | shear viscosity across fibres (relaxed) | 0.5–1.5 | Pa*s | Voigt shear viscosity | Gennisson et al. 2010, Ultrasound Med Biol 36:789, doi:10.1016/j.ultrasmedbio.2010.02.013 | as above | C | no adjustment: no sex data; x3 when loaded | read in full (per tissue report) | T8 |
| `muscle.visc.prony_tau` | Prony time constants | [0.6, 6, 30, 60, 300] | s | NLV-SYM / Prony series | Van Loocke et al. 2009, J Biomech 42:1038, doi:10.1016/j.jbiomech.2009.02.022 (Table 2) | porcine muscle, < 2 h post mortem, unconfined compression, ex vivo | D | no adjustment: no human in vivo relaxation series exists; for animation drop 60/300 s; flagged | read in full (per tissue report) | T8 |
| `muscle.visc.prony_g_fibre` | Prony weights, fibre direction | [0.374, 0.285, 0.077, 0.078, 0.101] | 1 | Prony series | Van Loocke et al. 2009, J Biomech 42:1038, doi:10.1016/j.jbiomech.2009.02.022 (Table 2) | porcine, ex vivo | D | no adjustment; flagged | read in full (per tissue report) | T8 |
| `muscle.visc.prony_g_cross` | Prony weights, cross-fibre | [0.097, 0.532, 0.072, 0, 0.086] | 1 | Prony series | Van Loocke et al. 2009, J Biomech 42:1038, doi:10.1016/j.jbiomech.2009.02.022 (Table 2) | porcine, ex vivo | D | no adjustment; flagged | read in full (per tissue report) | T8 |
| `muscle.visc.short_long_ratio` | instantaneous / long-term transverse modulus | 7–13.9 | 1 | QLV check (5.7/0.81 preconditioned; 8.5/0.61 not) | Palevski et al. 2006, J Biomech Eng 128:782 | porcine gluteus, fresh, ex vivo | D | no adjustment; flagged | abstract | T8 |
| `muscle.dyn.soft_tissue_freq` | natural frequency of the relaxed leg soft-tissue package (check) | 15 (3–55) | Hz | check for T8 (whole package: muscle + fat + skin) | Nigg-group review, Sports Med Open 2022, doi:10.1186/s40798-022-00524-w; range: Schmitt & Gunther 2011, doi:10.1007/s00419-010-0458-z | in vivo adults (mixed; details not read) | C | no adjustment: no sex data; the 0.98 m figure rings higher by size scaling (state it, do not retune) | abstract | T8 |
| `muscle.dyn.damping_ratio` | damping ratio of the thigh soft-tissue package | **FIT**  | 1 | Rayleigh / Prony damping in E-TIS | Wakeling & Nigg 2001 (values not read) | in vivo, not read | N | no measured value | *unverified* | T8 |

### Tendon

| id | parameter | value | unit | model | source | population | tier | adjustment for the profile | status | tests |
|---|---|---|---|---|---|---|---|---|---|---|
| `tendon.E` | Young's modulus, free tendon | 6.1e8 (5.3e8–6.9e8) | Pa | stiff tets or 1D springs, k = E*A/L from her geometry | Onambele, Burgess & Pearson 2007, J Orthop Res 25:1635, doi:10.1002/jor.20404 | 10 women 21.3 +/- 0.9 y, in vivo ultrasound, patellar tendon, isometric ramps (tibial motion corrected) | A | no adjustment: direct young-female value (men in the same study 0.68 GPa; Hansen 2006 mixed/unstated 1.09 GPa) | abstract | T3, T6 |
| `tendon.patellar.strain_mvc` | patellar tendon strain at MVC (check) | 0.106 (0.096–0.116) | 1 | check | Onambele, Burgess & Pearson 2007, J Orthop Res 25:1635, doi:10.1002/jor.20404 | 10 women 21 y, in vivo | A | no adjustment: direct | abstract | T6 |
| `tendon.patellar.stiffness` | patellar tendon stiffness (check only; geometry-dependent) | 1.053e6 (9.45e5–1.161e6) | N/m | check | Onambele, Burgess & Pearson 2007, J Orthop Res 25:1635, doi:10.1002/jor.20404 | 10 women 21 y, in vivo | A | no adjustment: direct; the elf's value comes from E*A/L of her own tendon | abstract | T6 |
| `tendon.density` | density | - | kg/m^3 | inertia | no source read | - | N | gap: no value read; tendon mass is small next to muscle | - | T8 |

### Aponeurosis

| id | parameter | value | unit | model | source | population | tier | adjustment for the profile | status | tests |
|---|---|---|---|---|---|---|---|---|---|---|
| `aponeurosis.strain_long` | longitudinal strain at high force (check) | 0.014 (0.01–0.018) | 1 | anisotropic shell | Magnusson et al. 2003, Acta Physiol Scand, doi:10.1046/j.1365-201x.2003.01048.x | n = 5, in vivo ultrasound, gastrocnemius distal aponeurosis (sex not read) | C | no adjustment: no sex data for aponeurosis found; flagged | abstract | T3, T5 |
| `aponeurosis.transverse_ratio` | transverse / longitudinal strain during active force | 4 | 1 | anisotropic shell | Azizi & Roberts 2009, https://pmc.ncbi.nlm.nih.gov/articles/PMC2754367 | turkey, in vivo sonomicrometry | D | no adjustment: animal only; flagged | abstract | T5 |
| `aponeurosis.E_long` | longitudinal modulus | **FIT**  | Pa | anisotropic shell | none measured | - | N | no measured value | - | T3, T5 |

### Fat (subcutaneous adipose tissue)

| id | parameter | value | unit | model | source | population | tier | adjustment for the profile | status | tests |
|---|---|---|---|---|---|---|---|---|---|---|
| `fat.mu` | Ogden shear modulus mu1 (small strain) | 1167 (330–1600) | Pa | 1st-order Ogden, nearly incompressible | Al-Dirini et al. 2016, Ann Biomed Eng 44:2805, doi:10.1007/s10439-016-1560-3 (constants read from Table 1 of Wang et al. 2021, doi:10.1155/2021/4496416); agrees with Then et al. 2007, Technol Health Care 15:385 (G_inf 1182 Pa) | one seated person, in vivo buttock MRI + inverse FE (sex not verified; the same group's 2017 cohort is 10 men); Then: one subject, gluteal, in vivo | C | no adjustment: no adult sex, age or BMI effect on subcutaneous-fat stiffness was found (only a paediatric SWE study, and Alkhouli 2013 reports no strong BMI dependence, unverified); kept and flagged | abstract / secondary (Wang 2021 read in full; attribution to Al-Dirini *unverified*) | T4, T5, T7 |
| `fat.alpha` | Ogden exponent alpha1 | 16.2 (16.2–20) | 1 | 1st-order Ogden | Al-Dirini et al. 2016, Ann Biomed Eng 44:2805, doi:10.1007/s10439-016-1560-3 (constants read from Table 1 of Wang et al. 2021, doi:10.1155/2021/4496416); Comley & Fleck 2012 (porcine, alpha 20) | as fat.mu | C | no adjustment: no adult sex, age or BMI effect on subcutaneous-fat stiffness was found (only a paediatric SWE study, and Alkhouli 2013 reports no strong BMI dependence, unverified); kept and flagged | abstract / secondary | T5, T7 |
| `fat.bulk_physical` | physical bulk modulus (derived rho*c^2) | 1.91e9–2.08e9 | Pa | volumetric term; too stiff for VBD | acoustic tables c = 1450-1480 m/s, rho = 911-950 kg/m^3 (https://pmc.ncbi.nlm.nih.gov/articles/PMC4521448/table/Tab3) | tissue acoustics, population not read | N | no adjustment: derived from cited c and rho | *unverified* | T2 |
| `fat.solver.nu` | Poisson's ratio used by the solver | 0.49–0.499 | 1 | volumetric penalty | heel fat pad indentation fits, arXiv:2312.04108; Lanzl 2025 nu = 0.499 | heel pad in vivo / ex vivo | C | no adjustment: numerical choice inside the cited band; check fat thickness loss against Mogale 2023 | abstract | T5, T7 |
| `fat.density` | density | 950 (911–950) | kg/m^3 | inertia | Woodard & White 1986, Br J Radiol 59:1209 (values as tabulated against ICRU 44 in PMC3396587, Table 10); acoustic tables 911-950; Al-Dirini model 920 | reference adult tissue composition, sex not stated | C | no adjustment: no sex effect on adipose density found | abstract | T8, T9 |
| `fat.E_tension_initial` | tensile modulus, initial (check) | 1600 (800–2400) | Pa | check of the Ogden fit | Alkhouli et al. 2013, Am J Physiol Endocrinol Metab, doi:10.1152/ajpendo.00111.2013 | 44 surgical patients, human SAT, ex vivo (sex split not read) | D | no adjustment: no adult sex, age or BMI effect on subcutaneous-fat stiffness was found (only a paediatric SWE study, and Alkhouli 2013 reports no strong BMI dependence, unverified); kept and flagged | abstract | T4 |
| `fat.E_tension_30pct` | tensile modulus at <= 30 % strain (check) | 11700 (5300–18100) | Pa | check of the Ogden stiffening | Alkhouli et al. 2013, Am J Physiol Endocrinol Metab, doi:10.1152/ajpendo.00111.2013 | as above | D | no adjustment: no adult sex, age or BMI effect on subcutaneous-fat stiffness was found (only a paediatric SWE study, and Alkhouli 2013 reports no strong BMI dependence, unverified); kept and flagged | abstract | T4 |
| `fat.fibre_k1` | septa fibre stiffness k1 (optional anisotropy) | 429 | Pa | neo-Hooke + 3 circular fibre families | Lanzl et al. 2025, JMBBM 169:107042, doi:10.1016/j.jmbbm.2025.107042, fit to Sommer et al. 2013 | human abdominal fat, ex vivo | D | no adjustment: no adult sex, age or BMI effect on subcutaneous-fat stiffness was found (only a paediatric SWE study, and Alkhouli 2013 reports no strong BMI dependence, unverified); kept and flagged | read in full (per tissue report) | T7 |
| `fat.fibre_k2` | septa fibre exponent k2 (optional anisotropy) | 11.6 | 1 | neo-Hooke + 3 circular fibre families | Lanzl et al. 2025 (as above) | human abdominal fat, ex vivo | D | no adjustment: no adult sex, age or BMI effect on subcutaneous-fat stiffness was found (only a paediatric SWE study, and Alkhouli 2013 reports no strong BMI dependence, unverified); kept and flagged | read in full (per tissue report) | T7 |
| `fat.rate_limit` | strain rate below which fat is rate-independent | 10 | 1/s | validity of the quasi-static solve | Comley & Fleck 2012, Int J Impact Eng 46:1, doi:10.1016/j.ijimpeng.2011.12.009 | porcine, ex vivo | D | no adjustment; flagged | *unverified* | T7, T8 |
| `fat.relaxation` | relaxation (two-exponential, constants) | **FIT**  | s | 2-term Prony series | Alkhouli et al. 2013, Am J Physiol Endocrinol Metab, doi:10.1152/ajpendo.00111.2013 (two exponentials, constants not retrieved); Geerligs 2008 power law (porcine) | ex vivo | N | no measured constants available to us | - | T8 |
| `fat.superficial_deep_ratio` | stiffness ratio superficial / deep SAT | - | 1 | two fat layers split by the membranous layer | strain elastography, Australas J Ultrasound Med, doi:10.1002/ajum.12100 (qualitative only) | 210 pregnant women, in vivo | N | gap: only 'superficial stiffer' is known; keep one fat material | *unverified* | T7 |
| `fat.damping_ratio` | fat-only damping | **FIT**  | 1 | E-TIS damping | none found | - | N | gap | - | T8 |

### Skin

| id | parameter | value | unit | model | source | population | tier | adjustment for the profile | status | tests |
|---|---|---|---|---|---|---|---|---|---|---|
| `skin.matrix_mu` | GOH matrix shear modulus | **FIT** 2.014e5 (25000–2.6e5) | Pa | GOH membrane shell (neo-Hooke matrix + 2 dispersed fibre families) | Ni Annaidh et al. 2012, Ann Biomed Eng 40:1666, doi:10.1007/s10439-012-0542-3; JMBBM doi:10.1016/j.jmbbm.2011.08.016; in vivo suction range 25-260 kPa (Kalra 2016, unverified) | 7 donors aged 81-97, back skin, uniaxial, ex vivo | D | not applied: Agache 1980 (138 people 3-89 y, in vivo torsion) found Young's modulus about doubles above age 30, so young skin is softer, but the factor is from another method and its numbers are unverified; treat 0.20 MPa as an upper bound | read in full (per tissue report) | T6 |
| `skin.k1` | GOH fibre stiffness k1 (upper bound) | **FIT** 2.436e8 | Pa | GOH | Ni Annaidh et al. 2012, Ann Biomed Eng 40:1666, doi:10.1007/s10439-012-0542-3; JMBBM doi:10.1016/j.jmbbm.2011.08.016 | 7 donors 81-97 y, ex vivo | D | not applied (see skin.matrix_mu); upper bound | read in full (per tissue report) | T6 |
| `skin.k2` | GOH fibre exponent k2 | **FIT** 0.1327 | 1 | GOH | Ni Annaidh et al. 2012, Ann Biomed Eng 40:1666, doi:10.1007/s10439-012-0542-3; JMBBM doi:10.1016/j.jmbbm.2011.08.016 | 7 donors 81-97 y, ex vivo | D | no adjustment | read in full (per tissue report) | T6 |
| `skin.kappa` | GOH fibre dispersion kappa | 0.1404 (0.1376–0.1432) | 1 | GOH | Ni Annaidh et al. 2012, Ann Biomed Eng 40:1666, doi:10.1007/s10439-012-0542-3; JMBBM doi:10.1016/j.jmbbm.2011.08.016 (from histology) | 7 donors 81-97 y, ex vivo | D | no adjustment | read in full (per tissue report) | T6 |
| `skin.fibre_angle` | mean fibre angle to the Langer line | 0.716 | rad | GOH; field direction from the Langer-line map | Ni Annaidh et al. 2012, Ann Biomed Eng 40:1666, doi:10.1007/s10439-012-0542-3; JMBBM doi:10.1016/j.jmbbm.2011.08.016 | 7 donors 81-97 y, ex vivo | D | no adjustment | *unverified* | T6 |
| `skin.prestress` | in vivo in-plane pre-stress (orthotropic) | **FIT** 28000–92000 | Pa | rest shape smaller than the mesh | Flynn, Taberner & Nielsen 2011, Ann Biomed Eng 39:1935, doi:10.1007/s10439-011-0292-7 | volunteers, in vivo forearm and upper arm micro-robot (n, sex, age not in abstract) | C | no adjustment: no sex or age effect found; no thigh value | abstract | T6 |
| `skin.thickness_thigh` | skin thickness, thigh (epidermis + dermis) | 0.001683 (0.00129–0.00207) | m | shell thickness | Wang Y. et al. 2025, Quant Imaging Med Surg 15:5218, doi:10.21037/qims-2024-2637 (Table 4); cf. Derraik et al. 2014, PLoS ONE 9:e86637, doi:10.1371/journal.pone.0086637 | 33 healthy women (all 18-74 y, women's median 45), Chinese, in vivo 22/75 MHz ultrasound, right thigh: dermis 1.5645 +/- 0.3902 mm + epidermis 0.1189 mm | C | no adjustment: Wang 2025 found no thigh age correlation; Derraik 2014 (diabetic adults) gives -6 um/yr dermis, which would add about 0.1 mm at 28 vs 44 y, inside the range; women thinner than men (1.56 vs 1.94 mm dermis) is already in the source | read in full | T1 |
| `skin.thickness_buttock` | skin thickness, buttock | **FIT**  | m | shell thickness | none found | - | N | gap: no measurement found (trunk 3-7 mm from Herlin 2015 does not apply) | - | T1, T7 |
| `skin.thickness_knee` | skin thickness, knee | **FIT**  | m | shell thickness | none found | - | N | gap: no measurement found | - | T1, T6 |
| `skin.bending_D` | bending stiffness (derived D = E t^3 / 12(1-nu^2)) | **FIT** 5.3e-5–0.00053 | N*m | shell bending | derived from skin.thickness_thigh and E = 0.1-1 MPa, nu = 0.5; no measurement exists | - | N | no measured value | derived | T6 |
| `skin.density` | density | 1090 (1090–1100) | kg/m^3 | inertia | Woodard & White 1986, Br J Radiol 59:1209 (values as tabulated against ICRU 44 in PMC3396587, Table 10); ICRU 44 (1989) 1100 | reference adult tissue composition, sex not stated | C | no adjustment: no sex effect found | abstract | T8 |
| `skin.failure_stretch` | failure stretch along / across Langer lines (check) | 1.46 (1.46–1.52) | 1 | limit check; flag elements > 1.45 | Ni Annaidh et al. 2012, Ann Biomed Eng 40:1666, doi:10.1007/s10439-012-0542-3; JMBBM doi:10.1016/j.jmbbm.2011.08.016 | 7 donors 81-97 y, ex vivo | D | no adjustment | read in full (per tissue report) | T6 |
| `skin.visc_tau1` | QLV first time constant | **FIT** 0.19 (0.09–0.29) | s | QLV / Kelvin-Voigt | https://pmc.ncbi.nlm.nih.gov/articles/PMC4372409 | excised thin specimens 211-671 um (species not checked) | D | no adjustment | *unverified* | T8 |
| `skin.friction_dry` | friction, dry skin against counter-materials | 0.5 (0.2–0.5) | 1 | Coulomb contact | Derler & Gerhardt 2012, Tribol Lett 45:1, doi:10.1007/s11249-011-9854-y | literature review, mixed populations and sites | C | no adjustment: Veijgen, Masen & van der Heide 2013, Tribol Lett 49:251, doi:10.1007/s11249-012-0062-1 report sex differences in friction that are mostly explained by hydration; no numeric sex factor | abstract | T7, T9 |
| `skin.friction_moist` | friction, moist / sweaty skin | > 1 | 1 | Coulomb contact | Derler & Gerhardt 2012, Tribol Lett 45:1, doi:10.1007/s11249-011-9854-y | review, mixed | C | no adjustment (as above) | abstract | T6, T7 |
| `skin.friction_textile` | friction, skin against textile (dynamic to static) | 0.25–0.9 | 1 | Coulomb contact | Tribol Lett 2021, doi:10.1007/s11249-021-01560-5 (chest 0.90/0.79, forearm 0.31/0.25); trousers vs seat: Lamsal 2023, doi:10.1007/s11340-023-01022-0 | in vivo, handheld device (population not read) | C | no adjustment: no thigh/buttock value; no sex factor | abstract | T7, T9 |
| `skin.friction_self` | skin-on-skin friction (back of knee, calf on thigh) | **FIT** 0.5–1 | 1 | Coulomb self-contact | bracket only, from Derler & Gerhardt 2012, Tribol Lett 45:1, doi:10.1007/s11249-011-9854-y | - | N | gap: never measured | - | T6 |

### Deep and superficial fascia

| id | parameter | value | unit | model | source | population | tier | adjustment for the profile | status | tests |
|---|---|---|---|---|---|---|---|---|---|---|
| `fascia.E_long` | tensile modulus along the limb (stiff sheet) | 5.57e7–1.919e8 | Pa | anisotropic membrane shell (2-family HGO) | Bonaldi et al. 2023, Bioengineering 10:226, doi:10.3390/bioengineering10020226 | 4 fresh-frozen donors aged 54-89, fascia lata, ex vivo, 0.5 %/s | D | no adjustment: no sex or age data on fascia moduli found; flagged (older donors) | read in full (per tissue report) | T5, T6 |
| `fascia.E_trans` | tensile modulus across (second sheet) | 4.5e6–1.89e7 | Pa | anisotropic membrane shell | Bonaldi et al. 2023, Bioengineering 10:226, doi:10.3390/bioengineering10020226 | as above (one outlier 119.9 MPa excluded) | D | no adjustment (as above) | read in full (per tissue report) | T5, T6 |
| `fascia.hgo_C1` | HGO constant C1 (per-donor spread) | 1.1e5–6.8e5 | Pa | 2-family HGO, elastic region <= 3-4 % strain | Bonaldi et al. 2023, Bioengineering 10:226, doi:10.3390/bioengineering10020226 Table 3 | 4 donors 54-89 y, ex vivo | D | no adjustment: donor mean and spread are the prior | read in full (per tissue report) | T5, T6 |
| `fascia.hgo_k11` | HGO constant k11 (per-donor spread) | 2.8e6–5.7e6 | Pa | 2-family HGO, elastic region <= 3-4 % strain | Bonaldi et al. 2023, Bioengineering 10:226, doi:10.3390/bioengineering10020226 Table 3 | 4 donors 54-89 y, ex vivo | D | no adjustment: donor mean and spread are the prior | read in full (per tissue report) | T5, T6 |
| `fascia.hgo_k21` | HGO constant k21 (per-donor spread) | 42–173 | 1 | 2-family HGO, elastic region <= 3-4 % strain | Bonaldi et al. 2023, Bioengineering 10:226, doi:10.3390/bioengineering10020226 Table 3 | 4 donors 54-89 y, ex vivo | D | no adjustment: donor mean and spread are the prior | read in full (per tissue report) | T5, T6 |
| `fascia.hgo_k12` | HGO constant k12 (per-donor spread) | 7.7e5–2e6 | Pa | 2-family HGO, elastic region <= 3-4 % strain | Bonaldi et al. 2023, Bioengineering 10:226, doi:10.3390/bioengineering10020226 Table 3 | 4 donors 54-89 y, ex vivo | D | no adjustment: donor mean and spread are the prior | read in full (per tissue report) | T5, T6 |
| `fascia.hgo_k22` | HGO constant k22 (per-donor spread) | 115–579 | 1 | 2-family HGO, elastic region <= 3-4 % strain | Bonaldi et al. 2023, Bioengineering 10:226, doi:10.3390/bioengineering10020226 Table 3 | 4 donors 54-89 y, ex vivo | D | no adjustment: donor mean and spread are the prior | read in full (per tissue report) | T5, T6 |
| `fascia.hgo_kappa` | HGO constant kappa (per-donor spread) | 0.0004–0.19 | 1 | 2-family HGO, elastic region <= 3-4 % strain | Bonaldi et al. 2023, Bioengineering 10:226, doi:10.3390/bioengineering10020226 Table 3 | 4 donors 54-89 y, ex vivo | D | no adjustment: donor mean and spread are the prior | read in full (per tissue report) | T5, T6 |
| `fascia.sheet_angle` | angle between the two collagen sheets | 1.309–1.396 | rad | fibre family 2 direction | Bonaldi et al. 2023, Bioengineering 10:226, doi:10.3390/bioengineering10020226 | 4 donors, ex vivo | D | no adjustment | read in full (per tissue report) | T1 |
| `fascia.thickness_lata` | fascia lata thickness (histology, without epimysium) | 0.000944 (0.000788–0.0011) | m | shell thickness | histology cited in Bonaldi et al. 2023, Bioengineering 10:226, doi:10.3390/bioengineering10020226; in vivo US 1.88 mm incl. epimysium: Stecco et al. 2011, doi:10.1007/s00276-011-0876-9 | cadaver histology (population not read); US: 22 volunteers | D | no adjustment: no sex comparison found; a female-only study reports thicker lower-limb fascia in young than old women (numbers not read) | abstract | T1 |
| `fascia.thickness_crural` | crural fascia thickness (US) | 0.00078–0.00121 | m | shell thickness | Pirri et al. 2021, Diagnostics 11:177, doi:10.3390/diagnostics11020177 (Table 2) | 15 healthy adults, in vivo US (sex split not read) | C | no adjustment: no sex data | read in full (per tissue report) | T1 |
| `fascia.thickness_epimysium` | epimysial fascia thickness (tibialis anterior) | 0.00035 (0.00027–0.00043) | m | muscle surface layer | Pirri et al. 2021, Diagnostics 11:177, doi:10.3390/diagnostics11020177 (Table 2) | 15 healthy adults, in vivo US | C | no adjustment | read in full (per tissue report) | T1 |
| `fascia.thickness_superficial` | superficial (membranous) fascia thickness, limbs | 0.0006–0.001 | m | thin stiff shell tied to SAT | 2025 fascia review (Pirri et al., snippet) | review of US studies | C | no adjustment | *unverified* | T1 |
| `fascia.visc_residual_60s` | residual stress fraction after 60 s relaxation | 0.56 | 1 | 2-term Prony | Bonaldi et al. 2023, Bioengineering 10:226, doi:10.3390/bioengineering10020226 Table 2 | 4 donors, ex vivo | D | no adjustment | read in full (per tissue report) | T5, T8 |
| `fascia.visc_residual_300s` | residual stress fraction after 300 s relaxation | 0.43 | 1 | 2-term Prony | Bonaldi et al. 2023, Bioengineering 10:226, doi:10.3390/bioengineering10020226 Table 2 | 4 donors, ex vivo | D | no adjustment | read in full (per tissue report) | T5 |
| `fascia.prestress` | in vivo pre-tension | **FIT**  | Pa | rest shape | none found | - | N | gap: no measurement; replaces the 'shrinkage' knob | - | T5 |
| `fascia.density` | density | - | kg/m^3 | inertia | no source read | - | N | gap | - | T8 |

### Interfaces

| id | parameter | value | unit | model | source | population | tier | adjustment for the profile | status | tests |
|---|---|---|---|---|---|---|---|---|---|---|
| `iface.muscle_muscle.normal` | muscle <-> muscle, normal direction | - | - | form: bilateral (no separation) sliding contact (FEBio sliding-elastic tension=true; Teran 2005 fascia links) | Teran et al. 2005, doi:10.1109/TVCG.2005.42; Maas & Sandercock 2010, PMC2853902 | simulation + anatomy | M | no adjustment | read in full (per tissue report) | T5 |
| `iface.muscle_muscle.k_t` | muscle <-> muscle tangential shear stiffness per area | **FIT** 3e6–1e8 | Pa/m | cohesive tangential spring (areolar tissue, neurovascular tracts) | seed range derived: loose-tissue modulus 3-5 kPa / thickness 0.05-1 mm (Mount Sinai abstract; Stecco) | - | N | no measured value | derived | T5, T6 |
| `iface.muscle_muscle.slide_check` | relative aponeurosis slide MG vs soleus (check) | -0.0017–0.0037 | m | check | Bojsen-Moller et al. 2004 | 8 adults, in vivo ultrasound (sex not read) | C | no adjustment: no female data | abstract | T5 |
| `iface.muscle_fascia.friction` | muscle (epimysium) <-> deep fascia friction | **FIT**  | 1 | sliding, no separation, low friction (HA layer) | none measured; HA layer: Stecco et al. 2011 | - | N | gap | - | T6 |
| `iface.muscle_fascia.ha_film` | hyaluronan film thickness between fascia and muscle | 5e-5 | m | interface layer | Roman et al. 2013, doi:10.7556/jaoa.2013.021 citing Stecco | cadaver | D | no adjustment | *unverified* | T1 |
| `iface.fascia_fat.shear` | deep fascia <-> deep fat, tied through a compliant layer: shear stiffness | **FIT**  | Pa/m | tied (cohesive shear spring) or DAT as its own soft tets | Stecco reappraisal of the superficial fascia; Wilke & Tenberg 2020, J Anat, doi:10.1111/joa.13283 | - | N | no measured value | - | T6 |
| `iface.skin_fat.slip_check` | skin + SAT + superficial fascia slip (check; modelled as embedded / tied) | 0.00072 (0–0.00153) | m | skin membrane embedded in the outer fat tets | Wilke & Tenberg 2020, J Anat, doi:10.1111/joa.13283 | 11 adults, in vivo US cross-correlation (sex not read) | C | no adjustment | abstract | T6 |
| `iface.septa.toe_strain` | retinacula cutis profundus tether: slack (toe) strain | **FIT**  | 1 | tension-only tethers SM -> deep fascia, rest length = local DAT thickness from her geometry | none measured (Li et al. 2013 notes toe region) | - | N | gap | - | T6, T9 |
| `iface.septa.stiffness` | retinacula cutis tether stiffness (per region) | **FIT**  | N/m | tension-only tethers | none measured | - | N | gap | - | T6, T9 |
| `iface.septa.coherence_thigh` | septa orientation coherence, thigh (density prior) | 0.711 | 1 | tether layout prior | Li & Ahn 2011, PLoS ONE 6:e23987, doi:10.1371/journal.pone.0023987 | 25 adults, in vivo 10 MHz US (sex split not read) | C | no adjustment | abstract | T1 |
| `iface.gluteal_fold.anchor` | medial gluteal fold: dermis anchored to the ischium | - | - | form: zero-length hard anchors, density fading laterally | Si et al. 2024, Plast Reconstr Surg 154:241e, doi:10.1097/PRS.0000000000010723 | 20 fresh female buttocks, dissection (ex vivo) | D | no adjustment: female source | abstract | T7 |
| `iface.muscle_bone.footprint_check` | quadriceps tendon footprint on the patella (check) | 0.0005025 (0.000336–0.000611) | m^2 | tied over the full atlas footprint | Knee 2023 cadaver 3D CT/MRI (via BVS mdl-37399632) | cadavers (sex not read) | D | no adjustment: footprint comes from her warped atlas; female size difference not known | abstract | T1 |
| `iface.muscle_bone.ischial_slide` | gluteus maximus slide off the ischial tuberosity, seated (check) | 0.01374 | m | frictionless sliding over the ischial bursa | Zappala et al. 2024, J Biomech 163:111913, doi:10.1016/j.jbiomech.2023.111913 | 10 healthy subjects, semi-recumbent MR-DVC; built on Al-Dirini 2017 scans of 10 men | C | no adjustment: male-only data; no female value found; flagged | abstract | T7 |
| `iface.articular.friction` | bursa / articular cartilage friction | 0.001–0.01 | 1 | treated as frictionless | cartilage tribology (as quoted in the link report) | - | D | no adjustment | *unverified* | T6, T7 |
| `iface.patella.shift_max` | patella shift over flexion (check) | 0.003 | m | patella as a free body on quadriceps and patellar tendon springs, frictionless trochlear contact (Lenhart 2015) | Nha et al. 2008, J Orthop Res, doi:10.1002/jor.20554 | 8 knees, in vivo dual fluoroscopy + MRI, weight-bearing (sex not read) | C | no adjustment: no female tracking data found | abstract | T6 |
| `iface.patella.tilt_max` | patella tilt over flexion (check) | 0.105 | rad | as above | Nha et al. 2008 | as above | C | no adjustment | abstract | T6 |
| `iface.patella.lateral_max` | patella lateral displacement (check) | 0.006 | m | as above | Nha et al. 2008 | as above | C | no adjustment | abstract | T6 |

## 4. Values that must be fitted (no measured source)

Each is fitted once, the fit and its data recorded in this table, never set by hand.

| id | fit to |
|---|---|
| `muscle.passive.fibre_P1` | Maeda 2021 female VL shear modulus at 3 knee angles (rows mu_along*) + Zenodo 7081639 muscle-tendon length split |
| `muscle.passive.fibre_P2` | as muscle.passive.fibre_P1 |
| `muscle.passive.lambda_ofl` | geometry: myoLegs/Rajagopal L_opt vs atlas fibre length; checked by the T3 force-length scene |
| `muscle.dyn.damping_ratio` | Dyna / DFAUST 4D jiggle (research licence; use female subjects only) for frequency and decay |
| `aponeurosis.E_long` | fit to aponeurosis.strain_long (Magnusson 2003) with DL_Track_US / Zenodo 7081639 ultrasound |
| `fat.relaxation` | Alkhouli 2013 full-text relaxation curves if obtained; else Dyna/DFAUST decay (female subjects) |
| `fat.damping_ratio` | Dyna / DFAUST (female subjects); fat and muscle are mixed in it |
| `skin.matrix_mu` | FEMTO-ST CutiScan suction DIC (one 28-y man, forearm) + Rupani 2025 knee/thigh stretch DIC (15 people) |
| `skin.k1` | as skin.matrix_mu |
| `skin.k2` | as skin.matrix_mu |
| `skin.prestress` | Rupani 2025 thigh/knee DIC (and retraction data if found) |
| `skin.thickness_buttock` | no dataset found; MULTiS (100 people, CC BY) covers limbs only - needs a new source |
| `skin.thickness_knee` | no dataset found - needs a new source |
| `skin.bending_D` | fold wavelength behind the knee: Rupani 2025 DIC, DFAUST surfaces (female subjects) |
| `skin.visc_tau1` | FEMTO-ST CutiScan suction curves |
| `skin.friction_self` | crease behaviour in T6 against Rupani 2025 and 3DCP/TUCH self-contact meshes; record the fit |
| `fascia.prestress` | no direct dataset; candidate: van Riel 2026 cuff MRI (thigh shape at 0 mmHg) - flagged |
| `iface.muscle_muscle.k_t` | Bojsen-Moller et al. 2004 (doi:10.1152/japplphysiol.00084.2004, n = 8, MG-Sol +3.7 / -1.7 mm) + Huijing 2011 soleus strain; thigh target missing |
| `iface.muscle_fascia.friction` | Wilke & Tenberg 2020, J Anat, doi:10.1111/joa.13283: muscle 5.70 +/- 2.43 mm vs superficial fascia 0.72 +/- 0.81 mm (n = 11, paper values; data not shared); Wang 2023 ITB shear strain |
| `iface.fascia_fat.shear` | Wilke & Tenberg 2020, J Anat, doi:10.1111/joa.13283 coupling ratio (about 1:8) |
| `iface.septa.toe_strain` | Cereatti et al. 2017, J Biomech 62:5, doi:10.1016/j.jbiomech.2017.02.004 (thigh RMSD 7.6-8.4 mm, peak 23.4-28.4 mm, sit-to-stand up to 72.3 mm) + CAMS-Knee |
| `iface.septa.stiffness` | as iface.septa.toe_strain |

## 5. Where the male atlas and male myoLegs conflict with the profile

The atlas (BodyParts3D) is one adult male; myoLegs inherits Rajagopal et al. 2016 (IEEE TBME 63:2068, PMC5507211), whose
musculotendon parameters combine 21 cadaver lower limbs with MRI muscle volumes of 24 young adults (Handsfield et al. 2014,
J Biomech 47:631; the sex split of both sets was not confirmed here) and whose bone geometry is an averaged adult male of about
75 kg / 170 cm (search snippet, *unverified*). Every result must state this until the corrections below are made.

| conflict | what is known for women | correction found | applied? |
|---|---|---|---|
| **relative muscle volumes** | Women carry relatively more mass in gluteus maximus and medius, tibialis anterior and posterior and EDL; men relatively more in rectus femoris and semimembranosus. Uniform scaling of all F_max does not capture this; no open model reflects the female distribution | Maarleveld et al. 2024, *What the %PCSA?*, arXiv:2411.00071 (review of 57 studies) **[abstract]**; per-muscle factors not extracted | no: needs the per-muscle table from the paper |
| **female musculoskeletal model** | YONI: lower-limb OpenSim model from MRI of 9 young women aged 28.1 ± 2.6 y (matches the profile); scaled YONI fit a female subject with RMS 9.7 mm vs 24.7 mm for scaled Rajagopal | Gouka 2025, TU Delft MSc thesis, https://repository.tudelft.nl/record/uuid:cdeecb96-a27d-4493-8c10-f669e6cca2d1 **[abstract]**; licence and data access *unverified* | no: candidate source for female bones, attachments and volumes (ask before any download) |
| **muscle force** | Specific tension does not differ by sex (women 57.3 ± 13, men 55 ± 11 N/cm²), so female forces differ through PCSA only | O'Brien et al. 2010 **[abstract]** | yes: `muscle.active.sigma_max` kept sex-neutral; F_max must come from her own volumes |
| **passive muscle stiffness** | Women lower than men at long muscle lengths (VL at full knee flexion 43.0 vs 56.7 kPa, ages 30–39) and at 90° (RF, VM, VL; Akagi 2019) | Maeda 2021; Akagi 2019 **[read in full]** | yes: female values used directly |
| **tendon** | Patellar tendon in young women: lower stiffness (1053 vs 1652 N/mm), higher strain (10.6 vs 9.0 %), modulus 0.61 vs 0.68 GPa | Onambele et al. 2007 **[abstract]** | yes: female E used; myoLegs tendon slack lengths and stiffness are male-derived |
| **fat distribution** | Thigh subcutis in women about 1.8× men's (17.7 vs 9.8 mm, adults with diabetes, mean BMI 28); the atlas has no subject fat | Derraik et al. 2014 **[read in full]** | by design: fat depth comes from her own surface (`skin_fat.npz`), not the atlas |
| **skin** | Thigh dermis thinner in women (1.56 vs 1.94 mm) | Wang et al. 2025 **[read in full]** | yes: `skin.thickness_thigh` is the female value |
| **pelvis shape** | Female pelvis wider with a wider aperture; bispinous width greater, biacetabular width not significantly different (Kurki 2011, via review); greater trochanter more medial and abductor moment arm shorter in women; hip-joint-centre adjustment changed hip moments significantly in sex-specific Rajagopal variants | Frontiers Bioeng Biotechnol 2022, doi:10.3389/fbioe.2022.863194 **[abstract]**; review snippets *unverified* | no: the warp fits her outer shape but the atlas pelvis interior stays male; flag in T7/T9 |
| **reference data for tests** | Zappalà 2024 / Al-Dirini 2017 buttock MRI are 10 men; Sejersted 1984 compartment pressure 7 men; men's seated peak pressure about 10 kPa higher than women's (TU Delft) | see rows | no: T7 magnitudes are male; use female subsets where they exist (Mogale 2023 has 13 women of 26; DFAUST female subjects) |

## 6. Biggest gaps

1. **Interfaces have no measured stiffness or friction at all**: muscle↔muscle shear, muscle↔fascia friction, fascia↔fat shear, septa
   tethers, skin-on-skin friction. All are **FIT**, and the thigh-specific sliding target (rectus vs vasti vs hamstrings) is missing.
2. **Skin stiffness for a young woman**: the only GOH fit is from donors aged 81–97; in vivo pre-stress is from the arm; no thigh,
   buttock or knee value. Buttock and knee skin thickness: no source found.
3. **Fat**: the in vivo constants come from one or two subjects of unconfirmed sex; no adult sex or BMI effect on stiffness;
   no relaxation constants; no fat-only damping; superficial/deep ratio unknown.
4. **Fascia**: all moduli from 4 elderly donors ex vivo; no pre-tension; no sex data.
5. **Damping**: muscle relaxation series is porcine ex vivo; thigh damping ratio unmeasured (fit to Dyna/DFAUST).
6. **Model constants** P1, P2, λ_ofl are whole-muscle fits, and λ_ofl depends on the male atlas rest lengths.
7. **Densities** of tendon and fascia: no source read.

## 7. What this pass verified or added

- Persad 2024 recommendation of 26.8 N/cm² is now confirmed at abstract level (was *unverified*).
- New tier-A sources: Maeda et al. 2021 (female VL shear modulus by knee angle, full text) and Onambele et al. 2007 (young-female
  patellar tendon, abstract).
- New sex evidence: O'Brien 2010 (specific tension, no difference), Akagi 2019 (young women less stiff at 90°), Wang 2025 and
  Derraik 2014 (thigh skin thickness by sex), Maarleveld 2024 (female muscle-mass distribution), Gouka 2025 (female model).
- Densities now cited (Woodard & White 1986; ICRU 44), replacing the uncited 1100 kg/m³ for skin.
- Eby 2015 (Clin Biomech 30:22, 133 adults 21–94) checked: sex and age mattered only above 60 y and at full elbow extension, so no
  adjustment is needed for a 28-year-old [abstract].
- Not resolved: Al-Dirini 2016 subject sex and original table, Hansen 2006 and Gennisson 2010 populations (abstracts do not give
  them), Agache 1980 numbers, Bonaldi donor sexes.
