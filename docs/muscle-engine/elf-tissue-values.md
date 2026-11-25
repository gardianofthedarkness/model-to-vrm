# Tissue material values for the elf

Every physical value in `e-mat.json` (version 0.1, 2026-11-24). Units as given; a **k** or **M** after a number means ×10³ or ×10⁶ of that unit, so "800 Pa" is 800 pascals and "201 k Pa" is 201,400 pascals. "—" means not set; a range alone means only a range was found. **Used** marks values the current teacher or 1-D model reads; the rest are collected but not used yet. Profile: {"sex": "female", "age_years": 28, "health": "healthy", "build": "normal; BMI-equivalent computed from her own mesh volume and height (plan.md 2.1)", "source_priority": ["A: women ~20-35, in vivo", "B.

## Muscle

| Parameter | Value | Unit | Range | Used | Source |
|---|---|---|---|---|---|
| `muscle.passive.mu_along` shear modulus along fibres, relaxed, short length (G1 target | 4160 | Pa | 3190, 5130 |  | Maeda et al. 2021, Int J Environ Res Public Health 18:8947 |
| `muscle.passive.mu_along_knee90` shear modulus along fibres, moderately stretched (knee 90 de | 8630 | Pa | 6810, 10.4 k |  | Maeda et al. 2021, Int J Environ Res Public Health 18:8947 |
| `muscle.passive.mu_along_fullflex` shear modulus along fibres, strongly stretched (knee full fl | 43 k | Pa | 33.6 k, 52.5 k |  | Maeda et al. 2021, Int J Environ Res Public Health 18:8947 |
| `muscle.passive.mu_cross` shear modulus across fibres, relaxed (G2) | 800 | Pa | 50, 1030 | teacher | Gennisson et al. 2010, Ultrasound Med Biol 36:789 |
| `muscle.passive.fibre_P1` passive fibre stress coefficient P1 | 0.05 | 1 (fraction of sigma_max; normalisation unverified) | — |  | Blemker, Pinsky & Delp 2005, J Biomech 38:657; Blemker & Del |
| `muscle.passive.fibre_P2` passive fibre exponent P2 | 6.6 | 1 | — |  | Blemker, Pinsky & Delp 2005, J Biomech 38:657; Blemker & Del |
| `muscle.passive.lambda_ofl` fibre stretch at optimal length relative to mesh rest | 1.4 | 1 | 1.07, 1.4 |  | Blemker, Pinsky & Delp 2005, J Biomech 38:657; Blemker & Del |
| `muscle.passive.bulk_physical` physical bulk modulus | 2e+03 M | Pa | — |  | muscle is about 80 % water |
| `muscle.solver.K_over_G` bulk-to-shear ratio used by the solver | 100 | 1 | 100, 2000 | teacher | FEBio muscle example uses K/G = 2000 |
| `muscle.check.volume_change` volume change during contraction (check) | 0.001 | 1 (upper bound) | — |  | Baskin & Paolini 1967 |
| `muscle.density` density | 1050 | kg/m^3 | 1040, 1060 | teacher | Woodard & White 1986, Br J Radiol 59:1209 |
| `muscle.active.sigma_max` peak isometric fibre stress (specific tension) | 268 k | Pa | 171 k, 573 k | 1-D model, act1 | Persad et al. 2024, J Appl Physiol 137:945 |
| `muscle.active.fl_halfwidth` active force-length: half-width of the central parabola | 0.5 | 1 (of lambda/lambda_ofl) | — | act1 | Blemker, Pinsky & Delp 2005, J Biomech 38:657; Blemker & Del |
| `muscle.active.fv_vmax` max shortening velocity | 10 | L_opt/s | — |  | Zajac 1989; Millard et al. 2013 |
| `muscle.active.fv_ecc_plateau` eccentric force plateau | — | 1 (of F_iso) | 1.4, 1.8 |  | Zajac 1989; Millard et al. 2013 |
| `muscle.active.stiffening_direction` how stiffness rises with activation | — | - | — |  | Gennisson et al. 2010, Ultrasound Med Biol 36:789 |
| `muscle.active.mu_ratio_20pctMVC` shear stiffness ratio, rest to 20 % MVC (check) | 2.58 | 1 | 2.11, 2.58 | teacher (effort stiffness) | Bensamoun et al. 2008 |
| `muscle.active.mu_torque_linearity` linearity of shear modulus vs torque (check, R^2) | 0.982 | 1 | — |  | Bouillard, Nordez & Hug 2011, PLoS ONE 6:e29261 |
| `muscle.active.thickness_gain` muscle thickness gain rest to contraction (check) | — | 1 | 0.17, 0.2 |  | vastus medialis at 90 deg knee, https://pmc.ncbi.nlm.nih.gov |
| `muscle.active.transverse_load_loss` force loss under transverse load 1.1-3.2 N/cm^2 (optional) | 0.13 | 1 | 0.13, 0.15 |  | Siebert et al. 2016, J Biomech |
| `muscle.visc.eta_along` shear viscosity along fibres (relaxed) | — | Pa*s | 0.6, 1.6 |  | Gennisson et al. 2010, Ultrasound Med Biol 36:789 |
| `muscle.visc.eta_cross` shear viscosity across fibres (relaxed) | — | Pa*s | 0.5, 1.5 |  | Gennisson et al. 2010, Ultrasound Med Biol 36:789 |
| `muscle.visc.prony_tau` Prony time constants | 0.6, 6, 30, 60, 300 | s | — |  | Van Loocke et al. 2009, J Biomech 42:1038 |
| `muscle.visc.prony_g_fibre` Prony weights, fibre direction | 0.374, 0.285, 0.077, 0.078, 0.101 | 1 | — |  | Van Loocke et al. 2009, J Biomech 42:1038 |
| `muscle.visc.prony_g_cross` Prony weights, cross-fibre | 0.097, 0.532, 0.072, 0, 0.086 | 1 | — |  | Van Loocke et al. 2009, J Biomech 42:1038 |
| `muscle.visc.short_long_ratio` instantaneous / long-term transverse modulus | — | 1 | 7, 13.9 |  | Palevski et al. 2006, J Biomech Eng 128:782 |
| `muscle.dyn.soft_tissue_freq` natural frequency of the relaxed leg soft-tissue package (ch | 15 | Hz | 3, 55 |  | Nigg-group review, Sports Med Open 2022 |
| `muscle.dyn.damping_ratio` damping ratio of the thigh soft-tissue package | — | 1 | — |  | Wakeling & Nigg 2001 |

## Tendon and aponeurosis

| Parameter | Value | Unit | Range | Used | Source |
|---|---|---|---|---|---|
| `tendon.E` Young's modulus, free tendon | 610 M | Pa | 530 M, 690 M |  | Onambele, Burgess & Pearson 2007, J Orthop Res 25:1635 |
| `tendon.patellar.strain_mvc` patellar tendon strain at MVC (check) | 0.106 | 1 | 0.096, 0.116 |  | Onambele, Burgess & Pearson 2007, J Orthop Res 25:1635 |
| `tendon.patellar.stiffness` patellar tendon stiffness (check only; geometry-dependent) | 1.05 M | N/m | 945 k, 1.16 M |  | Onambele, Burgess & Pearson 2007, J Orthop Res 25:1635 |
| `aponeurosis.strain_long` longitudinal strain at high force (check) | 0.014 | 1 | 0.01, 0.018 |  | Magnusson et al. 2003, Acta Physiol Scand |
| `aponeurosis.transverse_ratio` transverse / longitudinal strain during active force | 4 | 1 | — |  | Azizi & Roberts 2009, https://pmc.ncbi.nlm.nih.gov/articles/ |
| `aponeurosis.E_long` longitudinal modulus | — | Pa | — |  | none measured |
| `tendon.density` density | — | kg/m^3 | — |  | no source read |

## Fat

| Parameter | Value | Unit | Range | Used | Source |
|---|---|---|---|---|---|
| `fat.mu` Ogden shear modulus mu1 (small strain) | 1167 | Pa | 330, 1600 | teacher | Al-Dirini et al. 2016, Ann Biomed Eng 44:2805 |
| `fat.alpha` Ogden exponent alpha1 | 16.2 | 1 | 16.2, 20 |  | Al-Dirini et al. 2016, Ann Biomed Eng 44:2805 |
| `fat.bulk_physical` physical bulk modulus (derived rho*c^2) | — | Pa | 1.91e+03 M, 2.08e+03 M |  | acoustic tables c = 1450-1480 m/s, rho = 911-950 kg/m^3 |
| `fat.solver.nu` Poisson's ratio used by the solver | — | 1 | 0.49, 0.499 |  | heel fat pad indentation fits, arXiv:2312.04108; Lanzl 2025  |
| `fat.density` density | 950 | kg/m^3 | 911, 950 | teacher | Woodard & White 1986, Br J Radiol 59:1209 |
| `fat.E_tension_initial` tensile modulus, initial (check) | 1600 | Pa | 800, 2400 |  | Alkhouli et al. 2013, Am J Physiol Endocrinol Metab |
| `fat.E_tension_30pct` tensile modulus at <= 30 % strain (check) | 11.7 k | Pa | 5300, 18.1 k |  | Alkhouli et al. 2013, Am J Physiol Endocrinol Metab |
| `fat.fibre_k1` septa fibre stiffness k1 (optional anisotropy) | 429 | Pa | — |  | Lanzl et al. 2025, JMBBM 169:107042 |
| `fat.fibre_k2` septa fibre exponent k2 (optional anisotropy) | 11.6 | 1 | — |  | Lanzl et al. 2025 |
| `fat.rate_limit` strain rate below which fat is rate-independent | 10 | 1/s | — |  | Comley & Fleck 2012, Int J Impact Eng 46:1 |
| `fat.relaxation` relaxation (two-exponential, constants) | — | s | — |  | Alkhouli et al. 2013, Am J Physiol Endocrinol Metab |
| `fat.superficial_deep_ratio` stiffness ratio superficial / deep SAT | — | 1 | — |  | strain elastography, Australas J Ultrasound Med |
| `fat.damping_ratio` fat-only damping | — | 1 | — |  | none found |

## Skin

| Parameter | Value | Unit | Range | Used | Source |
|---|---|---|---|---|---|
| `skin.matrix_mu` GOH matrix shear modulus | 201 k | Pa | 25 k, 260 k | teacher (skin layer) | Ni Annaidh et al. 2012, Ann Biomed Eng 40:1666 |
| `skin.k1` GOH fibre stiffness k1 (upper bound) | 244 M | Pa | — |  | Ni Annaidh et al. 2012, Ann Biomed Eng 40:1666 |
| `skin.k2` GOH fibre exponent k2 | 0.1327 | 1 | — |  | Ni Annaidh et al. 2012, Ann Biomed Eng 40:1666 |
| `skin.kappa` GOH fibre dispersion kappa | 0.1404 | 1 | 0.1376, 0.1432 |  | Ni Annaidh et al. 2012, Ann Biomed Eng 40:1666 |
| `skin.fibre_angle` mean fibre angle to the Langer line | 0.716 | rad | — |  | Ni Annaidh et al. 2012, Ann Biomed Eng 40:1666 |
| `skin.prestress` in vivo in-plane pre-stress (orthotropic) | — | Pa | 28 k, 92 k |  | Flynn, Taberner & Nielsen 2011, Ann Biomed Eng 39:1935 |
| `skin.thickness_thigh` skin thickness, thigh (epidermis + dermis) | 0.001683 | m | 0.00129, 0.00207 | teacher (skin layer) | Wang Y. et al. 2025, Quant Imaging Med Surg 15:5218 |
| `skin.thickness_buttock` skin thickness, buttock | — | m | — |  | none found |
| `skin.thickness_knee` skin thickness, knee | — | m | — |  | none found |
| `skin.bending_D` bending stiffness (derived D = E t^3 / 12(1-nu^2)) | — | N*m | 5.3e-05, 0.00053 |  | derived from skin.thickness_thigh and E = 0.1-1 MPa, nu = 0. |
| `skin.density` density | 1090 | kg/m^3 | 1090, 1100 |  | Woodard & White 1986, Br J Radiol 59:1209 |
| `skin.failure_stretch` failure stretch along / across Langer lines (check) | 1.46 | 1 | 1.46, 1.52 |  | Ni Annaidh et al. 2012, Ann Biomed Eng 40:1666 |
| `skin.visc_tau1` QLV first time constant | 0.19 | s | 0.09, 0.29 |  | https://pmc.ncbi.nlm.nih.gov/articles/PMC4372409 |
| `skin.friction_dry` friction, dry skin against counter-materials | 0.5 | 1 | 0.2, 0.5 |  | Derler & Gerhardt 2012, Tribol Lett 45:1 |
| `skin.friction_moist` friction, moist / sweaty skin | — | 1 | 1, — |  | Derler & Gerhardt 2012, Tribol Lett 45:1 |
| `skin.friction_textile` friction, skin against textile (dynamic to static) | — | 1 | 0.25, 0.9 |  | Tribol Lett 2021 |
| `skin.friction_self` skin-on-skin friction (back of knee, calf on thigh) | — | 1 | 0.5, 1 |  | bracket only, from Derler & Gerhardt 2012, Tribol Lett 45:1 |

## Fascia

| Parameter | Value | Unit | Range | Used | Source |
|---|---|---|---|---|---|
| `fascia.E_long` tensile modulus along the limb (stiff sheet) | — | Pa | 55.7 M, 192 M |  | Bonaldi et al. 2023, Bioengineering 10:226 |
| `fascia.E_trans` tensile modulus across (second sheet) | — | Pa | 4.5 M, 18.9 M |  | Bonaldi et al. 2023, Bioengineering 10:226 |
| `fascia.hgo_C1` HGO constant C1 (per-donor spread) | — | Pa | 110 k, 680 k |  | Bonaldi et al. 2023, Bioengineering 10:226 |
| `fascia.hgo_k11` HGO constant k11 (per-donor spread) | — | Pa | 2.8 M, 5.7 M |  | Bonaldi et al. 2023, Bioengineering 10:226 |
| `fascia.hgo_k21` HGO constant k21 (per-donor spread) | — | 1 | 42, 173 |  | Bonaldi et al. 2023, Bioengineering 10:226 |
| `fascia.hgo_k12` HGO constant k12 (per-donor spread) | — | Pa | 770 k, 2 M |  | Bonaldi et al. 2023, Bioengineering 10:226 |
| `fascia.hgo_k22` HGO constant k22 (per-donor spread) | — | 1 | 115, 579 |  | Bonaldi et al. 2023, Bioengineering 10:226 |
| `fascia.hgo_kappa` HGO constant kappa (per-donor spread) | — | 1 | 0.0004, 0.19 |  | Bonaldi et al. 2023, Bioengineering 10:226 |
| `fascia.sheet_angle` angle between the two collagen sheets | — | rad | 1.309, 1.396 |  | Bonaldi et al. 2023, Bioengineering 10:226 |
| `fascia.thickness_lata` fascia lata thickness (histology, without epimysium) | 0.000944 | m | 0.000788, 0.0011 | teacher (when fascia exists) | histology cited in Bonaldi et al. 2023, Bioengineering 10:22 |
| `fascia.thickness_crural` crural fascia thickness (US) | — | m | 0.00078, 0.00121 |  | Pirri et al. 2021, Diagnostics 11:177 |
| `fascia.thickness_epimysium` epimysial fascia thickness (tibialis anterior) | 0.00035 | m | 0.00027, 0.00043 |  | Pirri et al. 2021, Diagnostics 11:177 |
| `fascia.thickness_superficial` superficial (membranous) fascia thickness, limbs | — | m | 0.0006, 0.001 |  | 2025 fascia review |
| `fascia.visc_residual_60s` residual stress fraction after 60 s relaxation | 0.56 | 1 | — |  | Bonaldi et al. 2023, Bioengineering 10:226 |
| `fascia.visc_residual_300s` residual stress fraction after 300 s relaxation | 0.43 | 1 | — |  | Bonaldi et al. 2023, Bioengineering 10:226 |
| `fascia.prestress` in vivo pre-tension | — | Pa | — |  | none found |
| `fascia.density` density | — | kg/m^3 | — |  | no source read |

## Interfaces between tissues

| Parameter | Value | Unit | Range | Used | Source |
|---|---|---|---|---|---|
| `iface.muscle_muscle.normal` muscle <-> muscle, normal direction | — | - | — |  | Teran et al. 2005 |
| `iface.muscle_muscle.k_t` muscle <-> muscle tangential shear stiffness per area | — | Pa/m | 3 M, 100 M |  | seed range derived: loose-tissue modulus 3-5 kPa / thickness |
| `iface.muscle_muscle.slide_check` relative aponeurosis slide MG vs soleus (check) | — | m | -0.0017, 0.0037 |  | Bojsen-Moller et al. 2004 |
| `iface.muscle_fascia.friction` muscle (epimysium) <-> deep fascia friction | — | 1 | — |  | none measured; HA layer: Stecco et al. 2011 |
| `iface.muscle_fascia.ha_film` hyaluronan film thickness between fascia and muscle | 5e-05 | m | — |  | Roman et al. 2013 |
| `iface.fascia_fat.shear` deep fascia <-> deep fat, tied through a compliant layer: sh | — | Pa/m | — |  | Stecco reappraisal of the superficial fascia; Wilke & Tenber |
| `iface.skin_fat.slip_check` skin + SAT + superficial fascia slip (check; modelled as emb | 0.00072 | m | 0, 0.00153 |  | Wilke & Tenberg 2020, J Anat |
| `iface.septa.toe_strain` retinacula cutis profundus tether: slack (toe) strain | — | 1 | — |  | none measured |
| `iface.septa.stiffness` retinacula cutis tether stiffness (per region) | — | N/m | — |  | none measured |
| `iface.septa.coherence_thigh` septa orientation coherence, thigh (density prior) | 0.711 | 1 | — |  | Li & Ahn 2011, PLoS ONE 6:e23987 |
| `iface.gluteal_fold.anchor` medial gluteal fold: dermis anchored to the ischium | — | - | — |  | Si et al. 2024, Plast Reconstr Surg 154:241e |
| `iface.muscle_bone.footprint_check` quadriceps tendon footprint on the patella (check) | 0.0005025 | m^2 | 0.000336, 0.000611 |  | Knee 2023 cadaver 3D CT/MRI |
| `iface.muscle_bone.ischial_slide` gluteus maximus slide off the ischial tuberosity, seated (ch | 0.01374 | m | — |  | Zappala et al. 2024, J Biomech 163:111913 |
| `iface.articular.friction` bursa / articular cartilage friction | — | 1 | 0.001, 0.01 |  | cartilage tribology |
| `iface.patella.shift_max` patella shift over flexion (check) | 0.003 | m | — |  | Nha et al. 2008, J Orthop Res |
| `iface.patella.tilt_max` patella tilt over flexion (check) | 0.105 | rad | — |  | Nha et al. 2008 |
| `iface.patella.lateral_max` patella lateral displacement (check) | 0.006 | m | — |  | Nha et al. 2008 |
