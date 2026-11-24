# Builds e-mat.json and e-mat.md from one list (single source of truth).
import json, sys, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "muscle-engine")

# source tier: A = women ~20-35 in vivo; B = mixed-sex young adults in vivo; C = any adult in vivo;
# D = ex vivo human / animal; M = model constant (no tissue population); G = from her own geometry; N = no source
P = []
def p(id, tissue, param, value, unit, rng, model, source, population, tier, adjustment, status, tests,
      fit=None, flag=False, note=""):
    P.append(dict(id=id, tissue=tissue, parameter=param, value=value, unit=unit, range=rng, model=model,
                  source=source, population=population, tier=tier, adjustment=adjustment, status=status,
                  tests=tests, fit_required=fit is not None, fit_dataset=fit, flagged=flag, note=note))

# ---------------------------------------------------------------- sources (short keys)
MAEDA = "Maeda et al. 2021, Int J Environ Res Public Health 18:8947, doi:10.3390/ijerph18178947"
GENN = "Gennisson et al. 2010, Ultrasound Med Biol 36:789, doi:10.1016/j.ultrasmedbio.2010.02.013"
PALEV = "Palevski et al. 2006, J Biomech Eng 128:782"
BLEM = "Blemker, Pinsky & Delp 2005, J Biomech 38:657; Blemker & Delp 2005, Ann Biomed Eng 33:661, doi:10.1007/s10439-005-1433-7; FEBio muscle material https://febiosoftware.github.io/febio-feature-manual/features/solid_material_muscle_material"
PERSAD = "Persad et al. 2024, J Appl Physiol 137:945, doi:10.1152/japplphysiol.00296.2024"
OBRIEN = "O'Brien et al. 2010, Exp Physiol 95:202, doi:10.1113/expphysiol.2009.048967"
AKAGI = "Akagi et al. 2019, Sci Rep 9:16718, doi:10.1038/s41598-019-53375-z"
VLOOCKE = "Van Loocke et al. 2009, J Biomech 42:1038, doi:10.1016/j.jbiomech.2009.02.022 (Table 2)"
ONAMB = "Onambele, Burgess & Pearson 2007, J Orthop Res 25:1635, doi:10.1002/jor.20404"
HANSEN = "Hansen et al. 2006, Clin Biomech 21:54, doi:10.1016/j.clinbiomech.2005.07.008"
WW = "Woodard & White 1986, Br J Radiol 59:1209 (values as tabulated against ICRU 44 in PMC3396587, Table 10)"
NIA = "Ni Annaidh et al. 2012, Ann Biomed Eng 40:1666, doi:10.1007/s10439-012-0542-3; JMBBM doi:10.1016/j.jmbbm.2011.08.016"
AGACHE = "Agache et al. 1980, Arch Dermatol Res 269:221, doi:10.1007/BF00406415"
FLYNN = "Flynn, Taberner & Nielsen 2011, Ann Biomed Eng 39:1935, doi:10.1007/s10439-011-0292-7"
WANG25 = "Wang Y. et al. 2025, Quant Imaging Med Surg 15:5218, doi:10.21037/qims-2024-2637 (Table 4)"
DERRAIK = "Derraik et al. 2014, PLoS ONE 9:e86637, doi:10.1371/journal.pone.0086637"
DERLER = "Derler & Gerhardt 2012, Tribol Lett 45:1, doi:10.1007/s11249-011-9854-y"
VEIJ = "Veijgen, Masen & van der Heide 2013, Tribol Lett 49:251, doi:10.1007/s11249-012-0062-1"
BONALDI = "Bonaldi et al. 2023, Bioengineering 10:226, doi:10.3390/bioengineering10020226"
PIRRI = "Pirri et al. 2021, Diagnostics 11:177, doi:10.3390/diagnostics11020177 (Table 2)"
WILKE = "Wilke & Tenberg 2020, J Anat, doi:10.1111/joa.13283"
CEREATTI = "Cereatti et al. 2017, J Biomech 62:5, doi:10.1016/j.jbiomech.2017.02.004"
ALDIRINI = "Al-Dirini et al. 2016, Ann Biomed Eng 44:2805, doi:10.1007/s10439-016-1560-3 (constants read from Table 1 of Wang et al. 2021, doi:10.1155/2021/4496416)"
THEN = "Then et al. 2007, Technol Health Care 15:385"
ALKH = "Alkhouli et al. 2013, Am J Physiol Endocrinol Metab, doi:10.1152/ajpendo.00111.2013"
ZAPP = "Zappala et al. 2024, J Biomech 163:111913, doi:10.1016/j.jbiomech.2023.111913"

NOSEXFAT = ("no adjustment: no adult sex, age or BMI effect on subcutaneous-fat stiffness was found (only a paediatric "
            "SWE study, and Alkhouli 2013 reports no strong BMI dependence, unverified); kept and flagged")

# ================================================================ MUSCLE, passive
p("muscle.passive.mu_along", "muscle", "shear modulus along fibres, relaxed, short length (G1 target)",
  4160, "Pa", [3190, 5130], "Blemker-Delp transversely isotropic (G1); small-strain SWE value, mapping to the large-strain constant is a T4 fit",
  MAEDA, "9 women 30-39 y (34.8 +/- 3.0), BMI 20.3, Japanese, not regularly exercising; in vivo SWE; vastus lateralis, knee full extension",
  "A", "no adjustment: direct female 30-39 value; same study found no sex effect at full extension (Gennisson 2010 brachialis 5.4 kPa agrees)",
  "read in full", ["T4", "T5"])
p("muscle.passive.mu_along_knee90", "muscle", "shear modulus along fibres, moderately stretched (knee 90 deg)",
  8630, "Pa", [6810, 10450], "Blemker-Delp passive fibre curve (check of P1, P2, lambda_ofl)", MAEDA,
  "9 women 30-39 y, BMI 20.3; in vivo SWE; vastus lateralis, knee 90 deg flexion", "A",
  "no adjustment: direct female value. Akagi 2019 (18 men, 23 women, 21-22 y, VL/VM/RF at 90 deg) found men > women (p<0.001) and supports a female-specific value; its kPa values are only in a figure",
  "read in full", ["T3", "T4"])
p("muscle.passive.mu_along_fullflex", "muscle", "shear modulus along fibres, strongly stretched (knee full flexion)",
  43020, "Pa", [33580, 52460], "Blemker-Delp passive fibre curve (check)", MAEDA,
  "9 women 30-39 y, BMI 20.3; in vivo SWE; vastus lateralis, knee full flexion", "A",
  "no adjustment: direct female value; in this study women were lower than men at full flexion (p=0.001; men 30-39: 56.7 kPa)",
  "read in full", ["T3", "T6"])
p("muscle.passive.mu_cross", "muscle", "shear modulus across fibres, relaxed (G2)",
  800, "Pa", [50, 1030], "Blemker-Delp transversely isotropic (G2)",
  GENN + "; " + PALEV,
  "Gennisson: healthy volunteers, in vivo SSI, brachialis (n, sex, age not in abstract); Palevski: porcine gluteus, fresh, ex vivo, preconditioned long-term 0.81 kPa",
  "C", "no adjustment: no sex or age data for cross-fibre shear found; value set at Palevski's long-term figure inside Gennisson's in vivo band; flagged",
  "read in full (Gennisson, per tissue report); abstract (Palevski)", ["T4"], flag=True)
p("muscle.passive.fibre_P1", "muscle", "passive fibre stress coefficient P1", 0.05, "1 (fraction of sigma_max; normalisation unverified)",
  None, "Blemker-Delp passive fibre law f_p = P1(exp(P2(l/l_ofl-1))-1)", BLEM,
  "model constant matched to whole-muscle behaviour, no tissue population", "M",
  "no adjustment: model constant; refit to the female passive SWE angle series", "read in full (FEBio manual, per tissue report)",
  ["T3"], fit="Maeda 2021 female VL shear modulus at 3 knee angles (rows mu_along*) + Zenodo 7081639 muscle-tendon length split", flag=True)
p("muscle.passive.fibre_P2", "muscle", "passive fibre exponent P2", 6.6, "1", None,
  "Blemker-Delp passive fibre law", BLEM, "model constant, no tissue population", "M",
  "no adjustment: model constant; refit as for P1", "read in full (FEBio manual, per tissue report)", ["T3"],
  fit="as muscle.passive.fibre_P1", flag=True)
p("muscle.passive.lambda_ofl", "muscle", "fibre stretch at optimal length relative to mesh rest", 1.4, "1", [1.07, 1.4],
  "Blemker-Delp (1.4 in papers; 1.07 in the FEBio example)", BLEM, "model constant; really a property of the atlas rest pose",
  "M", "no adjustment: must be computed per muscle from myoLegs optimal fibre length against the warped atlas fibre length (both male, see conflicts)",
  "abstract (PMC2857665 confirms 1.4)", ["T3"],
  fit="geometry: myoLegs/Rajagopal L_opt vs atlas fibre length; checked by the T3 force-length scene", flag=True)
p("muscle.passive.bulk_physical", "muscle", "physical bulk modulus", 2.0e9, "Pa", None,
  "volumetric term U(J); physical value, too stiff for VBD", "muscle is about 80 % water (Van Loocke 2009); K of water used as the estimate",
  "none (water)", "N", "no adjustment: physical estimate, not a measurement of muscle", "unverified", ["T2"], flag=True,
  note="the solver uses muscle.solver.K_over_G instead")
p("muscle.solver.K_over_G", "muscle", "bulk-to-shear ratio used by the solver", 100, "1", [100, 2000],
  "volumetric penalty; stability choice, not a material fact", "FEBio muscle example uses K/G = 2000 (k = 1 MPa, G = 500 Pa)",
  "solver setting", "M", "no adjustment: numerical setting, judged only by the T2 volume check", "read in full (FEBio manual, per tissue report)",
  ["T2"], note="pass: volume change < 1 % (muscle.check.volume_change)")
p("muscle.check.volume_change", "muscle", "volume change during contraction (check)", 0.001, "1 (upper bound)", None,
  "check target", "Baskin & Paolini 1967 (as quoted in the muscle report)", "not read", "N",
  "no adjustment", "unverified", ["T2"], flag=True)
p("muscle.density", "muscle", "density", 1050, "kg/m^3", [1040, 1060], "inertia", WW + "; ICRU 44 (1989) 1040",
  "reference adult tissue composition, sex not stated", "C", "no adjustment: no sex effect on muscle density found",
  "abstract (comparison table seen via search, paper not read)", ["T8", "T9"])

# ================================================================ MUSCLE, active
p("muscle.active.sigma_max", "muscle", "peak isometric fibre stress (specific tension)", 2.68e5, "Pa", [1.71e5, 5.73e5],
  "Blemker-Delp active fibre stress sigma_max * a * f_L(lambda)", PERSAD + "; range ends: gracilis direct 171 kPa (Drexel 2023), " + OBRIEN,
  "systematic review: 29 in vivo human studies, 95 values (2-73 N/cm^2), mixed sex", "B",
  "no adjustment: O'Brien 2010 found no sex difference in in vivo quadriceps specific tension (women 57.3 +/- 13, men 55 +/- 11 N/cm^2, 20 adults); women differ in PCSA, not stress",
  "abstract", ["T3", "T5"])
p("muscle.active.fl_halfwidth", "muscle", "active force-length: half-width of the central parabola", 0.5, "1 (of lambda/lambda_ofl)",
  None, "f = 1-4(1-x)^2 for 0.6<x<1.4; 9(x-0.4)^2 below; 9(x-1.6)^2 above; x = lambda/lambda_ofl", BLEM,
  "model curve (sarcomere-based), no tissue population", "M",
  "no adjustment: no sex effect on the force-length shape was searched for or found; flagged", "read in full (FEBio manual, per tissue report)",
  ["T3"], flag=True)
p("muscle.active.fv_vmax", "muscle", "max shortening velocity", 10, "L_opt/s", None,
  "Hill force-velocity; take the curve from MyoSuite to stay consistent with E-ACT", "Zajac 1989; Millard et al. 2013, doi:10.1115/1.4023390",
  "model constant", "M", "no adjustment: use MyoSuite's curve", "unverified", ["T9"], flag=True)
p("muscle.active.fv_ecc_plateau", "muscle", "eccentric force plateau", None, "1 (of F_iso)", [1.4, 1.8],
  "Hill force-velocity (MyoSuite curve)", "Zajac 1989; Millard et al. 2013", "model constant", "M",
  "no adjustment: use MyoSuite's curve", "unverified", ["T9"], flag=True)
p("muscle.active.stiffening_direction", "muscle", "how stiffness rises with activation", None, "-", None,
  "form: stiffening only along fibres, produced by the active fibre stress itself; no extra isotropic g_max term; cross-fibre stays at G2",
  GENN, "in vivo brachialis, healthy volunteers (n, sex not in abstract): along-fibre x20 from 0 to 4 kg, cross-fibre constant", "C",
  "no adjustment: no sex data on activation stiffening found; flagged", "read in full (per tissue report)", ["T4"], flag=True)
p("muscle.active.mu_ratio_20pctMVC", "muscle", "shear stiffness ratio, rest to 20 % MVC (check)", 2.58, "1", [2.11, 2.58],
  "check of the active fibre term", "Bensamoun et al. 2008 (MRE, vastus medialis; 3.69->9.52 kPa 1D, 4.36->9.22 kPa 2D)",
  "in vivo MRE; n, sex, age not read", "C", "no adjustment: no sex data found; flagged", "abstract", ["T4"], flag=True)
p("muscle.active.mu_torque_linearity", "muscle", "linearity of shear modulus vs torque (check, R^2)", 0.982, "1", None,
  "check: indentation/SWE stiffness must rise linearly with force", "Bouillard, Nordez & Hug 2011, PLoS ONE 6:e29261, doi:10.1371/journal.pone.0029261",
  "11 subjects, in vivo SSI, hand muscles (sex, age not in abstract)", "C", "no adjustment", "abstract", ["T4"], flag=True)
p("muscle.active.thickness_gain", "muscle", "muscle thickness gain rest to contraction (check)", None, "1", [0.17, 0.20],
  "check for T5 bulging", "vastus medialis at 90 deg knee, https://pmc.ncbi.nlm.nih.gov/articles/PMC3872419 ; saturation above ~30 % MVC: Hodges et al. 2003, doi:10.1002/mus.10375",
  "in vivo ultrasound; population and contraction level not read", "C", "no adjustment", "abstract", ["T5"], flag=True)
p("muscle.active.transverse_load_loss", "muscle", "force loss under transverse load 1.1-3.2 N/cm^2 (optional)", 0.13, "1", [0.13, 0.15],
  "optional second-order term", "Siebert et al. 2016, J Biomech (PubMed 26976226); Siebert 2014", "rat gastrocnemius, in situ / isolated", "D",
  "no adjustment: animal only", "abstract", ["T5", "T7"], flag=True)

# ================================================================ MUSCLE, damping
p("muscle.visc.eta_along", "muscle", "shear viscosity along fibres (relaxed)", None, "Pa*s", [0.6, 1.6], "Voigt shear viscosity (high-frequency)",
  GENN, "in vivo brachialis (n, sex not in abstract)", "C", "no adjustment: no sex data; x10 when loaded", "read in full (per tissue report)", ["T8"], flag=True)
p("muscle.visc.eta_cross", "muscle", "shear viscosity across fibres (relaxed)", None, "Pa*s", [0.5, 1.5], "Voigt shear viscosity",
  GENN, "as above", "C", "no adjustment: no sex data; x3 when loaded", "read in full (per tissue report)", ["T8"], flag=True)
p("muscle.visc.prony_tau", "muscle", "Prony time constants", [0.6, 6, 30, 60, 300], "s", None, "NLV-SYM / Prony series",
  VLOOCKE, "porcine muscle, < 2 h post mortem, unconfined compression, ex vivo", "D",
  "no adjustment: no human in vivo relaxation series exists; for animation drop 60/300 s; flagged", "read in full (per tissue report)", ["T8"],
  flag=True, note="Bosboom 2001 in vivo rat single tau = 6.01 s agrees with the 6 s term")
p("muscle.visc.prony_g_fibre", "muscle", "Prony weights, fibre direction", [0.374, 0.285, 0.077, 0.078, 0.101], "1", None,
  "Prony series", VLOOCKE, "porcine, ex vivo", "D", "no adjustment; flagged", "read in full (per tissue report)", ["T8"], flag=True)
p("muscle.visc.prony_g_cross", "muscle", "Prony weights, cross-fibre", [0.097, 0.532, 0.072, 0.000, 0.086], "1", None,
  "Prony series", VLOOCKE, "porcine, ex vivo", "D", "no adjustment; flagged", "read in full (per tissue report)", ["T8"], flag=True)
p("muscle.visc.short_long_ratio", "muscle", "instantaneous / long-term transverse modulus", None, "1", [7.0, 13.9],
  "QLV check (5.7/0.81 preconditioned; 8.5/0.61 not)", PALEV, "porcine gluteus, fresh, ex vivo", "D",
  "no adjustment; flagged", "abstract", ["T8"], flag=True)
p("muscle.dyn.soft_tissue_freq", "muscle", "natural frequency of the relaxed leg soft-tissue package (check)", 15, "Hz", [3, 55],
  "check for T8 (whole package: muscle + fat + skin)", "Nigg-group review, Sports Med Open 2022, doi:10.1186/s40798-022-00524-w; range: Schmitt & Gunther 2011, doi:10.1007/s00419-010-0458-z",
  "in vivo adults (mixed; details not read)", "C",
  "no adjustment: no sex data; the 0.98 m figure rings higher by size scaling (state it, do not retune)", "abstract", ["T8"], flag=True)
p("muscle.dyn.damping_ratio", "muscle", "damping ratio of the thigh soft-tissue package", None, "1", None,
  "Rayleigh / Prony damping in E-TIS", "Wakeling & Nigg 2001 (values not read)", "in vivo, not read", "N",
  "no measured value", "unverified", ["T8"],
  fit="Dyna / DFAUST 4D jiggle (research licence; use female subjects only) for frequency and decay", flag=True)

# ================================================================ TENDON / APONEUROSIS
p("tendon.E", "tendon", "Young's modulus, free tendon", 6.1e8, "Pa", [5.3e8, 6.9e8], "stiff tets or 1D springs, k = E*A/L from her geometry",
  ONAMB, "10 women 21.3 +/- 0.9 y, in vivo ultrasound, patellar tendon, isometric ramps (tibial motion corrected)", "A",
  "no adjustment: direct young-female value (men in the same study 0.68 GPa; Hansen 2006 mixed/unstated 1.09 GPa)", "abstract", ["T3", "T6"])
p("tendon.patellar.strain_mvc", "tendon", "patellar tendon strain at MVC (check)", 0.106, "1", [0.096, 0.116], "check",
  ONAMB, "10 women 21 y, in vivo", "A", "no adjustment: direct", "abstract", ["T6"])
p("tendon.patellar.stiffness", "tendon", "patellar tendon stiffness (check only; geometry-dependent)", 1.053e6, "N/m", [0.945e6, 1.161e6], "check",
  ONAMB, "10 women 21 y, in vivo", "A", "no adjustment: direct; the elf's value comes from E*A/L of her own tendon", "abstract", ["T6"])
p("aponeurosis.strain_long", "aponeurosis", "longitudinal strain at high force (check)", 0.014, "1", [0.010, 0.018], "anisotropic shell",
  "Magnusson et al. 2003, Acta Physiol Scand, doi:10.1046/j.1365-201x.2003.01048.x", "n = 5, in vivo ultrasound, gastrocnemius distal aponeurosis (sex not read)", "C",
  "no adjustment: no sex data for aponeurosis found; flagged", "abstract", ["T3", "T5"], flag=True)
p("aponeurosis.transverse_ratio", "aponeurosis", "transverse / longitudinal strain during active force", 4, "1", None, "anisotropic shell",
  "Azizi & Roberts 2009, https://pmc.ncbi.nlm.nih.gov/articles/PMC2754367", "turkey, in vivo sonomicrometry", "D",
  "no adjustment: animal only; flagged", "abstract", ["T5"], flag=True)
p("aponeurosis.E_long", "aponeurosis", "longitudinal modulus", None, "Pa", None, "anisotropic shell",
  "none measured", "-", "N", "no measured value", "-", ["T3", "T5"],
  fit="fit to aponeurosis.strain_long (Magnusson 2003) with DL_Track_US / Zenodo 7081639 ultrasound", flag=True)
p("tendon.density", "tendon", "density", None, "kg/m^3", None, "inertia", "no source read", "-", "N",
  "gap: no value read; tendon mass is small next to muscle", "-", ["T8"], flag=True)

# ================================================================ FAT
p("fat.mu", "fat", "Ogden shear modulus mu1 (small strain)", 1167, "Pa", [330, 1600], "1st-order Ogden, nearly incompressible",
  ALDIRINI + "; agrees with " + THEN + " (G_inf 1182 Pa)",
  "one seated person, in vivo buttock MRI + inverse FE (sex not verified; the same group's 2017 cohort is 10 men); Then: one subject, gluteal, in vivo",
  "C", NOSEXFAT, "abstract / secondary (Wang 2021 read in full; attribution to Al-Dirini unverified)", ["T4", "T5", "T7"], flag=True)
p("fat.alpha", "fat", "Ogden exponent alpha1", 16.2, "1", [16.2, 20], "1st-order Ogden", ALDIRINI + "; Comley & Fleck 2012 (porcine, alpha 20)",
  "as fat.mu", "C", NOSEXFAT, "abstract / secondary", ["T5", "T7"], flag=True)
p("fat.bulk_physical", "fat", "physical bulk modulus (derived rho*c^2)", None, "Pa", [1.91e9, 2.08e9], "volumetric term; too stiff for VBD",
  "acoustic tables c = 1450-1480 m/s, rho = 911-950 kg/m^3 (https://pmc.ncbi.nlm.nih.gov/articles/PMC4521448/table/Tab3)",
  "tissue acoustics, population not read", "N", "no adjustment: derived from cited c and rho", "unverified", ["T2"], flag=True)
p("fat.solver.nu", "fat", "Poisson's ratio used by the solver", None, "1", [0.49, 0.499], "volumetric penalty",
  "heel fat pad indentation fits, arXiv:2312.04108; Lanzl 2025 nu = 0.499", "heel pad in vivo / ex vivo", "C",
  "no adjustment: numerical choice inside the cited band; check fat thickness loss against Mogale 2023", "abstract", ["T5", "T7"], flag=True)
p("fat.density", "fat", "density", 950, "kg/m^3", [911, 950], "inertia", WW + "; acoustic tables 911-950; Al-Dirini model 920",
  "reference adult tissue composition, sex not stated", "C", "no adjustment: no sex effect on adipose density found", "abstract", ["T8", "T9"])
p("fat.E_tension_initial", "fat", "tensile modulus, initial (check)", 1600, "Pa", [800, 2400], "check of the Ogden fit",
  ALKH, "44 surgical patients, human SAT, ex vivo (sex split not read)", "D", NOSEXFAT, "abstract", ["T4"], flag=True)
p("fat.E_tension_30pct", "fat", "tensile modulus at <= 30 % strain (check)", 11700, "Pa", [5300, 18100], "check of the Ogden stiffening",
  ALKH, "as above", "D", NOSEXFAT, "abstract", ["T4"], flag=True)
p("fat.fibre_k1", "fat", "septa fibre stiffness k1 (optional anisotropy)", 429, "Pa", None, "neo-Hooke + 3 circular fibre families",
  "Lanzl et al. 2025, JMBBM 169:107042, doi:10.1016/j.jmbbm.2025.107042, fit to Sommer et al. 2013", "human abdominal fat, ex vivo",
  "D", NOSEXFAT, "read in full (per tissue report)", ["T7"], flag=True)
p("fat.fibre_k2", "fat", "septa fibre exponent k2 (optional anisotropy)", 11.6, "1", None, "neo-Hooke + 3 circular fibre families",
  "Lanzl et al. 2025 (as above)", "human abdominal fat, ex vivo", "D", NOSEXFAT, "read in full (per tissue report)", ["T7"], flag=True)
p("fat.rate_limit", "fat", "strain rate below which fat is rate-independent", 10, "1/s", None, "validity of the quasi-static solve",
  "Comley & Fleck 2012, Int J Impact Eng 46:1, doi:10.1016/j.ijimpeng.2011.12.009", "porcine, ex vivo", "D",
  "no adjustment; flagged", "unverified", ["T7", "T8"], flag=True)
p("fat.relaxation", "fat", "relaxation (two-exponential, constants)", None, "s", None, "2-term Prony series",
  ALKH + " (two exponentials, constants not retrieved); Geerligs 2008 power law (porcine)", "ex vivo", "N",
  "no measured constants available to us", "-", ["T8"],
  fit="Alkhouli 2013 full-text relaxation curves if obtained; else Dyna/DFAUST decay (female subjects)", flag=True)
p("fat.superficial_deep_ratio", "fat", "stiffness ratio superficial / deep SAT", None, "1", None, "two fat layers split by the membranous layer",
  "strain elastography, Australas J Ultrasound Med, doi:10.1002/ajum.12100 (qualitative only)", "210 pregnant women, in vivo", "N",
  "gap: only 'superficial stiffer' is known; keep one fat material", "unverified", ["T7"], flag=True)
p("fat.damping_ratio", "fat", "fat-only damping", None, "1", None, "E-TIS damping", "none found", "-", "N",
  "gap", "-", ["T8"], fit="Dyna / DFAUST (female subjects); fat and muscle are mixed in it", flag=True)

# ================================================================ SKIN
p("skin.matrix_mu", "skin", "GOH matrix shear modulus", 2.014e5, "Pa", [2.5e4, 2.6e5], "GOH membrane shell (neo-Hooke matrix + 2 dispersed fibre families)",
  NIA + "; in vivo suction range 25-260 kPa (Kalra 2016, unverified)", "7 donors aged 81-97, back skin, uniaxial, ex vivo", "D",
  "not applied: Agache 1980 (138 people 3-89 y, in vivo torsion) found Young's modulus about doubles above age 30, so young skin is softer, but the factor is from another method and its numbers are unverified; treat 0.20 MPa as an upper bound",
  "read in full (per tissue report)", ["T6"],
  fit="FEMTO-ST CutiScan suction DIC (one 28-y man, forearm) + Rupani 2025 knee/thigh stretch DIC (15 people)", flag=True)
p("skin.k1", "skin", "GOH fibre stiffness k1 (upper bound)", 2.436e8, "Pa", None, "GOH", NIA, "7 donors 81-97 y, ex vivo", "D",
  "not applied (see skin.matrix_mu); upper bound", "read in full (per tissue report)", ["T6"], fit="as skin.matrix_mu", flag=True)
p("skin.k2", "skin", "GOH fibre exponent k2", 0.1327, "1", None, "GOH", NIA, "7 donors 81-97 y, ex vivo", "D",
  "no adjustment", "read in full (per tissue report)", ["T6"], fit="as skin.matrix_mu", flag=True)
p("skin.kappa", "skin", "GOH fibre dispersion kappa", 0.1404, "1", [0.1376, 0.1432], "GOH", NIA + " (from histology)", "7 donors 81-97 y, ex vivo", "D",
  "no adjustment", "read in full (per tissue report)", ["T6"], flag=True)
p("skin.fibre_angle", "skin", "mean fibre angle to the Langer line", 0.716, "rad", None, "GOH; field direction from the Langer-line map",
  NIA, "7 donors 81-97 y, ex vivo", "D", "no adjustment", "unverified", ["T6"], flag=True)
p("skin.prestress", "skin", "in vivo in-plane pre-stress (orthotropic)", None, "Pa", [2.8e4, 9.2e4], "rest shape smaller than the mesh",
  FLYNN, "volunteers, in vivo forearm and upper arm micro-robot (n, sex, age not in abstract)", "C",
  "no adjustment: no sex or age effect found; no thigh value", "abstract", ["T6"],
  fit="Rupani 2025 thigh/knee DIC (and retraction data if found)", flag=True)
p("skin.thickness_thigh", "skin", "skin thickness, thigh (epidermis + dermis)", 1.683e-3, "m", [1.29e-3, 2.07e-3], "shell thickness",
  WANG25 + "; cf. " + DERRAIK,
  "33 healthy women (all 18-74 y, women's median 45), Chinese, in vivo 22/75 MHz ultrasound, right thigh: dermis 1.5645 +/- 0.3902 mm + epidermis 0.1189 mm",
  "C", "no adjustment: Wang 2025 found no thigh age correlation; Derraik 2014 (diabetic adults) gives -6 um/yr dermis, which would add about 0.1 mm at 28 vs 44 y, inside the range; women thinner than men (1.56 vs 1.94 mm dermis) is already in the source",
  "read in full", ["T1"])
p("skin.thickness_buttock", "skin", "skin thickness, buttock", None, "m", None, "shell thickness", "none found", "-", "N",
  "gap: no measurement found (trunk 3-7 mm from Herlin 2015 does not apply)", "-", ["T1", "T7"],
  fit="no dataset found; MULTiS (100 people, CC BY) covers limbs only - needs a new source", flag=True)
p("skin.thickness_knee", "skin", "skin thickness, knee", None, "m", None, "shell thickness", "none found", "-", "N",
  "gap: no measurement found", "-", ["T1", "T6"], fit="no dataset found - needs a new source", flag=True)
p("skin.bending_D", "skin", "bending stiffness (derived D = E t^3 / 12(1-nu^2))", None, "N*m", [5.3e-5, 5.3e-4], "shell bending",
  "derived from skin.thickness_thigh and E = 0.1-1 MPa, nu = 0.5; no measurement exists", "-", "N",
  "no measured value", "derived", ["T6"], fit="fold wavelength behind the knee: Rupani 2025 DIC, DFAUST surfaces (female subjects)", flag=True)
p("skin.density", "skin", "density", 1090, "kg/m^3", [1090, 1100], "inertia", WW + "; ICRU 44 (1989) 1100",
  "reference adult tissue composition, sex not stated", "C", "no adjustment: no sex effect found", "abstract", ["T8"])
p("skin.failure_stretch", "skin", "failure stretch along / across Langer lines (check)", 1.46, "1", [1.46, 1.52], "limit check; flag elements > 1.45",
  NIA, "7 donors 81-97 y, ex vivo", "D", "no adjustment", "read in full (per tissue report)", ["T6"], flag=True)
p("skin.visc_tau1", "skin", "QLV first time constant", 0.19, "s", [0.09, 0.29], "QLV / Kelvin-Voigt",
  "https://pmc.ncbi.nlm.nih.gov/articles/PMC4372409", "excised thin specimens 211-671 um (species not checked)", "D",
  "no adjustment", "unverified", ["T8"], fit="FEMTO-ST CutiScan suction curves", flag=True)
p("skin.friction_dry", "skin", "friction, dry skin against counter-materials", 0.5, "1", [0.2, 0.5], "Coulomb contact",
  DERLER, "literature review, mixed populations and sites", "C",
  "no adjustment: " + VEIJ + " report sex differences in friction that are mostly explained by hydration; no numeric sex factor",
  "abstract", ["T7", "T9"], flag=True)
p("skin.friction_moist", "skin", "friction, moist / sweaty skin", None, "1", [1.0, None], "Coulomb contact", DERLER,
  "review, mixed", "C", "no adjustment (as above)", "abstract", ["T6", "T7"], flag=True)
p("skin.friction_textile", "skin", "friction, skin against textile (dynamic to static)", None, "1", [0.25, 0.90], "Coulomb contact",
  "Tribol Lett 2021, doi:10.1007/s11249-021-01560-5 (chest 0.90/0.79, forearm 0.31/0.25); trousers vs seat: Lamsal 2023, doi:10.1007/s11340-023-01022-0",
  "in vivo, handheld device (population not read)", "C", "no adjustment: no thigh/buttock value; no sex factor", "abstract", ["T7", "T9"], flag=True)
p("skin.friction_self", "skin", "skin-on-skin friction (back of knee, calf on thigh)", None, "1", [0.5, 1.0], "Coulomb self-contact",
  "bracket only, from " + DERLER, "-", "N", "gap: never measured", "-", ["T6"],
  fit="crease behaviour in T6 against Rupani 2025 and 3DCP/TUCH self-contact meshes; record the fit", flag=True)

# ================================================================ DEEP FASCIA
p("fascia.E_long", "fascia", "tensile modulus along the limb (stiff sheet)", None, "Pa", [5.57e7, 1.919e8], "anisotropic membrane shell (2-family HGO)",
  BONALDI, "4 fresh-frozen donors aged 54-89, fascia lata, ex vivo, 0.5 %/s", "D",
  "no adjustment: no sex or age data on fascia moduli found; flagged (older donors)", "read in full (per tissue report)", ["T5", "T6"], flag=True)
p("fascia.E_trans", "fascia", "tensile modulus across (second sheet)", None, "Pa", [4.5e6, 1.89e7], "anisotropic membrane shell",
  BONALDI, "as above (one outlier 119.9 MPa excluded)", "D", "no adjustment (as above)", "read in full (per tissue report)", ["T5", "T6"], flag=True)
for k, lo, hi, u in [("C1", 0.11e6, 0.68e6, "Pa"), ("k11", 2.8e6, 5.7e6, "Pa"), ("k21", 42, 173, "1"),
                     ("k12", 0.77e6, 2.0e6, "Pa"), ("k22", 115, 579, "1"), ("kappa", 0.0004, 0.19, "1")]:
    p("fascia.hgo_" + k, "fascia", "HGO constant " + k + " (per-donor spread)", None, u, [lo, hi], "2-family HGO, elastic region <= 3-4 % strain",
      BONALDI + " Table 3", "4 donors 54-89 y, ex vivo", "D", "no adjustment: donor mean and spread are the prior", "read in full (per tissue report)",
      ["T5", "T6"], flag=True)
p("fascia.sheet_angle", "fascia", "angle between the two collagen sheets", None, "rad", [1.309, 1.396], "fibre family 2 direction",
  BONALDI, "4 donors, ex vivo", "D", "no adjustment", "read in full (per tissue report)", ["T1"], flag=True)
p("fascia.thickness_lata", "fascia", "fascia lata thickness (histology, without epimysium)", 9.44e-4, "m", [7.88e-4, 1.10e-3], "shell thickness",
  "histology cited in " + BONALDI + "; in vivo US 1.88 mm incl. epimysium: Stecco et al. 2011, doi:10.1007/s00276-011-0876-9",
  "cadaver histology (population not read); US: 22 volunteers", "D",
  "no adjustment: no sex comparison found; a female-only study reports thicker lower-limb fascia in young than old women (numbers not read)",
  "abstract", ["T1"], flag=True)
p("fascia.thickness_crural", "fascia", "crural fascia thickness (US)", None, "m", [7.8e-4, 1.21e-3], "shell thickness",
  PIRRI, "15 healthy adults, in vivo US (sex split not read)", "C", "no adjustment: no sex data", "read in full (per tissue report)", ["T1"], flag=True)
p("fascia.thickness_epimysium", "fascia", "epimysial fascia thickness (tibialis anterior)", 3.5e-4, "m", [2.7e-4, 4.3e-4], "muscle surface layer",
  PIRRI, "15 healthy adults, in vivo US", "C", "no adjustment", "read in full (per tissue report)", ["T1"], flag=True)
p("fascia.thickness_superficial", "fascia", "superficial (membranous) fascia thickness, limbs", None, "m", [6e-4, 1.0e-3], "thin stiff shell tied to SAT",
  "2025 fascia review (Pirri et al., snippet)", "review of US studies", "C", "no adjustment", "unverified", ["T1"], flag=True)
p("fascia.visc_residual_60s", "fascia", "residual stress fraction after 60 s relaxation", 0.56, "1", None, "2-term Prony",
  BONALDI + " Table 2", "4 donors, ex vivo", "D", "no adjustment", "read in full (per tissue report)", ["T5", "T8"], flag=True)
p("fascia.visc_residual_300s", "fascia", "residual stress fraction after 300 s relaxation", 0.43, "1", None, "2-term Prony",
  BONALDI + " Table 2", "4 donors, ex vivo", "D", "no adjustment", "read in full (per tissue report)", ["T5"], flag=True)
p("fascia.prestress", "fascia", "in vivo pre-tension", None, "Pa", None, "rest shape", "none found", "-", "N",
  "gap: no measurement; replaces the 'shrinkage' knob", "-", ["T5"],
  fit="no direct dataset; candidate: van Riel 2026 cuff MRI (thigh shape at 0 mmHg) - flagged", flag=True)
p("fascia.density", "fascia", "density", None, "kg/m^3", None, "inertia", "no source read", "-", "N", "gap", "-", ["T8"], flag=True)

# ================================================================ INTERFACES
p("iface.muscle_muscle.normal", "interface", "muscle <-> muscle, normal direction", None, "-", None,
  "form: bilateral (no separation) sliding contact (FEBio sliding-elastic tension=true; Teran 2005 fascia links)",
  "Teran et al. 2005, doi:10.1109/TVCG.2005.42; Maas & Sandercock 2010, PMC2853902", "simulation + anatomy", "M",
  "no adjustment", "read in full (per tissue report)", ["T5"])
p("iface.muscle_muscle.k_t", "interface", "muscle <-> muscle tangential shear stiffness per area", None, "Pa/m", [3e6, 1e8],
  "cohesive tangential spring (areolar tissue, neurovascular tracts)",
  "seed range derived: loose-tissue modulus 3-5 kPa / thickness 0.05-1 mm (Mount Sinai abstract; Stecco)", "-", "N",
  "no measured value", "derived", ["T5", "T6"],
  fit="Bojsen-Moller et al. 2004 (doi:10.1152/japplphysiol.00084.2004, n = 8, MG-Sol +3.7 / -1.7 mm) + Huijing 2011 soleus strain; thigh target missing", flag=True)
p("iface.muscle_muscle.slide_check", "interface", "relative aponeurosis slide MG vs soleus (check)", None, "m", [-1.7e-3, 3.7e-3], "check",
  "Bojsen-Moller et al. 2004", "8 adults, in vivo ultrasound (sex not read)", "C", "no adjustment: no female data", "abstract", ["T5"], flag=True)
p("iface.muscle_fascia.friction", "interface", "muscle (epimysium) <-> deep fascia friction", None, "1", None,
  "sliding, no separation, low friction (HA layer)", "none measured; HA layer: Stecco et al. 2011", "-", "N", "gap", "-", ["T6"],
  fit=WILKE + ": muscle 5.70 +/- 2.43 mm vs superficial fascia 0.72 +/- 0.81 mm (n = 11, paper values; data not shared); Wang 2023 ITB shear strain", flag=True)
p("iface.muscle_fascia.ha_film", "interface", "hyaluronan film thickness between fascia and muscle", 5e-5, "m", None, "interface layer",
  "Roman et al. 2013, doi:10.7556/jaoa.2013.021 citing Stecco", "cadaver", "D", "no adjustment", "unverified", ["T1"], flag=True)
p("iface.fascia_fat.shear", "interface", "deep fascia <-> deep fat, tied through a compliant layer: shear stiffness", None, "Pa/m", None,
  "tied (cohesive shear spring) or DAT as its own soft tets", "Stecco reappraisal of the superficial fascia; " + WILKE, "-", "N",
  "no measured value", "-", ["T6"], fit=WILKE + " coupling ratio (about 1:8)", flag=True)
p("iface.skin_fat.slip_check", "interface", "skin + SAT + superficial fascia slip (check; modelled as embedded / tied)", 7.2e-4, "m", [0, 1.53e-3],
  "skin membrane embedded in the outer fat tets", WILKE, "11 adults, in vivo US cross-correlation (sex not read)", "C",
  "no adjustment", "abstract", ["T6"], flag=True)
p("iface.septa.toe_strain", "interface", "retinacula cutis profundus tether: slack (toe) strain", None, "1", None,
  "tension-only tethers SM -> deep fascia, rest length = local DAT thickness from her geometry", "none measured (Li et al. 2013 notes toe region)", "-", "N",
  "gap", "-", ["T6", "T9"],
  fit=CEREATTI + " (thigh RMSD 7.6-8.4 mm, peak 23.4-28.4 mm, sit-to-stand up to 72.3 mm) + CAMS-Knee", flag=True)
p("iface.septa.stiffness", "interface", "retinacula cutis tether stiffness (per region)", None, "N/m", None, "tension-only tethers",
  "none measured", "-", "N", "gap", "-", ["T6", "T9"], fit="as iface.septa.toe_strain", flag=True)
p("iface.septa.coherence_thigh", "interface", "septa orientation coherence, thigh (density prior)", 0.711, "1", None, "tether layout prior",
  "Li & Ahn 2011, PLoS ONE 6:e23987, doi:10.1371/journal.pone.0023987", "25 adults, in vivo 10 MHz US (sex split not read)", "C",
  "no adjustment", "abstract", ["T1"], flag=True)
p("iface.gluteal_fold.anchor", "interface", "medial gluteal fold: dermis anchored to the ischium", None, "-", None,
  "form: zero-length hard anchors, density fading laterally",
  "Si et al. 2024, Plast Reconstr Surg 154:241e, doi:10.1097/PRS.0000000000010723", "20 fresh female buttocks, dissection (ex vivo)", "D",
  "no adjustment: female source", "abstract", ["T7"])
p("iface.muscle_bone.footprint_check", "interface", "quadriceps tendon footprint on the patella (check)", 5.025e-4, "m^2", [3.36e-4, 6.11e-4],
  "tied over the full atlas footprint", "Knee 2023 cadaver 3D CT/MRI (via BVS mdl-37399632)", "cadavers (sex not read)", "D",
  "no adjustment: footprint comes from her warped atlas; female size difference not known", "abstract", ["T1"], flag=True)
p("iface.muscle_bone.ischial_slide", "interface", "gluteus maximus slide off the ischial tuberosity, seated (check)", 1.374e-2, "m", None,
  "frictionless sliding over the ischial bursa", ZAPP, "10 healthy subjects, semi-recumbent MR-DVC; built on Al-Dirini 2017 scans of 10 men", "C",
  "no adjustment: male-only data; no female value found; flagged", "abstract", ["T7"], flag=True)
p("iface.articular.friction", "interface", "bursa / articular cartilage friction", None, "1", [0.001, 0.01], "treated as frictionless",
  "cartilage tribology (as quoted in the link report)", "-", "D", "no adjustment", "unverified", ["T6", "T7"], flag=True)
p("iface.patella.shift_max", "interface", "patella shift over flexion (check)", 3e-3, "m", None,
  "patella as a free body on quadriceps and patellar tendon springs, frictionless trochlear contact (Lenhart 2015)",
  "Nha et al. 2008, J Orthop Res, doi:10.1002/jor.20554", "8 knees, in vivo dual fluoroscopy + MRI, weight-bearing (sex not read)", "C",
  "no adjustment: no female tracking data found", "abstract", ["T6"], flag=True)
p("iface.patella.tilt_max", "interface", "patella tilt over flexion (check)", 0.105, "rad", None, "as above", "Nha et al. 2008", "as above", "C",
  "no adjustment", "abstract", ["T6"], flag=True)
p("iface.patella.lateral_max", "interface", "patella lateral displacement (check)", 6e-3, "m", None, "as above", "Nha et al. 2008", "as above", "C",
  "no adjustment", "abstract", ["T6"], flag=True)

# ---------------------------------------------------------------- write JSON
profile = {
    "sex": "female", "age_years": 28, "health": "healthy", "build": "normal; BMI-equivalent computed from her own mesh volume and height (plan.md 2.1)",
    "source_priority": ["A: women ~20-35, in vivo", "B: mixed-sex young adults, in vivo", "C: any adult, in vivo",
                        "D: ex vivo human or animal", "M: model constant (no tissue population)", "N: no measured source"],
    "geometry": "fat depth from her surface (skin_fat.npz); muscle and bone shapes from the warped atlas (adult male, see e-mat.md section 5)",
}
doc = {"version": "0.1", "date": "2026-11-24", "units": "SI",
       "status_values": ["read in full", "abstract", "unverified", "derived", "-"],
       "profile": profile, "params": P}
with open(os.path.join(OUT, "e-mat.json"), "w", encoding="utf-8") as f:
    json.dump(doc, f, indent=1, ensure_ascii=False)

# ---------------------------------------------------------------- stats
n = len(P)
nA = sum(x["tier"] == "A" for x in P)
nadj = sum(x["adjustment"].lower().startswith(("applied", "adjusted")) for x in P)
nunv = sum("unverified" in x["status"] for x in P)
nfit = sum(x["fit_required"] for x in P)
nnull = sum(x["value"] is None and x["range"] is None for x in P)
tiers = {t: sum(x["tier"] == t for x in P) for t in "ABCDMN"}
print(n, nA, nadj, nunv, nfit, nnull, tiers)
json.dump(dict(n=n, nA=nA, nadj=nadj, nunv=nunv, nfit=nfit, nnull=nnull, tiers=tiers),
          open(os.path.join(OUT, "e-mat-stats.json"), "w"))

# ---------------------------------------------------------------- sex/age effect bookkeeping
EFFECT = {
 # direct female source; a sex effect in the literature confirms a female-specific value is needed
 "direct-female": ["muscle.passive.mu_along", "muscle.passive.mu_along_knee90", "muscle.passive.mu_along_fullflex",
                   "tendon.E", "tendon.patellar.strain_mvc", "tendon.patellar.stiffness", "skin.thickness_thigh",
                   "iface.gluteal_fold.anchor"],
 # cited sex study shows no difference -> value kept, citation recorded
 "checked-no-difference": ["muscle.active.sigma_max"],
 # a sex/age effect is published but could not be transferred as a number
 "found-not-applied": ["skin.matrix_mu", "skin.k1", "skin.friction_dry", "skin.friction_moist", "fascia.thickness_lata"],
}
eff = {i: k for k, ids in EFFECT.items() for i in ids}
for x in P:
    x["sex_age_effect"] = eff.get(x["id"], "none-found")
doc["params"] = P
with open(os.path.join(OUT, "e-mat.json"), "w", encoding="utf-8") as f:
    json.dump(doc, f, indent=1, ensure_ascii=False)
from collections import Counter
print(Counter(x["sex_age_effect"] for x in P))
print(sorted(Counter(x["tissue"] for x in P).items()))
