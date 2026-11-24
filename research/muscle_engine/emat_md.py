import json, os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "muscle-engine")
d = json.load(open(os.path.join(OUT, "e-mat.json"), encoding="utf-8"))
P = d["params"]

def fmt(v):
    if v is None: return "-"
    if isinstance(v, list): return "[" + ", ".join(fmt(x) for x in v) + "]"
    if isinstance(v, float) and (abs(v) >= 1e5 or (v != 0 and abs(v) < 1e-2)): return f"{v:.4g}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
    return f"{v:g}" if isinstance(v, float) else str(v)

def cell(s):
    return str(s).replace("|", "/").replace("\n", " ")

def val(x):
    v = fmt(x["value"])
    if x["range"] is not None:
        lo, hi = x["range"]
        r = f"{fmt(lo)}–{fmt(hi)}" if hi is not None else f"> {fmt(lo)}"
        v = r if x["value"] is None else f"{v} ({r})"
    if x["fit_required"]:
        v = "**FIT** " + (v if v != "-" else "")
    return v

TIER = {"A": "A women 20–35 in vivo", "B": "B mixed young in vivo", "C": "C adult in vivo", "D": "D ex vivo / animal",
        "M": "M model constant", "N": "N no measured source"}
GROUPS = [("muscle", "Muscle (passive, active, damping)"), ("tendon", "Tendon"), ("aponeurosis", "Aponeurosis"),
          ("fat", "Fat (subcutaneous adipose tissue)"), ("skin", "Skin"), ("fascia", "Deep and superficial fascia"),
          ("interface", "Interfaces")]

n = len(P)
nA = sum(x["tier"] == "A" for x in P)
nunv = sum("unverified" in x["status"] for x in P)
nfit = sum(x["fit_required"] for x in P)
nflag = sum(x["flagged"] for x in P)
from collections import Counter
eff = Counter(x["sex_age_effect"] for x in P)
tiers = Counter(x["tier"] for x in P)

L = []
w = L.append
w("# E-MAT v0.1: material table for the subject profile")
w("")
w("Built 2026-11-24 for [plan.md](plan.md) §4.1 (E-MAT), from the six reports in [tissue-research/](tissue-research/) plus a targeted")
w("search for sex and age data. Machine-readable copy: [e-mat.json](e-mat.json) (same rows, SI units, generated from one list).")
w("Nothing here is tuned. A value with no measured source is left empty and marked **FIT** with the dataset it must be fitted to.")
w("")
w("## 1. Subject profile")
w("")
w("A healthy woman of about 28, normal build (plan §2.1). Her BMI-equivalent comes from her own mesh; fat depth from her surface;")
w("muscle and bone shapes from the warped atlas, which is **male** (section 5).")
w("")
w("Source tiers, in order of preference: **A** women about 20–35 in vivo · **B** mixed-sex young adults in vivo · **C** any adult in vivo ·")
w("**D** ex vivo human or animal · **M** model constant (no tissue population) · **N** no measured source.")
w("")
w("## 2. Summary")
w("")
w(f"| count | |")
w(f"|---|---|")
w(f"| parameters (rows) | {n} |")
w(f"| tier A (women 20–35, in vivo) | {tiers['A']} |")
w(f"| tier B / C / D / M / N | {tiers['B']} / {tiers['C']} / {tiers['D']} / {tiers['M']} / {tiers['N']} |")
w(f"| numerically adjusted for sex or age | 0 (see below) |")
w(f"| direct female source used | {eff['direct-female']} |")
w(f"| sex effect checked, none found (value kept) | {eff['checked-no-difference']} |")
w(f"| sex/age effect published but not transferable (direction stated) | {eff['found-not-applied']} |")
w(f"| no sex/age data found (flagged) | {eff['none-found']} |")
w(f"| status *unverified* | {nunv} |")
w(f"| must be fitted to a dataset (**FIT**) | {nfit} |")
w(f"| flagged (weaker source than tier A/B, or a gap) | {nflag} |")
w("")
w("**Why no numeric adjustment.** Where a young-female in vivo value existed it was used directly (muscle along-fibre shear,")
w("patellar tendon, thigh skin thickness). Where a sex study existed for a mixed source it showed no difference (specific tension,")
w("O'Brien 2010). The remaining sex or age effects (skin stiffness with age, Agache 1980; friction by sex, Veijgen 2013; fascia")
w("thickness young vs old women) are from other methods or have no extractable factor, so applying them would be a guess. Their")
w("direction is written in the row.")
w("")
w("Columns: *value* is the central value with the cited range in brackets; a range alone means the source gives no single value.")
w("*Tier* is the source population class. *Tests* are the scenes in plan §6 that check the row.")
w("")
w("## 3. Table")
for key, title in GROUPS:
    rows = [x for x in P if x["tissue"] == key]
    w("")
    w(f"### {title}")
    w("")
    w("| id | parameter | value | unit | model | source | population | tier | adjustment for the profile | status | tests |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for x in rows:
        st = x["status"]
        st = st.replace("unverified", "*unverified*")
        w("| " + " | ".join(cell(c) for c in [f"`{x['id']}`", x["parameter"], val(x), x["unit"], x["model"], x["source"],
                                              x["population"], x["tier"], x["adjustment"], st, ", ".join(x["tests"])]) + " |")
w("")
w("## 4. Values that must be fitted (no measured source)")
w("")
w("Each is fitted once, the fit and its data recorded in this table, never set by hand.")
w("")
w("| id | fit to |")
w("|---|---|")
for x in P:
    if x["fit_required"]:
        w(f"| `{x['id']}` | {cell(x['fit_dataset'])} |")
w("")
w("## 5. Where the male atlas and male myoLegs conflict with the profile")
w("")
w("The atlas (BodyParts3D) is one adult male; myoLegs inherits Rajagopal et al. 2016 (IEEE TBME 63:2068, PMC5507211), whose")
w("musculotendon parameters combine 21 cadaver lower limbs with MRI muscle volumes of 24 young adults (Handsfield et al. 2014,")
w("J Biomech 47:631; the sex split of both sets was not confirmed here) and whose bone geometry is an averaged adult male of about")
w("75 kg / 170 cm (search snippet, *unverified*). Every result must state this until the corrections below are made.")
w("")
w("| conflict | what is known for women | correction found | applied? |")
w("|---|---|---|---|")
w("| **relative muscle volumes** | Women carry relatively more mass in gluteus maximus and medius, tibialis anterior and posterior and EDL; men relatively more in rectus femoris and semimembranosus. Uniform scaling of all F_max does not capture this; no open model reflects the female distribution | Maarleveld et al. 2024, *What the %PCSA?*, arXiv:2411.00071 (review of 57 studies) **[abstract]**; per-muscle factors not extracted | no: needs the per-muscle table from the paper |")
w("| **female musculoskeletal model** | YONI: lower-limb OpenSim model from MRI of 9 young women aged 28.1 ± 2.6 y (matches the profile); scaled YONI fit a female subject with RMS 9.7 mm vs 24.7 mm for scaled Rajagopal | Gouka 2025, TU Delft MSc thesis, https://repository.tudelft.nl/record/uuid:cdeecb96-a27d-4493-8c10-f669e6cca2d1 **[abstract]**; licence and data access *unverified* | no: candidate source for female bones, attachments and volumes (ask before any download) |")
w("| **muscle force** | Specific tension does not differ by sex (women 57.3 ± 13, men 55 ± 11 N/cm²), so female forces differ through PCSA only | O'Brien et al. 2010 **[abstract]** | yes: `muscle.active.sigma_max` kept sex-neutral; F_max must come from her own volumes |")
w("| **passive muscle stiffness** | Women lower than men at long muscle lengths (VL at full knee flexion 43.0 vs 56.7 kPa, ages 30–39) and at 90° (RF, VM, VL; Akagi 2019) | Maeda 2021; Akagi 2019 **[read in full]** | yes: female values used directly |")
w("| **tendon** | Patellar tendon in young women: lower stiffness (1053 vs 1652 N/mm), higher strain (10.6 vs 9.0 %), modulus 0.61 vs 0.68 GPa | Onambele et al. 2007 **[abstract]** | yes: female E used; myoLegs tendon slack lengths and stiffness are male-derived |")
w("| **fat distribution** | Thigh subcutis in women about 1.8× men's (17.7 vs 9.8 mm, adults with diabetes, mean BMI 28); the atlas has no subject fat | Derraik et al. 2014 **[read in full]** | by design: fat depth comes from her own surface (`skin_fat.npz`), not the atlas |")
w("| **skin** | Thigh dermis thinner in women (1.56 vs 1.94 mm) | Wang et al. 2025 **[read in full]** | yes: `skin.thickness_thigh` is the female value |")
w("| **pelvis shape** | Female pelvis wider with a wider aperture; bispinous width greater, biacetabular width not significantly different (Kurki 2011, via review); greater trochanter more medial and abductor moment arm shorter in women; hip-joint-centre adjustment changed hip moments significantly in sex-specific Rajagopal variants | Frontiers Bioeng Biotechnol 2022, doi:10.3389/fbioe.2022.863194 **[abstract]**; review snippets *unverified* | no: the warp fits her outer shape but the atlas pelvis interior stays male; flag in T7/T9 |")
w("| **reference data for tests** | Zappalà 2024 / Al-Dirini 2017 buttock MRI are 10 men; Sejersted 1984 compartment pressure 7 men; men's seated peak pressure about 10 kPa higher than women's (TU Delft) | see rows | no: T7 magnitudes are male; use female subsets where they exist (Mogale 2023 has 13 women of 26; DFAUST female subjects) |")
w("")
w("## 6. Biggest gaps")
w("")
w("1. **Interfaces have no measured stiffness or friction at all**: muscle↔muscle shear, muscle↔fascia friction, fascia↔fat shear, septa")
w("   tethers, skin-on-skin friction. All are **FIT**, and the thigh-specific sliding target (rectus vs vasti vs hamstrings) is missing.")
w("2. **Skin stiffness for a young woman**: the only GOH fit is from donors aged 81–97; in vivo pre-stress is from the arm; no thigh,")
w("   buttock or knee value. Buttock and knee skin thickness: no source found.")
w("3. **Fat**: the in vivo constants come from one or two subjects of unconfirmed sex; no adult sex or BMI effect on stiffness;")
w("   no relaxation constants; no fat-only damping; superficial/deep ratio unknown.")
w("4. **Fascia**: all moduli from 4 elderly donors ex vivo; no pre-tension; no sex data.")
w("5. **Damping**: muscle relaxation series is porcine ex vivo; thigh damping ratio unmeasured (fit to Dyna/DFAUST).")
w("6. **Model constants** P1, P2, λ_ofl are whole-muscle fits, and λ_ofl depends on the male atlas rest lengths.")
w("7. **Densities** of tendon and fascia: no source read.")
w("")
w("## 7. What this pass verified or added")
w("")
w("- Persad 2024 recommendation of 26.8 N/cm² is now confirmed at abstract level (was *unverified*).")
w("- New tier-A sources: Maeda et al. 2021 (female VL shear modulus by knee angle, full text) and Onambele et al. 2007 (young-female")
w("  patellar tendon, abstract).")
w("- New sex evidence: O'Brien 2010 (specific tension, no difference), Akagi 2019 (young women less stiff at 90°), Wang 2025 and")
w("  Derraik 2014 (thigh skin thickness by sex), Maarleveld 2024 (female muscle-mass distribution), Gouka 2025 (female model).")
w("- Densities now cited (Woodard & White 1986; ICRU 44), replacing the uncited 1100 kg/m³ for skin.")
w("- Eby 2015 (Clin Biomech 30:22, 133 adults 21–94) checked: sex and age mattered only above 60 y and at full elbow extension, so no")
w("  adjustment is needed for a 28-year-old [abstract].")
w("- Not resolved: Al-Dirini 2016 subject sex and original table, Hansen 2006 and Gennisson 2010 populations (abstracts do not give")
w("  them), Agache 1980 numbers, Bonaldi donor sexes.")
w("")
open(os.path.join(OUT, "e-mat.md"), "w", encoding="utf-8").write("\n".join(L))
print("written", len(L), "lines")
