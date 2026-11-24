# Subcutaneous adipose tissue (fat): mechanics in living people

Desk research for E-MAT (`docs/muscle-engine/plan.md`) and the sitting tests (`docs/muscle-engine/sit-physics-research-handoff.md`). Every number has a source. *unverified* means I saw the value only in a secondary source or a search abstract, not in the original paper. No data was downloaded.

## 1. Structure that matters for mechanics

| Feature | What it is | Numbers | Source |
|---|---|---|---|
| Adipocytes | Lipid-filled cells. Each sits inside a basement membrane and a collagen-fibril "reinforced basement membrane" (RBM), giving a foam-like matrix | diameter 50–80 µm | Lanzl et al. 2025, JMBBM 169:107042, [doi:10.1016/j.jmbbm.2025.107042](https://doi.org/10.1016/j.jmbbm.2025.107042) (read in full; open access) |
| Septa | Long collagen fibres running through the fat. They are the likely cause of the slight anisotropy | no spacing numbers in that paper | Lanzl 2025; Sommer et al. 2013, [doi:10.1016/j.actbio.2013.06.011](https://doi.org/10.1016/j.actbio.2013.06.011) |
| Superficial adipose layer (SAT-s) | Just under the dermis. Large lobules inside compact, organised septa (retinacula cutis) that tie the dermis to deeper planes | — | Lancerotto et al. 2011, Surg Radiol Anat, [doi:10.1007/s00276-010-0772-8](https://doi.org/10.1007/s00276-010-0772-8); Markman & Barton 1987, [PubMed 3602174](https://pubmed.ncbi.nlm.nih.gov/3602174/) |
| Membranous layer (superficial fascia; Scarpa's fascia on the abdomen) | Sheet that splits the superficial layer from the deep layer | — | Lancerotto 2011; Stecco review, [UniPD](https://www.research.unipd.it/handle/11577/150935) |
| Deep adipose layer (SAT-d) | Looser, more widely spaced septa; between the membranous layer and the muscle fascia; varies by region | thigh dSAT/sSAT thickness ratio ≈ 1.20 *unverified* (search summary of a thigh ultrasound study) | Markman & Barton 1987 |
| Superficial vs deep stiffness | Strain elastography finds SSAT "denser" (stiffer) than DSAT | qualitative, 210 pregnant women | Australas J Ultrasound Med, [doi:10.1002/ajum.12100](https://doi.org/10.1002/ajum.12100) *unverified* |
| Buttock under the ischium | When sitting, the tissue under the ischial tuberosity (IT) is mostly fat and connective tissue: gluteus maximus slides off the IT. 5 of 7 subjects did not sit on muscle | fat+skin under IT ≈ 13 ± 9 mm; loaded ≈ 10 mm (6 subjects) | Sonenblum et al., RESNA 2015, [html](https://www.resna.org/sites/default/files/conference/2015/wheelchair_seating/sonenblum.html) |
| Gluteus sliding | Gluteus maximus slides off the IT during sitting; the authors attribute this to the ischial bursa | mean 13.74 mm | Zappalá et al. 2024, J Biomech 163:111913, [ORCA](https://orca.cardiff.ac.uk/165053) *(abstract and search summary)* |

**For the engine:** model SAT as two layers, a stiffer superficial layer and a softer deep layer, split by a membranous sheet (that sheet belongs in the fascia file). Under the IT, fat and not muscle should carry the seat load once the gluteus has slid off. That is a direct check for the sitting tests.

## 2. Stretch, shear, compression: constitutive fits

| Study | Tissue / test | Model and constants | Notes |
|---|---|---|---|
| **Al-Dirini et al. 2016**, Ann Biomed Eng 44:2805, [doi:10.1007/s10439-016-1560-3](https://doi.org/10.1007/s10439-016-1560-3) | **in vivo**, one seated person; MRI loaded vs unloaded; inverse FE ("heuristic optimisation") | 1st-order Ogden fat **μ₁ = 1166.7 Pa, α₁ = 16.2**; muscle μ₁ = 1907.4 Pa, α₁ = 4.6; fat ρ = 920 kg/m³ | Values read from Table 1 of Wang et al. 2021, [doi:10.1155/2021/4496416](https://doi.org/10.1155/2021/4496416) (read in full), which cites them as its ref [20]. That ref [20] is Al-Dirini is *unverified* from the original table. Validation error 4.7 ± 4.4 mm |
| **Then et al. 2007**, Technol Health Care 15:385, [IOS](https://content.iospress.com/articles/technology-and-health-care/thc00473) | **in vivo** gluteal; MRI + MRI-safe indenter; inverse FE; Ogden (slightly compressible) | long-term shear modulus skin/fat **G∞ = 1182 Pa**; muscle G∞ = 1025 Pa | one subject, 7 data points (stated by the same group later) |
| Fougeron et al. 2020, [SAM ENSAM](https://sam.ensam.eu/bitstream/10985/18282/3/IBHGC_JBE_2020_FOUGERON.pdf) | in vivo thigh; freehand ultrasound indentation; Ogden; skin+fat+muscle lumped together | μ = 3.2 ± 1.3 kPa (muscle relaxed), 13.7 ± 6.5 kPa (contracted) *unverified* | not fat alone |
| Comley & Fleck 2010/2012, Int J Impact Eng 46:1, [doi:10.1016/j.ijimpeng.2011.12.009](https://doi.org/10.1016/j.ijimpeng.2011.12.009) | porcine fat; uniaxial compression from quasi-static to 5700 s⁻¹ (SHPB) | one-term Ogden, α = 20, μ = E/3; E ≈ 1 kPa at ~10⁻³ s⁻¹ → μ ≈ 0.33 kPa; E ≈ 3 MPa at ~10³ s⁻¹ *unverified* (secondary, Oxford/JoMMS pages) | **curve shape does not change with rate; only the magnitude does** |
| Sommer et al. 2013, Acta Biomater 9:9036, [doi:10.1016/j.actbio.2013.06.011](https://doi.org/10.1016/j.actbio.2013.06.011) | human abdominal fat, ex vivo; biaxial tension and triaxial shear | anisotropic fibre-reinforced hyperelastic. Read from the plotted data in Lanzl 2025: biaxial Cauchy stress ≈ 0.6 kPa at stretch 1.15 (1:1); shear stress ≈ 0.15–0.22 kPa at γ = 0.5 | nonlinear, anisotropic (septa), viscoelastic; a fit to one load type does not carry over to the others |
| Lanzl et al. 2025 (fit to Sommer data) | microstructural model: neo-Hooke + 3 circular fibre families | c = 0.00947 kPa, k₁ = 0.429 kPa, k₂ = 11.6; ν = 0.499 assumed (κ = 219.2 kPa) | read in full; shear fit too stiff (EEARTH 0.58–0.81) |
| Alkhouli et al. 2013, AJP Endo, [doi:10.1152/ajpendo.00111.2013](https://doi.org/10.1152/ajpendo.00111.2013) | human SAT vs omental fat, 44 surgical patients; tension and relaxation | SAT modulus initial **1.6 ± 0.8 kPa**, final (≤30 % strain) **11.7 ± 6.4 kPa**; omental 2.9 / 32 kPa | nonlinear to 30 % strain; relaxation fits **two exponentials** |
| Sun et al. 2021a/b, Acta Biomater 129:188; JMBBM 113 | human abdominal fat; dynamic compression and simple shear | 3rd-order Ogden + QLV; porcine is stiffer than human | constants not retrieved, *unverified* |
| Geerligs et al. 2008, Biorheology 45:677, [doi:10.3233/BIR-2008-0517](https://doi.org/10.3233/BIR-2008-0517) | porcine fat, rheometer, ≤0.1 % strain | linear **G ≈ 7.5 kPa at 10 rad/s, 37 °C**; power-law frequency dependence; phase angle independent of frequency and temperature | linear range only |
| Porcine MRI confined indentation (W. Sydney, [record](https://researchers.westernsydney.edu.au/en/publications/elastic-and-viscoelastic-properties-of-porcine-subdermal-fat-usin/)) | porcine subdermal fat | neo-Hooke μ = 0.53 ± 0.31 kPa *unverified* | |

**How these fit together.** At small strain, the in vivo gluteal values (1.17 kPa Ogden μ from Al-Dirini; 1.18 kPa G∞ from Then) agree within 2 % even though the methods differ. Ex vivo human tension gives 1.6 kPa initially. Porcine quasi-static gives 0.3–0.5 kPa. Geerligs' 7.5 kPa is a small-strain dynamic modulus at 10 rad/s, so it is higher. The large α (16–20) means fat is very soft at first and then stiffens sharply.

**Rate dependence.** A review reports compression to be rate-independent up to ~10 s⁻¹ and strongly rate-dependent above that (Comley & Fleck; *unverified* secondary). Sitting and walking are below 10 s⁻¹. Drop impacts are above it.

**Near-incompressibility.** No measured bulk modulus for whole fat was found. FE studies use ν = 0.45–0.4999. A heel fat-pad study found ν = 0.49 fits most indentation cases and 0.4999 fits high loads; one ν cannot fit every case ([arXiv:2312.04108](https://arxiv.org/abs/2312.04108)). Acoustic data give a physical upper bound: c ≈ 1450–1480 m/s and ρ ≈ 911–950 kg/m³ ([tables, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4521448/table/Tab3)) → K = ρc² ≈ 2 GPa (*my derivation*). That is about 10⁶ × μ, too stiff for VBD to use directly.

**Large compressive strain in sitting.** Under the IT, peak maximum principal strain is **59–67 %** in Wang 2021 FE models. Mogale et al. 2023 measured large compressions in vivo (see §4).

## 3. Viscoelasticity, damping, jiggle

- **Relaxation:** two-exponential stress relaxation in human SAT (Alkhouli 2013). The time constants were not retrieved (*unverified*). Porcine fat follows a power law (Geerligs 2008), so relaxation has no single time constant; a Prony series with 2–3 terms is the usual approximation.
- **Rate stiffening at impact:** above ~10 s⁻¹, stiffness rises by orders of magnitude, to E ≈ 3 MPa at 10³ s⁻¹ (*unverified*, secondary for Comley & Fleck).
- **Bulging between septa:** no quantitative measurement found. Sommer's anisotropy and the Lanzl fibre model are the nearest mechanical evidence. *Open question.*
- **Jiggle after impact:** soft-tissue packages of the leg vibrate at about **15 Hz** when relaxed; muscle activation mostly changes the *damping*, not the frequency (Nigg-group review, Sports Med Open 2022, [doi:10.1186/s40798-022-00524-w](https://doi.org/10.1186/s40798-022-00524-w)). In heel-toe running, thigh and shank wobbling-mass frequencies range from 3 to 55 Hz in the first 90 ms ([Schmitt & Günther 2011, Arch Appl Mech](https://doi.org/10.1007/s00419-010-0458-z)). Thigh soft tissue can move up to 7 cm (Lafortune, cited by Loughborough, *unverified*). Wobbling-mass models lower joint forces by up to ~50 % in drop landings (Pain & Challis 2006, [doi:10.1016/j.jbiomech.2004.10.036](https://doi.org/10.1016/j.jbiomech.2004.10.036)). **No fat-specific damping ratio was found.** These numbers describe the whole soft-tissue package (muscle + fat + skin).

## 4. Pressed: sitting and posture

| Study | Method | Key numbers |
|---|---|---|
| Linder-Ganz et al. 2007, J Biomech 40:1443, [doi:10.1016/j.jbiomech.2006.06.020](https://doi.org/10.1016/j.jbiomech.2006.06.020) | Open-MRI, 6 healthy subjects sitting vs not; IT sag used as a displacement boundary condition; FE with neo-Hooke (porcine muscle data) | **peak strains and stresses are in the gluteal muscle, not the fat or skin**, 3–5 mm distal to the bone–muscle interface under the IT |
| Al-Dirini et al. 2016 (above) | one subject, MRI-validated FE (28 muscles, intermuscular fat, SAT, skin) | error 4.7 ± 4.4 mm; fat Ogden μ 1.17 kPa, α 16.2 |
| Zappalá et al. 2024 (above) | MR digital volume correlation, 10 healthy subjects, semi-recumbent, progressive loading; reuses the Al-Dirini 2017 data | gluteus slides 13.74 mm; along the IT→seat path, peak compressive/shear strain **in the fat near the fat–muscle interface**; **DVC data released publicly** |
| Mogale et al. 2023, Front Phys 11, [doi:10.3389/fphy.2023.1209538](https://doi.org/10.3389/fphy.2023.1209538) | 26 able-bodied (13 F, BMI 24–25), upright ultrasound + 3 T MRI with a loader in lateral decubitus | compression under the IT: **upright seated US — total 64.2 ± 9.1 %, muscle+tendon 73.1 ± 13.2 %, skin+SAT 40.6 ± 18.3 %**; MRI at 30 % body weight — total 47 ± 7 %, MT 61.5 ± 16 %, SA 16 ± 23 %; at 50 % BW, MT 72 ± 4 % and no further SA reduction. Unloaded composition: MT 70 %, skin+SA 30 % |
| Sonenblum et al. (FONAR upright MRI) RESNA 2015; Sonenblum et al. 2021 Front Bioeng, [doi:10.3389/fbioe.2021.753897](https://doi.org/10.3389/fbioe.2021.753897) | seated on a foam cushion vs an IT-off-loading cushion; 35–43 subjects in the larger studies | bulk thickness under IT **5.6–32.1 mm**, mostly adipose, thickest in able-bodied people (search summary, *unverified*); SCI fat thicknesses of 6, 15 and 30 mm were used as model cases in Wang 2021 |
| Positional MRI seat study, [PubMed 34569437](https://www.pubmed.gov/34569437/) | 4 healthy males, seat-pan angle × foam × shear | deformation under the IT falls with a softer cushion and with less seat shear |
| Wang et al. 2021 (read in full) | FE, fat 6–30 mm under the IT | rigid seat: peak contact pressure 0.149 MPa (6 mm fat) down to 0.052 MPa (30 mm); contact area 0.062 → 0.100 m² (the paper prints "mm²"); soft cushion 15.3 kPa: 0.020 → 0.015 MPa |

**Indentation in vivo:** Then 2007 (gluteal, G∞ 1.18 kPa); Fougeron 2020 (thigh); a 2024 thigh two-layer inverse study reported a fat median μ ≈ 537 kPa (IQR ±1131), which conflicts with everything else by about 500× and shows how method-dependent inverse fits are (*unverified*; search summary).

**Lying vs standing / gravity sag:** no study was found that measures subcutaneous fat displacement between supine and upright. Organ shifts give the order of magnitude: liver ≈ 20 mm inferior from supine to seated; thorax 13–23 mm caudal (search summary, *unverified*). *Open question.*

## 5. Variation and the small figure

- **Thickness:** eight-site mean SAT is 3–10 mm in lean men (BMI < 28.5), with 3.2–12.4 kg total SAT (Störchle et al. 2018, Sci Rep, [doi:10.1038/s41598-018-34213-0](https://doi.org/10.1038/s41598-018-34213-0)). Under the IT: 6–32 mm (Sonenblum).
- **BMI:** Alkhouli found no strong BMI dependence of the modulus (*unverified*). Elastographic density correlates poorly with BMI (ajum.12100). Thicker fat spreads seat load: in Wang 2021 the peak pressure falls about 3× from 6 to 30 mm of fat.
- **Sex / region:** women carry more gluteofemoral fat; no sex difference in stiffness was found in these sources (*unverified*). Abdominal fat is the most tested; no sources for buttock- or thigh-specific ex vivo constants were found.
- **Age:** no fat-specific stiffness-vs-age data found. *Open question.*
- **For a 0.98 m stylised figure** (*my derivation from scaling, not a source*): stiffness constants (μ, α, ν, ρ) are material properties and do not scale with size. Gravity stress scales as ρgL. At about 0.58× human size with the same μ, sitting strains come out **lower** than in an adult. Natural jiggle frequency scales as √(μ/ρ)/L, so it comes out **about 1.7× higher** (~25 Hz from 15 Hz). To match adult *looks*, either accept this or state it openly as a stylisation choice. Do not tune μ by hand (rule: no hard-coding). Fat thickness should come from the figure's own geometry, checked against the ranges above scaled by L.

## 6. Datasets to verify a simulation

| Dataset | Contents | Size | Access / licence | URL |
|---|---|---|---|---|
| **Zappalá 2024 buttock DVC** | 3D displacement / strain fields of the buttock under progressive seated load, from MRI | 10 healthy subjects | stated as a public dataset; paper CC BY; data licence *unverified* | link in the paper: tinyurl.com/36x2npyt; [ORCA 165053](https://orca.cardiff.ac.uk/165053) |
| Mogale 2023 supplementary | US + MRI thickness, loaded and unloaded, skin+SA / MT | 26 + 1 subjects | CC BY; "included in the article/Supplementary Material" | [doi:10.3389/fphy.2023.1209538](https://doi.org/10.3389/fphy.2023.1209538) |
| Al-Dirini 2016/2017 MRI | seated loaded/unloaded MRI, FE model | 1 model subject / the 10 behind Zappalá | request from authors (Flinders/Adelaide) *unverified* | [Adelaide record](https://digital.library.adelaide.edu.au/items/89172083-441f-4edd-9e71-d0cee4d65cab) |
| Sonenblum upright MRI | seated buttock MRI, two cushions | 35–43 subjects | request from authors; mostly wheelchair users (medical, *treat carefully*) | [Front Bioeng 2021](https://doi.org/10.3389/fbioe.2021.753897) |
| Positional MRI seat study | seat angle × foam × shear, IT deformation | 4 subjects | published "for FE validation"; data access *unverified* | [PubMed 34569437](https://www.pubmed.gov/34569437/) |
| **Dyna / Dynamic FAUST** (MPI) | 4D registered meshes of soft-tissue jiggle (running, jumping), 60 fps | ~40 000 scans; 10 subjects (Dyna page) | registration; "research purposes", PS:License 1.0, non-commercial *unverified* | [dfaust.is.tue.mpg.de](https://dfaust.is.tue.mpg.de/), [Dyna](https://ps.is.tuebingen.mpg.de/code/dyna) |
| CAESAR / WEAR | 3D body scans, **standing and seated** poses, 73 landmarks | ~5000 adults 18–65 | paid: WEAR €1095 (€595 for members); CAESAR name owned by SAE | [bodysizeshape.com](https://bodysizeshape.com/page-1855750) |
| UK Biobank imaging | supine Dixon fat/water body MRI (no seated scans) | ~100 000 target | application + fee, approved research | ukbiobank.ac.uk *unverified terms* |

Note: Dyna, DFAUST and CAESAR are surface scans of real people. They should be used only as numerical targets (displacement curves, frequencies), not shown as pictures.

## 7. Recommended E-MAT entry (fat, SAT)

| Parameter | Value | Source |
|---|---|---|
| Model | 1st-order Ogden, nearly incompressible | Al-Dirini 2016; Comley & Fleck 2012 |
| μ (shear, small strain) | **1.17 kPa** (in vivo gluteal); range 0.33–1.6 kPa | Al-Dirini via Wang 2021; Then 2007 (1.18 kPa); Alkhouli 2013; Comley & Fleck |
| α | **16.2** (in vivo); 20 (porcine compression) | Al-Dirini via Wang 2021; Comley & Fleck |
| ν | 0.49 (normal load) to 0.499; physical K ≈ 2 GPa is impractical | arXiv:2312.04108; Lanzl 2025 |
| ρ | **920 kg/m³** (range 911–950) | Wang 2021 Table 1; acoustic tables |
| Superficial vs deep | superficial stiffer than deep; ratio *unknown*, so do **not** invent one. Leave it as one material until a ratio is found | ajum.12100; Lancerotto 2011 |
| Anisotropy | weak, from septa; optional fibre term (Lanzl k₁ = 0.429 kPa, k₂ = 11.6, ex vivo abdominal) | Lanzl 2025; Sommer 2013 |
| Viscoelasticity | quasi-static for strain rates < 10 s⁻¹ (sitting); 2-term Prony series, values *open* | Comley & Fleck; Alkhouli 2013 |
| Impact | rate stiffening above 10 s⁻¹; a soft-tissue package frequency of ~15 Hz (adult) is the target for T8 | Comley & Fleck; review s40798-022-00524-w |

### Dataset → plan test scene

| Test | Check against |
|---|---|
| T4 (indentation stiffness) | Then 2007 gluteal G∞; Fougeron 2020 thigh relaxed/contracted μ |
| T5 (loaded vs hanging) | Mogale 2023 compression % (fat 16–41 % vs muscle 62–73 %) |
| **T7 (paired chair)** | **Zappalá 2024 DVC fields**; Linder-Ganz 2007 (peak strain in muscle 3–5 mm under the IT); gluteus slide 13.7 mm; fat under the IT carries the load (Sonenblum); Al-Dirini 4.7 mm error as the bar |
| **T8 (drop and settle)** | **Dyna/DFAUST** jiggle trajectories (frequency, decay); 15 Hz, 3–55 Hz range |
| T9 (sit clip) | CAESAR/WEAR seated vs standing shape; Wang 2021 contact pressure vs fat thickness |

## 8. Open questions

1. Fat-only damping ratio in vivo: none found. Could Dyna be used to fit one (fat and muscle are mixed in it)?
2. Superficial/deep SAT stiffness ratio: only qualitative so far.
3. Original Al-Dirini 2016 table (the attribution is via Wang 2021); Sun 2021 Ogden + QLV constants; Alkhouli relaxation time constants. These need the full texts.
4. Supine ↔ upright sag of subcutaneous fat: no measurement found.
5. Septal bulging (cellulite-type relief) mechanics: no quantitative source.
6. Scaling a stylised 0.98 m figure: accept the physical result (smaller strain, faster jiggle) or make an explicit, documented stylisation choice. Do not hand-tune μ.
7. Licences for Zappalá DVC data and DFAUST need checking before any download (ask the user).
