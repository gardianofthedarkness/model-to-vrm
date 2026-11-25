# Flesh dynamics, round 0: cited numbers for the leg and hip flesh runtime

Target: healthy adult woman about 28. The runtime has skinning, a muscle bulge, and damped wobble modes per region, driven by bone acceleration; wobble stiffness rises with muscle activation.

Compiled 2026-11-24 from literature only. Every value below is from a paper abstract, table or full text that was actually read, unless it is marked **derived** (my arithmetic from cited values) or **secondary** (quoted by a later paper, original not read).

Access limits:
- The J Appl Physiol originals of Wakeling & Nigg 2001/2002 are not open access. Their absolute numbers are therefore **secondary**, taken from Friesenbichler's 2013 PhD thesis.
- The full texts of Schmitt & Günther 2011, Liu & Nigg 2000, Gruber 1998 and Pain & Challis 2001/2006 were not readable. Only their abstracts were used, plus one secondary parameter table.

Sex flags: **[M]** = male-only, **[mixed]** = both sexes pooled, **[F]** = female-only, **[?]** = sex not reported.

Conventions:
- Most soft-tissue vibration (STV) papers fit `s(t) = a·e^(−c·t)·sin(2π·f·t + φ)`, where c is an amplitude decay rate in s⁻¹.
- The **derived** damping ratio is `ζ = c / sqrt(c² + (2πf)²)`, and the decay time to 1/e amplitude is `τ = 1/c`.
- Some papers (Wakeling, Enders, Martínez) instead fit the decay of signal *power*: `P = P0·e^(−d·t)`. In that case the amplitude rate is about d/2, so these "damping coefficients" are not directly comparable.

---

## Summary tables (recommended values for the woman about 28)

### 1. Wobble modes: frequency, damping, amplitude

| Region / mode | Recommended | Basis (source, population) |
|---|---|---|
| Thigh front (quadriceps), relaxed natural frequency | **8–11 Hz** (rest); **about 14 Hz** in landing impacts | Wakeling & Nigg 2001 (secondary, via Friesenbichler 2013) [mixed 7F/7M]; Zhang 2019 Table 2: 13.8–14.2 Hz [M] |
| Thigh back (hamstrings), frequency | **about 16–20 Hz** in landing | Zhang 2019 Table 2: 16.2 Hz (anticipated), 19.8 Hz (unanticipated) [M] |
| Calf (triceps surae / gastrocnemius), frequency | relaxed **10–20 Hz**; running **about 10.5 Hz** (dominant); search band 7–40 Hz | Wakeling & Nigg 2001 (secondary) [mixed]; Behling 2021 Table 1: 10.5 ± 2.7 Hz [mixed, 15F/17M, 27 y] |
| Thigh damping ratio ζ | **about 0.12–0.16** (quadriceps), **about 0.08–0.11** (hamstrings) | **Derived** from Zhang 2019: c = 10.7–13.9 s⁻¹ (quadriceps), 9.4–11.7 s⁻¹ (hamstrings) [M] |
| Calf damping ratio ζ | **wide spread: 0.1–0.7**. Running gastrocnemius medialis (GM) fits give c ≈ 60 s⁻¹ → ζ ≈ 0.66 (τ ≈ 16 ms) | **Derived** from Behling 2021 [mixed]. The value is model-fit dependent; see notes |
| Thigh decay time τ (1/e amplitude) | **about 70–110 ms** | **Derived** from Zhang 2019 (1/c) [M] |
| Wobble frequency band in heel-toe running (thigh and shank wobbling masses) | **3–55 Hz** overall; skin-marker data **25–40 Hz** for shank and thigh muscles | Schmitt & Günther 2011 abstract [?]; Christensen 2017, citing Günther 2003 [?] |
| Wobble displacement amplitude (centre of mass relative to bone) | **3 mm to 40 mm**; rotation **2.2–11.4°** | Schmitt & Günther 2011 abstract, heel-toe running, first 90 ms [?] |
| Peak soft-tissue acceleration, landing (30 cm) | quadriceps **7.8 g** (anticipated) to 15.7 g (unanticipated); hamstrings 4.4–7.4 g | Zhang 2019 Table 2 [M] |
| Peak wobbling-mass acceleration, running | shank about 320 m/s², thigh about 160 m/s²; peaks lag the bone peak by about 5 ms (shank) and about 20 ms (thigh) | Christensen 2017 Discussion, citing Günther 2003 [?] |
| Buttock / gluteal wobble | **NOT FOUND**. No cited gluteal STV frequency, damping or amplitude data | — |
| Female-specific wobble data | **Only mixed-sex**: Wakeling & Nigg 2001 (7F/7M), Behling 2021 (15F/17M), Friesenbichler 2011 (7F/3M). No sex split was found. Wakeling 2002 reports no effect of sex on muscle tuning (as cited by Martínez 2019) | — |

### 2. Activation dependence (stiffness rises with activation)

| Effect | Value | Source |
|---|---|---|
| Frequency and damping vs muscle force | Both **increase** with force and with shortening velocity. Both are larger normal to the skin than along the segment axis. Also depend on muscle length and on the individual | Wakeling & Nigg 2001, J Appl Physiol, abstract [mixed 7F/7M, n=14] |
| Quadriceps tested at | 0, 50 and 100% maximum voluntary force; knee at 20, 40 and 60° | Wakeling & Nigg 2001, J Biomech, abstract |
| Anticipated vs unanticipated landing (more vs less pre-activation) | Damping c **+30%** (quadriceps 10.67→13.86 s⁻¹) and **+25%** (hamstrings 9.37→11.68 s⁻¹). Frequency not significantly different (quadriceps 14.2→13.8 Hz, hamstrings 19.8→16.2 Hz) | Zhang 2019 Table 2 [M, n=12] |
| Strategy actually used in vivo | Raising damping is reported far more often than shifting frequency | Play 2022 review; Martínez 2019 |
| Isolated muscle, active vs passive eigenfrequencies | 163/265/399 Hz vs 139/215/286 Hz → ratio **1.17–1.40**. **Derived** stiffness ratio (∝ f²) **1.4–1.95** | Christensen 2023 [rat gastrocnemius, ex vivo] |
| Running speed 7.2→19.6 km/h | STV amplitude +144%, frequency +27%, damping +27% | Play 2022 review, citing Boyer & Nigg 2004 and Trama 2019 |
| Fatigue | Vibration intensity and damping coefficient rise; settling time stays constant | Khassetarash 2019 abstract [?, n=11]; Friesenbichler 2011: frequency shifts 2.4–4.8 Hz in a few peaks [mixed 7F/3M] |
| Mode structure | Several modes, all underdamped. Lower modes have about 100% higher damping *ratios* than higher modes (abstract wording) | Khassetarash 2015, J Biomech, abstract [?] |

**Runtime recommendation (my synthesis, not a measured law):**
- Scale the wobble *damping* strongly with activation: c about ×1.25–1.3 from low to high pre-activation.
- Scale *frequency* weakly: Christensen's isolated muscle gives at most ×1.2–1.4. In vivo frequency shifts are small and inconsistent.
- Wakeling & Nigg 2001 give the in vivo direction (both rise). Their magnitudes per %MVC are not available to me.

### 3. Quasi-static bulge (from the delegated literature pass)

| Muscle | Rest → MVC thickness | Population | Source |
|---|---|---|---|
| Rectus femoris | 28.8 → 33.9 mm (**+17.7%**). Reaches +12% at 25% MVC, so it saturates early | [M] n=15, 24 y | Chauhan 2013 Table 1 |
| Rectus femoris | 21.5 → 28.2 mm (**+31%**), knee 60° | [mixed] 20F/20M, 23 y | Şekir 2022 Table 2 |
| Vastus medialis | **+16.5 to +19.5%** at MVC; +15.5% at 30% MVC | [mixed] 23F/10M, 64 y, knee osteoarthritis | Chen 2013 Table 2 |
| Vastus lateralis | 27.1 → 25.7 mm (**−5%**) at 20–30% MVC; pennation 15→21° from rest to MVC | [mixed] 7F/5M; [M] | Hagoort 2021; Chauhan 2013 |
| Gastrocnemius lateralis | 15.2 → 22.0 mm (**+45%**) | [M] n=6, 28 y | Maganaris 1998 Table 2 |
| Soleus | 15.1 → 21.0 mm (**+39%**) | [M] n=6 | Maganaris 1998 Table 3 |
| Gastrocnemius medialis | 17.1 → 16.9 mm (**about 0%**; shape change goes into pennation 22→43°) | [M] n=6; Narici 1996 agrees | Maganaris 1998 Table 1 |
| Thigh / calf girth, relaxed vs contracted | **NOT FOUND** | — | — |
| Hamstrings thickness change | **NOT FOUND** | — | — |

Recommendation:
- **Anterior thigh: +20% thickness at MVC** (range 15–30%), on a saturating curve with about 65% of the change by 25–30% MVC.
- **Lateral thigh: about 0%**.
- **Calf: soleus and gastrocnemius lateralis +40%, gastrocnemius medialis 0%.**
- Volume is conserved: thickness gain is offset by narrowing in the other cross-fibre direction (Randhawa & Wakeling 2018; Eng 2018).
- Therefore girth change should be an *output* of the constant-volume bulge, not a tuned number.

### 4. Fat and skin (from the delegated literature pass)

| Quantity | Recommended (woman about 28, BMI about 21–22) | Basis |
|---|---|---|
| Subcutaneous fat thickness, front thigh | **13–15 mm** | Lackner 2026: 12.8 mm median [F] n=27, 24 y; Cannataro 2025: 14.9 ± 4.9 mm [F] 28–40 y, n=180 |
| Subcutaneous fat thickness, lateral / medial / posterior thigh | **about 13 / about 14 / about 9–10 mm** | Lipometer site ratios front:lateral:rear:inner = 10.3 : 10.4 : 7.2 : 11.2 mm [F] n=21, 24.8 y, BMI 19.9 (Kruschitz 2013). Scaled to front value: **derived** |
| Subcutaneous fat thickness, calf (medial) | **7–8 mm** | Lackner 2026: 8.0 mm [F] |
| Subcutaneous fat thickness, buttock (dorsogluteal) | **20–25 mm** | Joo & Sohng 2010 CT: women 24.9 ± 5.5 mm (n=128); Pak J Med Sci 2019: 20.9 mm normal weight [mixed] |
| Fat shear modulus | **1.2 kPa** quasi-static (confirmed); about 3 kPa at 90 Hz | Then 2007: G∞ = 1182 Pa, gluteal skin + fat, in vivo [?]; Chakouch 2015 MRE: 3.04 kPa [mixed 7F/22M] |
| Fat nonlinearity | One-term Ogden μ ≈ 1.17 kPa, α ≈ 16. Tensile modulus 1.6 → 11.7 kPa by 30% strain | Al-Dirini 2016 (via PMC8654536); Alkhouli 2013 [?] |
| Skin thickness, thigh / buttock | **1.6–1.7 mm** / about 2.4 mm | 2025 QIMS: women thigh 1.67 ± 0.39 mm [F subgroup n=33]; Gibney 2010 buttock 2.4 mm [mixed] |
| Skin (dermis) shear modulus | about 20 kPa (forearm, **derived** from C10 = 9.4 kPa); no thigh-specific value found | Hendriks 2003 [?] |
| Female-specific fat modulus | **NOT FOUND** | — |

### 5. Wobbling-mass fraction

| Quantity | Value | Source |
|---|---|---|
| Liu & Nigg 4-mass running model | Lower body: rigid 6.15 kg + wobbling 6.0 kg → **49% wobbling** (**derived**). Upper body: rigid 12.58 kg + wobbling 50.34 kg → **80% wobbling** (**derived**). k1–k5 = 6, 6, 10, 10, 18 kN/m; c1 = 300, c2 = 650, c4 = 1900 N·s/m | Liu & Nigg 2000, as tabulated by Nikooyan & Zadpoor 2011 and the COMSOL "lumped human body" model (**secondary**) |
| Effect of rigid fraction on loading | +10% rigid mass fraction → peak vertical GRF +0.73 BW. Pain & Challis 2004: +20% bone/soft-tissue ratio → +13% peak impact force | Gittoes, Kerwin & Brewin 2007 (ISBS) [F, 1 subject, 22 y, 69 kg] |
| Female landing wobbling mass model | Shank, thigh and trunk each split into wobbling + rigid masses (subject-specific component inertia model) | Gittoes, Brewin & Kerwin 2005 (ISBS) [F, n=2] — fractions not printed in the paper |
| Per-segment thigh / shank fraction from Gruber 1998, Pain & Challis 2006, Schmitt & Günther 2011 | **NOT OBTAINED** (full texts not accessible). Gruber's fractions were optimised to match GRF (noted in later reviews) | — |

Recommendation:
- Treat the "wobbling" share of thigh and shank as the soft tissue (fat + muscle), i.e. segment mass minus bone. Liu & Nigg's lower-body 49% is a lumped fit, not anatomy.
- Better: take per-segment fat/lean/bone mass from the character's own tissue layers. Holmes 2005 gives DXA regressions for thigh and leg fat, lean and wobbling mass in university-age men and women, but the numbers were not obtained.

---

## Per-source notes

### Soft-tissue vibration

- **Wakeling JM, Nigg BM (2001). Modification of soft tissue vibrations in the leg by muscular activity.** J Appl Physiol 90:412–420. https://doi.org/10.1152/jappl.2001.90.2.412 (PubMed 11160036)
  - Muscles: triceps surae, tibialis anterior, quadriceps.
  - Design: free vibration after a tap, during isometric and isotonic contractions.
  - Frequency and damping rise with force and with shortening velocity. Both are larger normal to the skin.
  - Participants: 14 healthy (7 male, 7 female), per the JSSM Table 2 summary (https://www.jssm.org/tablejssm-10-19.xml/table002.xml).
  - Absolute values (secondary, Friesenbichler 2013 thesis p. 68): relaxed triceps surae 10–20 Hz, quadriceps 8–11 Hz. Overall STV frequency content about 10–50 Hz (thesis p. 39).
  - Full text not open; the per-%MVC values were not obtained.

- **Wakeling JM, Nigg BM (2001). Soft-tissue vibrations in the quadriceps measured with skin mounted transducers.** J Biomech 34:539–543. https://doi.org/10.1016/S0021-9290(00)00203-7
  - Quadriceps at 0/50/100% maximum voluntary force; knee at 20/40/60°.
  - Underdamped. Method resolution: 2% (frequency), 7% (damping).
  - Abstract only.

- **Wakeling JM, Nigg BM, Rozitis AI (2002).** J Appl Physiol 93:1093–1103. https://doi.org/10.1152/japplphysiol.00142.2002
  - n=20, vibration platform 10–65 Hz; quadriceps, hamstrings, tibialis anterior, triceps surae.
  - Muscle activity and damping rise when the input is near each tissue's natural frequency. Natural frequency does not track the input.
  - The JSSM table lists a 10-male cohort at 5 mm amplitude for this paper; Martínez 2019 cites it for no effect of sex on tuning.

- **Wakeling JM, Liphardt AM, Nigg BM (2003).** J Biomech 36:1761–1769. https://doi.org/10.1016/S0021-9290(03)00216-1
  - Walking, n=40. A soft shoe moved the input closer to resonance, but force transfer did not rise, so damping must have increased.
  - The damping rise matched higher biceps femoris and lateral gastrocnemius activity.
  - Soft-tissue mass was estimated from anthropometry. No numbers in the abstract.

- **Zhang S, Fu W, Liu Y (2019).** Changes in lower-limb biomechanics, soft tissue vibrations, and muscle activation during unanticipated bipedal landings. J Hum Kinet. https://doi.org/10.2478/hukin-2019-0003 (PMC6714375)
  - **[M]** 12 male collegiate basketball players, 23.7 y, 70 kg; 30 cm drop.
  - Table 2 (anticipated drop jump / unanticipated drop landing):
    - Peak acceleration: quadriceps 7.82/15.66 g; hamstrings 4.42/7.41 g.
    - Frequency: quadriceps 13.83/14.24 Hz; hamstrings 16.17/19.75 Hz.
    - Damping c: quadriceps 13.86/10.67 s⁻¹; hamstrings 11.68/9.37 s⁻¹.
    - Time to peak: 49–62 ms.
  - Model: `s = a·e^(−ct)·sin(2πf·t+φ)`.
  - Read in full via Europe PMC.

- **Deng L et al. (2021).** Compression garments reduce soft tissue vibrations and muscle activations during drop jumps. Sensors 21:5644. https://doi.org/10.3390/s21165644 (PMC8402353)
  - **[M]** n=12.
  - Damping of quadriceps and hamstrings rises from 30 to 60 cm drop height and with compression shorts. Values are in Fig. 6 only, not tabulated.

- **Behling A-V, Giandolini M, von Tscharner V, Nigg BM (2021).** PLoS ONE 16:e0256296. https://doi.org/10.1371/journal.pone.0256296 (PMC8370632)
  - **[mixed]** 32 analysed (15 F), 27 y (18–49), BMI 23.8; treadmill 8.4 km/h; gastrocnemius medialis.
  - Table 1 (hard / soft shoe): frequency 10.5 ± 2.7 / 10.4 ± 3.0 Hz; c 60.7 ± 21.7 / 57.5 ± 23.1 s⁻¹. Search band 7–40 Hz, c range 5–120 s⁻¹.
  - Within-person change between shoes: Δc −43 to +65 s⁻¹; Δf −3 to +4 Hz.
  - The c is very high compared with the landing data. It comes from a 100 ms window fit, so treat the derived ζ ≈ 0.66 as an upper bound.

- **Martínez A, Lam CK, von Tscharner V, Nigg BM (2019).** Soft tissue vibration dynamics after an unexpected impact. Physiol Rep 7:e13990. https://doi.org/10.14814/phy2.13990 (PMC6339545)
  - **[mixed]** 17M/4F; repeated knee-curl heel strikes; gastrocnemius medialis.
  - Power-decay damping. All 21 participants adapted. Damping on the long axis changed in 18/21 (17 immediately); frequency changed in 15/21.
  - Absolute values are shown in figures only.

- **Friesenbichler B (2013).** PhD thesis, University of Calgary. https://prism.ucalgary.ca/handle/11023/1175
  - Ch. 3, prolonged running (published as Friesenbichler et al. 2011, J Biomech 44:116):
    - **[mixed]** 7F (31.7 y, 60 kg) / 3M; triceps surae.
    - Fatigue raised vibration intensity at many spectral peaks. Frequency shifts of 2.4–4.8 Hz were rare.
  - Ch. 5, whole-body vibration platform at 2.5 mm amplitude (**[M]** n=16):
    - Triceps surae transmissibility 2.33 / 1.88 / 1.23 at 10 / 17 / 28 Hz.
    - Quadriceps transmissibility 2.33 / 1.17 / 0.46 at 10 / 17 / 28 Hz.
    - This is consistent with quadriceps resonance near 10 Hz.

- **Boyer KA, Nigg BM (2005)**, ISB abstract 0998. https://media.isbweb.org/images/conf/2005/abstracts/0998.pdf
  - One subject's natural frequencies: quadriceps 15.23, hamstrings 20.95, triceps surae 22.85 Hz.

- **Boyer & Nigg (2006)** J Biomech 39:645 (https://doi.org/10.1016/j.jbiomech.2005.01.027).
  - The dominant frequency is the same across the quadriceps compartment, but the acceleration peak varies by site, so treat the compartment as one mode.

- **Boyer & Nigg (2007)** J Biomech Eng 129:594 (https://doi.org/10.1115/1.2746384).
  - **[M]** Pendulum heel impacts. Damping changes track EMG.

- **Khassetarash A et al. (2015).** Damping and energy dissipation in soft tissue vibrations during running. J Biomech 48:204–209. https://doi.org/10.1016/j.jbiomech.2014.11.051
  - Gastrocnemius in sprinting; multiple modes (partly ensemble empirical mode decomposition), all underdamped.
  - Damping ratio differs about 100% between lower and higher modes. Absolute ζ is not in the abstract.

- **Khassetarash A et al. (2019).** J Biomech 93:140–146. https://doi.org/10.1016/j.jbiomech.2019.06.026
  - n=11 runners, 4.29 m/s to exhaustion.
  - Damping coefficient, intensity and energy dissipation rise with fatigue; settling time stays constant.

- **Enders H, von Tscharner V, Nigg BM (2014).** J Sci Med Sport 17:218. https://doi.org/10.1016/j.jsams.2013.03.015
  - Gastrocnemius medialis at 3.5 m/s. Shod rearfoot strike had 93% lower (power-decay) damping than shod forefoot strike.
  - Enders 2012 (sprint, n=8): hard surface damping 7.02 s⁻¹ lower than soft (search snippet).

- **Trama R, Blache Y, Hautier C (2019).** J Biomech 82:171. https://doi.org/10.1016/j.jbiomech.2018.10.023
  - n=20, gastrocnemius medialis and vastus lateralis, four running speeds.
  - Higher speed raises all STV power and shifts it to higher frequencies.
  - Trama 2020 (MSSE 52:112): landings and side cuts give up to 3× the input energy of running.

- **Play MC et al. (2022).** Soft tissue vibrations in running: a narrative review. Sports Med Open 8:131. https://doi.org/10.1186/s40798-022-00524-w (PMC9588116)
  - Impact excitation 10–30 Hz; soft tissue about 15 Hz relaxed; bone 200–900 Hz.
  - Speed 7.2→19.6 km/h: amplitude +144%, frequency +27%, damping +27%.
  - Damping increases are reported more than frequency shifts.

- **Schmitt S, Günther M (2011).** Human leg impact: energy dissipation of wobbling masses. Arch Appl Mech 81:887–897. https://doi.org/10.1007/s00419-010-0458-z
  - Heel-toe running, first 90 ms, stance-leg shank and thigh wobbling masses.
  - Abstract (via search snippet): frequencies about 3–55 Hz; centre-of-mass excursions 3 mm–4 cm; rotations 2.2–11.4°.
  - Dissipation, thigh: <50 J horizontal, <15 J vertical. Shank: <10 J horizontal, <5 J vertical. Rotation about ±1 J.
  - Full text (masses, sex) not read.

- **Christensen KB, Günther M, Schmitt S, Siebert T (2017).** Sci Rep 7:13299. https://doi.org/10.1038/s41598-017-13630-7 (PMC5643554)
  - Discussion, citing Günther et al. 2003 (J Mech Med Biol 3:309): human running peak acceleration shank about 320 and thigh about 160 m/s².
  - Muscle-mass acceleration peaks lag the bone peak by about 5 ms (shank) and 20 ms (thigh).
  - Human shank and thigh muscle range 25–40 Hz; scaling from rat predicts about 35 Hz.

- **Christensen KB et al. (2023).** Sci Rep 13:19575. https://doi.org/10.1038/s41598-023-45821-w
  - Rat gastrocnemius ex vivo: active eigenfrequencies 163/265/399 Hz vs passive 139/215/286 Hz.
  - The difference is attributed to cross-bridge attachment. Gives the activation→stiffness direction and size.

- **Gluteal region:** no STV study was found. The closest is pelvic floor accelerometry in 17 women (PubMed 34974336), which does not fit this use. The buttock wobble needs either new data (e.g. a 4D scan of jogging) or the judge loop.

### Wobbling-mass models

- **Liu W, Nigg BM (2000).** J Biomech 33:219–224. https://doi.org/10.1016/S0021-9290(99)00151-7
  - Parameters read from the COMSOL "lumped human body" documentation, which cites Nikooyan & Zadpoor 2011 (Proc IMechE H 225:1121, https://doi.org/10.1177/0954411911424210): https://doc.comsol.com/6.4/doc/com.comsol.help.models.mbd.lumped_human_body/lumped_human_body.html
  - Body mass of the model is about 75 kg, probably male (**[?]**, likely M).

- **Gruber K, Ruder H, Denoth J, Schneider K (1998).** J Biomech 31:439–444. https://doi.org/10.1016/S0021-9290(98)00033-5
  - Abstract: rigid-body inverse dynamics gives "completely incorrect" knee and hip loads during impact. No fractions in the abstract.

- **Pain MTG, Challis JH (2006).** J Biomech 39:119–124. https://doi.org/10.1016/j.jbiomech.2004.10.036
  - 43 cm drop landing; subject-specific wobbling inertia; measured soft-tissue frequency response used as a criterion.
  - Joint loads up to about 50% lower than with a rigid model. Fractions not in the abstract.
  - Pain & Challis 2001 (J Appl Biomech 17:326): lower-leg soft-tissue impact force up to 300 N (search snippet).

- **Gittoes, Brewin & Kerwin (2005, ISBS); Gittoes, Kerwin & Brewin (2007, ISBS).** https://ojs.ub.uni-konstanz.de/cpa/article/view/1201 and https://ojs.ub.uni-konstanz.de/cpa/article/view/478
  - The only female-subject wobbling-mass landing models found (two women, 22 and 24 y).

### Bulge, fat, skin

Full per-source notes are in the two delegated passes. The key references:
- Maganaris 1998, J Physiol 512:603, https://doi.org/10.1111/j.1469-7793.1998.603be.x
- Chauhan 2013, SpringerPlus 2:548, https://doi.org/10.1186/2193-1801-2-548
- Şekir 2022, https://doi.org/10.47447/tjsm.0585
- Hagoort 2021, https://doi.org/10.1186/s12938-021-00967-4
- Narici 1996, https://doi.org/10.1113/jphysiol.1996.sp021685
- Lackner 2026, https://doi.org/10.1113/ep092957
- Cannataro? 2025, https://doi.org/10.1186/s12967-025-06159-1 (first author unconfirmed)
- Kruschitz 2013, https://doi.org/10.1371/journal.pone.0072002
- Joo & Sohng 2010, https://doi.org/10.4040/jkan.2010.40.2.247
- Then 2007, https://doi.org/10.3233/thc-2007-15601
- Chakouch 2015, https://doi.org/10.1371/journal.pone.0138873
- Alkhouli 2013, https://doi.org/10.1152/ajpendo.00111.2013
- Skin thickness 2025 QIMS, https://doi.org/10.21037/qims-2024-2637
- Hendriks 2003, https://doi.org/10.1034/j.1600-0846.2003.00019.x
- Muscle shear-wave elastography in women: Yoshiko 2023 (https://doi.org/10.1210/jendso/bvad032); vastus lateralis passive 9.1 kPa at knee 90° [F]

---

## Gaps to close in round 1

1. Wakeling & Nigg 2001 per-%MVC frequency and damping tables, quadriceps and triceps surae (needs library access).
2. Any gluteal soft-tissue wobble data.
3. Female-only STV frequency and damping.
4. Per-segment wobbling fractions: Holmes 2005 DXA regressions, Gittoes & Kerwin component inertia model, Schmitt & Günther 2011 masses.
5. Girth change relaxed→contracted.
6. Hamstrings bulge.
