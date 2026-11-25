# Running and jump-landing EMG timing: reference values for checking myoLegs activations (r2)

Purpose: we get leg muscle activations from motion (inverse dynamics plus static optimisation in MyoSuite myoLegs) for jogging and a jump-landing. This file collects **cited** EMG timing and relative amplitude to check them against. It also lists the activation-dynamics time constants, the electromechanical delay and open datasets.

Rules used here:
- Every number below comes from the source named next to it. I read the source as an abstract (PubMed), as PMC full text (via NCBI E-utilities), or on the publisher or dataset page.
- When a value exists only in a paper's figures, or I could not reach the full text, the entry says **not found** / **figure only**. No numbers were filled in from memory.
- "% cycle" means % of the stride from ipsilateral foot strike (0%) to the next foot strike (100%), unless noted otherwise.
- Sex is flagged for every population (F = female, M = male).

---

## 1. Context: phase boundaries in running at moderate speed

| Quantity | Value | Condition | Population | Source |
|---|---|---|---|---|
| Stance as % of stride | 30–37% in "normal" running; 46% or 57% in slow "jogging" | treadmill, running 1.25–4.5 m/s | n = 10, sex not stated in abstract | Gazendam & Hof 2007, Gait Posture 25:604–614, doi:10.1016/j.gaitpost.2006.06.013 (abstract) |
| Toe-off | 40% of gait cycle | 3.96 m/s treadmill | 1 M, 65.9 kg | Hamner, Seth & Delp 2010, J Biomech 43:2709, doi:10.1016/j.jbiomech.2010.06.025, PMC2973845 |
| Contact 288 ± 42 ms, swing 452 ± 45 ms, cadence 163 ± 10 steps/min (so stance is about 39% of the stride; this ratio is derived here, not printed in the paper) | 2.65 ± 0.31 m/s treadmill; 82.5% rearfoot strikers | 135 (78 M, 57 F), 30 ± 5 y | Santuz et al. 2018, Front Physiol 9:1509, doi:10.3389/fphys.2018.01509 |
| Support (contact) time 260 ms jogging, 220 ms running, 140 ms sprinting (620 ms walking) | overground | n and sex not in abstract | Mann, Moran & Dougherty 1986, Am J Sports Med 14:501, doi:10.1177/036354658601400614 (abstract) |

Check: at about 3 m/s, foot strike is 0% and toe-off falls at about 35–40%. Anything active from about 85–100% is pre-activation in late swing.

---

## 2. Muscle-by-muscle timing and amplitude in running

### 2a. The only per-muscle numeric table found in open full text: Chumanov et al. 2012

Chumanov, Wille, Michalski & Heiderscheit 2012, *Changes in muscle activation patterns when running step rate is increased*, Gait Posture 36:231–235, doi:10.1016/j.gaitpost.2012.02.023, PMC3387288.

- Population: **45 healthy adults, 20 F / 25 M**, 32.7 ± 15.5 y, 69.5 ± 13.1 kg. Treadmill at preferred speed **2.9 ± 0.5 m/s**, preferred step rate 172.6 ± 8.8 steps/min. Surface EMG at 2000 Hz.
- Values are the **mean EMG in a window divided by the mean EMG over the whole stride** (1.0 = stride average), at the preferred step rate (Table 1). The authors define the loading response as 0% to peak knee flexion (about 15% GC). They also note that 80–90% and 90–100% GC correspond to about 100 and 50 ms before foot contact.
- Windows were chosen per muscle; "—" means the window was not reported for that muscle.

| Muscle | Loading response (0–~15%) | 30–50% (pre-/early swing) | 70–80% (mid-late swing) | 80–90% | 90–100% |
|---|---|---|---|---|---|
| Vastus lateralis | **3.0 ± 0.9** | — | — | 0.5 ± 0.4 | 1.4 ± 0.8 |
| Rectus femoris | **2.3 ± 1.0** | 0.5 ± 0.3 | — | 1.0 ± 0.7 | 0.7 ± 0.4 |
| Tibialis anterior | 1.0 ± 0.5 | 0.8 ± 0.5 | — | **1.5 ± 0.6** | 0.8 ± 0.3 |
| Medial gastrocnemius | **2.0 ± 0.8** | — | — | 0.5 ± 0.5 | 1.0 ± 0.7 |
| Lateral hamstring (BF) | 1.0 ± 0.5 | — | 1.1 ± 0.8 | **2.2 ± 1.0** | 1.6 ± 1.1 |
| Medial hamstrings (ST/SM) | 0.9 ± 0.5 | — | 1.0 ± 0.8 | **2.2 ± 1.0** | 1.7 ± 0.9 |
| Gluteus maximus | **1.9 ± 1.0** | — | — | 0.9 ± 0.6 | 1.7 ± 1.0 |
| Gluteus medius | **2.1 ± 0.9** | — | — | 0.8 ± 0.6 | 1.4 ± 0.7 |

What this says: the knee extensors, gluteals and medial gastrocnemius peak in early stance. The hamstrings peak at 80–90% (terminal swing). Every muscle except RF and TA shows pre-activation rising into the last 10% of swing. Running at a step rate 10% higher than preferred mostly raised late-swing activity, and the loading response did not change.

### 2b. Per-muscle summary (all sources)

Explicit onset and offset percentages for each muscle at 2.5–3.5 m/s (the classic tables by Mann 1986, Montgomery 1994, Reber 1993 and Novacheck 1998, and the Gazendam 2007 and Cappellini 2006 profiles) are **in figures or paywalled full text that I could not reach**. The entries below give what the abstracts and open full texts actually state.

| Muscle | Active phases in running | Relative amplitude | Source(s) |
|---|---|---|---|
| **Vastus lateralis / medialis** | Terminal swing plus loading response. The vasti act together for knee extension in "terminal swing and loading response". Peak in loading response; pre-activation at 90–100%. | LR 3.0× stride mean (VL). VL aEMG exceeds 100% MVC in pre-contact and braking at high speed (elite M). | Montgomery, Pink & Perry 1994, Am J Sports Med 22:272, doi:10.1177/036354659402200220 (abstract; 30 recreational runners, 3 paces, fine-wire, sex not in abstract). Chumanov 2012. Kyröläinen, Avela & Komi 2005, J Sports Sci 23:1101, PMID 16194986 (17 elite M). |
| **Rectus femoris** | Early stance (with the vasti). The abstract also says RF "assisted the iliacus with hip flexion" (swing). Chumanov sampled 30–50% for RF because RF is active around pre-swing and early swing. | LR 2.3×; 30–50% 0.5×; 80–90% 1.0× | Montgomery 1994 (abstract); Chumanov 2012 |
| **Biceps femoris LH** | Mid-to-late swing through early stance; peak 80–90%. "The hamstrings fired primarily to eccentrically control hip flexion." | 80–90% 2.2×, 90–100% 1.6×, LR 1.0× | Montgomery 1994; Chumanov 2012 |
| **Semitendinosus** (medial hamstrings) | Same as BF: peak 80–90%, continuing into foot strike. Hamner 2010 shows ST active around late swing and early stance (figure only). | 80–90% 2.2×, 90–100% 1.7× | Chumanov 2012; Hamner 2010 Fig. 4 (figure only) |
| **Gluteus maximus** | Late swing plus early stance. Mostly quiet in walking, "increases substantially in activity and alters its timing with respect to speed during running". Roles: controls trunk flexion on the stance side and decelerates the swing leg. | LR 1.9×, 90–100% 1.7×. **Sex: females had 40% higher peak and 53% higher mean GMax activation than males at 3.7 m/s** (20 F + 20 M). | Lieberman et al. 2006, J Exp Biol 209:2143, doi:10.1242/jeb.02255 (abstract; numeric timing in figures, full text not reached). Chumanov 2012. Willson, Petrowitz, Butler & Kernozek 2012, Clin Biomech 27:1052, doi:10.1016/j.clinbiomech.2012.08.008 |
| **Gluteus medius** | Late swing plus early stance (loading response). | LR 2.1×, 90–100% 1.4× | Chumanov 2012. Willson et al. 2011, Clin Biomech 26:735, doi:10.1016/j.clinbiomech.2011.02.012 (20 healthy F controls; onset and duration values in full text, not reached) |
| **Adductor magnus** | Its own synergy, "activated during the later stance or swing phases". Montgomery: AM gives pelvic stabilisation while assisting hip flexion and extension. | not quantified | Saito et al. 2018, Sci Rep 8:5979, doi:10.1038/s41598-018-24332-z, PMC5899144 (8 subjects: **7 M, 1 F**, 26.5 ± 6.9 y; 2.5/3.3/4.1 m/s). Montgomery 1994. |
| **Adductor longus** | Grouped with the hamstrings in one synergy "activated before the timing of right-heel contact" (late swing). Mann 1986: AL (and the hip flexors) **not** active during foot descent, contact and midsupport. | not quantified | Saito 2018; Mann 1986 (abstract) |
| **Medial / lateral gastrocnemius** | **86–125% of the stride**, i.e. from 86% of the previous stride through 25% of the current one (late swing plus early stance). This is the "calf group"; walking is different (26–55%). "Little or no activity in the gastrocnemius ... about the time of toe-off." | MG LR 2.0×, 90–100% 1.0× | Gazendam & Hof 2007 (abstract); Mann 1986 (abstract); Chumanov 2012 |
| **Soleus** | Same calf group, 86–125% (Gazendam). Santuz: the propulsion synergy (PL, GM, GL, SOL) is active in late stance. Reber: all posterior muscles peak in midstance (eccentric control of dorsiflexion). | not quantified | Gazendam 2007; Santuz 2018; Reber, Perry & Pink 1993, Am J Sports Med 21(6):805–810 (abstract via search summary; 15 runners, fine-wire, sex not in abstract) |
| **Tibialis anterior** | Swing (early swing and terminal swing pre-activation) plus loading response. Reber 1993: TA "fired above the fatigue threshold for 85% of the time", so it is active most of the cycle. Santuz: early-swing and late-swing synergies contain the dorsiflexors. | 80–90% 1.5×, LR 1.0×, 30–50% 0.8× | Reber 1993; Santuz 2018; Chumanov 2012 |
| **Peroneus longus** | Propulsion synergy, late stance. | not quantified | Santuz 2018 |

Speed effect: the averaged EMG of GMax, VL, BF, gastrocnemius and TA all rose with speed, "especially in the pre-contact and braking phases" (Kyröläinen 2005; 17 elite M middle-distance runners). Montgomery 1994: "faster running paces lead to increased activity".

Synergy-level timing (useful when checking groups of muscles):
- Santuz 2018 (135 people, 57 F): four synergies, each tied to a phase: weight acceptance (knee extensors plus gluteals, early stance), propulsion (plantarflexors, late stance), early swing (dorsiflexors), and late swing / landing preparation (knee flexors plus dorsiflexors). Per-muscle centre-of-activity values are not printed in the text.
- Saito 2018 (2.5–4.1 m/s): quadriceps synergy at heel contact, followed by an MG synergy, then AM and TA synergies in later stance or swing, then an AL + hamstrings synergy before heel contact. Peak timings did not change with speed.
- Cappellini et al. 2006, J Neurophysiol 95:3426, doi:10.1152/jn.00081.2006: 32 muscles, 3–12 km/h, **8 subjects** (n taken from Hamner 2010). Five temporal components, as in walking; one stance component **shifts earlier in running**. Numeric peak timings are **not found** (full text blocked).

---

## 3. Jump-landing / drop-jump pre-activation (time before initial contact, IC)

| Task | Muscle onset before IC | Population | Source |
|---|---|---|---|
| 180° rotational jump landing (onto a force plate) | VM 35.7 ± 11.2 ms; RF 38.1 ± 14.8; VL 47.1 ± 20.0; **BF 115.6 ± 30.5; SM 136.5 ± 47.5** | **10 healthy F**, 23.5 ± 2.5 y, 50.3 kg, 158.5 cm | Bai et al. 2019, Asia-Pac J Sports Med Arthrosc Rehabil Technol 17:16, doi:10.1016/j.asmart.2019.01.001, PMC6700471 |
| 360° rotational jump landing | VM 45.3 ± 17.4; RF 42.4 ± 13.4; VL 48.8 ± 19.2; BF 132.2 ± 46.7; SM 140.7 ± 40.6 ms | same | same |
| Single-leg abrupt landing | Hamstrings about **200 ms** and quadriceps about **80 ms** before IC (as summarised by Bai 2019) | 24 athletes (sex split not in abstract) | Cowling, Steele & McNair 2003, Br J Sports Med 37:126, doi:10.1136/bjsm.37.2.126, PMC1724613 |
| Drop jump, before training | Hip extensors **87.1 ± 63.6 ms**; knee extensors **272.3 ± 113.8 ms** before contact (56.2 and 124.0 ms after 4 weeks of hip training) | **21 F** | Stearns-Reider, Straub & Powers 2021, J Mot Behav 53:750, doi:10.1080/00222895.2020.1863319 (abstract) |
| Single-leg hop / single-leg cross drop landing (healthy controls) | BF −152 ± 5 ms, ST −140 ± 4 ms (hop); BF −150 ± 11 ms (cross drop) | 18 control athletes (sex not in abstract) | Ghislieri et al. 2026, Med Sci Sports Exerc 58:1865, doi:10.1249/MSS.0000000000004010 (abstract) |
| Falls from 0.2–1.0 m (soleus, TA) | Onset of pre-landing EMG is **timed relative to the expected moment of touchdown**: it starts later after take-off from greater heights. SOL and TA co-contract before and after touchdown, and amplitude rises with height. | 6 subjects | Santello & McDonagh 1998, Exp Physiol 83:857, doi:10.1113/expphysiol.1998.sp004165 (abstract; ms values not in abstract) |
| General statement | "Many studies of landings in humans and other animals have found the duration of muscle preactivation to be constant, starting approximately **100 ms** before impact, irrespective of fall duration." | review statement (turkey study) | Konow & Roberts 2024, J Biomech 165:112022, doi:10.1016/j.jbiomech.2024.112022, PMC11342256 |
| Drop landing 30.5 / 45.8 cm, amplitude by sex | Pre-contact (−200 to 0 ms) peak RF: **F 33.6 ± 18.5 vs M 18.7 ± 8.2 %MVIC**. Post-contact GMax peak: **F 69.5 ± 30.2 vs M 98.0 ± 33.4 %MVIC**; mean F 37.5 vs M 53.9 %MVIC. | 13 F, 9 M collegiate athletes | Zazulak et al. 2005, J Orthop Sports Phys Ther 35:292, doi:10.2519/jospt.2005.35.5.292 |
| Drop jump 50 cm, VL | Pre-landing VL activity plus knee angular velocity at touchdown predicts knee SEC stiffness. Onset in ms is not in the abstract. | 9 M | Horita, Komi, Nicol & Kyröläinen 2002, Eur J Appl Physiol 88:76, doi:10.1007/s00421-002-0673-6 |
| Drop jump, triceps surae + VL | Pre-activation "started well before" contact, and its duration depended on the type of stretch exercise. Value not found. | — | Avela, Komi & Santos 1996 (as summarised in search; full text not reached) |

**Gastrocnemius onset in ms before drop-jump contact: not found as a cited number.** Studies use fixed windows instead: e.g. 150 ms before contact for SOL/MG, or 50 ms (Ruan & Li 2010). Konow & Roberts give the generic "about 100 ms". A 2016 meta-analysis of landing onset times (Theisen et al., PLoS One 11:e0155277, PMC4864320) has healthy-control onset values for VL, VM, the hamstrings and MG, but only in forest-plot figures.

Check for the landing simulation: quadriceps should switch on about 40–80 ms before IC (up to about 270 ms in untrained women), and hamstrings about 115–200 ms before IC. Women show relatively more quadriceps (RF) and less GMax activity than men.

---

## 4. Activation dynamics time constants and electromechanical delay

| Item | Value | Source |
|---|---|---|
| **MuJoCo / MyoSuite muscle (what our model actually uses)** | Default `timeconst` = (τ_act 0.01 s, τ_deact 0.04 s). Effective τ = τ_act·(0.5 + 1.5·act) when ctrl > act; τ = τ_deact/(0.5 + 1.5·act) otherwise. Optional `tausmooth` blends the switch. | MuJoCo modeling docs (https://mujoco.readthedocs.io/en/stable/modeling.html), following Millard et al. 2013, J Biomech Eng 135:021005, doi:10.1115/1.4023390 |
| Zajac 1989 / Winters & Woo 1990 (as used by OpenSim's first-order model) | τ_act = 10 ms, τ_deact = 40 ms ("can be assumed") | OpenSim docs, *First-Order Activation Dynamics*, citing Zajac 1989 Crit Rev Biomed Eng 17:359 (PMID 2676342) and Winters & Woo 1990. The Zajac original text was not reached, so ranges quoted elsewhere (e.g. "12–20 / 24–200 ms") are **not verified**. |
| Thelen 2003 | τ_act = 15 ms, τ_deact = 50 ms (young adults); τ_deact raised to 60 ms for older adults | Thelen 2003, J Biomech Eng 125:70, doi:10.1115/1.1531112. **Partly verified:** a reproduced excerpt gives 15/50 ms and 50→60 ms, but one table fragment was garbled. MyoConverter uses 0.015 / 0.05 s as its "OpenSim defaults" for Thelen2003Muscle. |
| Winters 1995 | Activation equations are those adopted by Thelen 2003, with different coefficients. **Numeric values not found.** | Winters 1995, Ann Biomed Eng 23:359, doi:10.1007/BF02584437 |
| Electromechanical delay, voluntary (elbow flexors) | 49.5 ms eccentric, 53.9 ms isometric, 55.5 ms concentric | Cavanagh & Komi 1979, Eur J Appl Physiol 42:159, doi:10.1007/BF00431022 |
| Electromechanical delay, electrically evoked, gastrocnemius medialis | 11.6 ± 1.5 ms (fascicle motion starts at 6.1 ms; the passive series-elastic part is 47.5% of EMD) | Nordez et al. 2009, J Appl Physiol 106:1970, doi:10.1152/japplphysiol.00221.2009 (n = 9) |
| Caveat | Voluntary-contraction EMD measurements "should be interpreted with caution": they include detection artefacts and are longer than the true delay. | Hug, Lacourpaille & Nordez 2011, Muscle Nerve 44:838, doi:10.1002/mus.22139 |

Comparing to EMG: with MuJoCo's 10/40 ms dynamics, simulated activation lags excitation by roughly 10–40 ms, and EMG leads force by about 10–50 ms (EMD). Before comparing onset timing, shift the simulated activation earlier by about 10–50 ms, or shift the EMG later. A static-optimisation solution has no activation dynamics, so its "activation" lines up with force and should lag EMG by about the EMD.

---

## 5. Open datasets with running EMG

| Dataset | Content | Population | Licence / size | Notes |
|---|---|---|---|---|
| **Van Hooren & Meijer 2024**, Data in Brief 54:110312, doi:10.1016/j.dib.2024.110312, PMC11737525. Data: OSF doi:10.17605/OSF.IO/7QBXC (https://osf.io/7qbxc/) | C3D files (markers at 100 Hz, GRF at 1000 Hz, surface EMG of **11 muscles**, Delsys Trigno), plus OpenSim .trc/.mot/.osim/.sto (IK, ID, muscle forces, joint contact) and .xlsx events | **19 (10 M, 9 F)**, 23.6 ± 3.7 y, 67.2 kg | Article CC BY 4.0. Data "can be downloaded without restrictions"; no dataset licence stated. Size not reported. | **Best match:** includes women aged about 20–35, EMG plus full kinematics plus kinetics, level running at **2.78, 3.00, 3.33, 4.00 and 5.00 m/s**, plus slopes ±3/±6° and cadence ±10. The list of the 11 muscles is not stated in the article text. |
| **Santuz et al. 2018** running EMG, Zenodo concept doi:10.5281/zenodo.1254380 (v1.2.0: doi:10.5281/zenodo.1435592) | Raw EMG (1000 Hz) for 13 right-leg muscles (GMed, GMax, TFL, RF, VM, VL, ST, BF, TA, PL, GM, GL, SOL), touchdown/lift-off times from a pressure plate, filtered time-normalised EMG (200 points: 100 stance + 100 swing), NMF synergies, R code | **135 (78 M, 57 F)**, 30 ± 5 y, 69 ± 11 kg; 2.2–3.0 m/s or preferred (mean 2.65 m/s) | **CC BY-SA 4.0**, about 1.4 GB (dataset.rar) | Largest EMG sample and right in our speed band, but **no kinematics**. Fits a timing-envelope check per muscle. A newer Zenodo version exists. |
| **Hamner & Delp 2013**, SimTK project *nmbl_running* (https://simtk.org/home/nmbl_running), paper J Biomech 46:780, PMC3979434 | Motion capture, GRF (instrumented treadmill), raw EMG archive, scaled OpenSim models, CMC results | **10 M** experienced runners; 2, 3, 4, 5 m/s | MIT Use Agreement; size not shown | Male only. The 2010 single-subject study recorded erector spinae, GMax, GMed, ST, VL, MG, SOL and TA; the 2013 muscle list was not verified. |
| Fukuchi et al. 2017, PeerJ 5:e3298, figshare | Kinematics plus kinetics, overground and treadmill running at several speeds | n not checked here | open access (PeerJ); licence not checked | **No EMG** is mentioned in the search summaries; not opened. |
| Riglet et al. 2024, Sci Data, doi:10.1038/s41597-024-03420-y | 63 markers plus GRF, walking and running, C3D/CSV | 30 young adults | open | A ResearchGate abstract mentions 14 EMG muscles, but the Sci Data metadata I saw does not. **EMG presence unverified.** |

No PhysioNet running-EMG dataset turned up in this search.

---

## 6. Gaps and suggested next step

- Per-muscle onset/offset (%) tables at 2.5–3.5 m/s exist in Gazendam & Hof 2007, Cappellini 2006, Montgomery 1994, Reber 1993 and Novacheck 1998. All of these are paywalled or in figures only, so they are **not extracted here**.
- The cleanest way to get female, speed-matched, numeric timing envelopes is to compute them from **Santuz 2018** (57 F, 2.2–3.0 m/s) and **Van Hooren 2024** (9 F, 2.78–3.33 m/s, with kinematics so our ID/SO pipeline can run on the same trials). That is data processing, not literature, so it was not done here.
- No cited gastrocnemius drop-jump onset in ms was found; only the generic "about 100 ms" and fixed analysis windows.
