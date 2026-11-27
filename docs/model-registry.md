# MARROW model registry

**MARROW** is the interior model: from a body's outer shell it generates what lies under it. Every trained model gets
a code here, so runs, scores and files never get mixed up.

## Version code

`MARROW-S<stage>.<release>-<variant>`

- **Stage:** `S0` early experiments before the cascade; `S1` coarse stage (named physics compartments, for games);
  `S2` fine stage (detailed tissue inside S1's compartments, for medicine).
- **Release:** goes up when the data, label set or architecture changes.
- **Variant tags:** `t8` / `t16` token size in mm; `h` hair and cloth augmentation; `k25` training steps in thousands;
  `s0` seed (left out when 0).
- Autoencoders: `MARROW-VAE-S<stage>.<release>-t<mm>`.

Storage: private dataset `hanseg-ct-mr` unless noted; `field_exp/<run tag>/...`.

## Models

| Code | What it is | Data and labels | Architecture | Steps | Run tag / path | Status and key scores (test heads) |
|---|---|---|---|---|---|---|
| MARROW-S0.1 | first full model ("F1"); the elf the user liked | 103 named-label heads (TotalSegmentator 61 + HaN-Seg 42), 64 named structures | conv autoencoder (group norm) + flow transformer, 8 mm | 40k | `totalsegmentator-ct-v201` m16_full/flow_v3 | coherent named structures, blocky; robust to hair on the elf |
| MARROW-VAE-S0.2-t8 | gated transformer VAE for tissue types | HaN-Seg + PDDCA, continuous CT field, 7 tissue types | patch transformer VAE, 8 mm, KL | 12k | `field_exp/ae_check2/tvae_s1` | gate: eye 0.89, air 0.79, thin fat 0.41, bone 0.94; over-calls thin fat about 5.6x by volume |
| MARROW-S0.2-t8-k10 | tissue-type model ("N") | as above | VAE-S0.2-t8 + flow transformer | 10k | `field_exp/suite2_tvae/N_s0`, `N_s1` | eye 0.83, thin fat 0.28, lean 0.81, bone 0.70 (air 0.69 inflated by the scan-edge bug) |
| MARROW-S0.2-t8-k30 | same, longer ("N 30k") | as above | as above | 30k | `field_exp/suite3/NL_s0` | best tissue model: bone 0.74, lean 0.83, eye 0.83, fat under skin 0.59, thin fat 0.29; gains level off after 20k |
| MARROW-S0.2-t8-k10-shell | N + original shell loss ("NS") | as above | + shell loss (wrong target) | 10k | `field_exp/suite2_tvae/NS_s0`, `NS_s1` | discarded: eye 0.08-0.12 |
| MARROW-S0.2-t8-k10-shellfix | N + fixed shell loss ("NST") | as above | + shell loss at zero on the truth, weighted by noise level | 10k | `field_exp/suite3/NST_s0` | parked: eye 0.74, bone 0.62; fewer fat pieces |
| MARROW-S0.2-t8-k10-p | N + SynthRAD masked faces ("NSP") | + 85 SynthRAD partial heads | with original shell loss | 10k | `field_exp/suite2_tvae/NSP_s0`, `NSP_s1` | inconclusive (shell loss confounds) |
| MARROW-VAE-S1.0-t16 | transformer VAE on the 14 S1 compartments | 103 named-label heads (TotalSegmentator 61 + HaN-Seg 42), exact compartment one-hot | patch transformer VAE, 16 mm, KL | 12k | `field_exp/marrow_s1_0_t16/ae_soft` | gate passed: brain 0.99, mandible 0.90, skull 0.89, eyes 0.89, fat layer 0.70 at volume 1.07; weak: tongue 0.37, other bone 0.43; teeth 1.56x and airway 1.49x by volume |
| MARROW-S1.0-t16-k25 | coarse stage, no hair augmentation | as above | VAE-S1.0-t16 + flow transformer | 25k | `field_exp/marrow_s1_0_t16/N_s0` | 15 test heads: brain 0.96, eyes 0.84, skull 0.78, chewing 0.78/0.77, mandible 0.78, tongue 0.74, other soft 0.70, spine 0.64, fat layer 0.61, airway 0.60, teeth 0.53, other bone 0.16. Colab T4, whole suite 2.9 h |
| MARROW-S1.0-t16h-k25 | same with hair and cloth shell augmentation (2 variants per train head) | as above | as above | 25k | `field_exp/marrow_s1_0_t16/NH_s0` | 0.01-0.04 below t16-k25 on clean test heads (brain 0.95, eyes 0.81, skull 0.76, tongue 0.74); its benefit shows only on shells with hair (elf). **Released 2026-11-27 as MARROW-1.0-head** (public, CC BY-NC 4.0): https://huggingface.co/gardianofthedarkness/MARROW-1.0-head; code `release/marrow-1.0-head/` |
| MARROW-VAE-S1.0-t8 | transformer VAE on the 14 S1 compartments, 8 mm | as t16 | patch transformer VAE, 8 mm, KL | 12k | `field_exp/marrow_s1_0_t8/ae_soft` | gate better than t16 everywhere: brain 0.99, skull 0.93, chewing 0.92/0.92, mandible 0.92, eyes 0.91, other soft 0.89, spine 0.89, teeth 0.81, fat layer 0.78, airway 0.77, tongue 0.62, other bone 0.56 |
| MARROW-S1.0-t8-k25 | coarse stage at 8 mm, no hair augmentation | as above | VAE-S1.0-t8 + flow transformer | 25k | `field_exp/marrow_s1_0_t8/N_s0` | 15 test heads: a tie with t16-k25 (brain 0.96, eyes 0.84, chewing 0.79/0.78, skull 0.78, mandible 0.78, other soft 0.71, tongue 0.66, spine 0.64, airway 0.61, fat layer 0.59, teeth 0.52, other bone 0.15); the better VAE does not carry through, the generator is the limit. HF L40S, whole suite 2.2 h |
| MARROW-S1.0-t8h-k25 | same with hair and cloth augmentation | as above | as above | 25k | `field_exp/marrow_s1_0_t8/NH_s0` | brain 0.95, eyes 0.81, chewing 0.78/0.76, skull 0.77, mandible 0.76, tongue 0.69 (missing on 2 heads), other soft 0.68, airway 0.60, spine 0.59, fat layer 0.57, teeth 0.50 |

## Planned

| Code | What | Status |
|---|---|---|
| MARROW-S1.1a | S1.0 + decoder fine-tuned on the generator's own codes | after S1.0 |
| MARROW-S1.1b | S1.0 encoder, decoder and generator fine-tuned together (KL holds the latent space) | after S1.1a |
| MARROW-S2.0 | fine stage: refiner inside S1's compartments, continuous CT field targets | after S1 is accepted |

## S1 compartments (game.py)

outside, fat layer, skull, mandible, teeth, eyes, brain, chewing muscles left, chewing muscles right, tongue, airway,
spine, other bone, other soft tissue.
