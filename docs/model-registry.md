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

## Planned

| Code | What | Status |
|---|---|---|
| MARROW-VAE-S1.0-t16, -t8 | transformer VAE on 14 exact physics compartments | next run |
| MARROW-S1.0-t16-k25, -t16h-k25 | coarse stage, 16 mm tokens, without / with hair and cloth augmentation | next run (first) |
| MARROW-S1.0-t8-k25, -t8h-k25 | same at 8 mm | next run (second) |
| MARROW-S1.1a | S1.0 + decoder fine-tuned on the generator's own codes | after S1.0 |
| MARROW-S1.1b | S1.0 encoder, decoder and generator fine-tuned together (KL holds the latent space) | after S1.1a |
| MARROW-S2.0 | fine stage: refiner inside S1's compartments, continuous CT field targets | after S1 is accepted |

## S1 compartments (game.py)

outside, fat layer, skull, mandible, teeth, eyes, brain, chewing muscles left, chewing muscles right, tongue, airway,
spine, other bone, other soft tissue.
