# Anatomy sources: atlas replacements and sex differences (research 2026-11-19)

Z-Anatomy (CC BY-SA 4.0) is built on BodyParts3D, which comes from a single adult male. No free female fork exists.

## Learned models that predict the inside of the body from its surface (all SMPL-based)

| model | gives | sexes | data | licence / access |
|---|---|---|---|---|
| [HIT](https://hit.is.tue.mpg.de/) (CVPR 2024), [code](https://github.com/MarilynKeller/HIT) | per point inside the body: fat (adipose) / lean (muscle + organs) / long bones (arms, legs, pelvis); poses and reshapes with SMPL | trained on 157 men + 241 women (MRI) | whole-body MRI | MPI research licence; download page needs an account |
| [SKEL](https://skel.is.tue.mpg.de/) (SIGGRAPH Asia 2023), [code](https://github.com/MarilynKeller/SKEL) | skeleton meshes + anatomical joints from SMPL shape; biomechanical pose | `skel_male.pkl`, `skel_female.pkl` | BioAMASS | non-commercial research; accounts on skel + smpl sites |
| [OSSO](https://osso.is.tue.mpg.de/) (CVPR 2022) | skeleton shape from the body surface (STAR) | learned from thousands of DXA scans | DXA | research |
| [BOSS](https://arxiv.org/abs/2303.04923) (2023) | bones + organs + skin, SMPL-based | - | CT | no code found |

None of them has individual muscles: the Z-Anatomy muscles would still be warped into HIT's lean-tissue region.
Our pipeline fits MHR, not SMPL, so SMPL would first be fitted to the MHR surface (a standard registration).

## Measured sex differences (useful as checks, not as hand rules)
- Skeletal muscle ([Janssen et al. 2000](https://pubmed.ncbi.nlm.nih.gov/10904038/), whole-body MRI, n = 468):
  men 33.0 kg vs women 21.0 kg; 38.4% vs 30.6% of body mass. The gap is larger in the upper body.
- Subcutaneous fat ([Maruyama et al. 1991](https://www.jstage.jst.go.jp/article/ahs1983/10/1/10_1_61/_article/-char/en), ultrasound at 54 sites, 59 men / 66 women):
  women thicker at every site; thicker near the trunk than on the limbs in both sexes. Leg sites show the biggest gap.
- Pelvis: the most sexually dimorphic bone. A lower-limb statistical shape model with sex differences:
  [Audenaert et al. 2019](https://www.frontiersin.org/articles/10.3389/fbioe.2019.00302), 271 CT scans.
- Breasts: volume 57-1672 mL in 111 women ([3D-scan study](https://pubmed.ncbi.nlm.nih.gov/28673203/)).
  The bust-underbust girth difference does not predict volume well.
