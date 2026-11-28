# Head data sources: what each has and what it lacks

Checked facts for every head source we hold or considered, for MARROW (shell to interior). Updated 2026-11-28.
Real scans stay in private repos; nothing here is per-person beyond counts.

## Summary

| Source | Heads | Modality | Face | Labels we have | Licence | Use in MARROW |
|---|---|---|---|---|---|---|
| HaN-Seg | 42 | CT + T1 MRI (radiotherapy) | intact | named compartments (face_v1, S1) + continuous CT field | CC BY-ND (private only) | S1.0 training and test; MRI labeller training |
| TotalSegmentator v2.0.1 | 61 complete heads | CT | intact | named compartments (S1) | CC BY 4.0 | S1.0 training and test |
| PDDCA | 48 | CT | intact (checked 48 of 48) | tissue types + eyes only; 9 expert organ masks | CC BY 3.0 | tissue model (S0.2); **not yet in S1** |
| SynthRAD2025 head-and-neck | 156 downloadable pairs (128 placed) | CT + T1 MRI | **removed** (wedge) | tissue types + eyes, face unknown | CC BY-NC 4.0 | partial heads only (NSP, inconclusive) |
| IXI | 581 (563 adults prepared) | T1 MRI (also T2, PD) | intact, chin at the edge | none (MRI labeller not yet passing) | CC BY-SA 3.0 | waiting on the labeller |
| CQ500 | 491 | CT | upper head only | none | CC BY-NC-SA 4.0 | upper head only, after a coverage fact |

## Per source

### HaN-Seg (private copy hanseg-ct-mr)
- **Has:** 42 radiotherapy head-and-neck CTs, each with a T1 MRI of the same person. Faces intact. Named compartments and
  the continuous CT field in the head frame. Our MR-to-CT registration (mri_game_job stage1) succeeds on all 42.
- **Lacks:** the top of the head is cut by the MRI field of view in 17 of 42 (ixi_norm_job reference). Dental metal streaks
  in some CTs (marked unknown). Licence forbids derivatives being shared as data, so private only.

### TotalSegmentator CT v2.0.1 (private copy totalsegmentator-ct-v201)
- **Has:** 61 heads complete under the strict completeness rule, CT, faces intact, named compartments from TotalSegmentator's
  head tasks. Whole-body CTs, so the neck and trunk below are present.
- **Lacks:** no MRI. Raw CTs for the continuous field not yet re-fetched for all 61.

### PDDCA (private copy pddca-ct; MedVision "pddca-lite" mirror of PDDCA v1.4.1, TCIA Head-Neck Cetuximab)
Checked 2026-11-28 on all 48 raw CTs (mid-sagittal views, values; private).
- **Has:**
  - 48 radiotherapy head-and-neck CTs, all with face, nose, lips and chin present. No defacing: the -1024 minimum
    occurs only as scattered noise in the air and outside the reconstruction circle, never as a block over the face.
  - Voxels about 1.1 x 1.1 x 3 mm (in-plane 0.98-1.27 mm, slices 3 mm). Body extent median 300 mm front-back,
    395 mm head to foot (228-582 mm); several reach the chest.
  - Expert organ masks (the "masks" of the dataset): brainstem, both parotids, both optic nerves, chiasm in 48 of 48;
    mandible in 40; left submandibular gland in 41, right in 36. Craniofacial landmarks for 33 heads (in MedVision).
  - Head frame placed for 48 of 48 (synthrad_face_job SOURCE=pddca), tissue types and eyes (eye median 18.9 ml).
- **Lacks:**
  - Named S1 compartments (skull, teeth, chewing muscles, tongue...): only the tissue-type labelling was run, which is
    why S1.0 used 103 heads, not 151. Adding PDDCA to S1 needs the same TotalSegmentator head tasks as HaN-Seg.
  - Top of the skull cut in a few heads (by eye: about 3-5 of 48).
  - 3 mm slices (as SynthRAD; HaN-Seg is 2 mm).
  - Dental metal streaks in some heads; a thin outline near the skin in a few views may be a radiotherapy immobilisation mask (not confirmed).
  - No MRI.
- **Bonus:** the 40 expert mandibles give an independent check of our mandible labels.

### SynthRAD2025 Task 1 head-and-neck (private copy synthrad2025)
Facts from the dataset paper (arXiv 2502.17609, Med Phys 2025) and our own check (plan v8 14.6).
- **Has:** CT + T1 MRI pairs, centres A (91 train) and C (65 train) open on Zenodo (record 15373853). Rigid
  registration (Elastix). Body mask. 128 of 156 placed in the head frame: centre A 85 complete, centre C 43 with the top
  of the skull cut.
- **Lacks:**
  - **The face, in CT and MRI.** Defaced on purpose: everything in front of the line from the front of the brain to the
    jaw corner (on the mid-sagittal slice, across the full width) is set to -1024 (CT) and 0 (MRI): nose, lips, chin and
    the front of the eyes. Our census missed it first; the eye volume (8.8 against 18.6 ml) gave it away.
  - Resampled to 1 x 1 x 3 mm. No deformable CT for training pairs. MRI likely contrast-enhanced (table garbled).
  - Centre D (65 + 10) came with a challenge-only licence and must not be used. Validation and test CTs are embargoed
    until 2030.
- **Use:** partial heads (face unknown); MRI-CT pairs for the MRI labeller (brain, skull, spine), not for face parts.

### IXI (private copy ixi-t1-t2)
- **Has:** 581 healthy volunteers' head MRIs (T1, T2, PD), three London sites (Guy's, Hammersmith, IOP) with different
  scanners; faces intact; 563 adults prepared at 1.5 mm (ixi_prep_v1) and brightness-normalised (ixi_norm_v1).
- **Lacks:** any labels; CT. Chin often at the scan edge. Bone, teeth and air are all dark on T1. Its brain-research
  contrast differs from the radiotherapy MRIs our labeller learned (labeller fails on brain in some HH and Guy's heads).
- **Licence:** CC BY-SA 3.0: a public model trained on it may have to be share-alike.

### CQ500 (private copy cq500-ct)
- **Has:** 491 head CTs, faces intact. **Lacks:** the jaw (only about the top 148 mm in view; 0 of 20 sampled with a whole
  jaw); labels. Feeding cut heads without a coverage fact hurt the lower face (test 2).

### Considered, not used
- **Defaced:** SynthRAD2023 brain, an open FDG PET/CT copy, the 120-person head CT-MRI set on Zenodo.
- **Face clipped / unsuitable:** VSDFullBody (post-mortem, 212-272 mm head field of view), MUG500+ (skull only),
  ToothFairy (dental CBCT, jaws only), HNTS-MRG.
- **Need an application or agreement:** TCIA head-and-neck collections (NIH controlled access since April 2025; likely
  ten times our full-face count), SegRap2023, HECKTOR, NMDID (minors included, derivative ban), OASIS-3.

## Full-face count
CT with named compartments today: 103 (HaN-Seg 42 + TotalSegmentator 61). With PDDCA labelled: 151.
