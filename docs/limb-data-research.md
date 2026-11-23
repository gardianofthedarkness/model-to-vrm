# Limb / whole-body anatomy data: candidate datasets (checked 2026-11-23)

Goal: fill the forearms, hands, shins/lower legs, feet and whole-body proportions that are missing from TotalSegmentator CT v2.0.1.
No data was downloaded, no account was registered and no terms were accepted.
**[UNVERIFIED]** marks a claim I could not confirm from a primary source in this session.

## 1. Comparison table

| Dataset | Modality | Coverage | N | Ages | Labels | Licence / terms | Access | Size | Flags |
|---|---|---|---|---|---|---|---|---|---|
| **VSDFullBody** (Virtual Skeleton Database, SMIR; Zenodo mirror) | Post-mortem CT | "Whole body". Whether the arms and hands are in the field of view is **[UNVERIFIED]** | About 109 archive entries, but some are duplicates or empty files. The true count is **[UNVERIFIED]** | Adults in the 20 subjects of the derived set (52 ± 21 y). Full range **[UNVERIFIED]** | None in this record | CC BY-NC-SA 3.0 CH: no commercial use; derivatives allowed under share-alike | Open Zenodo download | 116.3 GB | **Deceased** (forensic institutes of Bern and Zürich) |
| VSDFullBodyBoneReconstruction (Fischer et al., Sci Data 2023) | CT plus bone masks and meshes | Pelvis and the entire lower body, including tibia, fibula and foot bones | About 20 subjects (10 M, 10 F) | Adults | Lower-body bones | CC BY-NC-SA 4.0 | Open Zenodo download | 30.9 GB | Deceased |
| BoneHub vsd-feet-seg (Hugging Face) | CT plus masks | Feet and ankles | 30 CTs | Adults (from VSD) | 56 bones: distal tibia and fibula, tarsals, metatarsals, phalanges, left and right | CC BY-NC-SA 4.0 | Open download from Hugging Face | Not stated | Deceased |
| **NMDID** (New Mexico Decedent Image Database) | Whole-body CT, soft-tissue and bone kernels, 1 mm slices | Head to toe | 15,249 decedents (2010–17) | **Includes children** (299 under 1 year, 872 aged 1–18) | None (about 60 metadata variables) | Custom DUA. It bans commercial use, redistribution and **derivative works**. Scholarly figures are allowed with the credit line. AI training is not mentioned | Account with an institutional email, then a research request approved by a committee | Over 50 TB | **Deceased; no consent.** The ethics Q&A says deceased people are not human subjects. Faces must not be reconstructed |
| **Visible Human Male/Female** (NLM) | CT, MRI, cryosection | Head to toe | 2 | Adults | None from NLM. Third-party masks exist (DU Digital Commons, lower-extremity bones) | NLM Terms and Conditions (no licence since 2019); acknowledge NLM | Open download, no registration | Large | Deceased donors |
| Visible Korean | CT, MRI, sections plus segmentations | Whole body | A few cadavers | Adults | Organ outlines | KOGL Type 4 per an ETRI page **[UNVERIFIED for KISTI]** | Contact KISTI | ? | Deceased |
| ICRP Publication 110 phantoms | Voxel phantoms | Whole body | 2 (AM, AF) | Adults | 141 organs and tissues | Supplement to the publication; licence **[UNVERIFIED]** | Ships with the publication | Small | Built from 2 real people |
| XCAT (Duke CVIT) | NURBS phantoms | Whole body | Phantom library | Adults and children | Organs, bones, muscles | Paid licence (brain phantom: $400 in year 1); fee for the full XCAT **[UNVERIFIED]** | Licence request | ? | – |
| FDG-PET-CT-Lesions (autoPET) | PET/CT | Mostly skull base to mid-thigh | 900 patients, 1,014 studies | Adults (ages in the CSV) | Tumour lesions only | CC BY 4.0 (defaced images); NIH controlled access for non-defaced images | TCIA Data Retriever | About 419 GB per version | Cancer patients |
| SAROS (TCIA) | CT, 28 source collections | 300 abdomen, 300 thorax and 299 "whole-body" CTs, mostly cut off at the upper legs | 900 CTs, 882 patients | Adults **[UNVERIFIED]** | Sparse labels (every 5th slice): 13 body regions (subcutaneous tissue, muscle, bone and others) and 6 body parts (arms, legs, head, torso) | Labels CC BY 4.0; sources mixed (CC BY 3.0/4.0, CC BY-NC 3.0) | TCIA; some sources restricted | ? | – |
| CADS | CT, 40 sources | Head to knee | 22,022 | Mixed | 167 structures | Mixed per subset (CC BY, CC BY-NC, CC BY-NC-ND, custom) | Gated on Hugging Face (accept terms) | 1.92 TB | Includes TotalSegmentator |
| CT-ORG (TCIA) | CT | Abdomen, neck to pelvis, some whole body | 140 | Mixed | 6 classes: liver, lungs, bladder, kidney, bones, brain | CC BY 3.0 (from an older TCIA page) | TCIA | ? | – |
| TotalSegmentator MRI v2.0.0 | MRI | Mixed clinical; lower-leg coverage **[UNVERIFIED]** | 616 | Mixed | 50 structures (the "total" task) | **CC BY-NC-SA 2.0** | Open Zenodo download | 5.1 GB | – |
| SimTK / Brown carpal database | Wrist CT, released as surfaces | Wrist, distal radius and ulna | 90 healthy volunteers, 120 wrists | Adults **[UNVERIFIED]** | Carpal, radius and ulna surfaces plus kinematics; no soft tissue | "MIT Use Agreement" | SimTK download | ? | CT volumes may not be included |
| OAI (Osteoarthritis Initiative) | Knee MRI and radiographs | Knees | 4,796 | 45–79 | Few public labels | OAI DUA | NIMH Data Archive account (eRA Commons or Login.gov) | Large | – |
| UK Biobank imaging | Whole-body Dixon MRI and DXA | Neck to knee **[UNVERIFIED]** | About 100,000 | 45+ | Derived fat and muscle fields | UKB MTA; same terms for commercial users | Application and fee (imaging tier about £9,000) | Very large | – |
| NAKO | Whole-body MRI | Whole body | 30,000 | Adults | – | **[UNVERIFIED]** | Application | – | – |

There is also a pseudo-labelling tool. TotalSegmentator's `appendicular_bones` model covers the radius, ulna, carpals, metacarpals, phalanges, tibia, fibula, tarsals and metatarsals. Some of its tasks need a licence, which is free for non-commercial use. Whether its training data is public is **[UNVERIFIED]**.

No openly downloadable whole-body MRI with fat and muscle labels was found. The VIBESegmentator weights (NAKO and UK Biobank) segment subcutaneous fat and muscle, but only the weights are published, not the data.

## 2. Best options for forearms, hands, shins, feet and whole-body proportions

1. **VSDFullBody plus its derived sets.** These are the only open whole-body post-mortem CTs with ready-made lower-limb and foot bone labels (20 subjects with lower-body bones; 30 feet with 56 bones). The licence is non-commercial share-alike, which fits an academic project. The data is from deceased people, and arm coverage must be checked in `VSD_Comments.xlsx`.
2. **NMDID.** This is the largest head-to-toe CT set (15k scans) and the only source with proportions at population scale. You would filter to adults with the metadata and pseudo-label the scans with TotalSegmentator `appendicular_bones` and `total`. Caveats: committee approval is needed; the DUA's ban on derivative works may cover trained models, so ask NMDID@unm.edu in writing first; the data is from deceased people collected without consent.
3. **Visible Human Male/Female.** These are free, need no registration, and include CT, MRI and colour cryosections covering hands and feet. Useful as a reference for muscle and fat layers in the limbs, but there are only 2 subjects.
4. **SimTK carpal database.** Wrist bone shapes from 90 living adults. These are surfaces only.
5. **SAROS whole-body subset.** CC BY 4.0 labels of subcutaneous tissue, muscle and bone regions across the body. Useful for the fat-layer and muscle envelope target, though the legs are mostly cut off.

## 3. Not verified
- The count of VSDFullBody subjects, their full age range, and whether arms and hands are included.
- The current per-collection licence on the SAROS TCIA page (the page returned 404).
- The ICRP 110 licence, the full XCAT fee, and the Visible Korean access terms.
- UK Biobank coverage and fee details, and NAKO access.
- The structures covered by TotalSegmentator MRI, and whether the `appendicular_bones` training data is public.
- No open run-off CTA dataset was found. AMOS (abdomen only) was not checked.

## Sources
- https://www.cancerimagingarchive.net/collection/fdg-pet-ct-lesions/
- https://nmdid.unm.edu/
- https://nmdid.unm.edu/resources/data-use
- https://nmdid.unm.edu/docs/ethics_Q_and_A_for_website.pdf
- https://zenodo.org/records/8270365
- https://zenodo.org/records/8302449
- https://huggingface.co/BoneHub/TotalFeetSeg
- https://huggingface.co/datasets/BoneHub/vsd-feet-seg
- https://www.nlm.nih.gov/research/visible/getting_data.html
- https://digitalcommons.du.edu/visiblehuman/
- https://www.nature.com/articles/s41597-024-03337-6 (SAROS)
- https://huggingface.co/datasets/mrmrx/CADS-dataset
- https://arxiv.org/abs/2507.22953
- https://wiki.cancerimagingarchive.net/x/zgJdC (CT-ORG)
- https://zenodo.org/records/14710732 (TotalSegmentator MRI)
- https://github.com/wasserth/TotalSegmentator
- https://simtk.org/projects/carpal-database
- https://www.icru.org/report/joint-icrp-icru-publication-on-adult-reference-computational-phantoms-icrp-publication-110/
- https://cvit.duke.edu/resources/
- https://www.ukbiobank.ac.uk/use-our-data/fees/
- https://datacatalog.med.nyu.edu/dataset/10162 (OAI)
- https://arxiv.org/abs/2406.00125 (VIBESegmentator)
