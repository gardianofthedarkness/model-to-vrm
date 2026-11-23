# Interior model: experiment code

Scripts behind the results in [docs/interior-model-plan-v7.md](../../docs/interior-model-plan-v7.md) and
[docs/interior-model-plan-v6.md](../../docs/interior-model-plan-v6.md). Each `*_job.py` is a self-contained
`uv` script run as a Hugging Face Job; `launch_*.py` starts it (token from the local vault, passed as a job secret);
`test_*.py` is its local check on made-up data. Data and outputs live in private Hugging Face dataset repositories and
are not in this repository.

| Experiment | Job script | Output folder (private dataset repo) | Result |
|---|---|---|---|
| T0a measurement, thigh and pelvis | `t0a_thigh_job.py`, `t0a_pelvis_job.py` | `t0a/`, `t0a_v2/` | v5.1 handout 11b |
| T0b atlas placed into 409 pelvises | `t0b_pelvis_job.py` | `t0b_pelvis/` | v7 9.1 |
| Thigh round trip and Gaussian fits (dropped) | `t1_thigh_gauss_job.py`, `t1_figs_job.py`, `t1_reload_job.py` | `t1_thigh/` | v6 section 9, v7 9.2 |
| E1 MAISI autoencoder capacity | `e1_maisi_job.py`, `e1_figs_job.py` | `e1/` | v6 12.1 |
| Stage A: surface to brightness code (dropped target) | `stageA_job.py` | `stageA/` | v7 9.3 |
| Encoding of all 1,226 adult scans | `encode_job.py` | `codes_v1/` | v7 section 4 |
| Density within a tissue | `density_job.py` | `density_check/` | v7 9.4 |
| Variance decomposition, cortex thickness | `variance_job.py` | `variance_check/` | training-target proposal |
| Label volumes v1 | `labels_job.py` | `labels_v1/` | v7 3.1 |
| Licensed tissue labels (TotalSegmentator `tissue_4_types`) | `tissue_job.py` | `tissue_v1/` | v7 9.7 |
| **A2: surface to labels, pelvis** | `a2_job.py` | `a2/` | v7 9.6 (passed) |
| Head: MRI labels against CT (HaN-Seg) | `h3_job.py`, `h3b_job.py` | `h3/`, `h3b/` (in the HaN-Seg repo) | v7 9b.4, 9b.6 |
| Attention cost at whole-body token counts | `bench_job.py` | job log | v7 5.5 |
| PINN toy for tissue mechanics | `../pinn/pinn_ring.py`, `pinn_gravity.py` | local | v7 9.8 |

Rules these scripts follow: a smoke run on a small card with a few people before any full run; the card is chosen
from a cost and speed estimate; people are split by the dataset's official split before anything is trained; one
upload per job; no key, licence number or per-person data is written to this repository.

Data: TotalSegmentator CT v2.0.1 (CC BY 4.0), HaN-Seg (CC BY-ND 4.0), IXI (CC BY-SA 3.0). Models used under
research-only terms: NVIDIA MAISI autoencoder, GRACE, TotalSegmentator licensed tasks (academic licence).
