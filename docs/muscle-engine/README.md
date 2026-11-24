# Muscle engine

An anatomy-based engine for the elf: bones, muscles, fascia, fat and skin simulated together, driven by motion clips.

| document | what it is |
|---|---|
| [e-mat.md](e-mat.md) / [e-mat.json](e-mat.json) | material table for the subject profile (healthy woman ~28), 99 parameters with source populations; rebuilt by `research/muscle_engine/emat_build.py` then `emat_md.py` |
| [plan.md](plan.md) | the engine plan: modules, test scenes, phases, reuse, subject profile, decisions |
| [sit-physics-research-handoff.md](sit-physics-research-handoff.md) | how we got here: the sit-down research and Tests 1–5 (tet winding, block contraction, knee, bone winding, loaded vs hanging) |
| [tissue-research/muscle.md](tissue-research/muscle.md) | skeletal muscle: passive, active, damping, sitting, datasets |
| [tissue-research/fat.md](tissue-research/fat.md) | subcutaneous fat |
| [tissue-research/skin.md](tissue-research/skin.md) | skin |
| [tissue-research/link-muscle-bone-muscle.md](tissue-research/link-muscle-bone-muscle.md) | tendons, attachments, the patella, muscle-to-muscle sliding |
| [tissue-research/link-muscle-fat-fascia.md](tissue-research/link-muscle-fat-fascia.md) | deep fascia and how fat sits on the muscle compartment |
| [tissue-research/link-fat-skin.md](tissue-research/link-fat-skin.md) | retinacula cutis and how skin is bound to fat |

The tissue reports are desk research (2026-11-24). Values marked *unverified* are not used until checked. Their populations are
mixed; the subject profile in [plan.md §2.1](plan.md) decides which values apply.

Scripts: `research/rig_stage/`. Pictures: `docs/img/`.
