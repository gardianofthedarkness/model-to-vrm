# Anatomy pack, format v0 (E-PACK output)

One `.npz` per character and region (e.g. `pack_thigh_l.npz`). SI units, metres, the atlas rest frame.

| key | shape / type | meaning |
|---|---|---|
| `X` | (n, 3) float64 | node positions at rest |
| `T` | (m, 4) int64 | tetrahedra, **positive signed volume** (det[x1−x0, x2−x0, x3−x0] > 0) |
| `region` | (m,) int32 | region id of each tet |
| `region_names` | (r,) str | e.g. `Vastus lateralis muscle.l`, `fat`, `Femur.l` |
| `region_kind` | (r,) str | `bone`, `muscle`, `tendon`, `fascia`, `fat`, `skin` |
| `fib` | (m, 3) float64 | unit fibre direction per tet (zero where not muscle/tendon/fascia) |
| `bone_of_node` | (n,) int32 | for nodes moved kinematically: the bone region id; −1 = simulated |
| `interfaces` | json string | list of `{a, b, law}`; `law` in `tied`, `sliding`, `contact`, `attached` (see plan §5.2) |
| `source` | json string | provenance: inputs (atlas file + hash), tool versions, options, E-MAT version |

Rules checked by T1 (`t1_pack_check.py`):
1. every tet has positive volume, none degenerate (volume > 1e-6 of the median);
2. every region is one connected piece; its boundary is closed and manifold;
3. region volumes against the source surfaces (reported, with the change in %);
4. the surfaces that will be split for `sliding`/`contact` interfaces do not intersect themselves or each other, and keep a gap
   above the solver's contact distance (libuipc sanity check is the reference);
5. the fibre field is unit length in muscle tets;
6. the pack's element counts match what a solver builds from it (the Newton dropped-tet bug of handoff Test 1).
