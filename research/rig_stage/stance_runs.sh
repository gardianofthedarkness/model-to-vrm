#!/bin/sh
# Same bones (ARDY sit clip frame 56, standing), each thigh muscle at its MyoSuite activation for the left foot carrying the
# body ("loaded") and for the leg hanging ("hanging") - myo_single_leg.py.  Activation 0 runs are shared.  Quasi-static,
# gravity, bone contact, ACTIVE=stiff.  Two runs at a time on the GPU.
#   sh stance_runs.sh <python> <sim dir> <leg_transforms npz> <fits dir> <myo_single_leg.json>
PY=$1; S=$2; TRF=$3; FD=$4; MJ=$5; OUT=$S/stance; mkdir -p $OUT
list=$($PY -c "
import json; J=json.load(open('$MJ'))['cases']; seen=set()
for m in J['loaded']['activation']:
    if m.startswith('Sartorius'): continue      # its tet mesh overfills its surface by 60 %; activation ~0 in both cases
    for c in ('loaded','hanging'):
        a=J[c]['activation'][m]; tag=m.replace(' ','_').replace('.','_')+'_a%.4f'%a
        if tag in seen: continue
        seen.add(tag); print(m.replace(' ','_').replace('.','_')+'|'+'%.4f'%a)
")
run() { m=$(echo $1 | cut -d'|' -f1); a=$(echo $1 | cut -d'|' -f2); mesh=$S/mesh_$(echo $m | sed 's/_$//').npz
  ACTIVE=stiff STATIC=1 GRAVITY=1 CONTACT=1 HOLD=56 DUR=3 SAVE_EVERY=3 EXC="0.3:$a" $PY newton_muscle_active.py $mesh $TRF $FD $OUT/${m}_a$a.npz > $OUT/${m}_a$a.log 2>&1; echo "done $m $a"; }
i=0; for item in $list; do run "$item" & i=$((i+1)); if [ $((i % 2)) -eq 0 ]; then wait; fi; done; wait; echo ALL DONE
