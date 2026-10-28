#!/usr/bin/env bash
# Launch serum_reconstruction_v5.py model groups in parallel and wait for all.
# Usage: bash run_v5_groups.sh "<group1>" "<group2>" ...   (each group: comma-separated models)
cd "$(dirname "$0")"
PY=${PY:-python}   # e.g. PY=.venv/Scripts/python.exe
mkdir -p Results_v5/logs
pids=()
for g in "$@"; do
  log="Results_v5/logs/$(echo "$g" | tr ',+' '__').log"
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 N_JOBS=4 \
    $PY -u serum_reconstruction_v5.py --models "$g" ${TABPFN_ARGS} > "$log" 2>&1 &
  pids+=($!)
  echo "started [$g] pid $! -> $log"
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "all groups finished (status=$status)"
exit $status
