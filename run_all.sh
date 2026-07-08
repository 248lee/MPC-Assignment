#!/usr/bin/env bash
#
# run_all.sh -- evaluate every compatible (env, method) combination
# =================================================================
#
# Derives the full list of runnable (env, method) pairs from the registry
# (registry.compatible_methods), then calls evaluate.py on each. Incompatible
# pairs are never attempted; a pair that fails at runtime (e.g. a prior method
# when torch / sac_lqr.pt is unavailable) is reported but does NOT abort the
# rest of the sweep.
#
# Usage
# -----
#   ./run_all.sh                       # default horizons (1..20), seed 0
#   ./run_all.sh --horizons 1 5 10     # forward flags to evaluate.py
#   ./run_all.sh --force               # recompute even fresh caches
#   PYTHON=python3.14 ./run_all.sh     # override the interpreter
#
# Any extra arguments are forwarded verbatim to each `evaluate.py` call.

set -u
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

# capture forwarded flags before the loop clobbers the positional params
FORWARD=("$@")

# enumerate all compatible (env, method) pairs straight from the registry so
# this script stays correct if envs/methods are added or their compatibility
# changes.
mapfile -t PAIRS < <("$PYTHON" - <<'PY'
from registry import ENV_REGISTRY, compatible_methods
for env in ENV_REGISTRY:
    for method in compatible_methods(env):
        print(env, method)
PY
)

if [ "${#PAIRS[@]}" -eq 0 ]; then
    echo "No (env, method) pairs found -- is registry.py importable?" >&2
    exit 1
fi

total="${#PAIRS[@]}"
ok=0
fail=0
failed_pairs=()

echo "Running $total compatible (env, method) combinations..."
echo "Forwarding extra args to evaluate.py: ${FORWARD[*]}"
echo "======================================================================"

i=0
for pair in "${PAIRS[@]}"; do
    i=$((i + 1))
    env="${pair%% *}"                   # text before the space
    method="${pair##* }"                # text after the space
    echo ""
    echo "[$i/$total] env=$env  method=$method"
    echo "----------------------------------------------------------------------"
    if "$PYTHON" evaluate.py --env "$env" --method "$method" "${FORWARD[@]}"; then
        ok=$((ok + 1))
    else
        fail=$((fail + 1))
        failed_pairs+=("$env/$method")
    fi
done

echo ""
echo "======================================================================"
echo "Done: $ok succeeded, $fail failed (out of $total)."
if [ "$fail" -gt 0 ]; then
    echo "Failed pairs (often just missing torch / sac_lqr.pt for prior methods):"
    for p in "${failed_pairs[@]}"; do
        echo "  - $p"
    done
fi
echo "Results cached under ./results/  (one npz per env+method)."
