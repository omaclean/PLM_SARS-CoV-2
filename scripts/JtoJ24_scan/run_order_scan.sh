#!/usr/bin/env bash
#
# Driver for the J -> J.2.4 ordered-mutation scans.
#
# Runs, in order:
#   1. epistasis_order_scan.py  -- PLM probability shifts at each mutated site,
#                                  for every background genotype and every ordering
#   2. plant_order_scan.py      -- PLANT 3D embedding trajectory for every ordering
#
# Usage:
#   ./run_order_scan.sh [options] [-- <extra args forwarded to both scans>]
#
# Options:
#   --output-root DIR      parent directory for both scans
#                          (default: Results/JtoJ.2.4_scan)
#   --env-epistasis DIR    conda env prefix for the PLM scan (needs fair-esm + torch)
#   --env-plant DIR        conda env prefix for the PLANT scan (needs transformers + torch)
#   --env DIR              use one prefix for both scans
#   --epistasis-only       skip the PLANT scan
#   --plant-only           skip the PLM scan
#   --epistasis-args "..."  extra args for the PLM scan only
#   --plant-args "..."      extra args for the PLANT scan only
#   --dry-run              enumerate and validate only; no model is loaded
#   -h, --help             show this help
#
# Args after `--` go to BOTH scans, so they must be flags both accept. Use
# --plant-args / --epistasis-args for flags only one of them defines, such as
# the PLANT-only --restrict-to-window.
#
# Environments are addressed by absolute interpreter path (PREFIX/bin/python),
# never by `conda activate`: activation does not survive a non-interactive shell
# or an sbatch step, and the two model families cannot share one env anyway --
# fair-esm and EvolutionaryScale esm both install as the module `esm`.
#
# If no prefix is given, the first existing candidate below is used, and failing
# that the `python3` on PATH (with a warning).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUTPUT_ROOT="${REPO_ROOT}/Results/JtoJ.2.4_scan"
ENV_EPISTASIS=""
ENV_PLANT=""
RUN_EPISTASIS=true
RUN_PLANT=true
DRY_RUN=false
EXTRA_ARGS=()
EPISTASIS_ARGS=()
PLANT_ARGS=()

# Searched in order when --env / --env-epistasis / --env-plant are not supplied.
# plm_entropy first: it is the only env here carrying fair-esm 2.0.0, a recent
# transformers, plotly AND seaborn, so it runs both scans. plm_sars has no
# seaborn/sklearn and an older torch, so it only works for the PLANT scan.
# $CONDA_PREFIX is deliberately last -- inside an activated shell it is often the
# conda *base*, whose bin/python exists but has none of the model stack.
CANDIDATE_ENVS=(
  "/home3/oml4h/miniconda3/envs/plm_entropy"
  "${HOME}/miniconda3/envs/plm_entropy"
  "${HOME}/anaconda3/envs/plm_entropy"
  "/home3/oml4h/miniconda3/envs/plm_sars"
  "${HOME}/miniconda3/envs/plm_sars"
  "${HOME}/anaconda3/envs/plm_sars"
  "${CONDA_PREFIX:-}"
)

usage() { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)    OUTPUT_ROOT=$2; shift 2;;
    --env-epistasis)  ENV_EPISTASIS=$2; shift 2;;
    --env-plant)      ENV_PLANT=$2; shift 2;;
    --env)            ENV_EPISTASIS=$2; ENV_PLANT=$2; shift 2;;
    --epistasis-only) RUN_PLANT=false; shift;;
    --plant-only)     RUN_EPISTASIS=false; shift;;
    --epistasis-args) IFS=' ' read -r -a EPISTASIS_ARGS <<< "$2"; shift 2;;
    --plant-args)     IFS=' ' read -r -a PLANT_ARGS <<< "$2"; shift 2;;
    --dry-run)        DRY_RUN=true; shift;;
    -h|--help)        usage;;
    --)               shift; EXTRA_ARGS=("$@"); break;;
    *)                echo "Unknown option: $1 (use -- to forward args to the scans)" >&2; exit 2;;
  esac
done

pick_python() {
  local prefix=$1
  if [[ -n "${prefix}" ]]; then
    if [[ -x "${prefix}/bin/python" ]]; then
      echo "${prefix}/bin/python"; return 0
    fi
    echo "ERROR: no python at ${prefix}/bin/python" >&2
    return 1
  fi
  for candidate in "${CANDIDATE_ENVS[@]}"; do
    if [[ -n "${candidate}" && -x "${candidate}/bin/python" ]]; then
      echo "${candidate}/bin/python"; return 0
    fi
  done
  echo "Warning: no candidate conda env found; falling back to the python3 on PATH." >&2
  command -v python3
}

DRY_FLAG=()
if [[ "${DRY_RUN}" == true ]]; then
  DRY_FLAG=(--dry-run)
fi

mkdir -p "${OUTPUT_ROOT}"

if [[ "${RUN_EPISTASIS}" == true ]]; then
  PY_EPISTASIS="$(pick_python "${ENV_EPISTASIS}")"
  echo "=== Epistasis order scan ==="
  echo "Interpreter: ${PY_EPISTASIS}"
  "${PY_EPISTASIS}" "${SCRIPT_DIR}/epistasis_order_scan.py" \
    --output-dir "${OUTPUT_ROOT}/epistasis" \
    "${DRY_FLAG[@]}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    ${EPISTASIS_ARGS[@]+"${EPISTASIS_ARGS[@]}"}
fi

if [[ "${RUN_PLANT}" == true ]]; then
  PY_PLANT="$(pick_python "${ENV_PLANT}")"
  echo "=== PLANT order scan ==="
  echo "Interpreter: ${PY_PLANT}"
  "${PY_PLANT}" "${SCRIPT_DIR}/plant_order_scan.py" \
    --output-dir "${OUTPUT_ROOT}/plant" \
    "${DRY_FLAG[@]}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    ${PLANT_ARGS[@]+"${PLANT_ARGS[@]}"}
fi

echo "Done. Outputs under ${OUTPUT_ROOT}"
