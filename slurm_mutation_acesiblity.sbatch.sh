#!/bin/bash
# =============================================================================
# Mutational-accessibility evaluation of a fine-tuned PLM on influenza data.
# -----------------------------------------------------------------------------
# Takes a training run's checkpoint as an argument instead of the previous stack
# of commented-out hardcoded blocks, so any finished evotune run can be scored
# without editing this file. By default the results land inside that run's own
# output directory, next to its training logs.
#
#   sbatch slurm_mutation_acesiblity.sbatch.sh --checkpoint-dir DIR [options]
#
#   --raw-only             score the base model alone (no fine-tuned checkpoint)
#   --precomputed FILE     score a precomputed PLM probability matrix instead
#   --checkpoint-dir DIR   A directory containing model.safetensors (HF-Trainer
#                          path) or esmc_weights.pt (FSDP2 and LoRA paths),
#                          e.g. <run>/model/final_checkpoint
#   --output-dir DIR       default: <run_dir>/mutational_accessibility
#   --model-tag TAG        default: derived from the run directory name
#   --base-model NAME      default: inferred from the checkpoint path
#                          (esm-c6b | esm-c600m | esm-c300m | esm2_t33_650M_UR50D)
#   --model-layer N        default: the model's final layer, inferred
#   --guide-path CSV       default: Sequences/IAV_lineage_guide.csv
#   --mutation-model M     SC2 | H1N1 | H3N2                 (default H3N2)
#   --env-prefix DIR       override the auto-selected conda env
#   --test-mode            quick pass over a few targets
#   --regen-figures-only   re-plot from cached results
#   --force-recompute-plm  ignore cached PLM outputs
#   --no-biochem           skip the biochemical control report
#   --biochem-boot N       bootstrap replicates for it (default 200, ~15-25 min)
#   -h | --help
#
# Every run ends by writing a biochemical control report into
# <output-dir>/biochem — the confound ladder, its figures, and a generated
# REPORT.md. It asks whether the model's agreement with codon mutation
# probability survives controlling for biochemistry and for the genetic code.
# It is a post-processing step on the tables this job already wrote, so it
# cannot affect them, and a failure there is reported without failing the job.
#
# Example — score the finished ESM-C 600M flu run:
#   sbatch slurm_mutation_acesiblity.sbatch.sh --checkpoint-dir \
#     /projects/u6dr/OM/PLM_Out/magma_esmc_esmc_600m_1node_FLU/full_job5797361/model/final_checkpoint
# =============================================================================
#SBATCH --job-name=mut-access-flu
#SBATCH --output=/home/u6dr/omaclean.u6dr/PLM_SARS-CoV-2/Results/iav_mutational_accessibility/slurm-%j.out
#SBATCH --error=/home/u6dr/omaclean.u6dr/PLM_SARS-CoV-2/Results/iav_mutational_accessibility/slurm-%j.err
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --time=4:00:00

set -euo pipefail

WORKDIR=/home/u6dr/omaclean.u6dr/PLM_SARS-CoV-2

# Model generations need different environments and cannot share one: ESM-2 uses
# fair-esm, ESM-C uses EvolutionaryScale esm>=3, and both install as the module
# `esm`, so installing one shadows the other. ENV_PREFIX is therefore chosen from
# the base model below rather than fixed here. Override with --env-prefix.
ENV_ESM2=/projects/u6dr/OM/envs/plm_entropy                     # fair-esm 2.0.0
ENV_ESMC=/projects/u6dr/OM/envs/plm_isambardconda_esmc_fsdp2    # esm 3.2.3
ENV_PREFIX=

CHECKPOINT_ROOT=
RAW_ONLY=false
PRECOMPUTED=
OUTPUT_DIR=
MODEL_TAG=
BASE_MODEL_NAME=
MODEL_LAYER=          # empty => model's final layer, resolved at load time
GUIDE_PATH=${WORKDIR}/Sequences/IAV_lineage_guide.csv
MUTATION_MODEL=H3N2
REGEN_FIGURES_ONLY=false
FORCE_RECOMPUTE_PLM=false
TEST_MODE=false
RUN_BIOCHEM=true
BIOCHEM_BOOT=200
BIOCHEM_CODE=/home/u6dr/omaclean.u6dr/plm_entropy/esm-2-finetune_biochem/code

usage() { sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-dir) CHECKPOINT_ROOT=$2; shift 2;;
    --raw-only)       RAW_ONLY=true;      shift;;
    --precomputed)    PRECOMPUTED=$2;     shift 2;;
    --output-dir)     OUTPUT_DIR=$2;      shift 2;;
    --model-tag)      MODEL_TAG=$2;       shift 2;;
    --base-model)     BASE_MODEL_NAME=$2; shift 2;;
    --model-layer)    MODEL_LAYER=$2;     shift 2;;
    --guide-path)     GUIDE_PATH=$2;      shift 2;;
    --mutation-model) MUTATION_MODEL=$2;  shift 2;;
    --env-prefix)     ENV_PREFIX=$2;      shift 2;;
    --test-mode)      TEST_MODE=true;     shift;;
    --no-biochem)     RUN_BIOCHEM=false;  shift;;
    --biochem-boot)   BIOCHEM_BOOT=$2;    shift 2;;
    --biochem-code)   BIOCHEM_CODE=$2;    shift 2;;
    --regen-figures-only|--regen_figures_only) REGEN_FIGURES_ONLY=true; shift;;
    --force-recompute-plm|--force_recompute_plm) FORCE_RECOMPUTE_PLM=true; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1;;
  esac
done

if [[ -z "${CHECKPOINT_ROOT}" && "${RAW_ONLY}" != "true" && -z "${PRECOMPUTED}" ]]; then
  echo "ERROR: give one of --checkpoint-dir, --raw-only or --precomputed." >&2; usage; exit 1
fi
CHECKPOINT_ROOT=${CHECKPOINT_ROOT%/}
if [[ -n "${CHECKPOINT_ROOT}" && ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "ERROR: checkpoint dir does not exist: ${CHECKPOINT_ROOT}" >&2; exit 1
fi
if [[ -n "${PRECOMPUTED}" && ! -f "${PRECOMPUTED}" ]]; then
  echo "ERROR: precomputed matrix not found: ${PRECOMPUTED}" >&2; exit 1
fi

# ── Locate the training run directory that owns this checkpoint ───────────────
# Layout is <run_dir>/model/final_checkpoint (or /model/checkpoint-N), so walk up
# past the checkpoint and the model/ level to find the run root.
RUN_DIR=${CHECKPOINT_ROOT:-${OUTPUT_DIR}}
case "$(basename "${RUN_DIR}")" in
  final_checkpoint|checkpoint-*) RUN_DIR=$(dirname "${RUN_DIR}") ;;
esac
if [[ "$(basename "${RUN_DIR}")" == "model" ]]; then
  RUN_DIR=$(dirname "${RUN_DIR}")
fi
RUN_LABEL=$(basename "${RUN_DIR}")

# ── Infer the base model from the path unless told ────────────────────────────
# The name matters beyond labelling: _load_plm_runtime() selects the ESM-C
# architecture from it. It used to do `"esmc_600m" if "600m" in name else
# "esmc_300m"`, which silently built a 300M model for any other size; it now
# dispatches on 300m/600m/6b explicitly and raises on anything else.
if [[ -z "${BASE_MODEL_NAME}" ]]; then
  lower=$(echo "${CHECKPOINT_ROOT}${PRECOMPUTED}${OUTPUT_DIR}" | tr '[:upper:]' '[:lower:]')
  case "${lower}" in
    *esmc_6b*|*esm-c6b*|*_6b_*)     BASE_MODEL_NAME=esm-c6b ;;
    *esmc_600m*|*600m*|*esm-c600m*) BASE_MODEL_NAME=esm-c600m ;;
    *esmc_300m*|*300m*)             BASE_MODEL_NAME=esm-c300m ;;
    *esm2*|*esm-2*)                 BASE_MODEL_NAME=esm2_t33_650M_UR50D ;;
    *) echo "ERROR: could not infer --base-model from ${CHECKPOINT_ROOT}; pass it explicitly." >&2; exit 1 ;;
  esac
fi
# Pick the environment from the model generation unless one was given.
if [[ -z "${ENV_PREFIX}" ]]; then
  case "$(echo "${BASE_MODEL_NAME}" | tr '[:upper:]' '[:lower:]')" in
    *esmc*|*esm-c*) ENV_PREFIX=${ENV_ESMC} ;;
    *)              ENV_PREFIX=${ENV_ESM2} ;;
  esac
fi
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "ERROR: no python at ${ENV_PREFIX}/bin/python" >&2; exit 1
fi

[[ -z "${MODEL_TAG}" ]] && MODEL_TAG="${RUN_LABEL}"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${RUN_DIR}/mutational_accessibility"

JOB_ID=${SLURM_JOB_ID:-manual}
export GUIDE_PATH CHECKPOINT_ROOT OUTPUT_DIR MODEL_TAG BASE_MODEL_NAME MODEL_LAYER
LOG_DIR=${OUTPUT_DIR}/slurm_logs/${JOB_ID}
STDOUT_LOG=${LOG_DIR}/console.stdout.log
STDERR_LOG=${LOG_DIR}/console.stderr.log
RUN_LOG=${LOG_DIR}/run_mutational_accessibility.log

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

exec > >(tee -a "${STDOUT_LOG}")
exec 2> >(tee -a "${STDERR_LOG}" >&2)

cd "${WORKDIR}"

echo "[$(date -Is)] Mutational accessibility job started"
echo "job_id=${JOB_ID}"
echo "run_dir=${RUN_DIR}"
echo "run_label=${RUN_LABEL}"
echo "checkpoint_root=${CHECKPOINT_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "model_tag=${MODEL_TAG}"
echo "base_model_name=${BASE_MODEL_NAME}"
echo "model_layer=${MODEL_LAYER:-<final layer, inferred>}"
echo "guide_path=${GUIDE_PATH}"
echo "mutation_model=${MUTATION_MODEL}"
echo "test_mode=${TEST_MODE}  regen_figures_only=${REGEN_FIGURES_ONLY}  force_recompute_plm=${FORCE_RECOMPUTE_PLM}"
echo "biochem_report=${RUN_BIOCHEM}  biochem_boot=${BIOCHEM_BOOT}"
echo "hostname=$(hostname)  nodelist=${SLURM_NODELIST:-manual}"

env | sort > "${LOG_DIR}/environment.txt"
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job "${SLURM_JOB_ID}" > "${LOG_DIR}/slurm_job.txt" 2>&1 || true
fi

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  PYTHON_CMD=("${ENV_PREFIX}/bin/python")
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -p "${ENV_PREFIX}" --no-capture-output python)
else
  echo "Warning: ${ENV_PREFIX}/bin/python not found and conda is unavailable; falling back to PATH python" >&2
  PYTHON_CMD=(python)
fi
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export HF_HOME=/projects/u6dr/OM/hf_cache

nvidia-smi > "${LOG_DIR}/nvidia-smi.txt" 2>&1 || true
"${PYTHON_CMD[@]}" --version | tee "${LOG_DIR}/python_version.txt"

"${PYTHON_CMD[@]}" - <<'PY' | tee "${LOG_DIR}/preflight.txt"
import os, sys
from pathlib import Path

guide_path = Path(os.environ["GUIDE_PATH"])
checkpoint_root = Path(os.environ["CHECKPOINT_ROOT"]) if os.environ.get("CHECKPOINT_ROOT") else None
print('guide_exists=', guide_path.exists())
print('checkpoint_root_exists=', checkpoint_root.exists() if checkpoint_root else 'n/a')

errors = []
if not guide_path.exists():
    errors.append(f"ERROR: guide file not found: {guide_path}")
else:
    # Every per-lineage FASTA the guide points at must exist, or the run dies
    # part-way through instead of up front.
    import csv
    missing = []
    with guide_path.open() as fh:
        for row in csv.DictReader(fh):
            for key in ("fasta", "reference"):
                target = (row.get(key) or "").strip()
                if target and not Path(target).exists():
                    missing.append(f"{row.get('month', '?')}:{key}={target}")
    print('guide_rows_with_missing_files=', len(missing))
    if missing:
        errors.append("ERROR: guide references missing files: " + "; ".join(missing[:5]))

if checkpoint_root is None:
    print("no checkpoint supplied (raw-only or precomputed run)")
elif not checkpoint_root.exists():
    errors.append(f"ERROR: checkpoint root not found: {checkpoint_root}")
else:
    # Two names for the same state dict: the HF-Trainer path writes
    # model.safetensors, the FSDP2 path writes esmc_weights.pt, and the LoRA
    # entry point writes esmc_weights.pt after merging its adapters into the
    # base weights. _load_plm_runtime reads both, so either satisfies preflight.
    weight_names = ('model.safetensors', 'esmc_weights.pt')
    def weights_in(directory):
        return [name for name in weight_names if (directory / name).exists()]

    child_checkpoints = sorted(
        path.name for path in checkpoint_root.iterdir()
        if path.is_dir() and weights_in(path)
    )
    direct_weights = weights_in(checkpoint_root)
    print('discovered_checkpoints=', child_checkpoints)
    print('direct_weights=', direct_weights)
    if not child_checkpoints and not direct_weights:
        errors.append(
            f"ERROR: none of {' / '.join(weight_names)} found in {checkpoint_root} "
            "or any of its subdirectories."
        )

for msg in errors:
    print(msg, file=sys.stderr)
if errors:
    sys.exit(1)
PY
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  echo "ERROR: preflight checks failed — see ${LOG_DIR}/preflight.txt" >&2
  exit 1
fi

RUN_ARGS=(
  "${WORKDIR}/scripts/run_mutational_accessibility.py"
  --analysis-mode MONTHLY_GUIDE
  --guide-path "${GUIDE_PATH}"
  --mutation-model "${MUTATION_MODEL}"
  --output-dir "${OUTPUT_DIR}"
  --expect-protein-diversity
  --base-model "${BASE_MODEL_NAME}"
)
[[ -n "${CHECKPOINT_ROOT}" ]] && RUN_ARGS+=(--checkpoint-dir "${CHECKPOINT_ROOT}")
# --model-tag and --precomputed-plm-path are mutually exclusive in the analysis
# script's argparse, so only one may be passed. With a precomputed matrix the
# label falls back to the matrix filename stem.
if [[ -n "${PRECOMPUTED}" ]]; then
  RUN_ARGS+=(--precomputed-plm-path "${PRECOMPUTED}")
else
  RUN_ARGS+=(--model-tag "${MODEL_TAG}")
fi
# Only pin the layer if asked; otherwise the analysis resolves the final layer
# from the loaded model, which differs per size (33 / 30 / 36 / 80).
[[ -n "${MODEL_LAYER}" ]] && RUN_ARGS+=(--model-layer "${MODEL_LAYER}")
[[ "${TEST_MODE}" == "true" ]] && RUN_ARGS+=(--test-mode)
[[ "${REGEN_FIGURES_ONLY}" == "true" ]] && RUN_ARGS+=(--regen-figures-only)
[[ "${FORCE_RECOMPUTE_PLM}" == "true" ]] && RUN_ARGS+=(--force-recompute-plm)

{
  printf '%q' "${PYTHON_CMD[0]}"
  for arg in "${PYTHON_CMD[@]:1}" "${RUN_ARGS[@]}"; do printf ' %q' "$arg"; done
  printf '\n'
} > "${LOG_DIR}/command.txt"
cat "${LOG_DIR}/command.txt"

"${PYTHON_CMD[@]}" "${RUN_ARGS[@]}" 2>&1 | tee -a "${RUN_LOG}"

# ── Biochemical control report ────────────────────────────────────────────────
# Post-processing over the tables written above: the confound ladder, its two
# figures and a generated REPORT.md, into ${OUTPUT_DIR}/biochem. Deliberately
# non-fatal — the accessibility results are the deliverable and are already on
# disk by this point, so a plotting or dependency failure here must not mark a
# successful scoring run as failed. --test-mode runs are skipped: they score a
# handful of targets, and a ladder built on those would be noise wearing a
# report's clothing.
BIOCHEM_LOG=${LOG_DIR}/biochem_report.log
if [[ "${RUN_BIOCHEM}" != "true" ]]; then
  echo "[$(date -Is)] biochemical report skipped (--no-biochem)"
elif [[ "${TEST_MODE}" == "true" ]]; then
  echo "[$(date -Is)] biochemical report skipped (--test-mode: too few targets to control)"
elif [[ ! -f "${BIOCHEM_CODE}/run_report.py" ]]; then
  echo "[$(date -Is)] WARNING: biochemical report skipped, no run_report.py at ${BIOCHEM_CODE}" >&2
else
  echo "[$(date -Is)] biochemical control report (${BIOCHEM_BOOT} bootstrap replicates)"
  set +e
  "${PYTHON_CMD[@]}" "${BIOCHEM_CODE}/run_report.py" \
    --run-dir "${OUTPUT_DIR}" --label "${MODEL_TAG}" --boot "${BIOCHEM_BOOT}" \
    2>&1 | tee -a "${BIOCHEM_LOG}"
  BIOCHEM_RC=${PIPESTATUS[0]}
  set -e
  case "${BIOCHEM_RC}" in
    0) echo "[$(date -Is)] biochemical report in ${OUTPUT_DIR}/biochem/REPORT.md" ;;
    2) echo "[$(date -Is)] biochemical report had nothing to analyse (no combined_long_table.csv)" ;;
    *) echo "[$(date -Is)] WARNING: biochemical report failed (rc=${BIOCHEM_RC}) — see ${BIOCHEM_LOG}." \
            "The mutational-accessibility results above are unaffected." >&2 ;;
  esac
fi

echo "[$(date -Is)] Mutational accessibility job finished"
echo "[$(date -Is)] results in ${OUTPUT_DIR}"
