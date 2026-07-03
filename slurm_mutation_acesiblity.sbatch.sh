#!/bin/bash
#SBATCH --job-name=mut-access-esmc-flu
#SBATCH --output=/home3/oml4h/PLM_SARS-CoV-2/Results/iav_mutational_accessibility/slurm-%j.out
#SBATCH --error=/home3/oml4h/PLM_SARS-CoV-2/Results/iav_mutational_accessibility/slurm-%j.err
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=1:00:00

set -euo pipefail

REGEN_FIGURES_ONLY=false
FORCE_RECOMPUTE_PLM=false
for arg in "$@"; do
  case "$arg" in
    --regen-figures-only|--regen_figures_only)
      REGEN_FIGURES_ONLY=true
      ;;
    --force-recompute-plm|--force_recompute_plm)
      FORCE_RECOMPUTE_PLM=true
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

WORKDIR=/home3/oml4h/PLM_SARS-CoV-2
ENV_PREFIX=/home3/oml4h/miniconda3/envs/plm_entropy
JOB_ID=${SLURM_JOB_ID:-manual}

GUIDE_PATH='/home3/oml4h/PLM_SARS-CoV-2/Sequences/IAV_lineage_guide.csv'
CHECKPOINT_ROOT='/home3/oml4h/my_SC2_finetunes/myflu/full_job4243186/model/'
OUTPUT_DIR='/home3/oml4h/PLM_SARS-CoV-2/Results/iav_mutational_accessibility/esmc_flu_full_job4154992_epochX'
MODEL_TAG='ESMC_600M_FLU'
BASE_MODEL_NAME='esm-c600m'
MODEL_LAYER=33

# GUIDE_PATH='/home3/oml4h/PLM_SARS-CoV-2/Sequences/IAV_lineage_guide.csv'
# CHECKPOINT_ROOT='/home3/oml4h/my_SC2_finetunes/myflu/fulldatasetESMcMagma/model'
# OUTPUT_DIR='/home3/oml4h/PLM_SARS-CoV-2/Results/iav_mutational_accessibility/esmc_flu_fullrun_magma'
# MODEL_TAG='ESMC_600M_FLU'
# BASE_MODEL_NAME='esm-c600m'
# MODEL_LAYER=33



# ENV_PREFIX=/projects/u6dr/OM/envs/plm_entropy
# GUIDE_PATH='/home3/oml4h/PLM_SARS-CoV-2/Sequences/IAV_lineage_guide.csv'
# CHECKPOINT_ROOT='/home3/oml4h/my_SC2_finetunes/myflu/full_data_esm2_AdamW'
# OUTPUT_DIR='/home3/oml4h/PLM_SARS-CoV-2/Results/iav_mutational_accessibility/esm2_flu_full_AdamW'
# MODEL_TAG='ESM2_600M_FLU_AdamW'
# BASE_MODEL_NAME='esm2_t33_650M_UR50D'
# MODEL_LAYER=33

# ENV_PREFIX=/projects/u6dr/OM/envs/plm_entropy
# GUIDE_PATH='/home3/oml4h/PLM_SARS-CoV-2/Sequences/IAV_lineage_guide.csv'
# CHECKPOINT_ROOT='/home3/oml4h/my_SC2_finetunes/myflu/full_data_esm2_magma'
# OUTPUT_DIR='/home3/oml4h/PLM_SARS-CoV-2/Results/iav_mutational_accessibility/esm2_flu_full_magma'
# MODEL_TAG='ESM2_600M_FLU_magma'
# BASE_MODEL_NAME='esm2_t33_650M_UR50D'
# MODEL_LAYER=33



CHECKPOINT_ROOT='/home3/oml4h/hugging_face_downloads/model_weights_topublish/ESM2-HA80/'
OUTPUT_DIR='/home3/oml4h/PLM_SARS-CoV-2/Results/iav_mutational_accessibility/Lytras_OG'
MODEL_TAG='ESM2_650M_HA80'
BASE_MODEL_NAME='esm2_t33_650M_UR50D'
MODEL_LAYER=33

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
echo "workdir=${WORKDIR}"
echo "guide_path=${GUIDE_PATH}"
echo "checkpoint_root=${CHECKPOINT_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "model_tag=${MODEL_TAG}"
echo "base_model_name=${BASE_MODEL_NAME}"
echo "model_layer=${MODEL_LAYER}"
echo "regen_figures_only=${REGEN_FIGURES_ONLY}"
echo "force_recompute_plm=${FORCE_RECOMPUTE_PLM}"
echo "log_dir=${LOG_DIR}"
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

nvidia-smi > "${LOG_DIR}/nvidia-smi.txt" 2>&1 || true
cat "${LOG_DIR}/nvidia-smi.txt"
"${PYTHON_CMD[@]}" --version | tee "${LOG_DIR}/python_version.txt"
"${PYTHON_CMD[@]}" -c 'import sys; print(sys.executable)' | tee "${LOG_DIR}/python_path.txt"
conda info --envs > "${LOG_DIR}/conda_envs.txt" 2>&1 || true

"${PYTHON_CMD[@]}" - <<'PY' | tee "${LOG_DIR}/preflight.txt"
import os, sys
from pathlib import Path

guide_path = Path(os.environ["GUIDE_PATH"])
checkpoint_root = Path(os.environ["CHECKPOINT_ROOT"])
print('guide_exists=', guide_path.exists())
print('checkpoint_root_exists=', checkpoint_root.exists())

errors = []
if not guide_path.exists():
    errors.append(f"ERROR: guide file not found: {guide_path}")

if not checkpoint_root.exists():
    errors.append(f"ERROR: checkpoint root not found: {checkpoint_root}")
else:
    child_checkpoints = sorted(
        path.name for path in checkpoint_root.iterdir()
        if path.is_dir() and (path / 'model.safetensors').exists()
    )
    direct_weights = (checkpoint_root / 'model.safetensors').exists()
    print('discovered_checkpoints=', child_checkpoints)
    print('direct_model_safetensors=', direct_weights)
    if not child_checkpoints and not direct_weights:
        errors.append(
            f"ERROR: no model.safetensors found in {checkpoint_root} or any of its subdirectories"
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

{
  printf '%q' "${PYTHON_CMD[0]}"
  for arg in "${PYTHON_CMD[@]:1}"; do
    printf ' %q' "$arg"
  done
  printf ' %q' "/home3/oml4h/PLM_SARS-CoV-2/scripts/run_mutational_accessibility.py"
  printf ' --analysis-mode %q' "MONTHLY_GUIDE"
  printf ' --guide-path %q' "${GUIDE_PATH}"
  printf ' --mutation-model %q' "H3N2"
  printf ' --output-dir %q' "${OUTPUT_DIR}"
  printf ' --expect-protein-diversity'
  printf ' --model-tag %q' "${MODEL_TAG}"
  printf ' --base-model %q' "${BASE_MODEL_NAME}"
  printf ' --model-layer %q' "${MODEL_LAYER}"
  printf ' --checkpoint-dir %q' "${CHECKPOINT_ROOT}"
  if [[ "${REGEN_FIGURES_ONLY}" == "true" ]]; then
    printf ' --regen-figures-only'
  fi
  if [[ "${FORCE_RECOMPUTE_PLM}" == "true" ]]; then
    printf ' --force-recompute-plm'
  fi
  printf '\n'
} > "${LOG_DIR}/command.txt"

RUN_ARGS=(
  /home3/oml4h/PLM_SARS-CoV-2/scripts/run_mutational_accessibility.py
  --analysis-mode MONTHLY_GUIDE
  --guide-path "${GUIDE_PATH}"
  --mutation-model H3N2
  --output-dir "${OUTPUT_DIR}"
  --expect-protein-diversity
  --model-tag "${MODEL_TAG}"
  --base-model "${BASE_MODEL_NAME}"
  --model-layer "${MODEL_LAYER}"
  --checkpoint-dir "${CHECKPOINT_ROOT}"
)

if [[ "${REGEN_FIGURES_ONLY}" == "true" ]]; then
  RUN_ARGS+=(--regen-figures-only)
fi

if [[ "${FORCE_RECOMPUTE_PLM}" == "true" ]]; then
  RUN_ARGS+=(--force-recompute-plm)
fi

"${PYTHON_CMD[@]}" "${RUN_ARGS[@]}" 2>&1 | tee -a "${RUN_LOG}"

echo "[$(date -Is)] Mutational accessibility job finished"