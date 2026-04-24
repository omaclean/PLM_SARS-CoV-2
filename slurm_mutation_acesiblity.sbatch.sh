#!/bin/bash
#SBATCH --job-name=mut-access-esmc-flu
#SBATCH --output=/home/u6dr/omaclean.u6dr/PLM_SARS-CoV-2/Results/iav_mutational_accessibility/slurm-%j.out
#SBATCH --error=/home/u6dr/omaclean.u6dr/PLM_SARS-CoV-2/Results/iav_mutational_accessibility/slurm-%j.err
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=1:00:00

set -euo pipefail

WORKDIR=/home/u6dr/omaclean.u6dr/PLM_SARS-CoV-2
ENV_PREFIX=/projects/u6dr/OM/envs/plm_entropy
JOB_ID=${SLURM_JOB_ID:-manual}

GUIDE_PATH='/home/u6dr/omaclean.u6dr/PLM_SARS-CoV-2/Sequences/IAV_lineage_guide.csv'
CHECKPOINT_ROOT='/projects/u6dr/OM/PLM_Out/magma_esmc_esmc_600m_1node_FLU/full_job4243186/model'
OUTPUT_DIR='/home/u6dr/omaclean.u6dr/PLM_SARS-CoV-2/Results/iav_mutational_accessibility/esmc_flu_full_job4154992_epochX'
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
echo "log_dir=${LOG_DIR}"
echo "hostname=$(hostname)  nodelist=${SLURM_NODELIST:-manual}"

env | sort > "${LOG_DIR}/environment.txt"

if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job "${SLURM_JOB_ID}" > "${LOG_DIR}/slurm_job.txt" 2>&1 || true
fi

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "conda command not found in PATH" >&2
  exit 1
fi

conda activate "${ENV_PREFIX}"

export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

nvidia-smi > "${LOG_DIR}/nvidia-smi.txt" 2>&1 || true
cat "${LOG_DIR}/nvidia-smi.txt"
python --version | tee "${LOG_DIR}/python_version.txt"
which python | tee "${LOG_DIR}/python_path.txt"
conda info --envs > "${LOG_DIR}/conda_envs.txt" 2>&1 || true

python - <<'PY' | tee "${LOG_DIR}/preflight.txt"
import os
from pathlib import Path

guide_path = Path(os.environ["GUIDE_PATH"])
checkpoint_root = Path(os.environ["CHECKPOINT_ROOT"])
print('guide_exists=', guide_path.exists())
print('checkpoint_root_exists=', checkpoint_root.exists())
if checkpoint_root.exists():
  child_checkpoints = sorted(
    path.name for path in checkpoint_root.iterdir()
    if path.is_dir() and (path / 'model.safetensors').exists()
  )
  print('discovered_checkpoints=', child_checkpoints)
PY

{
  printf 'python %q' "/home/u6dr/omaclean.u6dr/PLM_SARS-CoV-2/scripts/run_mutational_accessibility.py"
  printf ' --analysis-mode %q' "MONTHLY_GUIDE"
  printf ' --guide-path %q' "${GUIDE_PATH}"
  printf ' --mutation-model %q' "H3N2"
  printf ' --output-dir %q' "${OUTPUT_DIR}"
  printf ' --expect-protein-diversity'
  printf ' --model-tag %q' "ESMC_600M_FLU"
  printf ' --base-model %q' "esm-c600m"
  printf ' --model-layer %q' "36"
  printf ' --checkpoint-dir %q' "${CHECKPOINT_ROOT}"
  printf '\n'
} > "${LOG_DIR}/command.txt"

python /home/u6dr/omaclean.u6dr/PLM_SARS-CoV-2/scripts/run_mutational_accessibility.py \
  --analysis-mode MONTHLY_GUIDE \
  --guide-path "${GUIDE_PATH}" \
  --mutation-model H3N2 \
  --output-dir "${OUTPUT_DIR}" \
  --expect-protein-diversity \
  --model-tag ESMC_600M_FLU \
  --base-model esm-c600m \
  --model-layer 36 \
  --checkpoint-dir "${CHECKPOINT_ROOT}" \
  2>&1 | tee -a "${RUN_LOG}"

echo "[$(date -Is)] Mutational accessibility job finished"