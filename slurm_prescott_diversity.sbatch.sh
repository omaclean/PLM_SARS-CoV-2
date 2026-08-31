#!/bin/bash
# =============================================================================
# ESCOTT/PRESCOTT diversity prediction for influenza A H3N2 HA.
# -----------------------------------------------------------------------------
# The ESCOTT-flavoured sibling of slurm_mutation_acesiblity.sbatch.sh. Same guide
# file, same codon mutational-accessibility model, same alpha sweep and the same
# output tables -- but the per-site amino-acid preference comes from ESCOTT
# (optionally frequency-modified into PRESCOTT) instead of a protein language
# model, so no GPU is needed and no checkpoint is scored.
#
#   sbatch slurm_prescott_diversity.sbatch.sh --output-dir DIR [options]
#
#   --output-dir DIR       default: Results/iav_prescott_diversity/<date>
#   --guide-path CSV       default: Sequences/IAV_lineage_guide.csv
#   --mutation-model M     SC2 | H1N1 | H3N2                 (default H3N2)
#   --structure PATH       structure behind the JET2 surrogate
#                          (default Sequences/6WXB-assembly1.cif)
#   --structure-role R     primary | extra                   (default primary)
#                          'extra' scores against the contemporary full-coverage
#                          J.2.4.1 model instead of 1968 6WXB.
#   --weight-mode W        structural | tjet                 (default structural)
#   --pc-mode P            interface_propensity | constant | zero
#   --sasa-context S       trimer | monomer                  (default trimer)
#   --parent-map-preset X  clade_evidence | brief_as_stated  (default clade_evidence)
#                          clade_evidence puts K under J.2.4; brief_as_stated
#                          restores the project brief's K <- J.2_int.
#   --parent-map STR       'child=parent,...' overrides on top of the preset
#   --parent-sensitivity / --no-parent-sensitivity
#                          also score the contested edge under the OTHER preset's
#                          parent, as an extra _parent<TOK> model row  (default on)
#   --coefficient-grid G   PRESCOTT coefficients c           (default 0.25,0.5,1.0)
#   --equation-grid G      PRESCOTT equations, 1/2/3/5       (default 2)
#   --frequency-cutoff-k G comma-separated k, Fc = log10(k/median depth)  (default 1)
#   --escott-temperature T softmax temperature               (default 1.0)
#   --alpha-step S         alpha grid step                   (default 0.1)
#   --trace-definition D   bootstrap | direct                (default bootstrap)
#   --trace-top-fraction F leave UNSET unless you mean it; jet_surrogate.py's own
#                          measured default (0.90) then applies. 0.30 leaves 77/566
#                          HA positions at trace == 0, i.e. scoring as pure noise.
#   --leakage-check / --no-leakage-check
#                          BLAST the deep evolutionary set against every evaluation
#                          panel (checks A/B/C)                        (default on)
#   --purge-leakage / --no-purge-leakage
#                          REMOVE deep-set sequences that are near neighbours of an
#                          evaluation target BEFORE ESCOTT sees the alignment. This
#                          is the point: the panels are 2025/26 GISAID and the deep
#                          set is 2024-cutoff NCBI, in a DIFFERENT accession
#                          namespace, so only alignment can find the overlap and only
#                          removal can stop it inflating the result.   (default on)
#   --fail-on-leakage      stop the job on residual leakage instead of reporting an
#                          inflated correlation                        (default off)
#   --leakage-min-identity X   drop at >= X% AA identity    (stage-1 default 99.0)
#   --leakage-max-hamming N    drop at <= N mismatches      (stage-1 default 10)
#                          The two are combined with OR and are NOT equivalent: on a
#                          ~550 aa HA, 10 mismatches is ~98.2% identity, so the
#                          Hamming rule is the stricter one and governs the purge.
#                          'none' disables either rule individually. Leave both UNSET
#                          unless you mean it; leakage_check.py owns the defaults.
#   --prepare-only         build the stage-1 INPUT tree (structure, MSAs, queries,
#                          parent frequency files) and the observed-diversity
#                          profiles, then stop before any ESCOTT scoring
#   --skip-prepare         run stage 2 only; score matrices must already exist
#   --test-mode            limit how much data is READ (targets/records) and nothing
#                          else -- no modelling parameter is changed
#   --dry-run              build inputs + observed diversity, then stop
#   --regen-figures-only   re-plot from cached tables
#   --force-recompute-scores  ignore every score/table cache
#   --env-prescott DIR     the conda env that runs the WHOLE pipeline
#   --env-analysis DIR     deprecated alias for --env-prescott; one env now carries
#                          both stages, so this defaults to --env-prescott
#   --workdir DIR          repository root
#   -h | --help
#
# Example -- the full five-lineage run:
#   sbatch slurm_prescott_diversity.sbatch.sh \
#     --output-dir /home3/oml4h/PLM_SARS-CoV-2/Results/iav_prescott_diversity/run1
#
# Example -- the structural sensitivity pass called for in CAVEATS.md item 3:
#   sbatch slurm_prescott_diversity.sbatch.sh --structure-role extra \
#     --output-dir .../run1_contemporary_structure
# =============================================================================
#SBATCH --job-name=prescott-div-flu
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00

set -euo pipefail

WORKDIR=/home3/oml4h/PLM_SARS-CoV-2

# ONE environment runs the whole pipeline. It carries stage 1's needs (prody,
# biotite+mkdssp, freesasa, R, the escott console script, mafft, BLAST) AND stage 2's
# (Functions_HuggingFace, which imports torch and esm at module scope). Addressed by
# absolute interpreter path -- never `conda activate`, which does not survive a
# non-interactive shell.
#
# NOTE if this env is ever rebuilt: pytorch must come from conda-forge, never pip. The
# pip wheel links the system libstdc++ and, because run_mutational_accessibility.py
# imports torch before pandas, a pip torch makes pandas fail with
# "GLIBCXX_3.4.29 not found".
ENV_PRESCOTT=/home3/oml4h/miniconda3/envs/PRESCOTT
# Kept only so existing --env-analysis invocations do not break; it now defaults to
# ENV_PRESCOTT below rather than to a separate analysis env.
ENV_ANALYSIS=

OUTPUT_DIR=
GUIDE_PATH=
MUTATION_MODEL=H3N2
STRUCTURE=
STRUCTURE_ROLE=primary
WEIGHT_MODE=structural
PC_MODE=interface_propensity
SASA_CONTEXT=trimer
PARENT_MAP_PRESET=clade_evidence
PARENT_MAP=
COEFFICIENT_GRID=0.25,0.5,1.0
EQUATION_GRID=2
FREQUENCY_CUTOFF_K=1
ESCOTT_TEMPERATURE=1.0
ALPHA_STEP=0.1
TRACE_DEFINITION=bootstrap
# Empty means "do not pass the flag", so jet_surrogate.py's own measured default wins.
# Never set this to 0.30 -- see the header.
TRACE_TOP_FRACTION=
PARENT_SENSITIVITY=true
LEAKAGE_CHECK=true
PURGE_LEAKAGE=true
FAIL_ON_LEAKAGE=false
# Empty means "do not pass the flag", so leakage_check.py's own defaults win -- the same
# discipline TRACE_TOP_FRACTION is under, and for the same reason.
LEAKAGE_MIN_IDENTITY=
LEAKAGE_MAX_HAMMING=
PREPARE_ONLY=false
SKIP_PREPARE=false
TEST_MODE=false
DRY_RUN=false
REGEN_FIGURES_ONLY=false
FORCE_RECOMPUTE_SCORES=false

usage() { sed -n '2,80p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)          OUTPUT_DIR=$2;             shift 2;;
    --guide-path)          GUIDE_PATH=$2;             shift 2;;
    --mutation-model)      MUTATION_MODEL=$2;         shift 2;;
    --structure)           STRUCTURE=$2;              shift 2;;
    --structure-role)      STRUCTURE_ROLE=$2;         shift 2;;
    --weight-mode)         WEIGHT_MODE=$2;            shift 2;;
    --pc-mode)             PC_MODE=$2;                shift 2;;
    --sasa-context)        SASA_CONTEXT=$2;           shift 2;;
    --parent-map-preset)   PARENT_MAP_PRESET=$2;      shift 2;;
    --parent-map)          PARENT_MAP=$2;             shift 2;;
    --coefficient-grid)    COEFFICIENT_GRID=$2;       shift 2;;
    --equation-grid)       EQUATION_GRID=$2;          shift 2;;
    --frequency-cutoff-k)  FREQUENCY_CUTOFF_K=$2;     shift 2;;
    --escott-temperature)  ESCOTT_TEMPERATURE=$2;     shift 2;;
    --alpha-step)          ALPHA_STEP=$2;             shift 2;;
    --trace-definition)    TRACE_DEFINITION=$2;       shift 2;;
    --trace-top-fraction)  TRACE_TOP_FRACTION=$2;     shift 2;;
    --parent-sensitivity)     PARENT_SENSITIVITY=true;   shift;;
    --no-parent-sensitivity)  PARENT_SENSITIVITY=false;  shift;;
    --leakage-check)          LEAKAGE_CHECK=true;        shift;;
    --no-leakage-check)       LEAKAGE_CHECK=false;       shift;;
    --purge-leakage)          PURGE_LEAKAGE=true;        shift;;
    --no-purge-leakage)       PURGE_LEAKAGE=false;       shift;;
    --fail-on-leakage)        FAIL_ON_LEAKAGE=true;      shift;;
    --leakage-min-identity)   LEAKAGE_MIN_IDENTITY=$2;   shift 2;;
    --leakage-max-hamming)    LEAKAGE_MAX_HAMMING=$2;    shift 2;;
    --env-prescott)        ENV_PRESCOTT=$2;           shift 2;;
    --env-analysis)        ENV_ANALYSIS=$2;           shift 2;;
    --workdir)             WORKDIR=$2;                shift 2;;
    --prepare-only)        PREPARE_ONLY=true;         shift;;
    --skip-prepare)        SKIP_PREPARE=true;         shift;;
    --test-mode)           TEST_MODE=true;            shift;;
    --dry-run)             DRY_RUN=true;              shift;;
    --regen-figures-only|--regen_figures_only)       REGEN_FIGURES_ONLY=true;      shift;;
    --force-recompute-scores|--force_recompute_scores) FORCE_RECOMPUTE_SCORES=true; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1;;
  esac
done

WORKDIR=${WORKDIR%/}
# One env for both stages. --env-analysis survives only as an alias.
[[ -z "${ENV_ANALYSIS}" ]] && ENV_ANALYSIS=${ENV_PRESCOTT}
[[ -z "${GUIDE_PATH}" ]] && GUIDE_PATH=${WORKDIR}/Sequences/IAV_lineage_guide.csv
[[ -z "${STRUCTURE}" ]]  && STRUCTURE=${WORKDIR}/Sequences/6WXB-assembly1.cif
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR=${WORKDIR}/Results/iav_prescott_diversity/$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${OUTPUT_DIR%/}

if [[ "${PREPARE_ONLY}" == "true" && "${SKIP_PREPARE}" == "true" ]]; then
  echo "ERROR: --prepare-only and --skip-prepare are mutually exclusive." >&2; exit 1
fi
if [[ ! -d "${WORKDIR}" ]]; then
  echo "ERROR: workdir does not exist: ${WORKDIR}" >&2; exit 1
fi
if [[ ! -x "${ENV_ANALYSIS}/bin/python" ]]; then
  echo "ERROR: no python at ${ENV_ANALYSIS}/bin/python" >&2; exit 1
fi
if [[ ! -x "${ENV_PRESCOTT}/bin/python3.10" ]]; then
  echo "ERROR: no python3.10 at ${ENV_PRESCOTT}/bin/python3.10" >&2; exit 1
fi

JOB_ID=${SLURM_JOB_ID:-manual}
LOG_DIR=${OUTPUT_DIR}/slurm_logs/${JOB_ID}
STDOUT_LOG=${LOG_DIR}/console.stdout.log
STDERR_LOG=${LOG_DIR}/console.stderr.log
RUN_LOG=${LOG_DIR}/run_prescott_diversity.log

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

exec > >(tee -a "${STDOUT_LOG}")
exec 2> >(tee -a "${STDERR_LOG}" >&2)

cd "${WORKDIR}"

echo "[$(date -Is)] PRESCOTT diversity job started"
echo "job_id=${JOB_ID}"
echo "workdir=${WORKDIR}"
echo "output_dir=${OUTPUT_DIR}"
echo "guide_path=${GUIDE_PATH}"
echo "mutation_model=${MUTATION_MODEL}"
echo "structure=${STRUCTURE}  role=${STRUCTURE_ROLE}"
echo "weight_mode=${WEIGHT_MODE}  pc_mode=${PC_MODE}  sasa_context=${SASA_CONTEXT}"
echo "parent_map_preset=${PARENT_MAP_PRESET}  parent_map=${PARENT_MAP:-<preset only>}"
echo "coefficient_grid=${COEFFICIENT_GRID}  equation_grid=${EQUATION_GRID}  k=${FREQUENCY_CUTOFF_K}"
echo "escott_temperature=${ESCOTT_TEMPERATURE}  alpha_step=${ALPHA_STEP}"
echo "trace_definition=${TRACE_DEFINITION}  trace_top_fraction=${TRACE_TOP_FRACTION:-<jet_surrogate default>}"
echo "parent_sensitivity=${PARENT_SENSITIVITY}"
echo "prepare_only=${PREPARE_ONLY}  skip_prepare=${SKIP_PREPARE}  test_mode=${TEST_MODE}  dry_run=${DRY_RUN}"
echo "regen_figures_only=${REGEN_FIGURES_ONLY}  force_recompute_scores=${FORCE_RECOMPUTE_SCORES}"
echo "env_prescott=${ENV_PRESCOTT}"
echo "env_analysis=${ENV_ANALYSIS}"
echo "hostname=$(hostname)  nodelist=${SLURM_NODELIST:-manual}"

env | sort > "${LOG_DIR}/environment.txt"
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job "${SLURM_JOB_ID}" > "${LOG_DIR}/slurm_job.txt" 2>&1 || true
fi

# biotite's DsspApp resolves `mkdssp` from PATH, so a bare interpreter path is not
# enough: without this the JET2 surrogate dies with FileNotFoundError('mkdssp').
export PATH="${ENV_PRESCOTT}/bin:${PATH}"
# prescott.py imports pyplot at module load and compute nodes have no display.
export MPLBACKEND=Agg
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# A stale ~/R library would shadow the env's seqinr and break escott's R step.
export R_LIBS_USER=

"${ENV_ANALYSIS}/bin/python" --version | tee "${LOG_DIR}/python_version.txt"
"${ENV_PRESCOTT}/bin/python3.10" --version | tee -a "${LOG_DIR}/python_version.txt"

export GUIDE_PATH ENV_PRESCOTT ENV_ANALYSIS WORKDIR REGEN_FIGURES_ONLY PARENT_MAP PARENT_MAP_PRESET PARENT_SENSITIVITY
export LEAKAGE_CHECK PURGE_LEAKAGE LEAKAGE_MIN_IDENTITY LEAKAGE_MAX_HAMMING
"${ENV_ANALYSIS}/bin/python" - <<'PY' | tee "${LOG_DIR}/preflight.txt"
import csv, os, shutil, sys
from pathlib import Path

workdir = Path(os.environ["WORKDIR"])
guide_path = Path(os.environ["GUIDE_PATH"])
env_prescott = Path(os.environ["ENV_PRESCOTT"])
env_analysis = Path(os.environ["ENV_ANALYSIS"])
regen_only = os.environ.get("REGEN_FIGURES_ONLY") == "true"
errors, notes = [], []

# 1. Every FASTA the guide points at must exist AND contain records. An existence-only
#    check is not enough here: at least one sibling panel in this tree
#    (H3N2_K_hard_nextle1_max5.fasta) is a zero-record file, which would only fail
#    much later, inside the observed-diversity pass.
print('guide_exists=', guide_path.exists())
if not guide_path.exists():
    errors.append(f"ERROR: guide file not found: {guide_path}")
else:
    with guide_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    print('guide_rows=', len(rows))
    for row in rows:
        label = (row.get("month") or row.get("label") or "?").strip()
        for key in ("fasta", "reference"):
            target = (row.get(key) or "").strip()
            if not target:
                continue
            path = Path(target)
            if not path.exists():
                errors.append(f"ERROR: {label}:{key} missing: {path}")
                continue
            with path.open() as handle:
                n_records = sum(1 for line in handle if line.startswith(">"))
            print(f'  {label}:{key} records={n_records}')
            if n_records == 0:
                errors.append(f"ERROR: {label}:{key} has zero FASTA records: {path}")

# 2. The plotting typo that kills every export_plots() call must not be re-introduced.
#    export_plots is called unguarded, so a dirty run_mutational_accessibility.py loses
#    the entire figure set after the tables have already been written.
import subprocess
rc = subprocess.run(
    ["git", "diff", "--quiet", "--", "scripts/run_mutational_accessibility.py"],
    cwd=workdir,
).returncode
print('run_mutational_accessibility_clean=', rc == 0)
if rc != 0:
    errors.append(
        'ERROR: scripts/run_mutational_accessibility.py has uncommitted changes. If this is '
        'the `ax.set_yscale("log").se` typo near line 3948, revert it before running -- it '
        'raises AttributeError inside export_plots and destroys every figure.'
    )

# 3. Stage-1 tooling.
for name in ("escott", "Rscript", "mkdssp", "mafft"):
    path = env_prescott / "bin" / name
    ok = path.exists() and os.access(path, os.X_OK)
    print(f'{name}_executable=', ok)
    if not ok:
        errors.append(f"ERROR: not executable: {path}")
prescott_py = Path("/home3/oml4h/PRESCOTT/prescott/prescott.py")
print('prescott_py_readable=', prescott_py.exists())
if not prescott_py.exists():
    notes.append(f"NOTE: {prescott_py} absent; the prescott parity cross-check will be skipped.")

for name in ("prepare_inputs.py", "jet_surrogate.py", "run_escott.py"):
    path = workdir / "scripts" / "prescott_iav" / name
    print(f'stage1_{name}=', path.exists())
    if not path.exists():
        errors.append(f"ERROR: stage-1 script missing: {path}")

# 4. Stage-2 imports. torch/esm are needed only because Functions_HuggingFace imports
#    them at module scope; nothing in this pipeline runs a model. The import ORDER
#    matters: run_mutational_accessibility.py imports torch before pandas, so a pip
#    (rather than conda-forge) torch makes pandas die with GLIBCXX_3.4.29 not found.
probe = subprocess.run(
    [str(env_analysis / "bin" / "python"), "-c",
     "import torch, esm, pandas, seaborn, statsmodels, adjustText, sklearn, Bio"],
    capture_output=True, text=True,
)
print('analysis_imports_ok=', probe.returncode == 0)
if probe.returncode != 0:
    errors.append("ERROR: analysis env imports failed: " + probe.stderr.strip().splitlines()[-1])

# 5. The deep, pre-cutoff MSA source -- the thing that keeps the 2025/26 test panels out
#    of the ESCOTT alignment.
deep = Path("/home3/oml4h/my_SC2_finetunes/myflu/full_job4243186/data/"
            "ncbiflu_HA_all_110424_noX_clu99_filt_HA-80.clean_input.fasta")
if not deep.exists():
    errors.append(f"ERROR: deep MSA source not found: {deep}")
else:
    with deep.open() as handle:
        n_deep = sum(1 for line in handle if line.startswith(">"))
    print('deep_msa_records=', n_deep)
    if n_deep < 1000:
        errors.append(f"ERROR: deep MSA source has only {n_deep} records; expected ~6433")

# 5b. Leakage screening needs BLAST. It runs inside prepare_inputs.py, i.e. after several
#     minutes of structure/query/mafft work, so a missing blastp would otherwise surface
#     late and leave a half-built inputs tree. It is a preflight because the alternative
#     -- silently skipping the screen -- means shipping a correlation nobody audited.
if os.environ.get("LEAKAGE_CHECK") == "true" or os.environ.get("PURGE_LEAKAGE") == "true":
    for tool in ("makeblastdb", "blastp"):
        found = shutil.which(tool, path=str(env_prescott / "bin")) or shutil.which(tool)
        print(f'{tool}=', found)
        if not found:
            errors.append(
                f"ERROR: {tool} not found; leakage screening needs BLAST+ in {env_prescott}. "
                "Install it, or pass --no-leakage-check --no-purge-leakage and accept that "
                "the run is UNAUDITED for test-set leakage."
            )
    # The two thresholds are OR-combined and NOT equivalent; a user who sets only one
    # will be governed by the other's default. Say the numbers out loud in the log.
    min_id = os.environ.get("LEAKAGE_MIN_IDENTITY") or "99.0 (leakage_check.py default)"
    max_ham = os.environ.get("LEAKAGE_MAX_HAMMING") or "10 (leakage_check.py default)"
    print(f'leakage_rule= drop if identity >= {min_id} OR hamming <= {max_ham}')
    notes.append(
        "NOTE: leakage purge ON. The Hamming rule is the stricter of the two on a ~550 aa "
        "HA (10 mismatches ~ 98.2% identity), so it governs unless you disable it with "
        "--leakage-max-hamming none. The purge is PER TARGET: each lineage gets its own "
        "purged MSA and ESCOTT is run once per target."
    )
else:
    notes.append(
        "WARNING: leakage screening is OFF. The deep MSA has NOT been checked against the "
        "2025/26 evaluation panels, which are in a different accession namespace and so "
        "cannot be screened by ID. Any correlation this run reports is unaudited."
    )

# 6. The resolved parent map must be acyclic and every parent must have a guide row.
if not regen_only and guide_path.exists():
    sys.path.insert(0, str(workdir / "scripts"))
    # Imported, NOT retyped. scripts/prescott_iav/constants.py is the single source of
    # truth for this map -- it is what prepare_inputs.py builds the frequency files
    # from -- so a local copy here could silently preflight a topology the run does
    # not use. The module is stdlib-only and imports anywhere.
    from prescott_iav import constants as prescott_constants
    presets = prescott_constants.DEFAULT_PARENT_MAPS
    preset_name = os.environ.get("PARENT_MAP_PRESET") or prescott_constants.DEFAULT_PARENT_MAP_PRESET
    if preset_name not in presets:
        errors.append(f"ERROR: unknown --parent-map-preset {preset_name!r}; known: {sorted(presets)}")
        preset_name = prescott_constants.DEFAULT_PARENT_MAP_PRESET
    parent_map = dict(presets[preset_name])
    for chunk in (os.environ.get("PARENT_MAP") or "").split(","):
        if "=" in chunk:
            child, parent = (part.strip() for part in chunk.split("=", 1))
            parent_map[child] = parent
    labels = {(row.get("month") or row.get("label") or "").strip() for row in rows}
    print('parent_map=', parent_map)
    for child, parent in parent_map.items():
        if child in labels and parent not in labels:
            errors.append(f"ERROR: parent {parent!r} of {child!r} has no guide row")
    # --parent-sensitivity scores the contested edge under the OTHER preset's parent,
    # so that parent needs a guide row too or stage 1 cannot build its frequency file.
    if os.environ.get("PARENT_SENSITIVITY") == "true":
        alt_edges = prescott_constants.sensitivity_edges_between_presets(preset_name)
        print('sensitivity_edges=', alt_edges)
        for child, parent in alt_edges.items():
            if child in labels and parent not in labels:
                errors.append(
                    f"ERROR: --parent-sensitivity alternate parent {parent!r} of {child!r} "
                    f"has no guide row; add one or pass --no-parent-sensitivity"
                )
    for child in parent_map:
        seen, cursor = {child}, parent_map.get(child)
        while cursor is not None:
            if cursor in seen:
                errors.append(f"ERROR: parent map contains a cycle through {cursor!r}")
                break
            seen.add(cursor)
            cursor = parent_map.get(cursor)

for msg in notes:
    print(msg)
for msg in errors:
    print(msg, file=sys.stderr)
if errors:
    sys.exit(1)
PY
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  echo "ERROR: preflight checks failed -- see ${LOG_DIR}/preflight.txt" >&2
  exit 1
fi

RUN_ARGS=(
  "${WORKDIR}/scripts/run_prescott_diversity.py"
  --analysis-mode MONTHLY_GUIDE
  --guide-path "${GUIDE_PATH}"
  --mutation-model "${MUTATION_MODEL}"
  --output-dir "${OUTPUT_DIR}"
  --expect-protein-diversity
  --prescott-python "${ENV_PRESCOTT}/bin/python3.10"
  --structure "${STRUCTURE}"
  --structure-role "${STRUCTURE_ROLE}"
  --weight-mode "${WEIGHT_MODE}"
  --pc-mode "${PC_MODE}"
  --sasa-context "${SASA_CONTEXT}"
  --parent-map-preset "${PARENT_MAP_PRESET}"
  --coefficient-grid "${COEFFICIENT_GRID}"
  --equation-grid "${EQUATION_GRID}"
  --frequency-cutoff-k "${FREQUENCY_CUTOFF_K}"
  --escott-temperature "${ESCOTT_TEMPERATURE}"
  --alpha-step "${ALPHA_STEP}"
  --trace-definition "${TRACE_DEFINITION}"
)
# Deliberately conditional: unset means the driver does not forward the flag at all and
# jet_surrogate.py's own measured default applies. Passing it unconditionally is exactly
# how a tuned 0.90 came to be overridden by 0.30.
[[ -n "${TRACE_TOP_FRACTION}" ]]             && RUN_ARGS+=(--trace-top-fraction "${TRACE_TOP_FRACTION}")
[[ "${PARENT_SENSITIVITY}" == "true" ]]      && RUN_ARGS+=(--parent-sensitivity)
[[ "${PARENT_SENSITIVITY}" != "true" ]]      && RUN_ARGS+=(--no-parent-sensitivity)
[[ -n "${PARENT_MAP}" ]]                     && RUN_ARGS+=(--parent-map "${PARENT_MAP}")
# Both booleans always forwarded, in both directions: which sequences ESCOTT was allowed
# to see is not something a reader should have to infer from the absence of a flag.
[[ "${LEAKAGE_CHECK}" == "true" ]]           && RUN_ARGS+=(--leakage-check)
[[ "${LEAKAGE_CHECK}" != "true" ]]           && RUN_ARGS+=(--no-leakage-check)
[[ "${PURGE_LEAKAGE}" == "true" ]]           && RUN_ARGS+=(--purge-leakage)
[[ "${PURGE_LEAKAGE}" != "true" ]]           && RUN_ARGS+=(--no-purge-leakage)
[[ "${FAIL_ON_LEAKAGE}" == "true" ]]         && RUN_ARGS+=(--fail-on-leakage)
# Thresholds only when set, so leakage_check.py stays the authority on the numbers.
[[ -n "${LEAKAGE_MIN_IDENTITY}" ]]           && RUN_ARGS+=(--leakage-min-identity "${LEAKAGE_MIN_IDENTITY}")
[[ -n "${LEAKAGE_MAX_HAMMING}" ]]            && RUN_ARGS+=(--leakage-max-hamming "${LEAKAGE_MAX_HAMMING}")
[[ "${TEST_MODE}" == "true" ]]               && RUN_ARGS+=(--test-mode)
[[ "${DRY_RUN}" == "true" ]]                 && RUN_ARGS+=(--dry-run)
[[ "${REGEN_FIGURES_ONLY}" == "true" ]]      && RUN_ARGS+=(--regen-figures-only)
[[ "${FORCE_RECOMPUTE_SCORES}" == "true" ]]  && RUN_ARGS+=(--force-recompute-scores)
# --prepare-only stops after stage 1; --dry-run then guarantees the driver builds the
# inputs and the observed-diversity profiles without entering the scoring loop.
[[ "${PREPARE_ONLY}" == "true" ]]            && RUN_ARGS+=(--dry-run)
[[ "${SKIP_PREPARE}" == "true" ]]            && RUN_ARGS+=(--no-auto-prepare)

{
  printf '%q' "${ENV_ANALYSIS}/bin/python"
  for arg in "${RUN_ARGS[@]}"; do printf ' %q' "$arg"; done
  printf '\n'
} > "${LOG_DIR}/command.txt"
cat "${LOG_DIR}/command.txt"

# --prepare-only builds the stage-1 INPUT tree (structure, MSAs, queries, parent
# frequency files) and then lets the driver run with --dry-run, which stops before any
# scoring. Every flag that changes what prepare_inputs writes is forwarded here, so the
# tree it builds is byte-identical to the one an ordinary run would build -- in
# particular --sensitivity-parent-map, without which the alternate-parent frequency file
# is absent and a later --parent-sensitivity run fails with FileNotFoundError.
if [[ "${PREPARE_ONLY}" == "true" ]]; then
  PREPARE_ARGS=(
    "${WORKDIR}/scripts/prescott_iav/prepare_inputs.py"
    --guide-path "${GUIDE_PATH}"
    --inputs-dir "${OUTPUT_DIR}/inputs"
    --structure "${STRUCTURE}"
    --parent-map-preset "${PARENT_MAP_PRESET}"
    --frequency-cutoff-k "${FREQUENCY_CUTOFF_K}"
    --drop-parent-reversions
  )
  [[ -n "${PARENT_MAP}" ]] && PREPARE_ARGS+=(--parent-map "${PARENT_MAP}")
  # Forwarded here as well as into RUN_ARGS: --prepare-only builds the inputs tree by
  # calling prepare_inputs.py DIRECTLY, so an unforwarded purge flag would leave a tree
  # whose MSAs were screened at different settings than the ones the run advertises.
  [[ "${LEAKAGE_CHECK}" == "true" ]] && PREPARE_ARGS+=(--leakage-check) || PREPARE_ARGS+=(--no-leakage-check)
  [[ "${PURGE_LEAKAGE}" == "true" ]] && PREPARE_ARGS+=(--purge-leakage) || PREPARE_ARGS+=(--no-purge-leakage)
  [[ "${FAIL_ON_LEAKAGE}" == "true" ]] && PREPARE_ARGS+=(--fail-on-leakage)
  [[ -n "${LEAKAGE_MIN_IDENTITY}" ]] && PREPARE_ARGS+=(--leakage-min-identity "${LEAKAGE_MIN_IDENTITY}")
  [[ -n "${LEAKAGE_MAX_HAMMING}" ]]  && PREPARE_ARGS+=(--leakage-max-hamming "${LEAKAGE_MAX_HAMMING}")
  if [[ "${PARENT_SENSITIVITY}" == "true" ]]; then
    # Let constants.py derive the disagreeing edges rather than naming K=J.2_int here:
    # a hard-coded edge in a batch script is precisely the copy that goes stale.
    SENSITIVITY_PRESET=$("${ENV_PRESCOTT}/bin/python" -c "
import sys; sys.path.insert(0, '${WORKDIR}/scripts')
from prescott_iav import constants as c
print(next((n for n in c.DEFAULT_PARENT_MAPS if n != '${PARENT_MAP_PRESET}'), ''))
")
    [[ -n "${SENSITIVITY_PRESET}" ]] && PREPARE_ARGS+=(--sensitivity-preset "${SENSITIVITY_PRESET}")
  fi
  [[ "${FORCE_RECOMPUTE_SCORES}" == "true" ]] && PREPARE_ARGS+=(--force)
  "${ENV_PRESCOTT}/bin/python3.10" "${PREPARE_ARGS[@]}" 2>&1 | tee -a "${RUN_LOG}"
fi

"${ENV_ANALYSIS}/bin/python" "${RUN_ARGS[@]}" 2>&1 | tee -a "${RUN_LOG}"

echo "[$(date -Is)] PRESCOTT diversity job finished"
echo "[$(date -Is)] results in ${OUTPUT_DIR}"
echo "[$(date -Is)] READ ${OUTPUT_DIR}/CAVEATS.md before reporting any number from this run"
