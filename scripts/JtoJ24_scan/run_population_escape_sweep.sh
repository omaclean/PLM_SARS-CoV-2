#!/bin/bash
# ===========================================================================
# Big population-escape run: the headline result plus a sensitivity grid.
#
# Every number this analysis produces moves with four choices that are
# assumptions, not measurements -- how fast immunity decays, how far
# cross-immunity reaches, whether a period's weight follows abundance or
# diversity, and the kernel shape. A single run is therefore not an answer, it
# is one cell of a grid. This runs the grid.
#
#   ./run_population_escape_sweep.sh                # headline + default grid
#   ./run_population_escape_sweep.sh --full         # the wide grid (~4x longer)
#   ./run_population_escape_sweep.sh --headline-only
#   ./run_population_escape_sweep.sh --dry-run      # print the commands, run nothing
#
# Re-running skips any variant whose output directory already holds a metadata
# file, so an interrupted job resumes where it stopped. --force overrides that.
#
# Outputs
#   <RUN_DIR>/population_escape/                    the headline run
#   <RUN_DIR>/population_escape_sweep/<tag>/        one directory per variant
#   <RUN_DIR>/population_escape_sweep/manifest.tsv  tag -> flags -> exit status
#   <RUN_DIR>/population_escape_sweep/logs/<tag>.log
# ===========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${SCAN_PYTHON:-/home3/oml4h/miniconda3/envs/plm_entropy/bin/python}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/Results/JtoJ.2.4_scan/plant}"
BACKGROUND_CSV="${BACKGROUND_CSV:-/home3/oml4h/hugging_face_downloads/PLANT_model/code/examples/backgrounds.csv}"

FULL=false
HEADLINE_ONLY=false
DRY_RUN=false
FORCE=false

usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)          FULL=true; shift ;;
    --headline-only) HEADLINE_ONLY=true; shift ;;
    --dry-run)       DRY_RUN=true; shift ;;
    --force)         FORCE=true; shift ;;
    --run-dir)       RUN_DIR=$2; shift 2 ;;
    --background-csv) BACKGROUND_CSV=$2; shift 2 ;;
    -h|--help)       usage ;;
    *) echo "Unknown option: $1 (use -h)" >&2; exit 2 ;;
  esac
done

# --- preflight -------------------------------------------------------------
for path in "$PYTHON" "$RUN_DIR/genotype_embeddings.csv" "$BACKGROUND_CSV"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing $path" >&2
    exit 1
  fi
done

SCAN="${SCRIPT_DIR}/plant_population_escape.py"
SWEEP_DIR="${RUN_DIR}/population_escape_sweep"
LOG_DIR="${SWEEP_DIR}/logs"
MANIFEST="${SWEEP_DIR}/manifest.tsv"

# The J lineage's own window. Quarterly is fine: with a 1-year half-life the
# landscape barely moves inside a quarter, so finer spacing buys nothing.
DATES=(
  --as-of 2022-01-01 --as-of 2022-04-01 --as-of 2022-07-01 --as-of 2022-10-01
  --as-of 2023-01-01 --as-of 2023-04-01 --as-of 2023-07-01 --as-of 2023-10-01
  --as-of 2024-01-01
)

# --- the grid --------------------------------------------------------------
# tag <TAB> flags. The headline settings are the first row of the grid too, so
# the sweep is self-contained if you only ever look at that directory.
VARIANTS=(
  "halflife0.5|--half-life 0.5"
  "halflife1.0|--half-life 1.0"
  "halflife2.0|--half-life 2.0"
  "scale1.0|--cross-immunity-scale 1.0"
  "scale2.0|--cross-immunity-scale 2.0"
  "scale4.0|--cross-immunity-scale 4.0"
  "within-unique|--within-period unique"
  "within-density|--within-period density --density-radius 0.25"
  "normalise-none|--normalise-by none"
  "kernel-sigmoid|--kernel sigmoid"
)

if [[ "$FULL" == true ]]; then
  VARIANTS+=(
    "halflife0.25|--half-life 0.25"
    "halflife5.0|--half-life 5.0"
    "scale0.5|--cross-immunity-scale 0.5"
    "scale8.0|--cross-immunity-scale 8.0"
    "kernel-linear|--kernel linear"
    "normalise-month|--normalise-by month"
    "within-density-fine|--within-period density --density-radius 0.05"
    "within-density-coarse|--within-period density --density-radius 1.0"
    "hl0.5-unique|--half-life 0.5 --within-period unique"
    "hl2.0-scale4|--half-life 2.0 --cross-immunity-scale 4.0"
    "sigmoid-unique|--kernel sigmoid --within-period unique"
    "maxage3|--max-age 3.0"
  )
fi

run_one() {
  local tag=$1 destination=$2
  shift 2
  local flags=("$@")

  if [[ "$FORCE" != true && -n "$(ls "${destination}"/run_metadata_*.json 2>/dev/null)" ]]; then
    printf '  [skip] %s (already done)\n' "$tag"
    printf '%s\t%s\tskipped\n' "$tag" "${flags[*]}" >> "$MANIFEST"
    return 0
  fi

  printf '  [run ] %-24s %s\n' "$tag" "${flags[*]}"
  if [[ "$DRY_RUN" == true ]]; then
    printf '%s\t%s\tdry-run\n' "$tag" "${flags[*]}" >> "$MANIFEST"
    return 0
  fi

  mkdir -p "$destination"
  "$PYTHON" "$SCAN" "$RUN_DIR" \
    --background-csv "$BACKGROUND_CSV" \
    --output-dir "$destination" \
    "${DATES[@]}" "${flags[@]}" \
    > "${LOG_DIR}/${tag}.log" 2>&1
  local status=$?
  printf '%s\t%s\t%s\n' "$tag" "${flags[*]}" "$status" >> "$MANIFEST"
  if [[ $status -ne 0 ]]; then
    printf '  [FAIL] %s -- see %s\n' "$tag" "${LOG_DIR}/${tag}.log" >&2
  fi
  return 0   # keep going: one bad variant must not abandon the rest
}

# --- go --------------------------------------------------------------------
mkdir -p "$SWEEP_DIR" "$LOG_DIR"
printf 'tag\tflags\tstatus\n' > "$MANIFEST"

echo "=========================================="
echo "  population escape sweep"
echo "=========================================="
echo "  python     : $PYTHON"
echo "  run dir    : $RUN_DIR"
echo "  background : $BACKGROUND_CSV"
echo "  dates      : ${#DATES[@]} flags -> $(( ${#DATES[@]} / 2 )) landscapes"
echo "  variants   : ${#VARIANTS[@]}$([[ "$FULL" == true ]] && echo ' (full grid)')"
echo "  started    : $(date -Is)"
echo ""

echo "Headline run (all defaults) -> ${RUN_DIR}/population_escape"
run_one "headline" "${RUN_DIR}/population_escape"

if [[ "$HEADLINE_ONLY" != true ]]; then
  echo ""
  echo "Sensitivity grid -> ${SWEEP_DIR}"
  for entry in "${VARIANTS[@]}"; do
    tag="${entry%%|*}"
    flag_string="${entry#*|}"
    read -r -a flags <<< "$flag_string"
    run_one "$tag" "${SWEEP_DIR}/${tag}" "${flags[@]}"
  done
fi

echo ""
echo "  finished   : $(date -Is)"
FAILED=$(awk -F'\t' 'NR>1 && $3 != "0" && $3 != "skipped" && $3 != "dry-run"' "$MANIFEST" | wc -l)
if [[ "$FAILED" -gt 0 ]]; then
  echo "  $FAILED variant(s) failed -- see $MANIFEST and $LOG_DIR" >&2
else
  echo "  all variants completed"
fi
echo ""
echo "Manifest: $MANIFEST"
echo "Compare across variants with the per-variant single_mutation_escape_by_date.csv,"
echo "which carries every date in one table."
exit 0
