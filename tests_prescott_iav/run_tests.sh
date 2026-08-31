#!/bin/bash
# ===========================================================================
# Run ONLY the PRESCOTT/ESCOTT influenza-diversity test suite.
#
# This mirrors the conventions of the repo-level ../run_tests.sh but is
# deliberately self-contained: it runs from this directory's own pytest.ini,
# never reads ../pytest.ini, and never touches ../tests/.
#
# It does NOT rely on `conda activate` or on whatever `python`/`pytest` happen
# to be on PATH -- the system ones lack this project's dependencies.  The
# PRESCOTT env interpreter is invoked by absolute path and its bin/ is
# prepended to PATH so the pipeline's subprocess children (escott, prescott,
# mafft, mkdssp, blastp, Rscript) resolve to the env's copies too.
#
#   ./run_tests.sh                 # the fast, offline default run
#   ./run_tests.sh -c              # + per-module coverage
#   ./run_tests.sh -s              # + the opt-in slow / real-data tests
#   ./run_tests.sh -t TestCommon   # one class, or -t test_common.py
# ===========================================================================

set -uo pipefail

PRESCOTT_ENV="${PRESCOTT_ENV:-/home3/oml4h/miniconda3/envs/PRESCOTT}"
PYTHON="$PRESCOTT_ENV/bin/python"
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

VERBOSE=""
COVERAGE=""
PARALLEL=""
SLOW=""
SPECIFIC_TEST=""
EXTRA=()

usage() {
    cat <<'USAGE'
Usage: ./run_tests.sh [OPTIONS] [-- EXTRA_PYTEST_ARGS]

Options:
  -v, --verbose     Verbose output (-vv, and show local variables on failure)
  -c, --coverage    Per-module coverage for prescott_iav + run_prescott_diversity
  -p, --parallel    Run in parallel with pytest-xdist (-n auto)
  -s, --slow        Also run the opt-in slow / real-data / real-binary tests
  -t, --test NAME   Run one file, class or test
                      -t test_common.py
                      -t test_common.py::TestSafeLabel
                      -t TestSafeLabel            (matched with -k)
  -h, --help        Show this message

Anything after `--` is passed straight through to pytest.

Examples:
  ./run_tests.sh
  ./run_tests.sh -c
  ./run_tests.sh -v -p -s
  ./run_tests.sh -t test_leakage_check.py
  ./run_tests.sh -- -k "cache and not slow"
USAGE
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)  VERBOSE="-vv --showlocals"; shift ;;
        -c|--coverage) COVERAGE="yes"; shift ;;
        -p|--parallel) PARALLEL="-n auto"; shift ;;
        -s|--slow)     SLOW="--run-slow"; shift ;;
        -t|--test)     SPECIFIC_TEST="${2:-}"; shift 2 ;;
        -h|--help)     usage; exit 0 ;;
        --)            shift; EXTRA=("$@"); break ;;
        *)             echo "Unknown option: $1"; echo "Use -h for help."; exit 1 ;;
    esac
done

# --- environment checks ----------------------------------------------------
if [ ! -x "$PYTHON" ]; then
    echo -e "${RED}Interpreter not found: $PYTHON${NC}" >&2
    echo "Set PRESCOTT_ENV to the conda env prefix if it lives elsewhere." >&2
    exit 1
fi

if ! "$PYTHON" -c "import pytest" 2>/dev/null; then
    echo -e "${RED}pytest is not installed in $PRESCOTT_ENV${NC}" >&2
    echo "Install it there -- do NOT pip install into the system python." >&2
    exit 1
fi

# The pipeline shells out to escott/prescott/mafft/mkdssp/blastp/Rscript.
# Prepending the env's bin/ makes those children the env's copies.
export PATH="$PRESCOTT_ENV/bin:$PATH"

# --- build the pytest command ----------------------------------------------
TARGET="$SUITE_DIR"
KFILTER=()
if [ -n "$SPECIFIC_TEST" ]; then
    if [[ "$SPECIFIC_TEST" == *.py* ]]; then
        TARGET="$SUITE_DIR/$SPECIFIC_TEST"
    else
        KFILTER=(-k "$SPECIFIC_TEST")
    fi
fi

COV_ARGS=()
if [ -n "$COVERAGE" ]; then
    COV_ARGS=(
        --cov=prescott_iav
        --cov=run_prescott_diversity
        --cov-branch
        --cov-report=term-missing
        "--cov-report=html:$SUITE_DIR/htmlcov_prescott_iav"
    )
fi

CMD=("$PYTHON" -m pytest "$TARGET")
[ -n "$VERBOSE" ]  && CMD+=($VERBOSE)
[ -n "$PARALLEL" ] && CMD+=($PARALLEL)
[ -n "$SLOW" ]     && CMD+=("$SLOW")
[ ${#KFILTER[@]}  -gt 0 ] && CMD+=("${KFILTER[@]}")
[ ${#COV_ARGS[@]} -gt 0 ] && CMD+=("${COV_ARGS[@]}")
[ ${#EXTRA[@]}    -gt 0 ] && CMD+=("${EXTRA[@]}")

echo "=========================================="
echo "  prescott_iav test suite"
echo "=========================================="
echo "  python : $PYTHON"
echo "  suite  : $SUITE_DIR"
echo "  config : $SUITE_DIR/pytest.ini"
echo ""
echo -e "${YELLOW}Running: ${CMD[*]}${NC}"
echo ""

# Run from the suite directory so pytest picks up THIS pytest.ini as rootdir
# config and never walks up to the repo-level one.
cd "$SUITE_DIR" || exit 1
"${CMD[@]}"
STATUS=$?

echo ""
if [ $STATUS -eq 0 ]; then
    echo -e "${GREEN}All tests passed.${NC}"
    if [ -n "$COVERAGE" ]; then
        echo -e "${GREEN}Coverage report: $SUITE_DIR/htmlcov_prescott_iav/index.html${NC}"
    fi
    if [ -z "$SLOW" ]; then
        echo "Note: the slow / real-data / real-binary tests were skipped."
        echo "      Run './run_tests.sh -s' before trusting a release."
    fi
else
    echo -e "${RED}Some tests failed (exit $STATUS).${NC}"
fi
exit $STATUS
