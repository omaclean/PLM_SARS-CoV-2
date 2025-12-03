#!/bin/bash
# Script to run tests for Functions_HuggingFace.py

set -e

echo "=========================================="
echo "Running Tests for Functions_HuggingFace"
echo "=========================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest is not installed. Installing..."
    pip install pytest pytest-cov pytest-xdist
fi

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse command line arguments
VERBOSE=""
COVERAGE=""
PARALLEL=""
SPECIFIC_TEST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -c|--coverage)
            COVERAGE="--cov=Functions_HuggingFace --cov-report=term --cov-report=html"
            shift
            ;;
        -p|--parallel)
            PARALLEL="-n auto"
            shift
            ;;
        -t|--test)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose     Run tests with verbose output"
            echo "  -c, --coverage    Generate coverage report"
            echo "  -p, --parallel    Run tests in parallel"
            echo "  -t, --test NAME   Run specific test or test class"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_tests.sh                              # Run all tests"
            echo "  ./run_tests.sh -v                           # Run with verbose output"
            echo "  ./run_tests.sh -c                           # Run with coverage"
            echo "  ./run_tests.sh -v -c -p                     # All options"
            echo "  ./run_tests.sh -t TestMutationFunctions     # Run specific class"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest tests/test_functions_huggingface.py"

if [ -n "$VERBOSE" ]; then
    PYTEST_CMD="$PYTEST_CMD $VERBOSE"
fi

if [ -n "$COVERAGE" ]; then
    PYTEST_CMD="$PYTEST_CMD $COVERAGE"
fi

if [ -n "$PARALLEL" ]; then
    PYTEST_CMD="$PYTEST_CMD $PARALLEL"
fi

if [ -n "$SPECIFIC_TEST" ]; then
    PYTEST_CMD="$PYTEST_CMD::$SPECIFIC_TEST"
fi

# Run tests
echo -e "${YELLOW}Running: $PYTEST_CMD${NC}"
echo ""

if eval $PYTEST_CMD; then
    echo ""
    echo -e "${GREEN}✅ All tests passed!${NC}"
    
    if [ -n "$COVERAGE" ]; then
        echo ""
        echo -e "${GREEN}📊 Coverage report generated in htmlcov/index.html${NC}"
    fi
    
    exit 0
else
    echo ""
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi
