#!/bin/bash
# Install testing dependencies into the active conda environment
# Run this after activating your conda environment:
#   conda activate plm_entropy  (or plm_sars)
#   ./install_test_deps.sh

set -e

echo "🔍 Checking current environment..."
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ Error: No conda environment is active!"
    echo "   Please run: conda activate plm_entropy"
    exit 1
fi

echo "✅ Active environment: $CONDA_DEFAULT_ENV"
echo ""
echo "📦 Installing test dependencies..."

# Install testing framework
pip install pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    pytest-xdist>=3.0.0 \
    pytest-timeout>=2.1.0 \
    pytest-mock>=3.10.0

# Install code quality tools
pip install flake8>=6.0.0 \
    black>=23.0.0 \
    pylint>=2.16.0 \
    isort>=5.12.0

# Optional but recommended
pip install coverage[toml]>=7.0.0 \
    pytest-html>=3.1.0

echo ""
echo "✅ Test dependencies installed successfully!"
echo ""
echo "🧪 You can now run tests with:"
echo "   ./run_tests.sh -v"
echo "   ./run_tests.sh -c    # with coverage"
