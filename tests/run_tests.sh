#!/bin/bash
# Test runner script for all unit tests

set -e

cd "$(dirname "$0")"
cd ..

echo "=========================================="
echo "Running JiT Segmentation Unit Tests"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

# Function to run a test
run_test() {
    local test_name=$1
    local test_file=$2
    
    echo -e "${YELLOW}Running: ${test_name}${NC}"
    if python "$test_file"; then
        echo -e "${GREEN}✓ PASSED: ${test_name}${NC}\n"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAILED: ${test_name}${NC}\n"
        ((FAILED++))
    fi
}

# Run individual tests
run_test "Model Architecture (JiT)" "tests/test_model_jit.py"
run_test "Denoiser (Image-Conditioned)" "tests/test_denoiser.py"
run_test "Evaluation Metrics" "tests/test_metrics.py"
run_test "OCTA Dataset Loading" "tests/test_dataset_octa.py"

# Print summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "${GREEN}Passed: ${PASSED}${NC}"
echo -e "${RED}Failed: ${FAILED}${NC}"
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed! ✓${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
