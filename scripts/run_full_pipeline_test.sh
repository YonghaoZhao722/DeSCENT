#!/bin/bash
# Run both Part 1 and Part 2 in sequence.
# Part 1: DEG + ReDeconv + condgen
# Part 2: Multimodal survival
#
# Usage: ./scripts/run_full_pipeline_test.sh [BRCA]
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/run_part1_deg_redeconv_condgen.sh" "${1:-BRCA}"
"$SCRIPT_DIR/run_part2_survival.sh" "${1:-BRCA}"
echo ""
echo "=== Full pipeline complete ==="
