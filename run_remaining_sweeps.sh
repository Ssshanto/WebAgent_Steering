#!/bin/bash
set -e

echo ">>> STARTING BATCH SWEEP <<<"
echo "Date: $(date)"

./run_phi.sh
./run_smollm.sh
./run_qwenvl.sh

echo ">>> ALL SWEEPS COMPLETE <<<"
echo "Date: $(date)"
