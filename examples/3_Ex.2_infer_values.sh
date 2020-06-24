#!/bin/bash
cd "$(dirname "$0")"

./NNApproximator -i data.csv -ni 3 -no 2 --inWeights myWeights --validate --validatePercentage 100 --outValues inferred_values.csv
