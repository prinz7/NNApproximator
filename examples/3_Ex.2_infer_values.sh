#!/bin/bash
cd "$(dirname "$0")"

./NNApproximator --input data.csv --numberIn 3 --numberOut 2 --inWeights myWeights --validate --validatePercentage 100 --outValues inferred_values.csv
