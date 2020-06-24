#!/bin/bash
cd "$(dirname "$0")"

./NNApproximator -i data.csv -ni 3 -no 2 --logScaling --outMinMax minMaxLog.csv --validate --validatePercentage 100
