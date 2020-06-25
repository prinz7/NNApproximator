#!/bin/bash
cd "$(dirname "$0")"

./NNApproximator --input data.csv --numberIn 3 --numberOut 2 --logScaling --outMinMax minMaxLog.csv --validate --validatePercentage 100
