#!/bin/bash
cd "$(dirname "$0")"

./NNApproximator --input data.csv --numberIn 3 --numberOut 2 --epochs 1500 --outWeights myWeights --printBehaviour
