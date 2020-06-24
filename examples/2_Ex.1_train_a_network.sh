#!/bin/bash
cd "$(dirname "$0")"

./NNApproximator -i data.csv -ni 3 -no 2 -e 1500 --outWeights myWeights --printBehaviour
