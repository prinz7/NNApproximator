#!/bin/bash
cd "$(dirname "$0")"

target_epochs=100
current_epoch=0
while [ ${current_epoch} -lt ${target_epochs} ]; do
    next_epoch=$((current_epoch + 10))
    if [ ! -f interm_epochs_${next_epoch} ]; then
        params="--input data.csv --numberIn 3 --numberOut 2 --epochs 10 --timeoutInHours 1 --outWeights interm_epochs_${next_epoch}"
        if [ -f interm_epochs_${current_epoch} ]; then
            params="${params} --inWeights interm_epochs_${current_epoch}"
        fi
        ./NNApproximator ${params}
        echo "" # line break
    fi
    current_epoch=${next_epoch}
done
