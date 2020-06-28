# Examples

This scripts run examples explained in https://github.com/prinz7/NNApproximator/wiki/Program-execution-examples. Example scripts should be executed in order.

## Thesis related information

 - A local copy of the wiki exists in Dropbox\Jannik\Final Files\Program.zip\Program\NNApproximator.wiki
 - More detailed information about parameters can be found in the thesis section 5.2.
 - Libtorch version used in thesis: 1.2.1, Also updated and tested with Libtorch 1.5.1.

## Additional notes for usage

 - When the program is interrupted, no intermediate results are stored and all progress is lost!
   - Workaround: Run program in a loop with small epochs and time-out. Pay attention to --inWeights and --outWeights and make sure to work on the latest results.
 - When a time-out is set, the currently running epoch is always finished. This can lead to an overall run-time (way) above the specified time-out.
 - Parameter --validatePercentage also requires parameter --validate to be set. Otherwise, there is no effect.
 - For inference, the input file needs values for all columns, even for the output columns. These can have any value. Also note, that R2 score will have no meaning in this scenario.

## More examples

Example command to continue the training with

 - 2 days timeout
 - fixed seed (777)
 - standard layout (2 hidden layers, 500 nodes each)
 - logarithmic scaling
 - nohup for running it in the background (doesn't stop when closing the terminal)
 
```
nohup ./NNApproximator -i data.csv -ni 3 -no 1 --epsilon 0.0000000001 -t 4 --numberOfDeteriorations 5 --learnRate 0.001 --timeoutInHours 48 --seed 777 --outWeights weights_log_new --logScaling --layers 2 --nodes 500 --inWeights weights_log_old > console_log
```

### Storing intermediate results along the way

```
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
```
