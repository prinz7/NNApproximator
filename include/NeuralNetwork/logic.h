#pragma once

#include "NeuralNetwork/networkanalyzer.h"
#include "NeuralNetwork/neuralnetwork.h"
#include "Utilities/constants.h"
#include "Utilities/programoptions.h"

namespace NeuralNetwork {

class Logic
{
public:
  /*
   * Performs the whole training/validation process depending on the options which the user set via the command line interface.
   */
  [[nodiscard]]
  bool performUserRequest(Utilities::ProgramOptions const& options);

private:
  /*
   * Trains the neural network with the given data. If the data vector is empty, no training is performed.
   */
  void trainNetwork(DataVector const& data);
  /*
   * Starts the interactive mode where the user can input values via the console. Following actions are performed with these values:
   * - normalization and scaling (if needed)
   * - infer output values with the inputted values
   * - denormalize and unscale inferred values (if needed)
   * - output results to the console
   *
   * This is an iterative process, until the user stops the process by typing 'q'.
   */
  void performInteractiveMode();
  /*
   * Infers values from the neural network depending on the inputted data and outputs the results to console.
   */
  void outputBehaviour(DataVector const& data);
  /*
   * Infers values from the neural network depending on the inputted data and outputs the results to a file in the given file path.
   */
  void saveValuesToFile(DataVector const& data, std::string const& outputPath);
  /*
   * Infers values from the neural network depending on the inputted data and
   * outputs the diff to the correct output to a file in the given file path.
   * The saved diff can be absolute or relative.
   */
  void saveDiffToFile(DataVector const& data, std::string const& outputPath, bool outputRelativeDifference);
  /*
   * Saves the minimum and maximum values from the current training data to the filepath which the user defined.
   * If the data got scaled, scaled min/max values are saved.
   */
  void saveMinMaxToFile() const;
  /*
   * Denormalizes an input tensor.
   * If limitValues is true, the output is limited by the current min/max output values.
   */
  void denormalizeInputTensor(torch::Tensor& tensor, bool limitValues = false);
  /*
   * Denormalizes an output tensor.
   * If limitValues is true, the output is limited by the current min/max output values.
   */
  void denormalizeOutputTensor(torch::Tensor const& inputTensor, torch::Tensor& outputTensor, bool limitValues = false);
  /*
   * Reverts the scaling on an output tensor.
   */
  void unscaleOutputTensor(torch::Tensor const& inputTensor, torch::Tensor& outputTensor) const;
  /*
   * Checks if the given min/max values are valid --> min != max
   */
  [[nodiscard]]
  bool minMaxValuesAreValid() const;

private:
  Network network {nullptr};
  std::unique_ptr<NetworkAnalyzer> analyzer {nullptr};
  Utilities::ProgramOptions options {};

  bool useMixedScaling = false;
  TensorDataType normalizedMixedScalingThreshold = Utilities::DefaultValues::MIXED_SCALING_THRESHOLD;

  MinMaxValues minMax = std::make_pair(MinMaxVector(), MinMaxVector());
  MinMaxVector& inputMinMax = minMax.first;
  MinMaxVector& outputMinMax = minMax.second;

  MixedMinMaxValues mixedScalingMinMax {};

  std::string inputFileHeader {};

  ProgressVector trainingProgress {};

  bool useBatchTraining = false;
  BatchMap batchedTrainingData = BatchMap();
};

}
