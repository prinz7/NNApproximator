#pragma once

#include "NeuralNetwork/networkanalyzer.h"
#include "NeuralNetwork/neuralnetwork.h"
#include "Utilities/constants.h"
#include "Utilities/programoptions.h"

namespace NeuralNetwork {

class Logic
{
public:
  [[nodiscard]]
  bool performUserRequest(Utilities::ProgramOptions const& options);

private:
  void trainNetwork(DataVector const& data);
  void performInteractiveMode();
  [[nodiscard]]
  std::vector<double> calculateR2ScoreAlternateDenormalized(DataVector const& testData);
  void outputBehaviour(DataVector const& data);
  void saveValuesToFile(DataVector const& data, std::string const& outputPath);
  void saveDiffToFile(DataVector const& data, std::string const& outputPath, bool outputRelativeDifference);
  void saveMinMaxToFile() const;
  void denormalizeInputTensor(torch::Tensor& tensor, bool limitValues = false);
  void denormalizeOutputTensor(torch::Tensor const& inputTensor, torch::Tensor& outputTensor, bool limitValues = false);
  void unscaleOutputTensor(torch::Tensor const& inputTensor, torch::Tensor& outputTensor);

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
