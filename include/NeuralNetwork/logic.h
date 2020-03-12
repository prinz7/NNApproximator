#pragma once

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
  double calculateMeanError(DataVector const& testData);
  [[nodiscard]]
  std::vector<double> calculateR2Score(DataVector const& testData);
  [[nodiscard]]
  std::vector<double> calculateR2ScoreAlternate(DataVector const& testData);
  [[nodiscard]]
  std::vector<double> calculateR2ScoreAlternateDenormalized(DataVector const& testData);
  void outputBehaviour(DataVector const& data);
  void saveValuesToFile(DataVector const& data, std::string const& outputPath);
  void saveDiffToFile(DataVector const& data, std::string const& outputPath, bool outputRelativeDifference);
  [[nodiscard]]
  torch::Tensor calculateDiff(torch::Tensor const& wantedValue, torch::Tensor const& actualValue) const;
  [[nodiscard]]
  torch::Tensor calculateRelativeDiff(torch::Tensor const& wantedValue, torch::Tensor const& actualValue) const;
  void saveMinMaxToFile() const;
  void denormalizeInputTensor(torch::Tensor& tensor, bool limitValues = false);
  void denormalizeOutputTensor(torch::Tensor const& inputTensor, torch::Tensor& outputTensor, bool limitValues = false);

private:
  Network network {nullptr};
  Utilities::ProgramOptions options {};

  bool useMixedScaling = false;
  TensorDataType normalizedMixedScalingThreshold = Utilities::DefaultValues::MIXED_SCALING_THRESHOLD;

  MinMaxValues minMax = std::make_pair(MinMaxVector(), MinMaxVector());
  MinMaxVector& inputMinMax = minMax.first;
  MinMaxVector& outputMinMax = minMax.second;

  MixedMinMaxValues mixedScalingMinMax {};

  std::string inputFileHeader {};

  ProgressVector trainingProgress {};
};

}
