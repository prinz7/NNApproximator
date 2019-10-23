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
  void normalizeWithFileData(DataVector& data, DataVector const& minMax);
  void trainNetwork(DataVector const& data);
  void performInteractiveMode();
  [[nodiscard]]
  double calculateMeanError(DataVector const& testData);
  [[nodiscard]]
  double calculateR2Score(DataVector const& testData);
  [[nodiscard]]
  double calculateR2ScoreAlternate(DataVector const& testData);
  [[nodiscard]]
  double calculateR2ScoreAlternateDenormalized(DataVector const& testData);
  [[nodiscard]]
  std::pair<DataVector, DataVector> splitData(DataVector const& inputData, double trainingPercentage) const;
  void outputBehaviour(DataVector const& data);
  void saveValuesToFile(DataVector const& data, std::string const& outputPath);
  void saveDiffToFile(DataVector const& data, std::string const& outputPath);
  [[nodiscard]]
  torch::Tensor calculateDiff(torch::Tensor const& input1, torch::Tensor const& input2) const;
  void saveMinMaxToFile() const;

private:
  Network network {nullptr};
  Utilities::ProgramOptions options {};

  std::pair<MinMaxVector, MinMaxVector> minMax = std::make_pair(MinMaxVector(), MinMaxVector());
  MinMaxVector& inputMinMax = minMax.first;
  MinMaxVector& outputMinMax = minMax.second;

  std::string inputFileHeader{};
};

}
