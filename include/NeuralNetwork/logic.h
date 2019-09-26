#pragma once

#include "NeuralNetwork/neuralnetwork.h"
#include "Utilities/constants.h"
#include "Utilities/programoptions.h"

namespace NeuralNetwork {

class Logic
{
public:
  [[nodiscard]]
  bool performUserRequest(const Utilities::ProgramOptions& options);

private:
  void trainNetwork(const DataVector& data);
  void performInteractiveMode();
  [[nodiscard]]
  double calculateMeanError(const DataVector& testData);

private:
  Network network {nullptr};
  Utilities::ProgramOptions options {};

  std::pair<MinMaxVector, MinMaxVector> minMax = std::make_pair(MinMaxVector(), MinMaxVector());
  MinMaxVector& inputMinMax = minMax.first;
  MinMaxVector& outputMinMax = minMax.second;
};

}
