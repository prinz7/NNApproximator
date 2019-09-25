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
  void trainNetwork(Network& network, const DataVector& data);
  [[nodiscard]]
  static double calculateMeanError(Network& network, const DataVector& testData);

private:
  Utilities::ProgramOptions options {};
};

}
