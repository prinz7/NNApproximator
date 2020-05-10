#pragma once

#include "NeuralNetwork/neuralnetwork.h"
#include "Utilities/constants.h"

namespace NeuralNetwork {

  class NetworkAnalyzer
  {
  public:
    explicit NetworkAnalyzer(Network& network);

  public:
    [[nodiscard]]
    double calculateMeanSquaredError(DataVector const& testData);
    [[nodiscard]]
    std::vector<double> calculateR2Score(DataVector const& testData);
    [[nodiscard]]
    std::vector<double> calculateR2ScoreAlternate(DataVector const& testData);

  public:
    [[nodiscard]]
    static torch::Tensor calculateDiff(torch::Tensor const& wantedValue, torch::Tensor const& actualValue);
    [[nodiscard]]
    static torch::Tensor calculateRelativeDiff(torch::Tensor const& wantedValue, torch::Tensor const& actualValue);
    [[nodiscard]]
    static torch::Tensor calculateCustomBatchLoss(DataVector const& trainedBatch, std::vector<torch::Tensor> const& predictions, TensorDataType normalizedThresholdCurrent);

  private:
    [[nodiscard]]
    static size_t getThresholdVoltageIndex(DataVector const& trainedBatch, TensorDataType normalizedThresholdCurrent);

  private:
    Network& network;
  };

}
