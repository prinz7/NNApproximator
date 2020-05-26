#pragma once

#include "NeuralNetwork/neuralnetwork.h"
#include "Utilities/constants.h"

namespace NeuralNetwork {

  using DenormalizeOutputTensorFunction = std::function<void(torch::Tensor const& inputTensor, torch::Tensor& outputTensor, bool limitValues)>;
  using UncaleOutputTensorFunction = std::function<void(torch::Tensor const& inputTensor, torch::Tensor& outputTensor)>;

  class NetworkAnalyzer
  {
  public:
    explicit NetworkAnalyzer(Network& network, DenormalizeOutputTensorFunction const& denormFunction, UncaleOutputTensorFunction const& unscaleFunction);

  public:
    [[nodiscard]]
    double calculateMeanSquaredError(DataVector const& testData);
    [[nodiscard]]
    std::vector<double> calculateR2Score(DataVector const& testData);
    [[nodiscard]]
    std::vector<double> calculateR2ScoreAlternate(DataVector const& testData);
    [[nodiscard]]
    std::vector<double> calculateR2ScoreAlternateDenormalized(DataVector const& testData);

  public:
    [[nodiscard]]
    static torch::Tensor calculateDiff(torch::Tensor const& wantedValue, torch::Tensor const& actualValue);
    [[nodiscard]]
    static torch::Tensor calculateRelativeDiff(torch::Tensor const& wantedValue, torch::Tensor const& actualValue);

  private:
    Network& network;
    DenormalizeOutputTensorFunction const& denormalizeOutputTensor;
    UncaleOutputTensorFunction const& unscaleOutputTensor;
  };

}
