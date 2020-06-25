#pragma once

#include "NeuralNetwork/neuralnetwork.h"
#include "Utilities/constants.h"

namespace NeuralNetwork {

  using DenormalizeOutputTensorFunction = std::function<void(torch::Tensor const& inputTensor, torch::Tensor& outputTensor, bool limitValues)>;
  using UnscaleOutputTensorFunction = std::function<void(torch::Tensor const& inputTensor, torch::Tensor& outputTensor)>;

  class NetworkAnalyzer
  {
  public:
    /*
     * Constructor of the NetworkAnalyzer class.
     * Required are a reference to neural network instance and two function pointers which denormalize and unscale an output tensor.
     */
    explicit NetworkAnalyzer(Network& network, DenormalizeOutputTensorFunction denormalizationFunction, UnscaleOutputTensorFunction unscaleFunction);

  public:
    /*
     * Calculates and returns the mean square error with the given data.
     */
    [[nodiscard]]
    double calculateMeanSquaredError(DataVector const& testData);
    /*
     * Calculates R2 for the given data.
     * WARNING: this method is numerical unstable. Use calculateR2ScoreAlternate to get a more stable output.
     */
    [[nodiscard]]
    std::vector<double> calculateR2Score(DataVector const& testData);
    /*
     * Calculates R2 for the given data.
     */
    [[nodiscard]]
    std::vector<double> calculateR2ScoreAlternate(DataVector const& testData);
    /*
     * Calculates R2 for the given data after the values are denormalized.
     */
    [[nodiscard]]
    std::vector<double> calculateR2ScoreAlternateDenormalized(DataVector const& testData);

  public:
    /*
     * Calculates the difference of the two given tensors.
     */
    [[nodiscard]]
    static torch::Tensor calculateDiff(torch::Tensor const& wantedValue, torch::Tensor const& actualValue);
    /*
     * Calculates the relative difference of the two given tensors.
     */
    [[nodiscard]]
    static torch::Tensor calculateRelativeDiff(torch::Tensor const& wantedValue, torch::Tensor const& actualValue);

  private:
    Network& network;
    DenormalizeOutputTensorFunction denormalizeOutputTensor;
    UnscaleOutputTensorFunction unscaleOutputTensor;
  };

}
