#include "NeuralNetwork/networkanalyzer.h"

namespace NeuralNetwork {
  NetworkAnalyzer::NetworkAnalyzer(Network& network_, DenormalizeOutputTensorFunction const& denormFunction, UncaleOutputTensorFunction const& unscaleFunction) :
    network(network_), denormalizeOutputTensor(denormFunction), unscaleOutputTensor(unscaleFunction)
  {
  }

  double NetworkAnalyzer::calculateMeanSquaredError(DataVector const& testData)
  {
    double error = 0;
    for (auto const& [x, y] : testData) {
      auto prediction = network->forward(x);
      auto loss = torch::mse_loss(prediction, y);
      error += loss.item<double>();
    }
    return error / testData.size();
  }

  std::vector<double> NetworkAnalyzer::calculateR2Score(DataVector const& testData)
  {
    if (testData.empty()) {
      return std::vector<double>();
    }

    std::vector<double> scores{};

    for (int64_t i = 0; i < testData[0].second.size(0); ++i) {
      double SQE = 0.0;
      double SQT = 0.0;

      TensorDataType y_cross = 0.0;
      for (auto const& [x, y] : testData) {
        (void) x;
        y_cross += y[i].item<TensorDataType>();
      }
      y_cross /= testData.size();

      for (auto const& [x, y] : testData) {
        auto prediction = network->forward(x);

        SQE += std::pow(prediction[i].item<TensorDataType>() - y_cross, 2.0);
        SQT += std::pow(y[i].item<TensorDataType>() - y_cross, 2.0);
      }

      scores.push_back(SQE / SQT);
    }

    return scores;
  }

  std::vector<double> NetworkAnalyzer::calculateR2ScoreAlternate(DataVector const& testData)
  {
    if (testData.empty()) {
      return std::vector<double>();
    }

    std::vector<double> scores{};

    for (int64_t i = 0; i < testData[0].second.size(0); ++i) {
      double SQR = 0.0;
      double SQT = 0.0;

      TensorDataType y_cross = 0.0;
      for (auto const& [x, y] : testData) {
        (void) x;
        y_cross += y[i].item<TensorDataType>();
      }
      y_cross /= testData.size();

      for (auto const& [x, y] : testData) {
        auto prediction = network->forward(x);
        TensorDataType yi = y[i].item<TensorDataType>();

        SQR += std::pow(yi - prediction[i].item<TensorDataType>(), 2.0);
        SQT += std::pow(yi - y_cross, 2.0);
      }

      scores.push_back(1.0 - (SQR / SQT));
    }

    return scores;
  }

  std::vector<double> NetworkAnalyzer::calculateR2ScoreAlternateDenormalized(DataVector const& testData)
  {
    if (testData.empty()) {
      return std::vector<double>();
    }

    std::vector<double> scores{};

    for (int64_t i = 0; i < testData[0].second.size(0); ++i) {
      double SQR = 0.0;
      double SQT = 0.0;

      TensorDataType y_cross = 0.0;
      for (auto const&[x, y] : testData) {
        auto yD = y.clone();
        denormalizeOutputTensor(x, yD, false);

        unscaleOutputTensor(x, yD);

        y_cross += yD[i].item<TensorDataType>();
      }
      y_cross /= testData.size();

      for (auto const&[x, y] : testData) {
        auto prediction = network->forward(x);
        auto yD = y.clone();
        denormalizeOutputTensor(x, yD, false);
        denormalizeOutputTensor(x, prediction, false);

        unscaleOutputTensor(x, yD);
        unscaleOutputTensor(x, prediction);

        TensorDataType yi = yD[i].item<TensorDataType>();

        SQR += std::pow(yi - prediction[i].item<TensorDataType>(), 2.0);
        SQT += std::pow(yi - y_cross, 2.0);
      }

      scores.push_back(1.0 - (SQR / SQT));
    }

    return scores;
  }

  torch::Tensor NetworkAnalyzer::calculateDiff(torch::Tensor const& wantedValue, torch::Tensor const& actualValue)
  {
    auto output = wantedValue.clone();

    for (int64_t i = 0; i < wantedValue.size(0); ++i) {
      output[i] -= actualValue[i].item<TensorDataType>();
    }

    return output;
  }

  torch::Tensor NetworkAnalyzer::calculateRelativeDiff(torch::Tensor const& wantedValue, torch::Tensor const& actualValue)
  {
    auto diff = calculateDiff(wantedValue, actualValue);

    for (int64_t i = 0; i < wantedValue.size(0); ++i) {
      diff[i] /= wantedValue[i].item<TensorDataType>();
    }

    return diff;
  }
}
