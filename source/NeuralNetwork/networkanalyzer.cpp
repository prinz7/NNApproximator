#include "NeuralNetwork/networkanalyzer.h"

namespace NeuralNetwork {
  NetworkAnalyzer::NetworkAnalyzer(Network& network_) :
    network(network_)
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

  torch::Tensor NetworkAnalyzer::calculateCustomBatchLoss(DataVector const& trainedBatch, std::vector<torch::Tensor> const& predictions, TensorDataType const normalizedThresholdCurrent)
  {
    auto thresholdIndex = getThresholdVoltageIndex(trainedBatch, normalizedThresholdCurrent);

    // calculate mse for whole batch:
//    std::vector<torch::Tensor> errors{};
//    for (size_t i = 0; i < trainedBatch.size(); ++i) {
//      auto diff = predictions[i].clone();
//      diff = diff.mul(-1.0);
//      diff = diff.add(trainedBatch[i].second);
//      diff = diff.pow(2.0);
//      errors.push_back(diff);
//    }
//    auto mse = torch::mean(torch::cat(errors));
//    mse = mse.mul(10.0);

    // Ioff (relative error):
    auto ioff = predictions[0].clone();
    ioff = ioff.div(trainedBatch[0].second);
    ioff = ioff.add(-1.0);
    ioff = ioff.abs();

    // Ion (relative error):
    auto ion = predictions[predictions.size() - 1].clone();
    ion = ion.div(trainedBatch[predictions.size() - 1].second);
    ion = ion.add(-1.0);
    ion = ion.abs();

    // Vt (relative error):
    auto vt = predictions[thresholdIndex].clone();
    vt = vt.div(trainedBatch[thresholdIndex].second);
    vt = vt.add(-1.0);
    vt = vt.abs();

    // X -- slope between Ioff and Vt (relative error):
    auto thresholdVoltage = trainedBatch[thresholdIndex].first[BATCH_VGS_INDEX].item<TensorDataType>();
    auto x_spice = trainedBatch[thresholdIndex].second.add(trainedBatch[0].second.mul(-1.0)).div(thresholdVoltage);
    auto x_nn = predictions[thresholdIndex].add(predictions[0].mul(-1.0)).div(thresholdVoltage);
    auto x = x_nn.div(x_spice);
    x = x.add(-1.0);
    x = x.abs();

    auto loss = ioff.add(vt);
//    auto loss = mse.add(ioff.add(ion));
//    if (thresholdVoltage > 0) {
//      loss = loss.add(vt.add(x));
//    }
    loss = loss.div(5.0);

    std::cout << "mse: " << " ioff: " << ioff << " ion: " << ion << " vt: " << vt << " x: " << x << " loss: " << loss << std::endl;

    return loss;
  }

  size_t NetworkAnalyzer::getThresholdVoltageIndex(DataVector const& trainedBatch, TensorDataType const normalizedThresholdCurrent)
  {
    size_t thresholdIndex = 0;

    for (size_t i = 1; i < trainedBatch.size(); ++i) {
      if (std::abs(trainedBatch[i].second.item<TensorDataType>() - normalizedThresholdCurrent) < std::abs(trainedBatch[thresholdIndex].second.item<TensorDataType>() - normalizedThresholdCurrent)) {
        thresholdIndex = i;
      }
    }

    return thresholdIndex;
  }
}
