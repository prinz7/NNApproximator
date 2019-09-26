#include "Utilities/datanormalizator.h"

namespace Utilities {

void DataNormalizator::Normalize(DataVector& data, std::pair<MinMaxVector, MinMaxVector>& minMaxVectors, TensorDataType const newMinValue, TensorDataType const newMaxValue)
{
  if (data.empty()) return;
  auto& inputMinMax = minMaxVectors.first;
  auto& outputMinMax = minMaxVectors.second;
  auto numberOfInputNodes = data.front().first.size(0);
  auto numberOfOutputNodes = data.front().second.size(0);

  // Initialize minMaxVectors:
  inputMinMax = MinMaxVector(numberOfInputNodes, std::make_pair(std::numeric_limits<TensorDataType>::max(), std::numeric_limits<TensorDataType>::min()));
  outputMinMax = MinMaxVector(numberOfOutputNodes, std::make_pair(std::numeric_limits<TensorDataType>::max(), std::numeric_limits<TensorDataType>::min()));

  // Calculate min and max:
  for (auto const& [inputTensor, outputTensor] : data) {
    for (int64_t i = 0; i < numberOfInputNodes; ++i) {
      inputMinMax[i].first = std::min(inputMinMax[i].first, inputTensor[i].item<TensorDataType>());
      inputMinMax[i].second = std::max(inputMinMax[i].second, inputTensor[i].item<TensorDataType>());
    }

    for (int64_t i = 0; i < numberOfOutputNodes; ++i) {
      outputMinMax[i].first = std::min(outputMinMax[i].first, outputTensor[i].item<TensorDataType>());
      outputMinMax[i].second = std::max(outputMinMax[i].second, outputTensor[i].item<TensorDataType>());
    }
  }

  for (auto& [inputTensor, outputTensor] : data) {
    Normalize(inputTensor, inputMinMax, newMinValue, newMaxValue);
    Normalize(outputTensor, outputMinMax, newMinValue, newMaxValue);
  }
}

void DataNormalizator::Normalize(torch::Tensor& tensor, MinMaxVector const& minMaxVector, TensorDataType const newMinValue, TensorDataType const newMaxValue)
{
  // Normalize data -- use '(X - min) / (max - min)' to get values between 0 and 1:
  for (int64_t i = 0; i < tensor.size(0); ++i) {
    TensorDataType normalizationFactor = newMaxValue - newMinValue;
    tensor[i] = ((tensor[i].item<TensorDataType>() - minMaxVector[i].first) / (minMaxVector[i].second - minMaxVector[i].first)) * normalizationFactor + newMinValue;
  }
}

void DataNormalizator::Denormalize(torch::Tensor& tensor, MinMaxVector const& minMaxVector, TensorDataType const oldMinValue, TensorDataType const oldMaxValue, bool const limitValues)
{
  if (tensor.size(0) != static_cast<int64_t>(minMaxVector.size())) {
    return;
  }

  TensorDataType normalizationFactor = oldMaxValue - oldMinValue;
  for (size_t i = 0; i < minMaxVector.size(); ++i) {
    if (limitValues) {
      tensor[i] = std::max(oldMinValue, (std::min(oldMaxValue, tensor[i].item<TensorDataType>())));
    }
    tensor[i] = (((tensor[i] - oldMinValue) / normalizationFactor) * (minMaxVector[i].second - minMaxVector[i].first)) + minMaxVector[i].first;
  }
}

}
