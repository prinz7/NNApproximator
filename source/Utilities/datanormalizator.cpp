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

  // Normalize data -- use '(X - min) / (max - min)' to get values between 0 and 1:
  TensorDataType normalizationFactor = newMaxValue - newMinValue;
  for (auto& [inputTensor, outputTensor] : data) {
    for (int64_t i = 0; i < numberOfInputNodes; ++i) {
      inputTensor[i] = ((inputTensor[i].item<TensorDataType>() - inputMinMax[i].first) / (inputMinMax[i].second - inputMinMax[i].first)) * normalizationFactor + newMinValue;
    }

    for (int64_t i = 0; i < numberOfOutputNodes; ++i) {
      outputTensor[i] = ((outputTensor[i].item<TensorDataType>() - outputMinMax[i].first) / (outputMinMax[i].second - outputMinMax[i].first)) * normalizationFactor + newMinValue;
    }
  }
}

}
