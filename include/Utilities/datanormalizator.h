#pragma once

#include "Utilities/constants.h"

namespace Utilities {

class DataNormalizator
{
public:
  static void CalculateMinMax(DataVector const& data, MinMaxValues& minMaxVectors);
  static void CalculateMixedMinMax(DataVector const& data, uint32_t thresholdVariable, TensorDataType threshold, MixedMinMaxValues& mixedMinMaxValues);
  [[nodiscard]]
  static std::optional<MinMaxValues> GetMinMaxFromFile(FilePath const& filePath, uint32_t numberOfInputVariables, uint32_t numberOfOutputVariables);
  [[nodiscard]]
  static std::optional<MixedMinMaxValues> GetMixedMinMaxFromFile(FilePath const& filePath, uint32_t numberOfInputVariables, uint32_t numberOfOutputVariables);

  static void Normalize(DataVector& data, std::pair<MinMaxVector const, MinMaxVector const> const& minMaxVectors, TensorDataType newMinValue = -0.5, TensorDataType newMaxValue = 0.5);
  static void Normalize(torch::Tensor& tensor, MinMaxVector const& minMaxVector, TensorDataType newMinValue = -0.5, TensorDataType newMaxValue = 0.5);
  static void Denormalize(torch::Tensor& tensor, MinMaxVector const& minMaxVector, TensorDataType oldMinValue = -0.5, TensorDataType oldMaxValue = 0.5, bool limitValues = false);

  static void ScaleLogarithmic(torch::Tensor& data);
  static void UnscaleLogarithmic(torch::Tensor& data);

  static void ScaleSquareRoot(torch::Tensor& data);
  static void UnscaleSquareRoot(torch::Tensor& data);
};

}
