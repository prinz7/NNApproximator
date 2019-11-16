#pragma once

#include "Utilities/constants.h"

namespace Utilities {

class DataNormalizator
{
public:
  static void CalculateMinMax(DataVector const& data, std::pair<MinMaxVector, MinMaxVector>& minMaxVectors);
  [[nodiscard]]
  static std::optional<std::pair<MinMaxVector, MinMaxVector>> GetMinMaxFromFile(FilePath const& filePath, uint32_t numberOfInputVariables, uint32_t numberOfOutputVariables);

  static void Normalize(DataVector& data, std::pair<MinMaxVector const, MinMaxVector const> const& minMaxVectors, TensorDataType newMinValue = -0.5, TensorDataType newMaxValue = 0.5);
  static void Normalize(torch::Tensor& tensor, MinMaxVector const& minMaxVector, TensorDataType newMinValue = -0.5, TensorDataType newMaxValue = 0.5);
  static void Denormalize(torch::Tensor& tensor, MinMaxVector const& minMaxVector, TensorDataType oldMinValue = -0.5, TensorDataType oldMaxValue = 0.5, bool limitValues = false);

  static void ScaleLogarithmic(torch::Tensor& data);
  static void UnscaleLogarithmic(torch::Tensor& data);

  static void ScaleSquareRoot(torch::Tensor& data);
  static void UnscaleSquareRoot(torch::Tensor& data);
};

}
