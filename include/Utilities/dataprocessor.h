#pragma once

#include "Utilities/constants.h"

namespace Utilities {

class DataProcessor
{
public:
  /*
   * Calculates the minimum and maximum values from the given data.
   */
  static void CalculateMinMax(DataVector const& data, MinMaxValues& minMaxVectors);
  /*
   * Calculates the minimum and maximum values from the given data with respect to an active mixed scaling.
   */
  static void CalculateMixedMinMax(DataVector const& data, uint32_t thresholdVariable, TensorDataType threshold, MixedMinMaxValues& mixedMinMaxValues);
  /*
   * Parses and returns the minimum and maximum values from the given file.
   */
  [[nodiscard]]
  static std::optional<MinMaxValues> GetMinMaxFromFile(FilePath const& filePath, uint32_t numberOfInputVariables, uint32_t numberOfOutputVariables);
  /*
   * Parses and returns the minimum and maximum values from the given file with respect to an active mixed scaling.
   */
  [[nodiscard]]
  static std::optional<MixedMinMaxValues> GetMixedMinMaxFromFile(FilePath const& filePath, uint32_t numberOfInputVariables, uint32_t numberOfOutputVariables);

  /*
   * Normalizes all given tensors.
   */
  static void Normalize(DataVector& data, std::pair<MinMaxVector const, MinMaxVector const> const& minMaxVectors, TensorDataType newMinValue = -0.5, TensorDataType newMaxValue = 0.5);
  /*
   * Normalizes the given tensor.
   */
  static void Normalize(torch::Tensor& tensor, MinMaxVector const& minMaxVector, TensorDataType newMinValue = -0.5, TensorDataType newMaxValue = 0.5);
  /*
   * Denormalizes the given tensor.
   */
  static void Denormalize(torch::Tensor& tensor, MinMaxVector const& minMaxVector, TensorDataType oldMinValue = -0.5, TensorDataType oldMaxValue = 0.5, bool limitValues = false);

  /*
   * Scales the tensor logarithmically.
   */
  static void ScaleLogarithmic(torch::Tensor& data);
  /*
   * Reverts the logarithmically scaling.
   */
  static void UnscaleLogarithmic(torch::Tensor& data);

  /*
   * Scales the tensor with the square root function.
   */
  static void ScaleSquareRoot(torch::Tensor& data);
  /*
   * Reverts the scaling with the square root function.
   */
  static void UnscaleSquareRoot(torch::Tensor& data);
};

}
